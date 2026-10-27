"""Training-time RTC: clean prefix, suffix-only loss, unchanged parameter shapes."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma
from openpi.models import model as model_lib
from openpi.models import pi0
from openpi.models.pi0_config import Pi0Config


def test_per_token_time_keeps_standard_time_embeddings_identical():
    time = jnp.array([0.2, 0.7])
    base = pi0.posemb_sincos(time, 32, 0.004, 4.0)
    tokens = pi0.posemb_sincos(jnp.repeat(time[:, None], 30, axis=1), 32, 0.004, 4.0)
    np.testing.assert_array_equal(tokens, jnp.repeat(base[:, None], 30, axis=1))


def test_adaptive_norm_accepts_token_time_without_new_parameters():
    norm = gemma.RMSNorm()
    x = jax.random.normal(jax.random.key(1), (2, 30, 4))
    cond = jax.random.normal(jax.random.key(2), (2, 4))
    variables = norm.init(jax.random.key(3), x, cond)
    # Nonzero modulation makes this sensitive to incorrect token broadcasting.
    variables["params"]["Dense_0"]["kernel"] = jax.random.normal(jax.random.key(4), (4, 12))
    variables["params"]["Dense_0"]["bias"] = jax.random.normal(jax.random.key(5), (12,))
    scalar = norm.apply(variables, x, cond)
    token = norm.apply(variables, x, jnp.repeat(cond[:, None], 30, axis=1))
    for a, b in zip(scalar, token, strict=True):
        np.testing.assert_allclose(np.broadcast_to(a, b.shape), b, atol=1e-6)
    shapes = jax.tree.map(lambda p: p.shape, variables)
    assert shapes == jax.tree.map(
        lambda p: p.shape, norm.init(jax.random.key(3), x, jnp.repeat(cond[:, None], 30, axis=1))
    )


def test_latency_conditioning_keeps_clean_prefix_and_normalizes_suffix_loss(monkeypatch):
    batch, horizon, dim = 64, 30, 32
    actions = jax.random.normal(jax.random.key(9), (batch, horizon, dim))
    captured = {}

    def suffix(obs, noisy, time):
        captured.update(noisy=noisy, time=time)
        return (
            jnp.zeros((batch, horizon, 4)),
            jnp.ones((batch, horizon), bool),
            jnp.array([True] + [False] * 29),
            None,
        )

    model = SimpleNamespace(
        action_horizon=horizon,
        rtc_training_max_delay=8,
        embed_prefix=lambda obs: (jnp.zeros((batch, 1, 4)), jnp.ones((batch, 1), bool), jnp.zeros(1, bool)),
        embed_suffix=suffix,
        PaliGemma=SimpleNamespace(llm=lambda *a, **kw: ((None, jnp.zeros((batch, horizon, dim))), None)),
        action_out_proj=lambda x: x,
    )
    observation = Pi0Config(pi05=True, action_horizon=horizon).fake_obs(batch_size=batch)
    monkeypatch.setattr(model_lib, "preprocess_observation", lambda rng, obs, **kw: obs)
    rng = jax.random.key(7)
    loss = pi0.Pi0.compute_loss(model, rng, observation, actions, train=True)
    prefix = np.asarray(captured["time"] == 0)
    delays = prefix.sum(axis=1)
    assert delays.min() == 0
    assert delays.max() == 8
    assert np.all(prefix == (np.arange(horizon)[None, :] < delays[:, None]))
    np.testing.assert_array_equal(np.asarray(captured["noisy"])[prefix], np.asarray(actions)[prefix])
    assert np.any(np.abs(np.asarray(captured["noisy"])[~prefix] - np.asarray(actions)[~prefix]) > 0.1)
    np.testing.assert_array_equal(np.asarray(loss)[prefix], 0)
    noise = jax.random.normal(jax.random.split(rng, 3)[1], actions.shape)
    expected = np.asarray(jnp.mean(jnp.square(noise - actions), axis=-1))
    expected = np.sum(expected * ~prefix, axis=1) / (horizon - delays)
    np.testing.assert_allclose(np.mean(loss, axis=1), expected, rtol=2e-6)
    model.action_out_proj = lambda x: jnp.asarray(prefix)[..., None] * jnp.full_like(x, 1e6)
    np.testing.assert_array_equal(pi0.Pi0.compute_loss(model, rng, observation, actions, train=True), loss)


def test_invalid_latency_training_configuration_is_rejected():
    with pytest.raises(ValueError, match="action horizon"):
        Pi0Config(pi05=True, action_horizon=30, rtc_training_max_delay=30)
    with pytest.raises(ValueError, match="Pi05"):
        Pi0Config(pi05=False, action_horizon=30, rtc_training_max_delay=8)
