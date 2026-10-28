import dataclasses
import os
import pathlib

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import numpy as np
import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.models.pi0_config import Pi0Config
from openpi.training import config as _config
from openpi.training import sharding

from . import train


class _TinyModel(nnx.Module):
    def __init__(self, rng):
        self.linear = nnx.Linear(2048, 1024, rngs=nnx.Rngs(rng))


@dataclasses.dataclass(frozen=True)
class _TinyConfig(Pi0Config):
    def create(self, rng):
        return _TinyModel(rng)


@dataclasses.dataclass(frozen=True)
class _Weights:
    mode: str

    def load(self, shape):
        return traverse_util.unflatten_dict(
            {
                key: np.full(value.shape, 0.125, value.dtype)
                if self.mode == "full" or (self.mode == "partial" and key[-1] == "kernel")
                else value
                for key, value in traverse_util.flatten_dict(shape).items()
            }
        )


@pytest.mark.parametrize("mode", ["full", "partial", "none"])
def test_sharded_checkpoint_initialization(mode):
    if jax.device_count() < 2:
        pytest.skip("Run with XLA_FLAGS=--xla_force_host_platform_device_count=8")
    cfg = _config.TrainConfig(name="test", model=_TinyConfig(), weight_loader=_Weights(mode), ema_decay=0.999)
    rng = jax.random.key(42)
    expected, _ = train.init_train_state(cfg, rng, sharding.make_mesh(1), resume=False)
    actual, _ = train.init_train_state(cfg, rng, sharding.make_mesh(jax.device_count()), resume=False)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    kernel = actual.params["linear"]["kernel"].value
    assert kernel.addressable_shards[0].data.size == kernel.size // jax.device_count()
    assert int(actual.step) == 0


@pytest.mark.parametrize("config_name", ["debug"])
def test_train(tmp_path: pathlib.Path, config_name: str):
    config = dataclasses.replace(
        _config._CONFIGS_DICT[config_name],  # noqa: SLF001
        batch_size=2,
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="test",
        overwrite=False,
        resume=False,
        num_train_steps=2,
        log_interval=1,
    )
    train.main(config)

    # test resuming
    config = dataclasses.replace(config, resume=True, num_train_steps=4)
    train.main(config)
