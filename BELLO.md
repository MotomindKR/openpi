# Bello RTC extensions

Based on upstream commit `215abfb217dbac7d5f1273282331b9b1866c0479`.

| Area | Change |
| --- | --- |
| JAX training | Clean action prefix, per-token flow times and suffix-only loss |
| PyTorch inference | Per-token time embeddings and adaptive normalization |
| Dataset loading | Explicit episode selection for session-based holdouts |
| Checkpoints | Save at completed optimizer-update intervals |

RTC conditioning preserves parameter shapes. The RTC license notice is in
[`LICENSE-RTC`](LICENSE-RTC).

In the upstream training environment:

```sh
JAX_PLATFORMS=cpu uv run pytest src/openpi/models/rtc_test.py
```
