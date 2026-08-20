---
name: minatar-env-fixes
description: Two bugs fixed to make MinAtar trainable — envs never registered, channels-last obs fed to channels-first CNN
metadata:
  type: project
---

To test on MinAtar (10M steps, CPU), two framework bugs had to be fixed (2026-08-20):

1. **MinAtar envs are never registered.** `minatar` declares a `gymnasium.envs` entry point, but gymnasium 1.x does not auto-load third-party entry points, so `gym.make("MinAtar/...")` raised `NamespaceNotFound`. Fixed in `src/oprl/envs/factory.py` by calling `minatar.gym.register_envs()` lazily inside `make_env` when the preset is `minatar` (kept inside a function so `import oprl` doesn't pull the optional dep — `tests/test_architecture.py` enforces that).

2. **Channel order is wrong.** MinAtar returns obs shaped `[H,W,C]=(10,10,4)` bool, but `CNNEncoder` assumes `[C,H,W]` — it treated the 10 rows as 10 input channels. Fixed via a new `EnvPreset.channels_first` flag (default False; True for the `minatar` preset) plus a `_ChannelsFirst` ObservationWrapper in the factory, and `CNNEncoder.forward` now skips `/255` when `x.max() <= 1.0` (bool obs would otherwise collapse to ~0).

**Why:** These are the actual MinAtar path bugs; the config preset existed but had `note: NOT reproduced yet` — nothing had ever been run against it.

**How to apply:** MinAtar now trains end-to-end. The 3 algorithms (spo / ga2e / ppo) are being run sequentially at 10M steps on `MinAtar/Breakout-v1`, CPU. Config for GA2E (which the CLI can't express via `--advantage '{...}'`) lives at `.runtmp/ga2e_minatar.yaml`; the driver is `.runtmp/run_all.sh`. Results land in `runs/{spo,ga2e,ppo}-minatar-MinAtar-Breakout-v1-seed1/`.

See [[minatar-config-status]].
