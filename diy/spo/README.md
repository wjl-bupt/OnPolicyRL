# SPO (DIY example)

**Simple Policy Optimization** (Xie & Zhang, ICML 2025) --
https://proceedings.mlr.press/v267/xie25m.html

Replaces PPO's ratio clipping with a direct KL penalty:

```
loss = -A * r + beta * KL(pi_old || pi_new),    KL_k3 = (r - 1) - log r
```

## Run it

```bash
python diy/spo/spo.py                                   # self-check, no env needed
uv run oprl train ppo --config diy/spo/config.yaml      # train
uv run oprl train ppo --config diy/spo/config.yaml --lr 1e-4   # CLI still overrides
```

## What this example demonstrates

- A published algorithm implemented in **one file that imports nothing from oprl**.
- Component hyperparameters (`beta`, `adaptive`) taken through `__init__` from the config
  spec, because a DIY component cannot add fields to `PPOConfig`. See
  [../README.md](../README.md).
- A `__main__` self-check that catches a sign error or a detached graph in a second,
  rather than after an hour of training that silently does not learn.

## Verification status

`tests/test_diy.py` asserts this implementation produces **numerically identical losses**
to the framework's built-in `spo` surrogate (with `adaptive=False`), and that it learns
CartPole. That makes it a genuine cross-check of the DIY path, not just an illustration.

Two honest caveats:

- The `adaptive` option is a KL-coefficient schedule from the original PPO paper's
  penalty variant, **not from the SPO paper**. It defaults to off so the default path
  stays faithful.
- Neither this nor the built-in `spo` is a paper reproduction. Single-seed CartPole only.
  A real comparison needs multiple seeds on MuJoCo -- see DESIGN.md §10.
