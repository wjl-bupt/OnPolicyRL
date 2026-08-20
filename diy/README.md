# diy/

Bring-your-own components. Nothing here is imported by the framework -- these are worked
examples of extending oprl **without editing `src/oprl/`**.

## Layout mirrors the extension points

One directory per component kind, matching `KINDS` in `src/oprl/registry.py`. The previous
layout was one directory per example (`diy/spo/`), which hid the thing that actually
matters: `spo.py` is not an algorithm, it is a **policy loss**. Reading it as an algorithm
sets the wrong expectations about what it has to implement.

```
diy/surrogates/   policy_loss    -- swap PPO's objective (~15 lines)
diy/advantages/   advantage      -- swap advantage/critic estimation
diy/algos/        algo           -- a whole new update rule (a train() function)
```

Run `uv run oprl components` to list the built-ins you are replacing, together with the
hyperparameters each accepts. `uv run oprl algos` does the same for algorithms.

## How it works

Any component named in a config can point at a local file instead of a registered name:

```yaml
surrogate: {from: ./diy/surrogates/spo.py:SimplePolicyOptimization, beta: 1.0}
```

The referenced class **inherits nothing and needs no registration** -- it only has to
satisfy the relevant protocol (structural subtyping). The same mechanism works for
`advantage`, `value_loss`, `network`, env wrappers, and now `algo`:

```bash
uv run oprl train ./diy/algos/my_algo.py:train --env CartPole-v1
```

## Your own hyperparameters go through `__init__`

Take your parameters as constructor arguments; they arrive from the spec dict:

```yaml
surrogate: {from: ./diy/surrogates/spo.py:SimplePolicyOptimization, beta: 2.0}
#                                                                  ^^^^^^^^ -> __init__(beta=2.0)
```

**This is now the convention for built-in components too.** Previously they read
`cfg.spo_beta` from `PPOConfig` while DIY components used `__init__` -- two conventions
for one job, and `PPOConfig` grew a field for every new surrogate. Eight such fields were
moved onto their components; see [`../doc/fix.md`](../doc/fix.md).

What remains on the Config is genuinely framework-wide: `clip_coef`, `gamma`, `lr`,
`_progress`. A surrogate still receives `cfg` in `__call__` and should read those from it.
So `oprl train ppo --help` stays an accurate list of the *framework's* hyperparameters
rather than a grab-bag of every experiment's knobs.

## Examples

| File | Kind | What it shows |
|---|---|---|
| [`surrogates/spo.py`](surrogates/spo.py) | `policy_loss` | A published surrogate reimplemented from scratch, cross-checked numerically against the built-in |
| [`advantages/ga2e.py`](advantages/ga2e.py) | `advantage` | Gradient-alignment lambda selection: needs the policy, backprop, and cross-iteration state during estimation |

`ga2e.py` is the interesting one for judging the design: it is the most demanding thing the
`AdvantageEstimator` protocol was shaped for (DESIGN.md §4.7), and it plugs in **without a
single framework edit**.

## Verify before you burn GPU hours

Each example ships a `__main__` self-check (`python diy/surrogates/spo.py`) and is covered
by `tests/test_diy.py`. An objective with a sign error or a detached graph runs to
completion without complaining -- it simply does not learn. Check the cheap things first.

`ga2e.py`'s self-check is worth a special mention: it asserts the alignment score peaks at
the lambda that generated the data, which is the property the whole method rests on.
