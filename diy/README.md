# diy/

Bring-your-own algorithms. Nothing here is imported by the framework -- these are
worked examples of implementing an algorithm **without editing `src/oprl/`**.

## How it works

Any component named in a config can point at a local file instead of a registered name:

```yaml
surrogate: {from: ./diy/spo/spo.py:SimplePolicyOptimization, beta: 1.0}
```

The referenced class **inherits nothing and needs no registration** -- it only has to
satisfy the relevant protocol (structural subtyping). The same mechanism works for
`advantage`, `value_loss`, `network` and env wrappers. Run `uv run oprl components` to
see the built-ins you are replacing.

## Adding your own hyperparameters

This is the one thing worth knowing before you start. A built-in component may read
`cfg.spo_beta`, because that field exists on `PPOConfig`. **Your component cannot add
fields to `PPOConfig`** -- a misspelled or unknown key there raises by design.

So take your parameters through `__init__` instead, and they arrive from the spec dict:

```yaml
surrogate: {from: ./diy/spo/spo.py:SimplePolicyOptimization, beta: 2.0}
#                                                            ^^^^^^^^ -> __init__(beta=2.0)
```

The component stays self-contained, and `oprl train --help` remains an accurate list of
the *framework's* hyperparameters rather than a grab-bag of every experiment's knobs.

## Examples

| Directory | Kind | What it shows |
|---|---|---|
| [`spo/`](spo/) | `policy_loss` | A published surrogate reimplemented from scratch, cross-checked against the built-in |

## Verify before you burn GPU hours

Each example ships a `__main__` self-check (`python diy/spo/spo.py`) and is covered by
`tests/test_diy.py`. An objective with a sign error or a detached graph runs to
completion without complaining -- it simply does not learn. Check the cheap things first.
