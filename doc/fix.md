# Architecture fixes — 2026-08-20

Eight changes, driven by one question: **what does it actually cost to add a new algorithm?**
The audit found the answer was "five hardcoded edits", and pulling that thread surfaced three
silent correctness bugs along the way.

Everything below is measured, not asserted. Where a measurement came out against the change,
it is recorded as such (see [GA2E's numbers](#63-measured-results)).

---

## 0. Summary

| # | Change | Kind | Why it mattered |
|---|---|---|---|
| 1 | `algo` added to the component registry | Architecture | Algorithms were the **only** extension point that could not be swapped from config |
| 2 | APO dispatch by capability, not by string | **Bug** | `surrogate: {name: apo}` silently disabled UARR |
| 3 | APO anchors on π_old | **Bug** | UARR had **zero gradient** — the term did nothing at all |
| 4 | `algos/_common.py` | Duplication | 51 of `vmpo.train()`'s 85 lines were byte-identical to `ppo.train()`, and had already diverged |
| 5 | Surrogate hyperparameters moved onto components | Consistency | Two conventions for one job; `PPOConfig` grew a field per algorithm |
| 6 | `diy/` reorganized; GA2E ported | Extension | 783-line SB3 subclass → 312 lines of logic, **zero framework edits** |
| 7 | SPO switched to the quadratic-penalty form | Correctness | Reduction bug returned a `[B]` vector instead of a scalar loss |
| 8 | APO `uarr_coef="auto"` | Feature | Per-sample weighting, with three defects in the first draft fixed |

Tests: **133 → 180** (+47). All pass, including `-m slow`. `ruff` clean.

---

## 1. `algo` is now a component kind

### The problem

`registry.py` had five kinds — `advantage`, `policy_loss`, `value_loss`, `encoder`,
`env_preset` — and every one of them could be swapped from a config file by name or by
`{from: ./my.py:MyClass}`. `algo` was not among them. Adding an algorithm meant editing
**five hardcoded sites**:

```
cli.py:27    _algo_module()          if name == "vmpo": ...
cli.py:55    _cmd_configs()          ["ppo", "vmpo"]
cli.py:191   argparse choices        ["ppo", "a2c", "vmpo"]
cli.py:199   argparse choices        ["ppo", "vmpo"]
algos/__init__.py:1                  from . import ppo, vmpo
```

So DESIGN.md §4.7's cheapest-sounding claim — *"a new algorithm is just a `train()`
function"* — was in practice its most expensive. The asymmetry was stark: a new advantage
estimator needed **zero** framework edits; a new algorithm needed five.

### The fix

`src/oprl/algos/base.py` (new, 138 lines). An algorithm is a **pair**, not a single object —
a `train()` function plus the Config class describing its hyperparameters — so unlike other
kinds it is looked up as a record rather than instantiated:

```python
@algo("ppo", PPOConfig, note="...")
def train(cfg, env, policy, log=None, estimator=None): ...
```

`alias()` covers entries that share a `train()` and differ only in hyperparameters. A2C was
already *documented* as "a config, not a code file" (§4.6) but `cli.py` still special-cased
it in two places; now that is literally true:

```python
alias("a2c", of="ppo", defaults=dict(num_epochs=1, num_minibatches=1,
                                     clip_coef=inf, gae_lambda=1.0))
```

`cli.py` no longer contains the string `"vmpo"` or `"a2c"` anywhere —
`test_cli_does_not_hardcode_algorithm_names` enforces that. Precedence is unchanged and
still strict: dataclass defaults < algo defaults < YAML < CLI, unknown keys raise.

### What it buys

```bash
uv run oprl algos                                    # new: list what is registered
uv run oprl train ./diy/algos/my_algo.py:train       # new: no registration at all
```

`+40` lines in `cli.py`, which is the honest cost: dispatch-by-registry is slightly longer
than an `if/else`, and it removes the need to ever touch the file again.

---

## 2 & 3. Two bugs in APO

### 2. Dispatch by string comparison

```python
# ppo.py:217, before
if cfg.surrogate == "apo" and cfg.apo_resample:
    cfg._resampled_logratio = _resample_logratio(policy, mb)
```

The algorithm file identified a specific surrogate by **string equality on the config
value**. But dict specs are a documented, first-class form — `config/ppo.yaml` uses one for
DAE (`advantage: {name: dae, horizon: 32}`). So writing `surrogate: {name: apo}` made the
comparison false, UARR was skipped, and APO silently degraded to plain PPO. Only
`diag/apo_degraded` recorded it, and nobody reads a diagnostic they have no reason to
suspect.

**Fix**: the surrogate object declares its own needs; the algorithm asks by attribute.

```python
prepare        = getattr(surrogate, "prepare", None)
on_rollout_end = getattr(surrogate, "on_rollout_end", None)
```

`ppo.py` now contains no algorithm-specific branch at all.

### 3. UARR had no gradient — the worse bug

Found while testing the fix above. The resampling did this:

```python
d      = policy.dist(mb.obs)     # current policy
a_new  = d.sample()
lp_old = d.log_prob(a_new)       # scored by the SAME policy
lp_new, _, _ = policy.evaluate(mb.obs, a_new)
return lp_new - lp_old           # === 0, identically
```

Sampling from π_θ and scoring under π_θ gives `logratio ≡ 0`, so
`uarr = (exp(0) - 1)² = 0`, and its gradient is zero too. Verified directly:

```
logratio: max|.| = 0.0
uarr value       = 0.0
sum |d(uarr)/dθ| = 0.0
```

**APO's entire contribution was a no-op.** It passed every existing test — including
`test_surrogate_learns` — because being exactly PPO is a perfectly good way to learn
CartPole.

**Fix**: APO's anchor is the *rollout* policy. `on_rollout_end` samples the anchor actions
once per iteration while θ is still θ_old and stores `log π_old`; `prepare` re-evaluates
those same actions per minibatch. Now:

```
apo                       degraded=0  uarr_max=0.00073  nonzero=33/36
{'name': 'apo'}           degraded=0  uarr_max=0.00056  nonzero=33/36
{'name':'apo','coef':0.5} degraded=0  uarr_max=0.00044  nonzero=33/36
{'name':'apo','resample':False}  degraded=1  uarr_max=0.0  nonzero=0/36
```

The 3 zero entries per run are epoch 0, where θ = θ_old — correct by construction.

> ⚠️ **Any prior APO result was actually plain PPO.** If you have APO numbers, they need
> re-running. `test_apo_uarr_has_a_gradient` now asserts a non-zero gradient, so this
> cannot regress silently.

---

## 4. `algos/_common.py` — shared plumbing, not a base class

`vmpo.train()` was 85 lines; **51 were byte-identical to `ppo.train()`** (device, seeding,
logger, timer, estimator lookup, normalizers, buffer, iteration count, lr annealing, logging
tail). That is the CleanRL failure mode §1 rejects, one size down.

It had already caused real divergence — `vmpo` built its buffer as:

```python
extra=getattr(estimator, "extra_fields", None) or None
```

thereby ignoring both config-declared `buffer.extra` fields **and** any estimator's
`resolve_fields()`. So `oprl train vmpo --advantage dae` could not work, because DAE's
`probs` field is env-dependent and arrives only via `resolve_fields`. Sharing the setup fixed
that as a side effect, and `VMPOConfig` gained the `buffer` field it was missing.

**What is shared**: `setup()`, `begin_iteration()`, `log_iteration()`, `finish()` —
mechanical plumbing only. **What is not**: the training loop. Every helper is called
explicitly from the algorithm file, so the loop still reads top to bottom with no hidden
state and no callbacks (§2, principle 5). The rule: *plumbing is shared, anything expressing
the update rule stays in the algorithm file.*

Result: identical lines between the two `train()` bodies went **51 → 36**, and the remainder
is genuinely structural (`for it in range(...)`, `opt.step()`). `ppo.py` is 227 lines, back
under its 300-line budget (§4.8).

---

## 5. One convention for component hyperparameters

`PPOConfig` carried eight surrogate-private fields (`rollback_alpha`, `spo_beta`,
`dpo_alpha`, `dpo_beta`, `mdpo_tk`, `rpe_alpha`, `apo_uarr_coef`, `apo_resample`) while
`diy/README.md` told users the opposite:

> **Your component cannot add fields to `PPOConfig`** — take your parameters through
> `__init__`.

Two conventions for one job, and the built-in path was the one that did not scale: §5's table
lists 17 algorithms, which would have meant 30+ single-algorithm fields on a dataclass shared
by all of them.

**All eight moved onto their components.** What stays on `PPOConfig` is genuinely
framework-wide: `clip_coef`, `gamma`, `lr`, `_progress`. Surrogates still receive `cfg` and
read those from it.

```yaml
surrogate: {name: dpo, alpha: 2.0, beta: 0.6}    # was: dpo_alpha / dpo_beta on PPOConfig
```

Discoverability was the one real objection — a dataclass field shows up in `--help`. So
`oprl components` now prints each component's constructor parameters:

```
policy_loss:
  apo, dpo, mdpo, ppo, ppo_kl, ppo_rpe, spo, tr_ppo
    apo: uarr_coef=0.1, resample=True
    dpo: alpha=2.0, beta=0.6
    spo: beta=1.0
```

Two further consequences:

- `get_surrogate()` now returns a **fresh instance** per call. A surrogate may hold
  per-minibatch state (APO's anchor), so sharing one object across concurrent runs in a
  process would let them interfere.
- **New surrogate `ppo_kl`**: PPO's clip *plus* a KL penalty. Distinct from `spo`, which
  *removes* the clip. It exists because GA2E's reference had a `kl_coef` path that was
  exactly this objective; without it those runs would have no counterpart here.

> ⚠️ **Breaking change.** `PPOConfig(spo_beta=1.0)` now raises. Move such settings into the
> surrogate spec. The error names the unknown field, so failures are immediate, not silent.

---

## 6. GA2E ported — 783 lines → 312 lines of logic, zero framework edits

### 6.1 What was needed

Nothing. Every requirement was already in the `AdvantageEstimator` protocol, because
DESIGN.md §4.7 says the protocol's shape *was determined by GA2E*. Verified item by item:

| GA2E needs | Already present |
|---|---|
| the policy during estimation | `compute(buf, policy, surrogate, cfg)` |
| scoring under the current surrogate | same signature |
| re-selection at epoch boundaries | `on_epoch_start`, called explicitly at `ppo.py:186` |
| cross-iteration λ EMA | `state_dict` / `load_state_dict` |
| trajectory-level train/val split | `buf.segments()` |
| ~20 private metrics | unmatched log prefixes default to mean — zero registration |
| a separate λ for the critic target | `compute` returns `(adv, returns, diag)` separately |

Placed at `diy/advantages/ga2e.py`, loaded with
`advantage: {from: ./diy/advantages/ga2e.py:GA2E}`. It imports **only torch**.

### 6.2 Three deliberate deviations

**(a) Correct truncation semantics — this changes the numbers.** The original derived one
mask from `episode_starts`:

```python
cont[:-1] = 1.0 - buf.episode_starts[1:]      # original: terminated and truncated collapsed
```

That is precisely the bug §4.1 exists to prevent. A time-limit truncation cuts the advantage
*recursion* but must **keep** the value bootstrap, because a truncated state still has value.
The port uses the framework's two-mask form.

> **This is why bit-for-bit agreement with the original is not a goal.** On any environment
> with a time limit — every MuJoCo task, CartPole at 500 steps — the two implementations
> *must* differ, and the original was wrong. DESIGN.md §12 item 11 asked whether to require
> bit-exactness; the answer is no, and the reason is that the old numbers were biased.

**(b) The yardstick is normalized like the candidates.** The original computed `g_val` from
raw advantages but each `g(λ)` from normalized ones, so the two sides of the cosine measured
different objectives. Both are normalized now, matching what PPO actually optimizes. This
mostly cancels in the argmax (`g_val` is constant across λ) but makes the reported cosine
meaningful rather than merely ordinal.

**(c) `epoch` mode uses the configured surrogate** instead of a hardcoded PPO clip. So
`--surrogate spo --advantage ga2e` scores λ under the objective actually being optimized.
This coupling is real and intended (§4.7 revision 2): **changing the surrogate changes the
selected λ, so the two are not independent knobs.** An ablation over both axes is not
separable — the file says so at the top.

### 6.3 Measured results

CartPole-v1, 60k steps, 3 seeds (1/2/3), `ent_coef=0.01`, mean ± sd of episode return:

| Configuration | Return |
|---|---|
| GAE λ=0.95 (default) | **218.8 ± 6.0** |
| GAE λ=0.88 | 227.2 ± 36.7 |
| GAE λ=0.90 | 212.3 ± 9.4 |
| GAE λ=1.00 | 160.6 ± 15.7 |
| GAE λ=0.80 | 127.2 ± 16.5 |
| **GAE λ=0.775** | **130.4 ± 31.5** |
| **GA2E** (mean λ_used = 0.775) | **124.4 ± 46.2** |

**GA2E does not beat GAE on CartPole.** Following §5.3 — *"failing to reproduce the claimed
improvement is labelled 'not reproduced' and the result is kept"* — here is the diagnosis
rather than a tuned-away result.

The last two rows are the decisive comparison. GA2E's λ averaged 0.775 across the run; GAE
*pinned* at 0.775 scores 130.4 ± 31.5, statistically indistinguishable from GA2E's
124.4 ± 46.2. So:

- **The port is faithful.** Given the λ it selects, it performs exactly as fixed-λ GAE does
  at that λ. The implementation is not losing anything.
- **λ selection is what underperforms** on this task. The alignment criterion drifts toward
  ~0.78 while the optimum is ~0.88–0.95, and `λ*` per-iteration is very noisy (0.1 → 1.0
  swings; the EMA is doing heavy lifting). On a task this short and dense-reward, the
  held-out gradient is a weak signal.

Supporting evidence that the machinery is correct, not merely plausible:

- `test_gae_matches_the_framework_implementation` — the port's GAE equals `oprl.gae`
  **exactly** (max abs diff 0.0) at λ ∈ {0, 0.5, 0.95, 1.0}.
- The self-check's self-alignment landscape is cleanly unimodal and peaks at exactly 1.0 at
  the yardstick's own λ:

```
   0.6  +0.983579  #######################################
   0.7  +1.000000  #######################################  <- yardstick
   0.8  +0.983008  #######################################
```

- `test_compute_leaves_the_policy_untouched` — the ~19 backward passes perturb no parameter
  and leave no stale gradient.

**Conclusion**: GA2E is functional and correctly ported; it is **not validated** as an
improvement. CartPole is the wrong task to judge it on (short horizon, dense reward, and
GAE's default λ already near-optimal). The honest next step is MinAtar or MuJoCo, where the
bias/variance trade-off λ controls actually bites.

### 6.4 What was dropped

| Original | Fate |
|---|---|
| `_train_with_spo_loss` (93 lines) | `--surrogate spo` |
| `_train_with_kl_loss` (101 lines) | `--surrogate ppo_kl` (added in §5) |
| `_train_per_epoch` (136 lines) | `on_epoch_start` + the shared `ppo.py` loop |
| `_deltas_cont`, `_trajectory_split` (~45 lines) | `buf.masks` + `buf.segments()` |
| `buffer.py` (516 lines, Lepski) | Not ported — superseded by gradient alignment |
| `_log_alignment` / `_dump_align` (~70 lines) | Not ported — `.npz` landscape logging is a plotting concern |

**783 → 600 lines total, of which 312 are logic** (168 docstring lines, 124 `__main__`
self-check). Against DESIGN.md §4.7's estimate of ~200 lines: close, and the gap is the
self-check, which is worth its length — it catches a sign error in a second rather than after
an hour of training that silently does not learn.

### 6.5 `diy/` reorganized by component kind

`diy/spo/` was misleading: `spo.py` is a **policy loss**, not an algorithm. The layout now
mirrors `registry.KINDS`:

```
diy/surrogates/spo.py     policy_loss
diy/advantages/ga2e.py    advantage
diy/algos/                algo  (empty — for a whole new update rule)
```

I did **not** merge `diy/` into `src/oprl/algos/`, for two reasons. First, they hold
different things: `algos/` is L2 update rules, `diy/` is user components of *any* kind —
merging would flatten two orthogonal axes (layer vs. authorship) into one directory where
`ppo.py` and a 15-line loss function sit side by side. Second, `pyproject.toml` declares
`packages = ["src/oprl"]`, so anything outside `src/oprl/` is not in the wheel; moving
`algos/` to a repo-root directory would break `oprl.algos.ppo` for anyone who
`pip install`s it. Also note `recipes/` is already claimed by §3 for runnable configs, so it
would be a confusing name for algorithms.

With change #1 in place the directory question is largely moot anyway: config can point
anywhere, so `diy/` vs `src/` is now just "published or not".

---

## 7. SPO: the quadratic-penalty form, reduced correctly

SPO's objective was switched mid-work to the advantage-weighted quadratic form used in the
PGAE reference:

```python
loss = -mean( A*r  -  beta * |A| * (r-1)^2 / (2*eps) )
```

The first draft reduced the two terms separately:

```python
loss = -(adv * ratio).mean() + (abs_adv / (2 * eps)) * ((ratio - 1) ** 2).mean()
#                               ^^^^^^^ per-sample [B]   ^^^^^^^^^^^^^^^^ scalar
```

Multiplying a per-sample `[B]` tensor by an already-reduced scalar yields a **`[B]` vector,
not a scalar loss**. It surfaced as `RuntimeError: a Tensor with 512 elements cannot be
converted to Scalar` — but only in the test that calls `.item()` on it. In training,
`loss.backward()` on a non-scalar would have raised elsewhere, and the reduction bug also
silently discards the per-sample `|A|` weighting, which is the whole point of the form: a
transition with a large advantage should be held closer to `r = 1` than one with a small
advantage.

**Fix**: one mean over the whole expression, `eps` floored at `1e-3` as in the reference
(the coefficient is `1/(2·eps)`, so a small `clip_coef` would otherwise become an enormous
penalty). `diy/surrogates/spo.py` was updated to match, since `test_diy.py` asserts the two
are numerically identical.

Two new tests: `test_spo_loss_is_a_scalar`, and a self-check assertion that doubling `|A|`
doubles the penalty — which fails if the weighting is ever factored back out.

> Note `(r-1)²/2` is the second-order expansion of the k3 KL estimator, so this is an
> advantage-*weighted* KL penalty with coefficient `1/eps`, rather than the uniform
> coefficient the previous k3 form used. The diagnostic key changed accordingly:
> `diag/kl_penalty` → `diag/spo_penalty`. (`ppo_kl` keeps `diag/kl_penalty`; it is a real KL.)

---

## 8. APO: `uarr_coef="auto"`

A per-sample UARR weight was added on top of the §3 fix. Intent (confirmed): weight each
sample by **0.5 · π_old(anchor action | s)**, so anchoring is enforced hardest where the
rollout policy already put real probability mass on the action being constrained.

The first draft had three defects:

1. **`uarr_coef=0.0` triggered the dynamic path.** The branch was
   `if self.uarr_coef <= 0.0 or self.uarr_coef >= 1.0`, so the value that most obviously
   reads as *"penalty off"* silently turned the dynamic weighting *on*. Now `"auto"` is an
   explicit sentinel and `0.0` means off.
2. **The weight came from the wrong action.** It used `logp_old`, the logprob of the action
   actually taken in the rollout. UARR constrains the **anchor** action, so the anchor's
   probability is what should set the weight. `prepare()` now caches
   `_anchor_logp_old` alongside the logratio.
3. **`diag/apo_degraded` stopped being a flag.** It was set to `logp_old.exp().mean()`, a
   probability magnitude (~0.33 in practice), making *"is UARR active?"* unanswerable from
   the logs — and it broke the §2 regression test, which is what caught it. It is 0/1 again,
   with the magnitude reported separately as `diag/uarr_coef`.

Also: the weight is `.detach()`ed. It is a weighting, not a quantity to backpropagate
through — without that, gradient would flow into π_old's parameters through the coefficient.

```yaml
surrogate: {name: apo, uarr_coef: auto}    # per-sample, weighted by the anchor's pi_old
surrogate: {name: apo, uarr_coef: 0.1}     # uniform (default)
surrogate: {name: apo, uarr_coef: 0.0}     # off
```

Four new tests cover the auto path, the `0.0`-means-off semantics, the flag, and rejection of
a negative coefficient.

---

## 9. Verification

```
pytest -q            180 passed          (was 133)
pytest -q -m slow     13 passed
ruff check           All checks passed
```

New tests, grouped by the claim they defend:

- `tests/algos/test_registry.py` (12) — registry dispatch, alias precedence, config
  strictness preserved, a file-based algorithm training end to end, and a guard that
  `cli.py` never re-hardcodes an algorithm name.
- `tests/algos/test_ga2e.py` (17) — GAE equals `oprl.gae` exactly; the two masks are not
  collapsed; folds are disjoint and deterministic; the policy is untouched by selection;
  `epoch` mode does not move the critic target; both refresh modes train.
- `tests/algos/test_surrogates.py` (+13, now 40) — UARR active in every spec form; UARR has a
  non-zero gradient; hyperparameters come from the spec and the eight removed fields stay
  off `PPOConfig`; `get_surrogate` returns independent instances.

## 10. Migration checklist

1. **`PPOConfig(spo_beta=…)` and the other seven fields now raise.** Move them into the
   surrogate spec: `surrogate: {name: spo, beta: 1.0}`.
2. **Re-run any APO experiment.** Prior results were plain PPO (§3).
3. **Re-run any SPO experiment.** The objective changed from a uniform k3 KL penalty to the
   advantage-weighted quadratic form (§7), and the diagnostic key is now
   `diag/spo_penalty`.
4. **GA2E numbers will differ from the PGAE original** on any env with a time limit. The new
   ones are correct (§6.2a).
5. `oprl train` no longer validates the algorithm name against a fixed list — a typo now
   surfaces as `no algo 'ppp' registered; available: [...]` from the registry.

## 11. Not done

- **`config/ga2e.yaml` presets** — deliberately absent until there is a validated
  configuration to record. A preset file asserting untested hyperparameters is worse than
  none.
- **GA2E on MinAtar / MuJoCo** — the measurement that would actually settle §6.3. CartPole
  cannot.
- **Dict-valued hyperparameters still cannot be passed on the command line.** `--network
  '{"advantage_head": true}'` is rejected by `_add_config_args`, which types every field as
  `str`/`int`/`float`/`bool`. Component specs therefore have to come from a YAML file or the
  Python API. Pre-existing, unrelated to these changes, and worth a follow-up: it is the one
  remaining place where the config file is strictly more capable than the CLI.
- **`num_epochs_critic` / `_progress`** still live on `PPOConfig` though they are arguably
  component concerns. Left alone: `_progress` is framework-supplied state, and moving
  `num_epochs_critic` would mean an estimator owning part of the training loop's structure.
- **`experiment.py`'s `Sweep.algo`** is still a plain string with no registry validation, so
  a typo in `experiments.yaml` fails at subprocess launch rather than at parse time.
