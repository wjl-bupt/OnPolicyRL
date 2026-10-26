# APO / GA2E review extensions

This change adds APO and GA2E diagnostics and extensions. It does not launch
performance experiments. Existing AAE/PAE modules and fixed-lambda results remain
available; no fixed-lambda campaign is added.

## APO action spaces

`apo` accepts a flat Box or zero-based Discrete action space. `ua_nums` must be an
integer >= 1. For Discrete(n), it must also be <= n-1; invalid values raise an error,
never silently clamp. An n=1 action space therefore cannot use APO.

For Discrete, first sample the executed action from the policy. Then draw unique
anchors from its complement, weighted by old-policy probabilities, using stable
Gumbel top-k sampling. Anchors never equal the executed action. Store original
categorical log probabilities, not the conditional proposal probabilities, because
the penalty measures the original policy ratio. This is a discrete extension, not
an assertion of identical sampling expectations to the continuous paper.

Continuous APO keeps the original sampling and density-weighted mean UARR loss.
Neither the original coefficient nor sum/mean semantics have been changed.

## GA2E actor update data

```yaml
algorithms:
  ga2e:
    train:
      update_data: candidate  # all (default) | candidate
```

**Value/critic always uses all valid rollout rows.** `update_data` affects only
policy surrogate, actor entropy and actor KL terms. Candidate mode retains full
minibatches and masks reference rows out of those actor terms. Actor advantage
normalization uses only candidate rows of each minibatch. A minibatch containing
no candidate rows performs only a value update.

GAE and returns still use complete trajectory segments; dropping reference rows
does not break the return recursion. Search, reference/candidate trajectory split,
EMA and critic target rules remain unchanged. Candidate membership is the ORIGINAL
split before the small-sample search fallback. Search can fall back to all rows,
but actor updates never silently promote reference rows to candidates. An empty
candidate set raises an error. Both rollout and epoch refresh modes are supported.

When candidate-mode actor KL early stopping fires, the remaining steps continue
value-only so the value pass covers all data. Default `all` preserves the original
stop-before-step behavior. With a shared trunk, value gradients can also change
policy features; candidate-only means actor LOSS isolation, not complete isolation
of the policy parameters from reference observations.

## Diagnostic configuration

Fields are accepted by APO and GA2E:

| Field | Default | Meaning |
|---|---:|---|
| `review_metrics` | true | Enable distribution and search diagnostics |
| `review_interval` | 1 | Distribution/parameter diagnostics every N rollout updates |
| `review_max_states` | 256 | Deterministic evenly spaced valid-state diagnostic sample |
| `review_actions` | 8 | Independent old-policy diagnostic action samples per state |
| `review_gradient_interval` | 0 | APO gradient decomposition every N updates; 0 disables |
| `review_reference_interval` | 0 | GA2E reference stability checks every N updates; 0 disables |
| `review_reference_resamples` | 4 | Whole-reference-trajectory bootstrap repetitions |
| `review_used_lambda` | false | Allow an extra gradient for the EMA-used lambda if not cached |

Independent diagnostic actions use a separate generator keyed by training seed and
update index, with replacement. They are distinct from training anchors in purpose,
but on discrete spaces their values can overlap. They do not change global RNG or
enter any training loss. Standard diagnostics are forward-only. The extra gradient
diagnostics do not update weights/optimizer state and are disabled by default.

GA2E search gradients are gradients of a LOSS. Parameter-displacement diagnostics
negate them before comparing with the ascent direction. The full PPO displacement
is compared against the initial search of that update (not a moving epoch reference).
Only actor/shared-feature parameters enter these displacement comparisons.

## Metric groups

- `review_candidates/lambda_*`: candidate cosine, unnormalized inner product,
  gradient norm, final search score and variance penalty. The selector is unchanged.
- `review_search/*`: cosine/inner-product winners, disagreement, score margin/range,
  reference norm and degeneracy, candidate count, sample/segment counts, overlap,
  segment-length distribution and rollout boundary segments.
- `review_lambda/*`: raw selection, EMA before/after, beta, actual used lambda,
  smoothing gap/change, boundary hit, and used-gradient score when available.
- `review_reference/*`: optional split-half direction agreement and whole-trajectory
  bootstrap score variability/winner agreement. Stability is NOT true-gradient accuracy.
- `review/executed/*`, `review/anchors/*`, `review/independent/*`: ratio and absolute
  log-ratio mean/std/max/P05/P50/P95/P99 and outside-clip fraction.
- `review/analytic_kl/*`, `review/entropy/*`, `review/policy_std/*`: sampled-state
  policy changes. Gaussian standard deviation metrics do not apply to categorical policies.
- `review/anchor_*`, `review/uarr_*`: old joint log density, effective coefficient,
  and weighted/unweighted per-anchor quadratic terms.
- `review/actor_parameter_delta_norm`, `review/update_vs_*`: actual update direction.
- `review/actor_update_samples`, `review/critic_update_samples`,
  `review/update_reference_overlap`: explicit actor/value data consumption.
- `review/optimizer_steps`, `review/epochs_entered`, `review/early_stop`,
  `review/actor_early_stop`: actual update work and stop reason indicators.
- `review_time/*`: rollout, estimation, update, search, optional diagnostics,
  evaluation and checkpoint times. Search is INCLUDED in estimation/update time;
  do not sum overlapping categories. Standard update time includes epoch search.
- `perf/session_wall_seconds`, `perf/session_steps`, `perf/end_to_end_sps`:
  elapsed time from training entry, including evaluation, diagnostics and saves.
  Startup/environment construction before training entry is excluded. Resume begins
  a new timed session; do not combine clocks without an explicit reconstruction.

`ga2e_search.jsonl` contains one complete candidate table per actual search, with
step/update, refresh mode, split counts and EMA before/after. Prefer this file over
mixing per-lambda scalar logs when search grids differ across epochs.

`evaluation_episodes.jsonl` stores raw completed episode returns/lengths and
termination/truncation flags. `eval/*` includes requested/actual counts, deterministic
flag, seed, raw-reward flag, median/P05/P95 and a wall timestamp for that evaluation.
Episode-level standard deviation is NOT training-seed uncertainty.

An optional programmatic `algo.set_review_probe(obs, actions, advantages,
source=..., seed=...)` accepts an independently collected current-policy batch for
one update and logs `review_probe/clipped_surrogate_gain`. It never collects extra
interactions automatically and never enters training. The caller is responsible
for independent trajectories. This measures a surrogate, NOT actual return gain.

## Offline aggregation

```bash
PYTHONPATH=src .venv-jax/bin/python -m workflow.review_report manifest.json --output runs/review_report
```

Manifest example (list EVERY expected seed, including missing/failed runs):

```json
{
  "runs": [
    {"path": "runs/example/ppo/seed1", "algorithm": "ppo", "environment": "Hopper-v5", "seed": 1, "budget_steps": 10000000, "eval_episodes": 100},
    {"path": "runs/example/ga2e/seed1", "algorithm": "ga2e", "environment": "Hopper-v5", "seed": 1, "budget_steps": 10000000, "eval_episodes": 100}
  ],
  "final_fraction": 0.1,
  "comparison_reference": "ppo",
  "tie_tolerance": 0.0,
  "thresholds": {"Hopper-v5": 2000},
  "bootstrap_seed": 0
}
```

Outputs `summary.json`, `summary.csv`, `curves.csv` from the same data. Final score
is the per-seed mean of complete evaluations in the final budget fraction, followed
by seed-level mean/std/bootstrap CI. The first checkpoint at/after the requested
budget is the terminal checkpoint; rollout overshoot remains visible. DONE plus a
complete terminal evaluation is required for a completed seed. Missing seeds are
counted, never filled or extrapolated. Duplicate seed identities and mixed budgets
within a group are errors. Non-finite/near-zero-mean CV is not interpreted as stability.

Raw AUC includes its observed domain. `equal_time` uses linear interpolation ONLY
inside the common observed wall-time interval; absent/reset clocks are excluded.
Win/tie/loss is based on complete-group means and the declared tolerance, not a
significance test. Source config/environment/manifest hashes are retained. Users
must verify protocol compatibility before assembling a manifest.

Cross-environment IQM is produced only with explicit
`normalization: {"Env": {"low": ..., "high": ..., "source": "..."}}`, full task
coverage and equal complete seed counts. There is no implicit min-max fitting to
the compared algorithms. Raw and normalized score columns are separate. This tool
does not rewrite historical data or claim the paper's existing figures are fixed.
