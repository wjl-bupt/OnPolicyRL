"""GA2E -- per-rollout lambda selection by gradient alignment.

Ported from `/data/workspace/PGAE/src/algo/ga2e/ga2e.py` (783 lines, a `stable_baselines3.PPO`
subclass) to an `AdvantageEstimator` (DESIGN.md §4.7). It imports **only torch**, like
`../surrogates/spo.py`.

## The method

Instead of estimating the lambda that minimizes advantage-MSE (an *estimation* objective,
which is systematically biased short), pick the lambda whose PPO update direction best
matches an near-unbiased one -- a *control* objective::

    lambda* = argmax_lambda  cos( g_val , g(lambda) )  -  beta * var(lambda)

  g_val     policy gradient on a held-out fold using near-unbiased advantages
            (lambda_val ~ 0.97, not 1.0, to avoid collinearity with the candidates)
  g(lambda) policy gradient on the train fold using GAE(lambda)

Search is a two-level uniform grid in lambda space: level 1 with step 0.1, then level 2
subdividing the winner's neighbourhood. lambda* is then EMA-smoothed across iterations,
which is the estimator's only cross-iteration state.

Cost: ~19 extra backward passes per rollout in `rollout` mode. `epoch` mode multiplies that
by `num_epochs` and is documented as ablation-only below.

## Three deliberate deviations from the original

**1. Correct truncation semantics (this changes the numbers).** The original collapsed both
episode-end kinds into one `cont` mask derived from `episode_starts`::

    cont[:-1] = 1.0 - buf.episode_starts[1:]      # original -- one mask

That is the bug DESIGN.md §4.1 exists to prevent: a time-limit truncation cuts the *advantage
recursion* but must **keep** the value bootstrap, because a truncated state still has value.
This port uses the framework's two-mask form (see `_delta_cont`). On environments with time
limits -- every MuJoCo task, CartPole at 500 steps -- results therefore differ from the
original, and the original was wrong. Bit-for-bit agreement is not a goal; see
`doc/fix.md`.

**2. The yardstick is normalized like the candidates.** The original computed `g_val` from
raw advantages but each `g(lambda)` from normalized ones, so the two sides of the cosine
measured different objectives. Since PPO's actual update uses normalized advantages
(`norm_adv="minibatch"`), both are normalized here. This mostly cancels in the argmax --
`g_val` is constant across lambda -- but it makes the reported cosine meaningful rather
than merely ordinal.

**3. `epoch` mode reuses the configured surrogate.** The original hardcoded PPO's clipped
objective for the clip-aware gradient. Here the real `surrogate` object is used, so
`--surrogate spo --advantage ga2e` scores lambda under the objective actually being
optimized. This coupling is real and intended (DESIGN.md §4.7, revision 2): changing the
surrogate changes the selected lambda, so the two are **not** independent knobs. Do not
read an ablation over both axes as separable.

## Dropped from the original (42% of it)

`_train_with_spo_loss` (93 lines) -> `--surrogate spo`. `_train_with_kl_loss` (101 lines) ->
`--surrogate ppo_kl`, added alongside this file. `_train_per_epoch` (136 lines) -> the
`on_epoch_start` hook plus the shared `ppo.py` loop. The 516-line Lepski `buffer.py` was
already superseded by gradient alignment and is not ported.

## Usage

    advantage: {from: ./diy/advantages/ga2e.py:GA2E, lambda_val: 0.97}
    advantage: {from: ./diy/advantages/ga2e.py:GA2E, refresh: epoch}   # expensive

Self-check (no environment needed)::

    python diy/advantages/ga2e.py
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["GA2E"]


class GA2E:
    """Gradient-alignment lambda selection.

    Satisfies the `AdvantageEstimator` protocol structurally -- it inherits nothing, not
    even `oprl.BaseEstimator`, so every method the protocol names is visible here.
    """

    # No extra buffer fields: GA2E only re-reads reward/value, which the base schema
    # already stores (DESIGN.md §4.3).
    extra_fields: dict = {}
    extra_policy_outputs: tuple = ()

    def __init__(
        self,
        cfg=None,
        *,
        lambda_min: float = 0.1,
        lambda_val: float = 0.97,
        level2_points: int = 10,
        val_every: int = 5,
        ema_beta: float = 0.9,
        two_level: bool = True,
        refresh: str = "rollout",
        var_penalty_beta: float = 0.0,
        n_var_folds: int = 3,
        grad_chunk_size: int = 0,
        value_ref_lambda: float | None = None,
    ):
        """
        Args:
            lambda_min: lower bound of the level-1 grid.
            lambda_val: the yardstick's near-unbiased lambda. Deliberately below 1.0 so the
                held-out gradient is not collinear with the lambda=1 candidate.
            level2_points: subdivisions of the level-2 neighbourhood (points = this + 1).
            val_every: take 1 trajectory into the validation fold every `val_every`
                trajectories, so frac_val ~ 1/val_every. Splitting by **whole trajectory**
                rather than at random is what keeps the two folds independent: adjacent
                transitions share a GAE recursion, so a random split leaks.
            ema_beta: smoothing of lambda* across iterations. The only cross-iteration state.
            two_level: False runs level 1 only (cheaper; for debugging or ablation).
            refresh: "rollout" (default) selects once per rollout, while theta is still
                theta_old so ratio == 1 and the clipped surrogate gradient equals vanilla
                PG -- clean semantics, ~19 backward passes. "epoch" re-selects at every PPO
                epoch boundary, fixing the first-order approximation that one advantage set
                is reused across epochs, at ~19 * num_epochs backward passes. **Expensive;
                use for ablation only** (DESIGN.md §7 optimizes time-to-conclusion, and this
                can exceed the cost of the main update).
            var_penalty_beta: weight on the variance term. 0 falls back to pure cosine
                alignment, exactly.
            n_var_folds: train mini-folds used to estimate Var[g(lambda)]; only read when
                `var_penalty_beta > 0`.
            grad_chunk_size: >0 accumulates the gradient in chunks of this many samples, so
                peak activation memory scales with the chunk rather than the whole fold.
                Needed only for large `num_envs` on image observations. Mathematically
                equivalent to the single-pass path up to floating-point summation order.
            value_ref_lambda: lambda for the **value regression target**. None reuses the
                policy's lambda*_EMA. A number (floored at 0.9) decouples the critic target
                from lambda*, so the critic is not chasing a moving target as lambda* jitters.
        """
        if refresh not in ("rollout", "epoch"):
            raise ValueError(f"refresh must be 'rollout' or 'epoch', got {refresh!r}")

        self.gamma = float(getattr(cfg, "gamma", 0.99)) if cfg is not None else 0.99
        self.lambda_min = float(lambda_min)
        self.lambda_val = float(lambda_val)
        self.level2_points = int(level2_points)
        self.val_every = int(val_every)
        self.ema_beta = float(ema_beta)
        self.two_level = bool(two_level)
        self.refresh = refresh
        self.var_penalty_beta = float(var_penalty_beta)
        self.n_var_folds = int(n_var_folds)
        self.grad_chunk_size = int(grad_chunk_size)
        self.value_ref_lambda = (
            None if value_ref_lambda is None else max(0.9, float(value_ref_lambda))
        )

        self._ema_lam: float | None = None      # the only cross-iteration state
        self._diag: dict[str, float] = {}
        # Cached per-rollout so `on_epoch_start` does not recompute what cannot change.
        self._cache: dict | None = None

    # ------------------------------------------------------------------ #
    #  AdvantageEstimator protocol
    # ------------------------------------------------------------------ #

    def compute(self, buf, policy=None, surrogate=None, cfg=None):
        """Select lambda, then return (advantages, value_targets, diagnostics)."""
        if policy is None:
            raise ValueError(
                "GA2E needs the policy: lambda selection differentiates the policy "
                "objective. Pure-function estimators may ignore it; this one cannot."
            )
        delta, cont, valid = self._delta_cont(buf)
        folds = self._trajectory_split(buf, valid)
        self._cache = {"delta": delta, "cont": cont, "folds": folds}

        # theta is still theta_old here, so ratio == 1 and the clipped surrogate gradient
        # coincides with vanilla PG. Passing surrogate=None takes the cheaper path.
        lam_star, best = self._select_lambda(buf, policy, delta, cont, folds,
                                             surrogate=None, cfg=cfg)
        lam_use = self._ema_update(lam_star)

        advantages = self._gae(delta, cont, lam_use)

        # The critic target may use its own lambda, decoupling it from lambda*'s jitter.
        ret_lam = self.value_ref_lambda if self.value_ref_lambda is not None else lam_use
        returns = (
            advantages + buf.values if abs(ret_lam - lam_use) < 1e-9
            else self._gae(delta, cont, ret_lam) + buf.values
        )

        diag = {
            "ga2e/lambda_star": float(lam_star),
            "ga2e/lambda_used": float(lam_use),
            "ga2e/lambda_returns": float(ret_lam),
            "ga2e/score_best": float(best),
            **self._diag,
        }
        return advantages, returns.detach(), diag

    def on_epoch_start(self, epoch, buf, policy=None, surrogate=None, cfg=None) -> None:
        """`epoch` mode: re-select lambda at theta_k and rewrite the advantages.

        A no-op in `rollout` mode, and for epoch 0 in either mode -- `compute()` has just
        run at this theta, so selecting again would only pay for the same answer.

        Only `buf.advantages` is rewritten; `buf.returns` deliberately stays as `compute()`
        set it, so the critic is not regressing onto a target that moves every epoch. At
        theta_k the ratio is no longer 1, so the real surrogate is used for the gradient.
        """
        if self.refresh != "epoch" or epoch == 0 or self._cache is None:
            return
        delta, cont, folds = (self._cache["delta"], self._cache["cont"],
                              self._cache["folds"])
        lam_star, best = self._select_lambda(buf, policy, delta, cont, folds,
                                             surrogate=surrogate, cfg=cfg)
        lam_use = self._ema_update(lam_star)
        buf.advantages = self._gae(delta, cont, lam_use)
        self._diag["ga2e/lambda_epoch"] = float(lam_use)
        self._diag["ga2e/score_best_epoch"] = float(best)

    def critic_loss(self, policy, mb, cfg):
        """GA2E does not change the critic's objective, only which lambda forms its target."""
        return None

    def resolve_fields(self, obs_space, act_space) -> dict:
        return {}

    def iteration_loss(self, policy, buf, cfg):
        return None

    def state_dict(self) -> dict:
        return {"ema_lam": self._ema_lam}

    def load_state_dict(self, d: dict) -> None:
        self._ema_lam = d.get("ema_lam")

    # ------------------------------------------------------------------ #
    #  GAE with correct two-mask semantics
    # ------------------------------------------------------------------ #

    def _delta_cont(self, buf) -> tuple[Tensor, Tensor, Tensor]:
        """TD residuals and the recursion mask -- both independent of lambda.

        Computing them once is what makes a 20-point lambda sweep cheap: only the backward
        passes cost anything.

        The two masks do different jobs, and conflating them is the most common on-policy
        bug (DESIGN.md §4.1)::

            delta_t = r_t + gamma * V(s_{t+1}) * (1 - terminated_t) - V(s_t)
            adv_t   = delta_t + gamma*lam * (1 - (terminated_t | truncated_t)) * adv_{t+1}

        Only `terminated` cuts the bootstrap: a truncated state still has value.
        """
        T, N = buf.T, buf.N
        m = buf.masks
        term = m.terminated[:T].to(torch.float32)
        trunc = m.truncated[:T].to(torch.float32)
        values = buf.values
        rewards = buf["reward"][:T]

        next_values = torch.cat(
            [values[1:], buf.bootstrap_value.reshape(1, N)], dim=0
        )
        delta = rewards + self.gamma * next_values * (1.0 - term) - values
        cont = 1.0 - torch.clamp(term + trunc, max=1.0)
        return delta, cont, m.valid[:T].bool()

    def _gae(self, delta: Tensor, cont: Tensor, lam: float) -> Tensor:
        """Standard backward scan. [T, N] in, [T, N] out."""
        adv = torch.zeros_like(delta)
        acc = torch.zeros_like(delta[0])
        gl = self.gamma * lam
        for t in range(delta.shape[0] - 1, -1, -1):
            acc = delta[t] + gl * cont[t] * acc
            adv[t] = acc
        return adv

    # ------------------------------------------------------------------ #
    #  Trajectory-level train/val split
    # ------------------------------------------------------------------ #

    def _trajectory_split(self, buf, valid: Tensor) -> tuple[Tensor, Tensor]:
        """Split into flat train / val indices, by **whole trajectory**.

        Uses `buf.segments()`, which already returns contiguous `(env, start, end)` spans
        and already excludes autoreset dummy steps. Assignment is deterministic -- every
        `val_every`-th segment goes to validation -- so the split adds no RNG consumption
        and does not perturb the run's seeding.

        Why not a random split: neighbouring transitions are coupled through the GAE
        recursion, so a per-transition split leaks information between folds and the
        held-out gradient stops being independent.
        """
        T, N = buf.T, buf.N
        segs = buf.segments()
        self._diag["ga2e/n_segments"] = float(len(segs))

        val_flat: list[Tensor] = []
        tr_flat: list[Tensor] = []
        device = valid.device
        for i, (n, start, end) in enumerate(segs):
            # Flat index in C order, matching buffer.iter_minibatches / _gather.
            idx = torch.arange(start, end, device=device) * N + n
            (val_flat if i % self.val_every == 0 else tr_flat).append(idx)

        val_idx = torch.cat(val_flat) if val_flat else torch.empty(0, dtype=torch.long,
                                                                  device=device)
        tr_idx = torch.cat(tr_flat) if tr_flat else torch.empty(0, dtype=torch.long,
                                                               device=device)

        total = float(T * N)
        # Degenerate case: too few trajectories to split. Using the whole rollout for both
        # folds makes the yardstick no longer held-out, so it is reported rather than hidden
        # -- a run showing split_degenerate=1 throughout should raise rollout_len or num_envs.
        if val_idx.numel() < 8 or tr_idx.numel() < 8:
            allv = torch.nonzero(valid.reshape(-1), as_tuple=False).squeeze(-1)
            self._diag["ga2e/split_degenerate"] = 1.0
            self._diag["ga2e/val_frac"] = 1.0
            self._diag["ga2e/train_frac"] = 1.0
            return allv, allv

        self._diag["ga2e/split_degenerate"] = 0.0
        self._diag["ga2e/val_frac"] = val_idx.numel() / total
        self._diag["ga2e/train_frac"] = tr_idx.numel() / total
        return tr_idx, val_idx

    # ------------------------------------------------------------------ #
    #  lambda search
    # ------------------------------------------------------------------ #

    def _select_lambda(self, buf, policy, delta, cont, folds, surrogate=None, cfg=None):
        """Two-level grid search maximizing the alignment score. Returns (lambda*, score)."""
        tr_idx, val_idx = folds

        adv_val = self._gae(delta, cont, self.lambda_val)
        g_val = self._policy_grad(buf, policy, adv_val, val_idx, surrogate, cfg)
        g_val = g_val / (g_val.norm() + 1e-8)

        cache: dict[float, float] = {}

        def score(lam: float) -> float:
            key = round(float(lam), 3)
            if key in cache:
                return cache[key]
            adv = self._gae(delta, cont, key)
            if self.var_penalty_beta <= 0.0:
                g = self._policy_grad(buf, policy, adv, tr_idx, surrogate, cfg)
                bias, var = float(g_val @ g / (g.norm() + 1e-8)), 0.0
            else:
                # Split the train fold to estimate the gradient's dispersion across
                # sub-samples: a lambda whose direction is right but unstable is worse than
                # its cosine alone suggests.
                chunks = [c for c in torch.chunk(tr_idx, max(2, self.n_var_folds))
                          if c.numel() >= 4]
                gs = torch.stack([
                    self._policy_grad(buf, policy, adv, c, surrogate, cfg) for c in chunks
                ])
                g_mean = gs.mean(0)
                bias = float(g_val @ g_mean / (g_mean.norm() + 1e-8))
                # Relative variance: tr(Cov[g]) over ||g_mean||^2, dimensionless so it is
                # comparable across lambda and across iterations.
                tr_cov = float(((gs - g_mean) ** 2).sum(1).mean())
                var = tr_cov / (float((g_mean**2).sum()) + 1e-12)
            cache[key] = bias - self.var_penalty_beta * var
            self._last_parts = (bias, var)
            return cache[key]

        # Level 1: uniform grid of step 0.1 over [lambda_min, 1.0].
        lo1 = int(round(self.lambda_min * 10))
        grid1 = [round(0.1 * i, 2) for i in range(lo1, 11)]
        lam1 = max(grid1, key=score)

        if not self.two_level:
            self._record_parts()
            self._diag["ga2e/n_evals"] = float(len(cache))
            return lam1, cache[round(lam1, 3)]

        # Level 2: subdivide [lam1 - 0.1, lam1 + 0.1], clipped to the valid range.
        lo = max(self.lambda_min, round(lam1 - 0.1, 3))
        hi = min(1.0, round(lam1 + 0.1, 3))
        n = self.level2_points
        grid2 = [round(lo + (hi - lo) * j / n, 3) for j in range(n + 1)]
        lam_star = max(grid2, key=score)

        self._record_parts()
        self._diag["ga2e/n_evals"] = float(len(cache))
        return lam_star, cache[round(lam_star, 3)]

    def _record_parts(self) -> None:
        if self.var_penalty_beta > 0.0 and hasattr(self, "_last_parts"):
            bias, var = self._last_parts
            self._diag["ga2e/score_bias"] = float(bias)
            self._diag["ga2e/score_var"] = float(var)

    def _ema_update(self, lam_star: float) -> float:
        self._ema_lam = (
            lam_star if self._ema_lam is None
            else self.ema_beta * self._ema_lam + (1.0 - self.ema_beta) * lam_star
        )
        return float(min(max(self._ema_lam, self.lambda_min), 1.0))

    # ------------------------------------------------------------------ #
    #  policy gradient on a subset
    # ------------------------------------------------------------------ #

    def _policy_grad(self, buf, policy, advantages, idx, surrogate=None, cfg=None) -> Tensor:
        """Flattened policy gradient over the transitions in `idx`.

        `surrogate=None` uses vanilla PG, valid where ratio == 1 (theta == theta_old). Given
        a surrogate, its own objective is used, which is required once theta has moved.

        Advantages are detached (they are constants here), and gradients are cleared both
        before and after so this never contaminates the real update. **No parameter is
        modified.**
        """
        adv = advantages.reshape(-1)[idx].detach()
        if adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        params = [p for p in policy.parameters() if p.requires_grad]
        obs_flat = _flatten_time(buf.obs)
        act_flat = buf["action"][: buf.T].reshape(buf.T * buf.N, *buf.schema["action"].shape)
        logp_old_flat = buf["logprob"][: buf.T].reshape(-1)

        M = adv.numel()
        chunk = self.grad_chunk_size if self.grad_chunk_size > 0 else M

        policy.zero_grad(set_to_none=True)
        for start in range(0, M, chunk):
            sel = idx[start : start + chunk]
            obs = _index(obs_flat, sel)
            logp, _, _ = policy.evaluate(obs, act_flat[sel])
            a = adv[start : start + chunk]
            if surrogate is None:
                per = -(logp * a)
            else:
                logp_old = logp_old_flat[sel]
                ratio = (logp - logp_old).exp()
                loss, _ = surrogate(ratio, logp, logp_old, a, cfg)
                # A surrogate returns an already-reduced mean; rescale so chunked and
                # single-pass paths produce the same total.
                per = loss * a.numel()
            (per.sum() / M).backward()

        grads = [
            (p.grad.detach().reshape(-1) if p.grad is not None
             else torch.zeros(p.numel(), device=adv.device))
            for p in params
        ]
        policy.zero_grad(set_to_none=True)
        return torch.cat(grads)


# --------------------------------------------------------------------------- #
#  small helpers (dict observations are handled without a tree utility import)
# --------------------------------------------------------------------------- #


def _flatten_time(obs):
    """[T, N, ...] -> [T*N, ...] for a tensor or a dict of tensors."""
    if isinstance(obs, dict):
        return {k: v.reshape(-1, *v.shape[2:]) for k, v in obs.items()}
    return obs.reshape(-1, *obs.shape[2:])


def _index(obs, idx):
    if isinstance(obs, dict):
        return {k: v[idx] for k, v in obs.items()}
    return obs[idx]


# --------------------------------------------------------------------------- #
#  self-check
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # What to test, and what NOT to test. The tempting check -- "generate data at
    # lambda_true, assert the score peaks there" -- does not hold on synthetic noise: with
    # random rewards there is no signal tied to any lambda, and the cosine between two
    # *different* folds is then dominated by how much longer-horizon sums happen to
    # correlate, which drifts monotonically toward lambda=1. A self-check that prints a
    # claim its own numbers contradict is worse than none.
    #
    # The sharp invariant is self-alignment: computed on the **same** fold, the score at the
    # yardstick's own lambda must be exactly 1.0 and must be the argmax. That single equality
    # pins the gradient sign, the flat-index mapping, and the advantage normalization -- any
    # of which being wrong breaks it.
    import torch.nn as nn

    torch.manual_seed(0)
    T, N, D, A = 64, 4, 5, 3

    class Pol(nn.Module):
        """Minimal Policy-protocol stand-in: only `evaluate` and `parameters` are used."""

        def __init__(self):
            super().__init__()
            self.net = nn.Linear(D, A)

        def evaluate(self, obs, action, state=None, valid=None):
            logits = self.net(obs.float())
            d = torch.distributions.Categorical(logits=logits)
            return d.log_prob(action), d.entropy(), torch.zeros(len(action))

    class Buf:
        """Stand-in for RolloutBuffer, exposing only what GA2E reads."""

        def __init__(self):
            from types import SimpleNamespace

            self.T, self.N = T, N
            self.schema = {"action": SimpleNamespace(shape=())}
            self._f = {
                "reward": torch.randn(T, N),
                "action": torch.randint(0, A, (T, N)),
                "logprob": torch.full((T, N), -1.1),
            }
            self.obs = torch.randn(T, N, D)
            self.values = torch.randn(T, N) * 0.1
            self.bootstrap_value = torch.zeros(N)
            term = torch.zeros(T, N, dtype=torch.bool)
            term[31, :] = True            # one mid-rollout episode boundary per env
            self.masks = SimpleNamespace(
                terminated=term, truncated=torch.zeros(T, N, dtype=torch.bool),
                valid=torch.ones(T, N, dtype=torch.bool),
            )

        def __getitem__(self, k):
            return self._f[k]

        def segments(self):
            return [(n, s, e) for n in range(N) for s, e in ((0, 32), (32, T))]

    pol, buf = Pol(), Buf()
    grid = [round(0.1 * i, 1) for i in range(1, 11)]

    # --- 1. self-alignment: same fold on both sides, yardstick at a grid point ---
    lam_yard = 0.7
    est = GA2E(lambda_min=0.1, lambda_val=lam_yard)
    delta, cont, valid = est._delta_cont(buf)
    fold = torch.arange(T * N)

    g_ref = est._policy_grad(buf, pol, est._gae(delta, cont, lam_yard), fold)
    g_ref = g_ref / (g_ref.norm() + 1e-8)
    scores = {}
    for lam in grid:
        g = est._policy_grad(buf, pol, est._gae(delta, cont, lam), fold)
        scores[lam] = float(g_ref @ g / (g.norm() + 1e-8))

    print(f"self-alignment score vs lambda (yardstick lambda={lam_yard}, single fold):")
    for lam in grid:
        mark = "  <- yardstick" if lam == lam_yard else ""
        bar = "#" * max(0, int(40 * (scores[lam] + 1) / 2))
        print(f"  {lam:>4.1f}  {scores[lam]:+.6f}  {bar}{mark}")

    peak = max(scores, key=scores.get)
    assert peak == lam_yard, (
        f"self-alignment peaks at {peak}, not at the yardstick's own {lam_yard}: the "
        "gradient sign, the flat-index mapping or the advantage normalization is wrong"
    )
    assert abs(scores[lam_yard] - 1.0) < 1e-5, (
        f"cos(g,g) = {scores[lam_yard]:.6f}, expected exactly 1.0"
    )
    print(f"\npeak at lambda={peak}, cos={scores[peak]:.6f}  OK")

    # --- 2. the full path runs and stays inside the grid ---
    est2 = GA2E(lambda_min=0.1, val_every=2)
    adv, ret, diag = est2.compute(buf, policy=pol, surrogate=None, cfg=None)
    assert adv.shape == (T, N) and ret.shape == (T, N)
    assert est2.lambda_min <= diag["ga2e/lambda_used"] <= 1.0
    assert diag["ga2e/split_degenerate"] == 0.0, "the 8 segments should split cleanly"
    print(f"compute() -> lambda_used={diag['ga2e/lambda_used']:.3f} "
          f"n_evals={diag['ga2e/n_evals']:.0f} "
          f"val_frac={diag['ga2e/val_frac']:.2f}")

    # --- 3. EMA is the only cross-iteration state, and it round-trips ---
    est2.load_state_dict({"ema_lam": 0.42})
    assert abs(est2._ema_update(0.42) - 0.42) < 1e-9
    assert est2.state_dict()["ema_lam"] is not None
    print("state_dict round-trip  OK")

    # --- 4. the two masks must do different things (the bug this port fixes) ---
    b_trunc, b_term = Buf(), Buf()
    b_trunc.masks.terminated = torch.zeros(T, N, dtype=torch.bool)
    b_trunc.masks.truncated = torch.zeros(T, N, dtype=torch.bool)
    b_trunc.masks.truncated[31, :] = True
    d_trunc, c_trunc, _ = est._delta_cont(b_trunc)
    d_term, c_term, _ = est._delta_cont(b_term)   # b_term terminates at t=31 by default

    assert not torch.allclose(d_trunc[31], d_term[31]), (
        "terminated and truncated produced the same TD residual -- the two masks have been "
        "collapsed, which is exactly the bug this port set out to fix"
    )
    assert torch.allclose(c_trunc[31], c_term[31]), (
        "both mask kinds must cut the advantage recursion identically"
    )
    print("two masks: terminated cuts the bootstrap, truncated keeps it; "
          "both cut the recursion  OK")
    print("\nself-check passed")
