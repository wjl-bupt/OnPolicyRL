"""Monitor -- report-only health + eval/checkpoint orchestration (DESIGN_v2 §2, §5.2).

The training loop calls `observe()` once per iteration and knows nothing else.
Three responsibilities live here, not in the loop:

* **anomaly detection** -- non-finite metrics (NaN/Inf) and KL explosion go to
  `log.event` (-> anomalies.jsonl + anomaly.log). Health is **report-only**:
  observe() NEVER returns a stop signal. What to do with a bad run is an
  analysis-stage decision, not the loop's.
* **evaluation** -- periodic AND final, on a FRESH independent env
  (common/evaluate), so eval never disturbs the training env's RNG/autoreset.
* **checkpointing** -- ``ckpt_count`` saves total by default (first rollout /
  evenly-spaced middles / final), into the run's model pool
  ``run_dir/model_pools/model_step-<global_step>.pt`` (one file per save, named
  by step -- never overwritten; resume picks the newest). This keeps the model
  pool tiny instead of one file every ``ckpt_interval`` updates; set
  ``ckpt_count<=0`` to fall back to the legacy periodic ``ckpt_interval``.

The final iteration is self-detected (`global_step >= total_steps`), so the loop
needs no extra line: the single existing `monitor.observe(...)` call in
`ppo.train()` is the entire L2 <-> L4 contact surface.

observe() is defensive: an eval or checkpoint failure is logged as an event and
swallowed, never raised into the loop -- losing a checkpoint is better than
losing the run. `eval_fn` / `save_fn` are injectable seams (default to the real
primitives) so the orchestration is testable without a live env.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

from common.checkpoint import save_checkpoint
from common.evaluate import evaluate


class Monitor:
    def __init__(self, policy, env_factory: Callable[[], object], *, log,
                 run_dir: str | Path, total_steps: int,
                 eval_interval: int = 10, eval_episodes: int = 10,
                 ckpt_count: int = 3, ckpt_interval: int = 20, eval_seed: int = 0,
                 kl_threshold: float = 0.5, meta: dict | None = None,
                 eval_fn: Callable = evaluate, save_fn: Callable = save_checkpoint):
        self.policy = policy
        self.env_factory = env_factory
        self.log = log
        self.run_dir = Path(run_dir)
        self.model_pool_dir = self.run_dir / "model_pools"
        self.total_steps = int(total_steps)
        self.eval_interval = int(eval_interval)
        self.eval_episodes = int(eval_episodes)
        self.ckpt_count = int(ckpt_count)
        self.ckpt_interval = int(ckpt_interval)
        self.eval_seed = int(eval_seed)
        self.kl_threshold = float(kl_threshold)
        self.meta = dict(meta or {})
        self._eval_fn = eval_fn
        self._save_fn = save_fn

        # count-mode checkpoint schedule (default): N = ckpt_count saves total =
        # first (iteration 1, right after the first rollout) + (N-2) evenly-spaced
        # middles + final. The middle target steps are fixed up front from
        # total_steps; observe() fires one when global_step first crosses it.
        n_mid = max(0, self.ckpt_count - 2)
        self._ckpt_mid_targets = (
            [self.total_steps * k / (self.ckpt_count - 1)
             for k in range(1, n_mid + 1)] if self.ckpt_count >= 2 else [])
        self._ckpt_next = 0                 # index of the next middle target to fire
        self._ckpt_saved: set[int] = set()  # global_steps already written (dedup)

    # -------------------------------- the hook --------------------------- #

    def observe(self, stats: dict, *, global_step: int, iteration: int) -> None:
        """Called once per update. Reports anomalies, then evaluates and/or
        checkpoints if this iteration is due or is the final one. Returns
        nothing -- there is no stop signal by design."""
        self._check_anomalies(stats, global_step, iteration)
        final = global_step >= self.total_steps
        if self._due(iteration, self.eval_interval, final):
            self._safe_eval(global_step, iteration)
        if self._ckpt_due(global_step, iteration, final):
            self._safe_ckpt(global_step, iteration)

    @staticmethod
    def _due(iteration: int, interval: int, final: bool) -> bool:
        if final:                       # the last iteration always evals + ckpts
            return True
        return interval > 0 and iteration % interval == 0

    def _ckpt_due(self, gs: int, iteration: int, final: bool) -> bool:
        """Count-mode checkpoint schedule (default): fire at the first iteration,
        at each evenly-spaced middle target as ``global_step`` crosses it, and at
        the final iteration -- ``ckpt_count`` files total. ``ckpt_count == 1``
        keeps only the final model; ``ckpt_count <= 0`` falls back to the legacy
        periodic ``ckpt_interval``. If one update spans several middle targets
        (coarse rollout / large N) they collapse into a single save -- fewer
        files, never more."""
        if self.ckpt_count <= 0:            # count disabled -> legacy periodic mode
            return self._due(iteration, self.ckpt_interval, final)
        if final:                           # the last iteration always checkpoints
            return True
        if self.ckpt_count == 1:            # only the final model is kept
            return False
        if iteration == 1:                  # first checkpoint, right after rollout 1
            return True
        fired = False
        while (self._ckpt_next < len(self._ckpt_mid_targets)
               and gs >= self._ckpt_mid_targets[self._ckpt_next]):
            self._ckpt_next += 1
            fired = True
        return fired

    # ------------------------------ anomalies ---------------------------- #

    def _check_anomalies(self, stats: dict, gs: int, it: int) -> None:
        for k, v in stats.items():
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if math.isnan(f) or math.isinf(f):
                self.log.event("nonfinite_metric", iteration=it, global_step=gs,
                               key=k, value=v, level="error")
        kl = stats.get("train/kl")
        if kl is not None:
            try:
                if float(kl) > self.kl_threshold:
                    self.log.event("kl_explode", iteration=it, global_step=gs,
                                   value=float(kl), threshold=self.kl_threshold)
            except (TypeError, ValueError):
                pass

    # --------------------------- eval / checkpoint ----------------------- #

    def _safe_eval(self, gs: int, it: int) -> None:
        try:
            metrics = self._eval_fn(self.policy, self.env_factory,
                                    episodes=self.eval_episodes, seed=self.eval_seed)
            for k, v in metrics.items():
                self.log.record(k, v)         # eval/* -> "last" agg, flushed next dump
        except Exception as e:  # noqa: BLE001 -- an eval must never kill training
            self.log.event("eval_failed", iteration=it, global_step=gs,
                           error=repr(e), level="error")

    def _safe_ckpt(self, gs: int, it: int) -> None:
        if gs in self._ckpt_saved:            # never write the same step twice
            return
        try:
            # model pool: one file per checkpoint, named by global_step (never
            # overwritten) -- resume picks the newest, earlier steps stay for
            # provenance / rollback.
            path = self.model_pool_dir / f"model_step-{gs}.pt"
            self._save_fn(path, self.policy,
                          global_step=gs, iteration=it, meta=self.meta)
            self._ckpt_saved.add(gs)
        except Exception as e:  # noqa: BLE001 -- losing a ckpt beats losing the run
            self.log.event("checkpoint_failed", iteration=it, global_step=gs,
                           error=repr(e), level="error")
