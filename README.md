# oprl

A lightweight, **research-workflow-first** framework for single-agent **on-policy** RL:
declare an experiment → run it with monitoring → get machine-readable results, curves and
a comparison table. The algorithm side is a pluggable space whose built-in surface is the
baseline surface (PPO clip + GAE, nothing else).

## Current state: **design phase, code cleared**

All code (`src/`, `tests/`, `config/`, `diy/`, `examples/`) was removed on 2026-09-16 by
decision of the author: the framework is being rebuilt from the design documents below,
and implementation is done by the author directly. This repository currently contains
**documentation only**; the previous implementation remains in git history
(`d26abdf` and earlier).

## The documents

| Document | Role |
|---|---|
| [`doc/algo_taxonomy.md`](doc/algo_taxonomy.md) | **Algorithm taxonomy** — on-policy branches by update mechanism (B1 PPO · B2 V-MPO · B3 TRPO reserved), each with its baseline as the branch foundation; variant lists; branch×slot availability |
| [`doc/logger_design.md`](doc/logger_design.md) | **Logger / metrics design** — SB3-referenced: prefix auto-aggregation, `metrics.jsonl` as the authoritative source, `event()` anomaly channel (report-only), `history()` sliding window; acceptance test list |
| [`doc/algo_design.md`](doc/algo_design.md) | **Algorithm-integration design (v3)** — unified component shape, `Batch`/`Rollout` contexts, `Estimate` + `Phase` slots, `UpdateRule` escape hatch; mapping table over known algorithm types; migration checklist (§10) |
| [`doc/DESIGN_v2.md`](doc/DESIGN_v2.md) | Framework v2 — "experiment lifecycle + algorithm space" positioning, the full research pipeline (declare → run → monitor → report → search → discover, interfaces reserved), budgets |
| [`doc/DESIGN.md`](doc/DESIGN.md) | v0.1 design rationale — the primitive-layer decisions (masks triple, declarative schema, Policy protocol, Logger) that still stand |
| [`doc/fix.md`](doc/fix.md) | Historical architecture-fix log (v0.1 era) |

## Rebuild entry points

1. **Algorithm layer** — implement against `doc/algo_design.md` §4–§8 (contexts, slots,
   escape hatch) following the migration checklist in §10.
2. **Lifecycle layer** — `doc/DESIGN_v2.md` §4–§8: experiment file → runner + monitor
   (report-only) + checkpoint → report trio (`results.json` + curves + summary over
   last-30%/10% and final evaluation) → space schema → discovery contract.
3. **Roadmap** — `doc/DESIGN_v2.md` §12: R1 (slimming) done as design; R2a (interface
   rebuild) → R2 (experiment + run) → R3 (report) → R4 (space) → R5 (discovery).
