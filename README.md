# OnPolicyRL

A native on-policy RL research framework: self-owned PyTorch models, rollout storage, algorithm updates, experiment grids, monitoring, evaluation and checkpoints.

Algorithms: **PPO / GA2E / SPO / RVL / DAE / V-MPO**.

```bash
cd ~/OnPolicyRL
bash scripts/run_integrated.sh --list
PYTHONPATH=src .venv-jax/bin/python -m oprl.cli configs/minatar_jax.yaml --plan
PYTHONPATH=src .venv-jax/bin/python -m oprl.cli configs/minatar_jax.yaml
```

One configuration per environment family and backend declares algorithms, environments, shared settings and sparse overrides. The compiler produces the jobs consumed by training and reporting. See [the family configuration protocol](doc/FAMILY_CONFIG.md). No Stable-Baselines3 runtime dependency or copied external algorithm package is used.

- `src/architecture/`: native MLP/CNN, categorical/Gaussian heads, advantage/relative-value heads.
- `src/protocol/`, `src/memory/`: policy/data contracts, masks, rollout storage and sampling.
- `src/onpolicy/{ppo,ga2e,spo,rvl,dae,vmpo}.py`: algorithm mathematics and native updates.
- `src/workflow/`, `src/common/`: experiment orchestration, logging, evaluation, checkpoints, reports.
- `src/oprl/`: installed command entry only.

See [implementation and migration details](doc/INTEGRATION.md) and [validation](doc/VALIDATION.md). Historical design documents remain under `doc/`; the integration guide describes implemented behavior and limitations. The MinAtar configuration declares 50 benchmark jobs; MuJoCo is a small smoke configuration. Classic Control preserves the previous local CartPole settings, including its parameter sweep.


## Code revisions and algorithm extensions

See [commit-based runs, editable local configs, MinAtar sticky actions, and algorithm discovery](doc/REVISIONS_AND_ALGORITHMS.md).

See [backend configurations and wrapper contracts](doc/BACKENDS_AND_WRAPPERS.md) for supported providers, semantic differences, installation and validation.
