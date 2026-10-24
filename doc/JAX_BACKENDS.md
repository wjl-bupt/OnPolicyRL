# JAX environment backends

OnPolicyRL owns its PyTorch models, buffers, PPO/GA2E updates and evaluation.
Only environment dynamics come from Gymnax (MinAtar) or Brax task definitions
with `backend='mjx'` (MuJoCo XLA physics). No SB3 or upstream trainer is used.

## Usage

Set `baseline.env_params.backend: jax`, `jax_platform: cpu`, and
`baseline.algo_params.device: cuda:0`. The simulator is JIT/vmap batched on CPU;
the policy and optimizer run on CUDA. The independent JAX Python environment is
`/root/OnPolicyRL/.venv-jax`; it reuses torch through a .pth pointing to .venv.

MinAtar IDs: `MinAtar/{Asterix,Breakout,Freeway,Seaquest,SpaceInvaders}-v0`.
The `backend: jax` field selects Gymnax, despite retaining familiar task names.
Use `use_minimal_action_set: false` for the full six actions.

MJX uses explicit task IDs `MJX/halfcheetah`, `MJX/ant`, `MJX/hopper`,
`MJX/walker2d`, `MJX/humanoid`, `MJX/swimmer`, `MJX/reacher`,
`MJX/inverted_pendulum`, `MJX/inverted_double_pendulum`, `MJX/pusher`.
These are Brax task definitions with MJX physics, NOT identical Gymnasium v5
benchmarks. The task implementation supplies rewards and physical parameters.
Example: `configs/mjx_inverted_pendulum_ppo.yaml` runs a short native PPO test.
Brax emits an upstream maintenance warning; actual physics uses MuJoCo MJX.

## Protocol and validation

`make_env(..., backend='jax', device='cuda:0')` returns the tensor adapter
directly. `reset()` returns torch observations. `step()` returns observations,
reward, separate terminated/truncated/valid masks, raw completed episodes, and
the pre-reset `_transition_obs`. Every transition is real (SameStep reset).
Fresh independent PRNG keys are used for each environment and each reset.
Images are channels-first 0/1 floats, never divided by 255.

Evaluation can set `monitor.eval_num_envs`. Each vector slot receives a fixed
episode quota, preventing short-episode selection bias. The JAX campaign uses
20 slots x 5 episodes = 100 independent-environment episodes per evaluation.
The evaluation PRNG stream is independent of training; deterministic actions.

Verified: all five MinAtar shape/action/seed contracts, truncation and pre-reset
observations, natural Breakout termination against Gymnax, MJX physics and
time limits, unbiased vector evaluation. End-to-end native PPO/GA2E GPU training
passed full 512 x 128 MinAtar rollout / six update epochs; MJX PPO passed two updates.
JAX CPU simulation is validated; an accelerator-enabled JAX installation is
required before requesting `jax_platform: gpu` (not installed/tested here).

## New MinAtar comparison

`configs/jax_minatar_compare/{ppo,ga2e}_{Game}.yaml`: 50 jobs, 5 environments,
seeds 0..4, 10,000,000 requested steps (10,027,008 after whole rollouts).
PPO **gae_lambda=0.95**, asserted at launch. GA2E retains the earlier algorithm
configuration. Network, batch sizes, learning rate, gamma, entropy and clipping
schedules retain the reference configuration. No reuse of legacy checkpoints.

Freeway uses Gymnax's native 2500-step truncation; other games use a 500000-step
external cap. Gymnax dynamics and random number generation differ from the
Python MinAtar package (no extra sticky-action wrapper is added). Do not merge
the new curves with the previous Gymnasium runs or attribute differences solely
to lambda. PPO and GA2E within this campaign have identical environment settings.

Output: `runs/jax_minatar_5seed_10m/`.
Manifest/frozen source/configs/versions/status/logs:
`runs/jax_minatar_5seed_10m/campaign_20261024/`.
The supervisor bounds new jobs to one per GPU, CPUs pinned separately, and
survives SSH disconnection. Existing legacy runs remain untouched.

Upstream sources inspected:
- https://github.com/RobertTLange/gymnax
- https://mujoco.readthedocs.io/en/latest/mjx.html
- https://github.com/google/brax/tree/main/brax/envs

Validated dependency versions: JAX/jaxlib 0.11.2, Gymnax 1.0.1,
Brax 0.14.2, MuJoCo/MJX 3.17.0, Gymnasium 1.3.0, torch 2.13.0+cu126.
