# oprl

A lightweight **single-agent, on-policy** deep RL framework. See [DESIGN.md](DESIGN.md)
for the full design rationale.

Core dependencies: `torch` + `numpy` + `gymnasium` + `pyyaml`.

> `pyyaml` is the fourth core dependency, added for `config/`. The design originally
> budgeted three -- hiding config loading behind an optional extra would mean the files
> in `config/` are unreadable by default, so the budget was broken deliberately.
> JSON configs do not need it (`config.py` imports yaml lazily).

## Install

```bash
uv sync --group dev
```

## Quick start

```bash
uv run oprl configs                                  # list presets
uv run oprl train ppo  --config classic
uv run oprl train ppo  --config mujoco --env HalfCheetah-v5 --lr 1e-4
uv run oprl train vmpo --config classic
uv run oprl sweep classic                            # a batch of experiments
uv run oprl components                               # what a config may name
```

As a library:

```python
import oprl
from oprl.algos import ppo

env = oprl.make_env("CartPole-v1", num_envs=8, seed=1)
policy = oprl.ActorCritic(env.obs_space, env.action_space, hidden=(64, 64))
cfg = ppo.PPOConfig(total_steps=100_000, num_envs=8, rollout_len=128)
ppo.train(cfg, env, policy, oprl.Logger(run_dir="runs/exp1"))
```

## Configuration

**One config file per algorithm** (`config/ppo.yaml`, `config/vmpo.yaml`), with one
section per environment family. A preset lists **only the keys that deviate from the
defaults**; everything else comes from the dataclass.

Precedence: `dataclass defaults < config section < CLI arguments`

```bash
uv run oprl train ppo --config mujoco --lr 1e-4 --num_envs 16
uv run oprl train ppo --surrogate dpo        # switching algorithms needs no config file
```

`config/ppo.yaml` is ~54 lines and fits on one screen:

```yaml
classic:            # CartPole / Acrobot / LunarLander ...
  note: measured -- CartPole-v1, 100k steps, seed=1, return ~200 (random ~20)
  total_steps: 100000
  num_epochs: 4
  ent_coef: 0.01

mujoco:             # HalfCheetah-v5 / Ant-v5 / Walker2d-v5 ...
  note: ported from CleanRL; NOT reproduced here yet
  num_envs: 1
  rollout_len: 2048
  num_minibatches: 32
  norm_obs: true
  norm_reward: true
```

Two behaviours worth knowing:

- **A misspelled hyperparameter raises with a hint** (`learning_rate` suggests `lr`).
  Silently ignoring it would let you believe a setting changed when it did not.
- **A saved resolved config replays directly**: `--config runs/xxx/config.yaml`.

Only the `classic` presets are measured on this framework. `mujoco`, `atari` and
`minatar` are starting points ported from CleanRL/SB3, marked as such in their `note`.

## Everything is assembled from configuration

Buffer fields, the network, the advantage estimator and both losses are declared in the
config -- optionally pointing at a `.py` file you wrote. No cascading code edits.

```yaml
advantage:  {from: ./my.py:MyAdvantage, lam_start: 0.8}
surrogate:  {from: ./my.py:MyPolicyLoss, beta: 0.3}
value_loss: {name: huber, delta: 2.0}
network:    {encoder: {from: ./my.py:MyEncoder, width: 32}}
buffer:
  extra:
    probs: {shape: [n_actions], dtype: float32}
```

A class referenced via `from` **inherits nothing and needs no registration** -- it only
has to satisfy the relevant protocol. `shape` accepts symbolic sizes (`n_actions`,
`obs_dim`) so configs need not hard-code environment dimensions.
Run `uv run oprl components` for the list of built-ins you are replacing.

Two places to look for working code:

- **[`diy/`](diy/)** -- bring-your-own components, one directory per extension point.
  [`diy/surrogates/spo.py`](diy/surrogates/spo.py) reimplements SPO (ICML 2025) in one
  standalone file that imports nothing from oprl, and `tests/test_diy.py` asserts it is
  **numerically identical** to the built-in `spo` surrogate.
  [`diy/advantages/ga2e.py`](diy/advantages/ga2e.py) is the harder case: gradient-alignment
  lambda selection, which needs the policy, backpropagation and cross-iteration state during
  estimation -- and plugs in with **no framework edit**.
- **[`examples/my_components.py`](examples/my_components.py)** -- one file touching all
  five extension points at once (encoder, advantage, policy loss, value loss, buffer field).

```bash
python diy/surrogates/spo.py                            # component self-check, no env needed
python diy/advantages/ga2e.py                           # ditto, prints the alignment landscape
uv run oprl train ppo --config diy/surrogates/spo.yaml  # train with it
```

### Algorithms are pluggable too

An algorithm is a registry entry like any other component, so adding one needs **no edit to
the framework**:

```bash
uv run oprl algos                                 # what is registered
uv run oprl train ./diy/algos/my_algo.py:train    # a train() function in your own file
```

See [`doc/fix.md`](doc/fix.md) for why this used to take five hardcoded edits.

## Experiment orchestration

Environments run one at a time; the seeds of one environment run in parallel with
adaptive GPU placement.

```bash
uv run oprl sweep mujoco --dry-run   # print the runs first
uv run oprl sweep mujoco
uv run oprl sweep mujoco --serial    # fully sequential, for a shared server
```

```
[oprl] 4 runs | mode=env_serial_seed_parallel | GPU=[0, 1]
[oprl] === CartPole-v1 (2 seeds in parallel) ===
  ▶ ppo-CartPole-v1-seed1 → gpu0 (pid 1913761)
  ▶ ppo-CartPole-v1-seed2 → gpu1 (pid 1913762)
```

GPU placement is **conservative admission control, not precise memory management** --
PyTorch's caching allocator makes exact prediction unrealistic. Each run goes to the card
with the most projected free memory that still fits (`per_run_mb`); when nothing fits, the
run queues rather than risking an OOM that would take down the whole batch. Child
processes are isolated with `CUDA_VISIBLE_DEVICES`. With no GPU it falls back to CPU.

## Layout

Organized into the three layers from DESIGN.md §3. **Single-responsibility modules stay
flat files; anything with multiple implementations becomes a package.**

```
src/oprl/
├─ types.py            L0  protocols (Masks/EnvAdapter/Policy), torch only
│
├─ schema.py           L1  Field / Op -- declarative buffer fields
├─ buffer.py               RolloutBuffer (GPU-resident)
├─ rollout.py              collect() -- the sampling loop shared by all algorithms
├─ registry.py             component registry and config-driven assembly
├─ config.py               Config base class and preset loading
├─ metrics.py              Timer / explained_variance
├─ logger.py               Logger + Sink (SB3-compatible API)
├─ norm.py                 obs / reward normalization
├─ tree.py                 map/index/stack over nested observations
├─ experiment.py           sweep orchestration and GPU scheduling
├─ advantages/             pluggable advantage estimators
│   ├─ base.py                 protocol + registry (no concrete algorithm)
│   ├─ gae.py                  GAE / Monte-Carlo
│   └─ dae.py                  DAE (NeurIPS 2022)  <- later: rvl.py / ga2e.py
├─ objectives/             pluggable objectives
│   ├─ ppo_family.py           PPO / TR-PPO / SPO / DPO / MDPO / RPE / APO
│   └─ value_loss.py           clipped / mse / huber
├─ nets/                   optional defaults, not a required path
│   ├─ encoders.py             MLP / CNN
│   └─ actor_critic.py         the default ActorCritic assembler
├─ envs/
│   ├─ presets.py              fixed wrapper stack per environment family
│   ├─ factory.py              make_env
│   └─ gym_vec.py              GymVecAdapter / TensorEnvAdapter
│
├─ algos/              L2  one update rule = one file
│   ├─ base.py                 algo registry: Algo record + @algo / alias
│   ├─ _common.py              setup / begin_iteration / log_iteration (plumbing only)
│   ├─ ppo.py                  (a2c is registered as an alias: PPO with other defaults)
│   └─ vmpo.py
└─ cli.py              L3  oprl train / configs / sweep / algos / components

config/                    hyperparameters, one file per algorithm
├─ ppo.yaml                default / classic / mujoco / atari / minatar / a2c
├─ vmpo.yaml               default / classic / mujoco
└─ experiments.yaml        sweep definitions

diy/                       bring-your-own components; nothing here is imported by oprl
├─ surrogates/spo.py       SPO reimplemented standalone, cross-checked against built-in
├─ advantages/ga2e.py      GA2E: gradient-alignment lambda selection
└─ algos/                  (empty) a whole new update rule goes here
```

Two architectural rules, checked automatically by `tests/test_architecture.py`:

- **L1 depends one way only and never imports `algos/`.** Otherwise the primitives get
  contaminated by an algorithm and `from oprl import gae` stops being independently usable.
- **Only torch / numpy / gymnasium at module level.** Optional backends (tensorboard,
  matplotlib, env suites) must be imported inside functions so `import oprl` stays light.

Surrogates live in `objectives/` rather than `algos/`: **they are primitives used by
algorithms, not algorithms themselves.** Putting them in `algos/` would break the
"one update rule = one file" criterion.

## Implemented

| Module | Notes |
|---|---|
| `oprl.gae` / `GAE` | GAE with the **terminated / truncated dual mask done correctly** |
| `oprl.RolloutBuffer` | GPU-resident, fields generated from a declarative `Schema` |
| `oprl.ActorCritic` | Default policy; swapping the encoder swaps the architecture |
| `oprl.Logger` | SB3-compatible (`record` / `record_mean` / `dump`) plus auto-aggregation |
| `oprl.envs` | Fixed wrapper stack per family, auto-detected from the env id |
| `oprl.experiment` | Sweeps: envs serial, seeds parallel, adaptive GPU placement |
| `algos.ppo` | PPO (a2c is a registered alias), pluggable surrogate / advantage / value loss |
| `algos.vmpo` | V-MPO (ICLR 2020): E-step plus learned Lagrangian multipliers |
| `algos.base` | Algo registry -- a new algorithm needs no framework edit |
| `objectives.ppo_family` | **8 published objectives**, ~15 lines each |
| `advantages` | Registry plus GAE and **DAE** (Direct Advantage Estimation) |

Discrete and continuous action spaces both verified (CartPole / Pendulum).

### Surrogate library

One `Surrogate` protocol buys 7 published algorithms; the same thing in CleanRL would
take 7 copy-pasted files.

| surrogate | paper | CartPole return @ 60k steps |
|---|---|---|
| `ppo` | Schulman et al. 2017 (baseline) | 201.7 |
| `tr_ppo` | Truly PPO, UAI 2019 | 198.9 |
| `spo` | Simple Policy Optimization, ICML 2025 | 206.9 |
| `dpo` | Discovered Policy Optimisation, NeurIPS 2022 | **227.3** |
| `mdpo` | Mirror Descent PO, ICLR 2022 | 188.0 |
| `ppo_rpe` | Relative Pearson Divergence, ICRA 2021 | 192.1 |
| `apo` | Anchored PO, Neural Networks 2026 | **221.4** |

A random policy scores about 20. **These are not paper reproductions** -- single seed,
CartPole only, untuned. They exist to verify each implementation actually learns.
Serious comparisons need multiple seeds on MuJoCo; see DESIGN.md §10.

### Advantage estimators

| estimator | paper | CartPole @ 60k, 3 seeds (mean ± sd) |
|---|---|---|
| `gae` | Schulman et al. 2015 (baseline) | **223.4 ± 24.9** |
| `dae` | Direct Advantage Estimation, NeurIPS 2022 | 206.0 ± 41.6 (`horizon=32`) |

DAE inverts GAE's dependency: a head predicts the advantage **directly**, and the value
function is fit to be consistent with it through a telescoping residual over each
trajectory. It therefore owns its own critic loss -- advantage and value learning are one
optimization, not two. `uv run oprl train ppo --config dae`

**DAE does not beat GAE on CartPole here.** The paper's gains are reported on Atari, which
this framework has not run yet, so treat the implementation as functional but not
validated. Shorter horizons are worse and far less stable (`h=8` scored 275 on one seed and
~50 on two others -- a spread that makes single-seed comparisons actively misleading).

Discrete actions only (the head needs one output per action). It is orthogonal to the
surrogate axis, so `--surrogate dpo --advantage dae` composes.

**LPO / Mirror Learning is not implemented**: it is a theoretical framework whose
instance uses a meta-learned neural drift, not a closed form. DPO is the closed-form
distillation of that result -- use DPO.

## Three key designs

**1. The buffer has no `done`.** Only `terminated` / `truncated` / `valid`, and no API
returns a collapsed flag. In GAE the bootstrap is cut only by `terminated` while the
recursion is cut by either -- merging them is the most widespread on-policy bug (the
reference CleanRL PPO does exactly that). Two tests in `tests/unit/test_gae.py` pin this
down.

**2. Swapping the network inherits nothing.** Algorithms depend on four protocol methods:

```python
class MyPolicy(nn.Module):          # inherits nothing from oprl
    is_recurrent = False
    def act(self, obs, state=None): ...
    def evaluate(self, obs, action, state=None, valid=None): ...
    def value(self, obs, state=None): ...
    def initial_state(self, batch): return None

ppo.train(cfg, env, MyPolicy(), log)   # works directly
```

**3. Buffer fields are declarative, with operators.**

```python
from oprl import Field, Op
buf = RolloutBuffer(T, N, obs_space, act_space, extra={
    "policy":   Field((n_act,), torch.float32, doc="DAE: full policy distribution"),
    "ep_start": Field((), torch.long, sample_op=Op.WHOLE, doc="never sliced"),
    "counter":  Field((), torch.float32, write_op=Op.ACCUMULATE),
})
print(buf.describe())    # the schema is self-documenting
```

`Op` is a **finite, predefined set** (STORE/ACCUMULATE/LAST x FLATTEN/WHOLE/SEQUENCE),
not a user callback -- the moment a schema can execute arbitrary code we are back to an
opaque config black box.

## Tests

```
tests/
├─ test_architecture.py   discipline: dependency direction, budget, protocol size, config drift
├─ test_diy.py            the diy/ examples, incl. DIY-vs-built-in numerical equivalence
├─ unit/                  primitives: test_gae.py  test_buffer.py  test_config.py
│                                     test_plugin.py  test_experiment.py
└─ algos/                 algorithms: test_train.py  test_surrogates.py
```

```bash
uv run pytest -m "not slow"   # 118 tests, ~18s
uv run pytest                 # 129 tests including real CartPole learning checks, ~110s
```

The learning tests are not smoke tests: they assert PPO, V-MPO and every surrogate beat a
random policy on CartPole, which catches algorithmic regressions rather than syntax errors.

## Not yet implemented

`SequenceSampler` (recurrent policies), RVL / GA2E estimators, `oprl.results` and
`oprl.plot` (the metrics -> archive -> figure pipeline), and real-hardware validation of
`TensorEnvAdapter` on Isaac / Brax. Roadmap in DESIGN.md §11.
