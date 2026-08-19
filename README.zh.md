# oprl

轻量级**单智能体 on-policy** 深度强化学习框架。设计文档见 [DESIGN.md](DESIGN.md)。

核心依赖四个：`torch` + `numpy` + `gymnasium` + `pyyaml`。

> `pyyaml` 是为 `config/` 加的第 4 个依赖。设计文档原定 3 个 ——
> 把配置加载藏在 optional extra 后面会让 `config/` 里的文件默认读不了，
> 所以选择破这个预算。`.json` 配置不需要它（`config.py` 里是延迟 import）。

## 安装

```bash
uv sync --group dev
```

## 快速开始

```bash
uv run oprl configs                                  # 列出预设及其说明
uv run oprl train ppo  --config classic
uv run oprl train ppo  --config mujoco --env HalfCheetah-v5 --lr 1e-4
uv run oprl train vmpo --config classic
uv run oprl train a2c  --env CartPole-v1   # A2C 是 PPO 的一组超参，不是独立代码
```

## 怎么改 PPO 超参

**一个算法一个配置文件**（`config/ppo.yaml`、`config/vmpo.yaml`），一族环境一个小节。
预设里**只写偏离默认值的项**，其余用代码默认值。

```bash
uv run oprl configs                                   # 看有哪些预设、各覆盖了什么
uv run oprl train ppo --config mujoco --env Ant-v5    # 用预设
uv run oprl train ppo --config mujoco --lr 1e-4       # 预设 + 命令行覆盖
uv run oprl train ppo --surrogate dpo                 # 换算法不需要配置文件
```

覆盖顺序：`dataclass 默认值 < config/<algo>.yaml 的小节 < CLI 参数`

`config/ppo.yaml` 全文只有 54 行，一屏能看完：

```yaml
classic:            # CartPole / Acrobot / LunarLander ...
  note: 已实测 CartPole-v1 100k 步 seed=1 → 回报 ~200（随机 ~20）
  total_steps: 100000
  num_epochs: 4
  ent_coef: 0.01

mujoco:             # HalfCheetah-v5 / Ant-v5 / Walker2d-v5 ...
  note: ⚠️ 移植自 CleanRL，未在本框架复现
  num_envs: 1
  rollout_len: 2048
  num_minibatches: 32
  norm_obs: true
  norm_reward: true
```

加自己的实验就加一个小节。**能在命令行一句话改完的，就不该建配置文件** ——
`--surrogate dpo` 不需要 `config/ppo/dpo.yaml`。

库内使用：

```python
from oprl.config import load_config
from oprl.algos.ppo import PPOConfig

cfg = load_config(PPOConfig, "mujoco", lr=1e-4, seed=3)
print(cfg.describe())   # 打印全部超参，* 标出偏离默认值的项
```

两点行为：

- **拼错超参会报错并给建议**（写 `learning_rate` → 提示 `lr`）。静默忽略会让你
  以为改了超参其实没改，是最难发现的实验错误之一。
- **落盘的 resolved config 可直接重跑**：`--config runs/xxx/config.yaml`。

⚠️ **只有 `classic` 预设是本框架实测过的**；`mujoco` / `atari` / `minatar` 是从
CleanRL/SB3 移植的起点值，`note` 里已标明未复现。

## 目录结构

按 DESIGN.md §3 的三层分层组织。**单一职责的留扁平文件，会有多个实现的才建包**：

```
src/oprl/
├─ types.py            L0  Protocol 定义（Masks/EnvAdapter/Policy），仅依赖 torch
│
├─ schema.py           L1  Field / Op —— 声明式 buffer 字段
├─ buffer.py               RolloutBuffer（GPU 常驻）
├─ rollout.py              collect() —— 所有算法共享的采样循环
├─ config.py               Config 基类
├─ metrics.py              Timer / explained_variance
├─ logger.py               Logger + Sink（SB3 API 兼容）
├─ norm.py                 Obs / Reward 归一化
├─ tree.py                 嵌套 obs 的 map/index/stack
├─ advantages/             可插拔 advantage 估计器
│   ├─ base.py                 协议 + 注册表（不含任何具体算法）
│   └─ gae.py                  GAE  ← 后续 dae.py / rvl.py / ga2e.py
├─ objectives/             可插拔策略目标
│   └─ ppo_family.py           PPO / TR-PPO / SPO / DPO / MDPO / RPE / APO
├─ nets/                   可选默认件（不是必经之路）
│   ├─ encoders.py             MLP / CNN
│   └─ actor_critic.py         默认 ActorCritic 组装器
├─ envs/
│   └─ gym_vec.py              GymVecAdapter / TensorEnvAdapter / make_env
│
├─ algos/              L2  一个更新规则 = 一个文件
│   ├─ ppo.py                  （a2c_config() 也在这里 —— A2C 是它的一组超参）
│   └─ vmpo.py
└─ cli.py              L3  oprl train / oprl configs

config/                    超参：一个算法一个文件，一族环境一个小节
├─ ppo.yaml                default / classic / mujoco / atari / minatar / a2c
└─ vmpo.yaml               default / classic / mujoco
```

**两条架构纪律，由 `tests/test_architecture.py` 自动检查**：

- **L1 单向依赖，绝不 import `algos/`** —— 否则原语被算法污染，`from oprl import gae`
  给别人项目用的独立性就没了。
- **顶层只 import torch / numpy / gymnasium** —— 可选后端（tensorboard/matplotlib/
  env 套件）必须延迟 import 到函数体内，保证 `import oprl` 不拉起它们。

`surrogates` 放在 `objectives/` 而不是 `algos/`：**它是被算法使用的原语，不是算法本身**。
放进 `algos/` 会让「一个更新规则 = 一个文件」的判据失效。

离散与连续动作均已跑通（CartPole / Pendulum）。

### Surrogate 算法库

一个 `Surrogate` 协议换来 7 个已发表算法 —— 在 CleanRL 里这需要 7 个复制粘贴的文件。

```bash
uv run oprl train ppo --env CartPole-v1 --surrogate dpo --advantage gae
```

| surrogate | 论文 | CartPole 60k 步实测回报 |
|---|---|---|
| `ppo` | Schulman et al. 2017（基线） | 201.7 |
| `tr_ppo` | Truly PPO, UAI 2019 | 198.9 |
| `spo` | Simple Policy Optimization, ICML 2025 | 206.9 |
| `dpo` | Discovered Policy Optimisation, NeurIPS 2022 | **227.3** |
| `mdpo` | Mirror Descent PO, ICLR 2022 | 188.0 |
| `ppo_rpe` | Relative Pearson Divergence, ICRA 2021 | 192.1 |
| `apo` | Anchored PO, Neural Networks 2026 | **221.4** |

随机策略约 20。**这不是论文复现**（单 seed、仅 CartPole、超参未调）—— 只用于验证
每个实现真的在学习。严肃的对照曲线需要多 seed + MuJoCo，见 DESIGN.md §10。

**LPO / Mirror Learning 未实现**：它是理论框架，其实例用元学习的神经网络 drift，
不是闭式公式。DPO 正是该结果的闭式蒸馏版 —— 用 DPO 即可。

## 三个关键设计

**1. buffer 里没有 `done`。** 只有 `terminated` / `truncated` / `valid` 三个 mask，
且没有任何 API 返回塌缩后的 `done`。GAE 里 bootstrap 只被 `terminated` 截断，
递推被两者任一截断 —— 把它们合并是最广泛的 on-policy bug（CleanRL 参考实现即如此）。
`tests/test_advantage.py` 有两个测试专门钉死这一点。

**2. 换网络架构不需要继承任何东西。** 算法只依赖 `Policy` 协议的四个方法：

```python
class MyPolicy(nn.Module):          # 不继承 oprl 的任何类
    is_recurrent = False
    def act(self, obs, state=None): ...
    def evaluate(self, obs, action, state=None, valid=None): ...
    def value(self, obs, state=None): ...
    def initial_state(self, batch): return None

ppo.train(cfg, env, MyPolicy(), log)   # 直接就能跑
```

**3. buffer 字段声明式扩展 + 算子形式，零样板代码。**

```python
from oprl import Field, Op
buf = RolloutBuffer(T, N, obs_space, act_space, extra={
    "policy":   Field((n_act,), torch.float32, doc="DAE: 完整策略分布"),
    "ep_start": Field((), torch.long, sample_op=Op.WHOLE, doc="不切分"),
    "counter":  Field((), torch.float32, write_op=Op.ACCUMULATE),
})
print(buf.describe())    # schema 自带文档，可打印可对比
```

`Op` 是**有限的预定义算子集合**（STORE/ACCUMULATE/LAST × FLATTEN/WHOLE/SEQUENCE），
不是用户回调 —— 一旦 schema 能执行任意代码，就重演了配置黑箱的错误。

**4. advantage 估计器可插拔**（参考 verl 的 `adv_estimator` 注册表）：

```python
@oprl.register("my_estimator")
class MyEstimator(oprl.BaseEstimator):
    def compute(self, buf, policy=None, surrogate=None, cfg=None):
        return adv, value_targets, {"diag/foo": 0.0}
```

协议比 verl 多三个方法：`critic_loss()`（DAE/RVL 要改 critic 学习目标）、
`on_epoch_start()`（GA2E 的 epoch 模式）、`state_dict()`（GA2E 的跨迭代 EMA）。

## 测试

```
tests/
├─ test_architecture.py   架构纪律（依赖方向、依赖预算、协议方法数、配置漂移/体积）
├─ unit/                  原语：test_gae.py  test_buffer.py  test_config.py
└─ algos/                 算法：test_train.py  test_surrogates.py
```

```bash
uv run pytest -m "not slow"   # 71 个：架构 + config + GAE 正确性 + smoke，约 4 秒
uv run pytest                 # 80 个：含 CartPole 真实学习验证，约 80 秒
```

学习能力测试不是 smoke test —— 它断言 PPO/V-MPO 在 CartPole 上必须显著超过随机策略
（随机约 20），用来拦住「算法退化」而不只是「语法错误」。

## 尚未实现

`SequenceSampler`（循环策略）、`AdvantageEstimator` 的 DAE/RVL/GA2E、`oprl.results` / `oprl.plot`（科研出图流水线）、
`TensorEnvAdapter` 的 Isaac/Brax 实机验证。路线图见 DESIGN.md §11。
