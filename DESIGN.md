# oprl — 轻量级 on-policy 深度强化学习框架设计

> 状态：设计草案 v0.1 · 2026-08-19
> 包名 `oprl`（可改：`opol` / `onrl` / `polar`）· 仅 PyTorch · 仅 on-policy · uv 管理

---

## 0. 一句话定位

**像 CleanRL 一样可以从上到下读完一个算法文件，但底层原语是库化、有测试、且把 on-policy 的经典正确性陷阱编码进类型里的。**

不是又一个 general-purpose RL 库。是一个**有意窄化到「单智能体 + on-policy」**的框架：砍掉 replay buffer、off-policy、offline、model-based、多智能体之后，抽象层数可以少一半，而这正是 SB3/Tianshou/TorchRL 复杂度的主要来源。省下的复杂度预算，全部投入到**benchmark 覆盖广度**（§6）和**可修改性**（§4.7）上。

---

## 1. 市面框架调研与对比

### 1.1 三个流派

**流派 A：单文件派（CleanRL / purejaxrl / cleanba）**

- 核心主张：一个算法 = 一个文件，读者不需要跳转，超参和逻辑全在眼前；论文复现友好。
- CleanRL 的真实价值不在代码，在 **open-rl-benchmark**——它把每个脚本的学习曲线、随机种子、硬件都存档了，这是它最难被替代的资产。
- 致命弱点：**零复用**。`ppo.py` / `ppo_continuous_action.py` / `ppo_atari.py` / `ppo_lstm.py` 之间大段重复。一个 GAE bug 要修 N 处；事实上 CleanRL 的参考 PPO **明确不 bootstrap truncated 状态**（官方 37 details 博文自己承认了），这个 bug 于是被复制到了全世界的 fork 里。
- 无法作为依赖使用。你不能 `import cleanrl`，只能 copy-paste，于是你的项目从第一天就 fork 了一份永不更新的死代码。

**流派 B：全家桶派（Stable-Baselines3 / Tianshou / skrl / RLlib）**

- SB3：文档和测试是全场最好的，RL Zoo 提供了大量已调好的超参（这是巨大的隐性资产）。但 `model.learn()` 是个黑箱大循环，想改训练流程只能靠 Callback 打洞；`BaseAlgorithm → OnPolicyAlgorithm → PPO` + `BasePolicy → ActorCriticPolicy → FeatureExtractor` 两条继承链交叉，改一个地方要读四个文件。性能不是它的目标（numpy VecEnv、CPU-GPU 来回搬）。
- Tianshou：`Trainer / Collector / Policy / Buffer` 的四分解法在概念上很漂亮，但为了同时容纳 on-policy 和 off-policy，付了明显的抽象税；`Batch` 这个自研的动态嵌套数据结构学习成本高、类型不友好、调试时看不清里面有什么。
- skrl：多后端（torch+jax）、Isaac Lab 集成是它的杀手锏，本质上是"Isaac 用户的 RL 库"，脱离 Isaac 生态后优势不明显。
- RLlib：为分布式和生产而生。新 API stack（RLModule / Learner / EnvRunner / ConnectorV2）比老的干净了，但整体依赖 Ray，起步成本极高。**只有当你真的需要多机异步时才值得**。

**流派 C：原语派（TorchRL / rlax+optax）**

- TorchRL：`TensorDict` + `TensorDictModule` + `Collector` + `objectives`，组合能力最强，官方血统，性能好。代价是极其啰嗦——一个 PPO 训练循环要装配十几个对象；`TensorDict` 是一套需要单独学的 DSL；API 在 0.x 阶段变动频繁。
- rlax/optax：只提供纯函数（loss、GAE），不管训练循环。是"库"而非"框架"，正确的心态但太薄。

### 1.2 对比表

| 维度 | CleanRL | SB3 | Tianshou | TorchRL | RLlib | skrl | purejaxrl/Stoix | **oprl（本框架）** |
|---|---|---|---|---|---|---|---|---|
| 定位 | 参考实现 | 全家桶 | 全家桶 | 原语 | 分布式生产 | Isaac 生态 | JAX 极速 | **单智能体 on-policy 专用** |
| 算法范围 | on+off | on+off | on+off | on+off | 全 | on+off+MARL | on-policy | **仅单智能体 on-policy** |
| 多智能体 | ✗ | ✗ | △ | △ | ✓✓ | ✓✓ | ✓(JaxMARL) | **✗ 明确不做（不留 agent 维度）** |
| 抽象层数 | 0 | 4–5 | 3–4 | 3 | 5+ | 3 | 1–2 | **2** |
| 代码复用 | ✗ 复制粘贴 | ✓ 继承 | ✓ 组合 | ✓✓ 组合 | ✓ | ✓ | ✗/△ | **✓ 函数复用，零继承** |
| 可读性（单算法） | ✓✓✓ | △ | △ | ✗ | ✗ | △ | ✓✓ | **✓✓** |
| 可扩展性 | ✗ | △ Callback | ✓ | ✓✓ | ✓ | ✓ | △ | **✓✓ 直接改算法文件** |
| 学习曲线 | 极低 | 低 | 中 | 高 | 很高 | 中 | 高（JAX） | **低** |
| 后端 | torch/jax | torch | torch | torch | torch/tf | torch+jax | jax | **torch only** |
| GPU 原生 env（Isaac/Brax/MJX） | △ 各写一份 | ✗ | △ | ✓ | △ | ✓✓ | ✓✓ | **✓✓ 一等公民** |
| torch.compile | ✗ | ✗ | ✗ | △ | ✗ | △ | n/a | **✓ 可选（默认关，§7.2）** |
| 循环网络（LSTM/GRU） | 单独文件 | contrib | ✓ | ✓ | ✓ | ✓ | △ | **✓ 无独立文件，换 Policy+Sampler** |
| 自定义网络架构 | 改算法文件 | △ 继承链+钩子 | ✓ | ✓✓ | △ Catalog | ✓ | 改脚本 | **✓✓ 窄 Protocol，可零继承手写** |
| 接入新算法的成本 | 复制整个文件 | 继承+覆写多钩子 | 实现 Policy 子类 | 装配多对象 | RLModule+Learner | 继承 Agent | 改脚本 | **写一个 `train()` 函数** |
| 依赖重量 | 极轻 | 轻 | 轻 | 中 | 极重(Ray) | 中 | 中 | **极轻（硬预算 3 个核心依赖）** |
| 已调好的超参 | ✓ | ✓✓ Zoo | △ | ✗ | △ | ✓ | △ | **✓ 计划内（见 §10）** |
| 复现基准曲线 | ✓✓ | ✓ | △ | △ | △ | ✓ | ✓ | **✓ 计划内（4 类各一，§6.1）** |
| benchmark 覆盖 | Atari+MuJoCo | 广（无 Isaac） | 广 | 中 | 广 | Isaac 为主 | JAX 系为主 | **✓✓ 含 MinAtar/MiniGrid/Isaac** |
| 近年 PPO 变体集成 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | △ | **✓✓ 17 个已发表，7 个各 ~15 行（§5）** |
| 日志系统 | 手写 wandb 调用 | Logger+Callback | 基础 | 基础 | Ray 集成 | 基础 | 手写 | **✓ 自研，**SB3 API 兼容**+自动聚合（§4.9）** |
| buffer 字段扩展 | 改脚本 | 继承 Buffer | 改 Batch | TensorDict | ✗ | 继承 | 改脚本 | **✓✓ 声明 `Schema`，零代码（§4.3）** |
| 科研出图流水线 | ✗（靠 wandb） | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓✓ 归档+统计正确的图（§8）** |
| 多机分布式 | ✗ | ✗ | △ | ✓ | ✓✓ | △ | ✗ | **✗ 明确不做**（单机，见 §7.3） |
| 迭代速度支持（早停/smoke/耗时归因） | ✗ | △ | ✗ | ✗ | △ | ✗ | ✗ | **✓✓ 一等目标（§7）** |

### 1.3 结论：市场空缺在哪

按上表，**"torch 原生 + 只做单智能体 on-policy + CleanRL 级可读 + 库级复用 + 对 GPU 原生 env 零拷贝 + compile 友好 + benchmark 覆盖广"** 这个组合目前没人占。

- 想要可读 → 只能 fork CleanRL，然后失去更新和测试；
- 想要库 → SB3（改不动 / 慢 / 无 Isaac）或 TorchRL（太重）；
- 想要快 → 得跳到 JAX（purejaxrl/Stoix），付静态 shape、编译时间、调试困难的代价；
- 想要 Isaac + 学术 benchmark 都能跑 → skrl 偏 Isaac，SB3 偏学术，**没有一个同时覆盖**；
- 想要**迭代快**（≠ 吞吐高）→ 没有任何框架把「早停、smoke 模式、耗时归因」当成一等功能（§7）。

**oprl 的赌注：单智能体 on-policy 算法族其实非常同构**（收集固定长度 rollout → 算 advantage → 多 epoch minibatch 更新）。同构度高到不需要继承体系，只需要一组好原语 + 每个更新规则一个扁平文件。**窄化省下的复杂度预算，直接兑换成 benchmark 覆盖广度和可修改性** —— 这是本框架的核心交易。

---

## 2. 设计原则（七条，按优先级）

1. **窄化优先（Scope discipline）**：**只做单智能体 on-policy**，且**不为 off-policy、不为多智能体预留任何接口**。任何"顺手也支持一下 DQN"、"要不要留个 replay buffer 抽象"、"advantage 模块加个 V-trace 以后好扩展"、"张量前面留个 agent 维度"的提议直接拒绝 —— 抽象层数立刻 +2，而 §1.3 的全部赌注就作废了。**两个可以被到处依赖的不变量：数据永远来自当前策略；张量第一维语义永远是 `[T, N_env]` 而非 `[T, N_env, N_agent]`。**
2. **函数优先，零继承（Composition, no inheritance）**：算法是**函数**，不是类的子类。没有 `BaseAlgorithm`、没有 `BasePolicy`。想改 PPO 就改 `algos/ppo.py`；想换网络就传一个满足 `Policy` 协议的 `nn.Module`，不需要继承我们任何东西。
3. **一个更新规则 = 一个文件**：网络结构、循环与否、观测形态、动作空间都不构成新算法，不开新文件。只有目标函数/更新机制真的不同才开（判据见 §4.6）。
4. **正确性写进类型（Correctness by construction）**：on-policy 的经典 bug（truncation bootstrap、autoreset 错位）不靠"文档提醒"，靠数据结构强制。见 §4.1。
5. **单向数据流，无隐藏状态**：训练循环在最外层且显式可见。不用 Callback 打洞；需要插入行为就直接写在循环里，或用一个极薄的 hook 列表。
6. **性能是设计约束而非事后优化，但优化目标是「拿到结论的时间」而非 SPS**：张量默认常驻 GPU；shape 全程静态以便 `torch.compile` 只编译一次；内置 timer 把耗时占比直接打出来。**不追吞吐排行榜**，判据见 §7。
7. **可复现是产品功能**：每次 run 落盘完整 resolved config + git hash + 依赖锁 + 种子；基准曲线进 CI 做回归。

**反目标（明确不做）**：**off-policy（含 replay buffer、target network、V-trace 等重要性采样校正、以及"为将来 off-policy 预留的抽象"）**、**多智能体（PettingZoo / MARL / self-play / 中心化 critic）**、offline RL、model-based、多机分布式、自研 env 套件、超参搜索框架（交给 wandb sweeps / optuna，不包进来）、多后端（不做 JAX）。

> 注 1：**异步采样（Sample Factory 式 APPO）也随之排除** —— 异步会引入 policy lag，使数据变成 off-policy，必须靠 V-trace 校正。这与"只做 on-policy"直接冲突，所以 §7 的并行方案只走同步向量化。放弃了 100k+ FPS 那条路，但按 §7 的目标（优化迭代时间而非 SPS）这个代价是可接受的 —— 换来的是全框架可以假设"buffer 里的数据严格由当前策略产生"。
>
> 注 2：**单智能体的收窄比想象中省得多。** 多智能体会强制：张量多一个 agent 维度（`[T,N,A,...]`，且 A 可变）、obs/action 变成 per-agent dict、需要 agent 间参数共享策略、需要中心化 critic（CTDE）的第二套观测通路、PettingZoo 的 AEC 顺序语义无法向量化。这些会渗透进 buffer、sampler、policy、adapter **四个**扩展点 —— 不是加一个模块，而是给每个模块都加一层。**明确不做，换来 §4.8 的行数预算能真的守住。** 若你将来需要 MARL，正确的做法是另开一个项目复用 L1 原语，而不是把这个框架撑开。

---

## 3. 分层架构

```
┌─────────────────────────────────────────────────────────┐
│ L3  entrypoints:  CLI (`oprl train` / `results` / `plot`)│  可选
│                   recipes/  可直接跑的配方脚本            │
│     results/      run 树 → 长表（§8.2）                   │
│     plot/         论文级图表（§8.3，matplotlib 在 extra） │
├─────────────────────────────────────────────────────────┤
│ L2  algos/        ppo.py + surrogates.py  (ppg.py 后续)  │  一个更新规则
│                   扁平、线性、自带 train loop、无继承      │  = 一个文件
├─────────────────────────────────────────────────────────┤
│ L1  primitives:                                          │
│     policy.py     Policy 协议 + 默认 ActorCritic（可换掉）│  有测试
│     buffer.py     RolloutBuffer（GPU 常驻，struct-of-T）  │  可单独 import
│     advantages/   AdvantageEstimator 协议：GAE/DAE/RVL/GA2E │
│     norm.py       RunningMeanStd / Obs & Reward 归一化     │
│     envs/         EnvAdapter 协议 + 3 个实现              │
│     nets/         encoders / heads / init（都只是默认件）  │
│     sampler.py    FlatSampler / SequenceSampler          │
│     logger.py     Logger（聚合）+ Sink 协议（§4.9）        │
│     config.py     dataclass config + CLI + YAML overlay   │
│     tree.py       嵌套 obs 的 map/index/stack（~100 行）  │
├─────────────────────────────────────────────────────────┤
│ L0  types.py      Protocol / TypedDict / 常量，仅依赖 torch│
└─────────────────────────────────────────────────────────┘
```

**只有两层是"框架"（L1 原语 + L2 算法）**。L1 每个模块都能被单独 import 到别人的项目里用（这是 CleanRL 做不到的）；L2 每个文件都能被单独复制出去改（这是 SB3 做不到的）。

### 3.1 目录结构

```
OnPolicyRL/
├─ pyproject.toml           # uv 管理，见 §9
├─ uv.lock                  # 提交进 git
├─ .python-version
├─ README.md  DESIGN.md
├─ src/oprl/
│  ├─ types.py
│  ├─ policy.py             # Policy 协议 + 默认 ActorCritic 组装器
│  ├─ buffer.py  schema.py  norm.py  sampler.py  tree.py
│  ├─ logger.py  sinks.py  config.py  timer.py  checkpoint.py  seeding.py
│  ├─ health.py             # 早停/诊断：KL 爆、熵塌缩、NaN、无改善（§7.1）
│  ├─ envs/    __init__.py  gym_vec.py  tensor_env.py  envpool.py
│  │            wrappers.py  presets.py  make.py    # presets/make = L3 便利层
│  ├─ nets/    encoders.py  heads.py  init.py       # 可选默认件，非必经之路
│  ├─ advantages/  gae.py  dae.py  rvl.py  ga2e.py   # 见 §4.7
│  ├─ results/   load.py  aggregate.py  cache.py     # 见 §8.2
│  ├─ plot/      curves.py  ablation.py  stats.py    # 见 §8.3
│  └─ algos/   ppo.py  surrogates.py  (ppg.py 后续)  # 见 §5
├─ configs/                 # YAML：每个 benchmark 家族一套已调好的超参
│  ├─ ppo/classic.yaml  ppo/minatar.yaml  ppo/atari.yaml
│  ├─ ppo/minigrid.yaml  ppo/mujoco.yaml  ppo/isaac.yaml
│  └─ a2c.yaml             # A2C 是 PPO 的一个配置，见 §4.6
├─ tests/
│  ├─ test_advantage.py     # GAE 对暴力参考实现
│  ├─ test_schema.py        # 字段声明 → 分配/写入/切分 正确（§4.3）
│  ├─ test_bootstrap.py     # V → 1/(1-γ) 的解析测试
│  ├─ test_autoreset.py     # 三种 autoreset 模式下 mask 正确
│  ├─ test_determinism.py   # 同种子逐位一致
│  └─ test_smoke.py         # 每个算法 1000 步不崩
├─ benchmarks/
│  ├─ run.py  reference_curves/   # 存档曲线，CI 回归用
└─ docs/
```

---

## 4. 核心抽象（逐个定稿）

### 4.1 `Transition` / mask 三元组 —— 最重要的设计决策

on-policy 最广泛的 bug 是把 `terminated` 和 `truncated` 塌缩成一个 `done`。GAE 里这两个 mask **作用不同**：

```
δ_t   = r_t + γ · V(s_{t+1}) · (1 − terminated_t) − V(s_t)      ← bootstrap 只被 terminated 截断
A_t   = δ_t + γλ · (1 − (terminated_t | truncated_t)) · A_{t+1}  ← 递推被任一种结束截断
```

外加 Gymnasium ≥1.0 默认的 **next-step autoreset**：每个 episode 边界后会插入一个**假 transition**（动作被丢弃、reward=0、obs 属于新 episode）。写进 buffer 不 mask 掉，就等于对一个从未执行的动作做策略梯度更新——不报错、不发散，只是按终止率成比例地污染梯度。

**设计决定**：buffer 里不存 `done`，存三个字段，且没有任何 API 让你拿到单一的 `done`：

```python
class Masks(NamedTuple):
    terminated: Tensor  # [T, N] bool — 真 MDP 终止，切断 bootstrap
    truncated:  Tensor  # [T, N] bool — 时限/外部截断，保留 bootstrap
    valid:      Tensor  # [T, N] bool — False = autoreset 产生的假步，从 loss 中剔除
```

`gae()` 的签名强制三者都传，缺一个是 TypeError。`valid` 由 `EnvAdapter` 根据自己声明的 autoreset 模式自动生成，算法作者不需要知道 autoreset 的存在。**把陷阱关进原语里，让算法层无从犯错**——这是 oprl 相对 CleanRL 最实质的价值。

### 4.2 `EnvAdapter` 协议 —— 让 GPU 原生 env 成为一等公民

其他框架的通病：把 Gymnasium 的 **numpy 批量接口**当成唯一抽象，于是 Isaac Lab / Brax / MJX 这些本来就返回 GPU 上 torch 张量的环境，被迫走一遍 `torch→numpy→torch`，性能论点当场作废。

```python
class EnvAdapter(Protocol):
    num_envs: int
    obs_space / action_space          # single-env 语义
    device: torch.device             # 数据实际所在设备
    def reset(self, seed: int | None) -> Obs: ...
    def step(self, action: Tensor) -> tuple[Obs, Tensor, Masks, Info]: ...
```

**契约：进出全部是 `torch.Tensor`，且已在 `self.device` 上。** 三个实现：

| 实现 | 目标 | 关键点 |
|---|---|---|
| `GymVecAdapter` | `gymnasium.vector.VectorEnv` | 读 `metadata["autoreset_mode"]`，三种模式各自正确生成 `valid` 与 bootstrap obs（`SAME_STEP` 从 `infos["final_obs"]` 稀疏取）；numpy→torch 一次拷贝 |
| `TensorEnvAdapter` | Isaac Lab / Brax / MJX / 自研 GPU env | **零拷贝直通**，`valid` 恒 True（这类 env 通常是 same-step 且自己给 final obs） |
| `EnvPoolAdapter` | envpool（Atari 高吞吐） | 异步 `send/recv` 模式，pin memory + non_blocking H2D |

`Obs` 类型 = `Tensor | dict[str, Tensor]`，嵌套操作由 `tree.py`（~100 行 map/index/stack）处理。**不引入 `tensordict` 依赖**——为了"轻量"这个卖点，核心依赖必须只有 `torch + numpy + gymnasium`。（可提供 optional adapter 给已经在用 tensordict 的人。）

### 4.3 `RolloutBuffer` + `Schema` —— 声明字段，不写代码

#### 问题：每个算法一个 buffer.py

参考实现里（`/data/workspace/PGAE/src/algo/`）有 **6 个独立的 `buffer.py`，共约 1978 行**（`ga2e` 516、`rvl` 458、`dae` 335、`pae` 273、`pgae` 191、`eapo` 134）。它们的差异其实只是**多存了几个字段**：DAE 要 `policies`/`start_indices`/`end_indices`，PAE 要 `train_mask`/`episode_starts`。**结构完全同构，却各自复制了一遍分配、写入、索引、минibatch 切分的逻辑。** 这是本框架要消除的最典型的重复。

#### 方案：字段用声明式 schema，buffer 自动生成

用户/算法**只声明字段的 shape 与 dtype**，分配、写入、GPU 常驻、minibatch 切分全部自动：

```python
@dataclass(frozen=True)
class Field:
    shape: tuple[int, ...] = ()          # 单个 env 单个 step 的形状，不含 [T, N]
    dtype: torch.dtype = torch.float32
    # 时间轴语义：多存一格用于 bootstrap（value 需要 V(s_T)）
    extra_step: bool = False
    # minibatch 是否切分此字段；False = 整体传递（如 episode 边界索引）
    per_sample: bool = True

Schema = dict[str, Field]
```

**核心字段由 `EnvAdapter` 自动推导，用户零配置**：

```python
def base_schema(env: EnvAdapter) -> Schema:
    """obs/action 的 shape 和 dtype 从 space 推出来，不需要用户告诉我们。"""
    return {
        "obs":        Field(env.obs_space.shape, torch.float32),
        "action":     Field(env.action_space.shape, action_dtype(env.action_space)),
        "logprob":    Field((), torch.float32),
        "reward":     Field((), torch.float32),
        "value":      Field((), torch.float32, extra_step=True),   # 需要 V(s_T)
        "terminated": Field((), torch.bool),
        "truncated":  Field((), torch.bool),
        "valid":      Field((), torch.bool),
    }
```

**算法只声明自己的增量**，这就是接入新算法时关于 buffer 的全部工作：

```python
# DAE 需要完整策略分布 + episode 边界
EXTRA = {
    "policy":       Field((n_actions,), torch.float32),
    "ep_start_idx": Field((), torch.long, per_sample=False),
}
buf = RolloutBuffer(env, cfg.rollout_len, extra=EXTRA)   # 就这一行

# GA2E 什么都不用加 —— 它只在 compute 阶段读 reward/value，不需要新字段
```

`Obs` 为 dict 时（MiniGrid），`base_schema` 自动展开成 `obs.image` / `obs.direction` 等子字段，由 `tree.py` 处理 —— **dict obs 不需要用户或算法写任何代码**。

#### 这样做的收益与代价

收益：
- **1978 行 → 一个 buffer + 每算法几行声明。**
- shape/dtype 集中在一处 → `torch.compile` 的静态形状保证、GPU 常驻策略、pinned memory 都只实现一次。
- schema 可打印、可对比、可写进 checkpoint 元数据 —— **「这个 checkpoint 的 buffer 布局是什么」变成可查询的**。
- 字段名拼错在**构造时**就报错（schema 已知全部合法 key），而不是训练到第 3 小时 `KeyError`。

代价，说清楚：
- 多了一层间接。`buf.obs` 变成动态属性而非静态字段，IDE 补全和静态类型检查会变弱。**对策**：`RolloutBuffer.__getattr__` 配 `TypedDict` 存根，并给核心字段留显式属性注解；minibatch 用 `NamedTuple` 而非 dict，保留补全。
- 这层抽象**必须保持在「声明式」而不滑向「配置式 DSL」**。禁止在 `Field` 上增加 `transform=` / `init_fn=` 之类回调 —— 一旦 schema 能执行代码，就重演了 SB3 配置黑箱的错误。**`Field` 永远只描述内存布局。**

#### 保留的其他决策

- 布局：**struct-of-tensors**，每字段一个预分配张量 `[T, N, ...]`，全程 GPU 常驻，**shape 静态**。
- 关键方法：`add(**kv)` / `compute_returns(estimator, ...)` / `iter_minibatches(sampler)`。
- 明确不做：变长 episode 存储、优先采样、CPU 分页。on-policy 固定 `[T, N]` 就够。
- 对比：SB3 的 `RolloutBuffer` 在 CPU 用 numpy 再逐 batch 搬到 GPU；我们直接在 GPU 上写入（具体收益待 §11 基准测定，不预先吹）。

### 4.4 `Sampler` —— 循环网络不是另一个算法，只是另一种切 minibatch 的方式

on-policy + RNN 的唯一真正差异是 **minibatch 不能打散时间轴**。两个 sampler 实现，`ppo.py` 里换的是一个对象，不是一个文件：

- `FlatSampler`：`[T,N]` 展平后随机切 minibatch（前馈网络的标准做法）。
- `SequenceSampler`：按 `(env, 时间块)` 切，保留时间轴 `[L, B]`，附带块起点的 `recurrent_state`，支持 burn-in。

sampler 由 `policy.is_recurrent` 自动选定（也可显式覆盖）。**`ppo.py` 里没有任何 `if rnn:` 分支**——差异全部被 `Sampler` 和 `Policy` 两个协议吸收掉了。

### 4.5 `Policy` 协议 —— 网络架构完全由你掌控

**这是框架与用户的主接口。** 其他框架在这里做得都不够好：SB3 强制你走 `BasePolicy → ActorCriticPolicy → FeatureExtractor` 的继承链，想换个结构要读四个文件、覆写若干个语义不清的钩子；CleanRL 则是把网络硬编码在算法文件里，改架构意味着改算法。

oprl 的做法：算法只依赖一个**极窄的 Protocol**，任何满足它的 `nn.Module` 都能用——包括你从零手写的。

```python
class Policy(Protocol):
    is_recurrent: bool                     # 决定默认 sampler，仅此而已

    # rollout 期：采样动作。recurrent_state 前馈实现里恒为 None
    def act(self, obs: Obs, state: RState | None = None
            ) -> tuple[Tensor, Tensor, Tensor, RState | None]: ...
            #          action,  logprob, value,  next_state

    # update 期：对已采集的动作重新求值（PPO 的 ratio 靠它）
    def evaluate(self, obs: Obs, action: Tensor, state: RState | None = None,
                 valid: Tensor | None = None
                 ) -> tuple[Tensor, Tensor, Tensor]: ...
            #          logprob, entropy, value

    def value(self, obs: Obs, state: RState | None = None) -> Tensor: ...
    def initial_state(self, batch: int) -> RState | None: ...
```

四个方法，语义各自完整。**框架不关心你内部是 MLP、CNN、Transformer、共享 trunk、还是双塔**——`evaluate` 返回三个张量就行。

三个使用层次，覆盖从「跑基线」到「魔改」的全谱：

**层次 1 —— 用默认组装器（跑基线，一行）**
```python
policy = ActorCritic.build(env, encoder="mlp", hidden=(64, 64), activation="tanh")
# encoder ∈ {"mlp", "cnn", "gru", "lstm", ...} 或直接传一个 nn.Module
```

**层次 2 —— 换 encoder，保留 head/分布/初始化逻辑（最常见的研究需求）**
```python
class MyEncoder(nn.Module):
    out_dim: int                          # 唯一契约
    def forward(self, obs) -> Tensor: ...
    # 若有循环状态，再实现 initial_state / 返回 (feat, next_state)

policy = ActorCritic(encoder=MyEncoder(...), env=env)   # head 自动按 action_space 配好
```
`ActorCritic` 只做三件事：调 encoder → 接 head → 建分布。它是**约 80 行的可读组装器，不是基类**；不满意可以整份复制出去改，因为算法不依赖它，只依赖 `Policy` 协议。

**层次 3 —— 整个 policy 自己写**
```python
class MyPolicy(nn.Module):     # 不继承 oprl 任何东西
    is_recurrent = False
    def act(self, obs, state=None): ...
    def evaluate(self, obs, action, state=None, valid=None): ...
    def value(self, obs, state=None): ...
    def initial_state(self, batch): return None

train(cfg, env, policy=MyPolicy(), log=logger)   # 直接就能跑
```

配套原则：
- `nets/` 里的 encoder/head 是**可选默认件，不是必经之路**。`import oprl.nets` 是可选的。
- 动作空间 → 分布的映射（Discrete→Categorical、Box→DiagGaussian/SquashedGaussian、MultiDiscrete→多头、动作 mask）由 `heads.py` 提供，但你可以完全绕过。
- 正交初始化、policy head 的 `std` 参数化方式（state-independent log_std vs 网络输出）、value head 是否共享 trunk —— 这些「已知重要的实现细节」全部是**默认件里的显式配置**，不是隐藏行为。
- **`valid` mask 透传到 `evaluate`**：RNN 实现需要它做正确的序列 masking；前馈实现忽略即可。

### 4.6 算法层的形状（**一个更新规则 = 一个文件**，但 surrogate 是可插拔的）

上一版设计列了 `ppo.py` / `ppo_rnn.py` / `a2c.py` 三个文件，这与「不要复制粘贴」自相矛盾。而 §5 的算法调研又暴露了另一个问题：**近年大量 PPO 改进（TR-PPO / SPO / DPO / MDPO / APO…）既不是"换个超参"，也不是"新的更新规则"，而是只替换了 surrogate 目标函数**；另有一类（DAE / RVL）只替换了 advantage/critic 的估计方式。原来的二分判据装不下它们。

修正后是**四档**判据：

| 差异所在 | 表现形式 | 例子 |
|---|---|---|
| 超参 / 网络 / 采样方式 | **config 或传对象**，零新代码 | A2C、RNN 策略、任意网络架构 |
| **只是 surrogate 目标不同** | **`surrogates.py` 里一个函数**（~15 行） | TR-PPO、SPO、DPO、MDPO、APO |
| **只是 advantage/critic 估计不同** | **实现 `AdvantageEstimator`**（§4.7） | DAE、RVL |
| 训练循环结构不同 | **新文件 + 新 `train()`** | PPG（aux phase）、V-MPO（EM+对偶变量） |

按此判据：
- `ppo_rnn.py` **不存在**。循环性 = 换 `Policy` 实现 + `SequenceSampler`（§4.4/4.5）。
- `a2c.py` **不存在**。A2C ≡ PPO 在 `num_epochs=1, num_minibatches=1, clip_coef=inf` 下的特例 —— 它是 `configs/a2c.yaml`，不是一份代码。
- **`trpo.py` 不做**（你的决定）。理由也站得住：TRPO 需要共轭梯度 + Fisher 向量积 + line search，是一套只服务于它自己的机械装置，复用价值低；而 §5 表里的 TR-PPO/SPO/MDPO 已经用 PPO 的代码路径拿到了 trust-region 的收益。
- `ppg.py` **存在**（后续）：它有独立的 auxiliary phase，训练循环结构真的不同。

**中间那一档是最大的杠杆** —— 一个 `Surrogate` 协议换来 §6 表里六七个算法，每个约 15 行：

```python
class Surrogate(Protocol):
    """给定 ratio/logp/advantage，返回策略损失。这是 §6 大部分算法的唯一差异点。"""
    def __call__(self, ratio: Tensor, logp: Tensor, logp_old: Tensor,
                 adv: Tensor, cfg: Config) -> tuple[Tensor, dict]: ...
            #     policy_loss, 诊断量（clipfrac / kl / ...）
```

`ppo_loss()` 内部只调 `cfg.surrogate(...)` 一次。`surrogates.py` 里每个实现都短到能和论文公式逐行对照 —— 这本身就是可信度：**读者能验证我们没写错**。

于是 M1–M2 阶段 `algos/` 下**只有 `ppo.py` + `surrogates.py`**：

```python
# src/oprl/algos/ppo.py 骨架（伪码，示意结构）

@dataclass
class PPOConfig(Config):
    total_steps: int = 1_000_000
    num_envs: int = 8;  rollout_len: int = 128
    num_epochs: int = 10;  num_minibatches: int = 4
    lr: float = 3e-4;  anneal_lr: bool = True
    gamma: float = 0.99;  gae_lambda: float = 0.95
    clip_coef: float = 0.2;  clip_vloss: bool = True
    ent_coef: float = 0.0;  vf_coef: float = 0.5
    max_grad_norm: float = 0.5;  norm_adv: Literal["batch","minibatch","none"] = "minibatch"
    target_kl: float | None = None
    norm_obs: bool = True;  norm_reward: bool = True
    seq_len: int | None = None      # 仅 recurrent policy 用；None = 自动

def train(cfg, env: EnvAdapter, policy: Policy, log: Logger):
    buf = RolloutBuffer(..., recurrent=policy.is_recurrent)
    sampler = make_sampler(cfg, policy)       # Flat 或 Sequence，唯一的分叉点
    opt = Adam(policy.parameters(), lr=cfg.lr)

    for it in range(num_iters):
        maybe_anneal_lr(opt, it, num_iters, cfg)
        collect(env, policy, buf, cfg.rollout_len, log)     # 原语，有测试
        buf.compute_returns(bootstrap_value, cfg.gamma, cfg.gae_lambda)
        for epoch in range(cfg.num_epochs):
            for mb in buf.iter_minibatches(sampler):
                loss, stats = ppo_loss(policy, mb, cfg)     # ~30 行，一眼看完
                step(opt, loss, policy, cfg.max_grad_norm)
            if early_stop_kl(stats, cfg.target_kl): break
        log.flush(global_step)      # 聚合后一次写出（§4.9）
```

`ppo_loss` 里对 `policy` 的调用只有 `policy.evaluate(mb.obs, mb.action, mb.state, mb.valid)` 一处 —— 这就是「架构可换」和「recurrent 不分叉」两件事在代码上的落点。

**每个超参都出现在 `PPOConfig` 里，没有藏在别处的默认值** —— 这是对「PPO 的 37 个实现细节」问题的正面解法：把所有已知重要的细节（obs/reward 归一化、advantage 归一化粒度、value clipping、正交初始化、lr annealing、grad clipping、KL early stop、entropy）都做成**显式、有默认值、可关闭**的配置项，而非硬编码。

### 4.7 扩展面（Extension surface）—— 可修改性的显式契约

"易于修改"不能只是口号，得能回答"改 X 要动几个文件"。框架**只有六个扩展点**，每个都是 `Protocol`（结构化子类型，不需要继承、不需要注册）：

| 你想改什么 | 动什么 | 要动几个文件 | 要继承吗 |
|---|---|---|---|
| 网络架构 | 传一个满足 `Policy` 的 `nn.Module`（或只换 encoder） | 1（你自己的） | ✗ |
| 新环境类型 | 实现 `EnvAdapter` | 1 | ✗ |
| minibatch 切法 | 实现 `Sampler` | 1 | ✗ |
| **PPO 目标函数** | 实现 `Surrogate`（~15 行） | 1 | ✗ |
| **advantage/critic 估计** | 实现 `AdvantageEstimator` | 1 | ✗ |
| 日志/可视化后端 | 实现 `Sink`（3 方法） | 1 | ✗ |
| **新算法（新更新规则）** | 复制 `ppo.py` 改，或新写一个 `train()` | 1 | ✗ |

关键性质：**这些 Protocol 之间互不引用**。换 encoder 不需要知道 buffer 长什么样；写 `EnvAdapter` 不需要知道 PPO 存在。

#### `AdvantageEstimator` —— 由 DAE / RVL / GA2E 逼出来的第六个扩展点

原设计假设 advantage 估计恒为 GAE。§5.2 的调研与**你现有的 GA2E** 一起推翻了这个假设，但三者的要求强度差别很大：

| 算法 | 需要什么 | 是否纯函数 |
|---|---|---|
| **DAE** | 直接回归 advantage，不走 `A = r + γV − V`；需 buffer 多存策略分布 | ✅ 纯函数 |
| **RVL** | critic 预测状态对反对称差值 `Δ(sᵢ,sⱼ)`，再重建 GAE | ✅ 纯函数 |
| **GA2E** | **梯度对齐选 λ**：`λ* = argmax_λ cos(g_val, g(λ))` | ❌ **需要 policy + 反向传播** |

**GA2E 是最苛刻的一个，它决定了协议的形状。** 其机制（`/data/workspace/PGAE/src/algo/ga2e/ga2e.py`）：

1. 按**轨迹**把 rollout 切成 train / val 两折（不是随机切 —— 要保持时间连续性）；
2. 在 val 折上用近无偏 λ_val≈0.97 算策略梯度 `g_val` 作**标尺（yardstick）**；
3. 在两级 λ 网格上（Level1 步长 0.1，Level2 在最优点邻域细分），对每个候选 λ 在 train 折上算 `g(λ)`，打分 `cos(g_val, g(λ)) − β·var`；
4. `argmax` 得 λ*，再跨迭代 EMA 平滑。

**这意味着约 19 次额外反向传播/rollout** —— 它根本不是「输入 rollout → 输出 advantage」的纯函数。

##### 三个被 GA2E 迫使的设计修正

**修正 1：协议必须能拿到 `policy`，并允许反向传播。**

```python
class AdvantageEstimator(Protocol):
    extra_fields: Schema = {}                    # buffer 需多存的字段（§4.3）
    extra_policy_outputs: tuple[str, ...] = ()   # Policy 需多输出的头

    def compute(self, buf: RolloutBuffer, bootstrap_value: Tensor,
                policy: Policy, surrogate: Surrogate, cfg: Config
                ) -> tuple[Tensor, Tensor, dict[str, float]]: ...
        #            advantages, value_targets, 诊断量

    def critic_loss(self, policy, mb, cfg) -> tuple[Tensor, dict]: ...

    # GA2E 在 epoch 边界重选 λ 用；其他实现为 no-op
    def on_epoch_start(self, epoch: int, buf, policy, surrogate, cfg) -> None: ...

    # 跨迭代状态（GA2E 的 λ EMA）。无状态实现返回 {}
    def state_dict(self) -> dict: ...
    def load_state_dict(self, d: dict) -> None: ...
```

`GAE` / `DAE` / `RVL` 忽略 `policy`、`surrogate`、`on_epoch_start`；只有 GA2E 用。

**修正 2：我之前声称的「`AdvantageEstimator` 与 `Surrogate` 完全正交」是错的 —— 至少对 GA2E 不成立。**

GA2E 的 λ 刷新有两种粒度，语义不同：
- `rollout` 模式：更新前选一次，此时 θ 未变 → ratio=1 → **clipped surrogate 梯度恰等于 vanilla PG**，语义干净、开销小；
- `epoch` 模式：每个 PPO epoch 边界重选，此时 **ratio≠1，`g(λ)` 必须用带 clip 的 surrogate 梯度**。

也就是说 **GA2E 的打分函数依赖当前 surrogate 的具体形式**。所以 `compute` 的签名里必须有 `surrogate` —— 这是一个真实的耦合，不能靠 API 措辞掩盖。`--surrogate=dpo --advantage=ga2e` 仍然可跑，但**语义上不是两个独立旋钮**：换 surrogate 会改变 GA2E 选出的 λ。这一点必须写进文档，否则用户会误读消融结果。

**修正 3：`on_epoch_start` 是对 §2 原则 5（「不用 Callback 打洞」）的一处让步。**

`epoch` 模式要在 PPO epoch 边界插入行为。折中办法是：**`ppo.py` 里显式写一行 `estimator.on_epoch_start(...)`，而不是注册回调**。它出现在训练循环里、能被读到、默认是 no-op —— 保住了「无隐藏行为」，代价是协议多一个方法。GA2E 还要求 **value 回归目标固定为第一轮 λ 的结果**（避免 value 训练追移动靶），这条得由 GA2E 自己在 `state_dict` 里记住。

##### 一个尚未解决的问题

`epoch` 模式的开销是 `~19 × n_epochs` 次反向传播 —— 在 `num_epochs=10` 时**可能超过 PPO 主更新本身的成本**。这与 §7「优化拿到结论的时间」直接冲突。倾向的处理：默认 `rollout` 模式，`epoch` 模式明确标注为「昂贵，仅用于消融」，并由 `timer` 单列 `time/lambda_select_frac` 让代价可见。

##### 迁移账（修正后）

我上一版给的「1300 行 → 250 行」是基于错误理解算的，作废。修正后：

| 现有代码 | 迁移后 |
|---|---|
| `buffer.py` 516 行（Lepski） | **不迁** —— 已被 `ga2e.py` 的 gradient-alignment 取代。若要保留作对照，另立 `advantages/lepski.py` |
| buffer 的分配/写入/索引 | 删 → `Schema` 自动（§4.3） |
| `_select_lambda` / `_alignment_scores` / `_trajectory_split` / `_policy_grad` / `_gae` | `advantages/ga2e.py`，**约 200 行**（真正的算法内容） |
| `_train_with_spo_loss` / `_train_with_kl_loss` | 删 → `--surrogate=spo` / `mdpo` |
| `_train_per_epoch` + PPO 主循环 | 删 → 共享 `ppo.py` + `on_epoch_start` |
| `_record_lambda` / `_log_alignment` / `_dump_align` | 保留指标语义，改用 `Logger`（§4.9，API 与 SB3 兼容 → 几乎不用改） |

**结论：`ga2e.py` 782 行 → 约 200 行，Lepski 那 516 行按需另立文件。** 验收标准仍是「3-seed 均值容差内重合」（numpy→torch 后浮点累加顺序变化，不可能逐位一致）。

#### 接入新算法的成本

新算法就是一个新的 `train(cfg, env, policy, log)` 函数 —— **没有基类要继承、没有注册表要登记、没有抽象方法要实现**。可复用的原语按需 import：

```python
# src/oprl/algos/my_algo.py —— 一个全新算法的全部仪式
from oprl import RolloutBuffer, gae, collect, FlatSampler, Config

@dataclass
class MyConfig(Config): ...          # 你自己的超参

def train(cfg, env, policy, log):
    ...                              # 想怎么写就怎么写
```

对照其他框架同一件事的成本：SB3 要继承 `OnPolicyAlgorithm` 并正确覆写 `train()`/`collect_rollouts()`，还得理解基类里 `_setup_model` / `_update_info_buffer` 等的隐含时序；Tianshou 要实现 `Policy` 子类并适配 `Trainer` 的生命周期；RLlib 要装配 `RLModule` + `Learner` + config builder。oprl 的成本是**写一个函数**。

代价我也说清楚：新算法要自己写训练循环，不会"免费继承"到 lr annealing、KL early stop 这些便利。这是刻意的取舍 —— 那些便利是 `ppo.py` 里几行可见的代码，复制它们比继承一个会隐式改变你行为的基类更好。**没有隐式行为，就没有"为什么我的算法多了一步 grad clip"这类问题。**

#### 可修改性的硬约束（对我们自己的纪律）

1. **`algos/` 单向依赖 `L1`，`L1` 绝不 import `algos/`**。反向依赖一出现，原语就被算法污染，L1 的独立可用性（`from oprl import gae` 给别人的项目用）就没了。CI 加依赖方向检查。
2. **每个 Protocol 方法数 ≤ 5**，超了就说明抽象错了。`Policy` 四个、`EnvAdapter` 三个、`Sampler` 一个、`Sink` 三个、`Surrogate` 一个、`AdvantageEstimator` 两个。
3. **不做插件注册表 / entry-point 发现机制**。传对象进去，不传字符串。字符串→类的映射（`encoder="mlp"`）只在便利函数 `ActorCritic.build()` 里出现，且始终有等价的传对象写法。
4. **配置只有 dataclass 一种来源**，不做 `**kwargs` 透传。`**kwargs` 是可修改性的假象 —— 它让参数无法被静态检查，也无法被 `--help` 列出。
5. **不引入依赖注入容器、不引入事件总线、不引入 Callback 体系**。要在训练循环里插东西，就在循环里写一行。

### 4.8 轻量化的量化目标

"轻量"也需要能被检验的指标，否则会随时间滑坡。设定硬预算，超了就要在 review 里被质疑：

| 指标 | 预算 | 说明 |
|---|---|---|
| 核心运行时依赖 | **3 个**（`torch` `numpy` `gymnasium`） | 其余全部进 extras |
| `import oprl` 耗时 | < 2s（基本是 torch 的时间） | 不在 `__init__` 里 import 可选后端 |
| L1 原语总行数 | < 2000 行 | 不含测试/注释；`logger.py`+`sinks.py` < 300 |
| `ppo.py` | < 300 行 | 超了说明该拆进 L1 |
| `surrogates.py` 每个函数 | < 20 行 | 必须能和论文公式逐行对照（§5.3） |
| 扩展点（Protocol）总数 | **≤ 6** | 已用满；再加要先删一个 |
| 每算法的 buffer 代码 | **0 行** | 只声明 `Schema`（§4.3）；对照：PGAE 现有 6 个 buffer 共 1978 行 |
| `ppo_loss()` | < 40 行 | 必须一屏读完 |
| 抽象层数（用户到张量） | ≤ 2 | 对照：SB3 4–5，RLlib 5+ |
| 读懂一个算法要跳转的文件数 | ≤ 3 | CleanRL 是 1，SB3 是 4+ |

最后一项是核心权衡：我们放弃 CleanRL 的"1 个文件"，换来复用和测试，但必须守住"≤3"。一旦滑到 5，就退化成 SB3 了，整个项目的立论也就没了。

### 4.9 `Logger` —— 自研，且不只是 wandb 的转发器

**为什么自研**：直接用 wandb/tensorboard 会让核心依赖变重（违反 §4.8 的 3 依赖预算），且把项目绑死在一个 SaaS 上。更重要的是 —— **RL 的日志需求和通用 ML 不同**，通用 logger 不解决这些：

- 指标产生频率不一：per-minibatch（loss、KL）、per-iteration（SPS、耗时占比）、per-episode（回报、长度，且**只有部分 env 在这一步结束**）。
- 一个 iteration 内有几十个 minibatch，逐个上报既慢又噪声大 —— **需要聚合而非转发**。
- episode 回报必须来自**未归一化**的奖励（否则曲线毫无意义，见 §10 的陷阱 6）。
- 诊断量（KL、clipfrac、熵、grad norm、explained variance）是判断「训练是否健康」的依据，必须默认就有，而不是用户自己想起来加。

#### 参考 SB3 的 Logger 设计（你的要求）

SB3 的 `logger` 有三点值得直接沿用 —— 它已经解决了 RL 日志的核心问题，且**你现有的 GA2E 代码就是按它写的**（`self.logger.record("ga2e/lambda_star", ...)`），沿用它意味着迁移时这些调用一行都不用改：

| SB3 的做法 | 为什么对 | 我们的处理 |
|---|---|---|
| `record(key, value)` 累积、`dump(step)` 一次写出 | 一个 iteration 内几十个 minibatch，逐个上报既慢又噪声大 | **直接沿用同名 API** |
| `record_mean(key, value)` 自动求均值 | per-minibatch 指标（loss/KL）天然要平均 | **沿用** |
| `Format` 分离（stdout/csv/tensorboard/json） | 后端可插拔，核心不绑死 | 改叫 `Sink`（同一概念，3 方法） |
| `key/subkey` 斜杠命名分组 | 天然形成 `train/`、`rollout/`、`ga2e/` 分区 | **沿用**，并据前缀推断聚合方式 |

三处我们要改进的：

1. **SB3 的 `record` 默认覆盖、`record_mean` 才平均** —— 用错了就静默丢数据（只留最后一个 minibatch 的值）。我们**按 key 前缀自动选聚合方式**（下表），`record` 即可，不需要用户记住选哪个。
2. **SB3 没有权威的机器可读输出** —— csv 会因 key 集合变化而错列，json 不是默认。我们**始终写 `metrics.jsonl`**，且它是 §8 的唯一数据源。
3. **SB3 的 logger 是 `self.logger` 全局挂在算法上** —— 我们显式传参（§2 原则 5），多 run 并行天然安全。

#### 三层结构

```python
# L1: Sink 协议 —— 一个后端只需三个方法
class Sink(Protocol):
    def write(self, metrics: dict[str, float], step: int) -> None: ...
    def write_media(self, key: str, value: Any, step: int) -> None: ...
    def close(self) -> None: ...

# 内置：ConsoleSink（默认，SB3 风格表格）/ JsonlSink（默认，权威数据源）
#       TensorBoardSink / WandbSink / CsvSink（均在 track extra）
```

```python
# L2: Logger —— 聚合 + 路由，这是自研的价值所在
class Logger:
    def __init__(self, sinks: list[Sink], run_dir: Path): ...

    # --- SB3 兼容 API（现有代码零改动迁移）---
    def record(self, key: str, value) -> None: ...        # 按前缀自动聚合
    def record_mean(self, key: str, value) -> None: ...   # 显式求均值
    def dump(self, step: int) -> None: ...                # = flush

    # --- 便利扩展 ---
    def add(self, **kv) -> None: ...                      # record 的批量版
    def add_episode(self, ret: float, length: int) -> None: ...  # 未归一化回报
    def media(self, key, value, step) -> None: ...
    def close(self) -> None: ...
```

聚合策略由 key 前缀约定，**不需要用户配置**：

| 前缀 | 聚合方式 | 例子 |
|---|---|---|
| `train/` `loss/` `grad/` | mean | `loss/policy`、`grad/norm` |
| `diag/` | mean + last | `diag/kl`、`diag/clipfrac` |
| `time/` | sum → 转占比 | `time/env`、`time/lambda_select`（§7.4） |
| `perf/` | last | `perf/sps` |
| `rollout/` `charts/` | 滑窗 mean | `rollout/ep_rew_mean`（SB3 同名） |
| 其他（如 `ga2e/`） | **mean，且允许算法自定义** | `ga2e/lambda_star`、`ga2e/score_bias` |

最后一行是关键：**算法的独有指标零登记成本**。GA2E 的 `ga2e/*`（20+ 个键）直接 `record` 即可，框架不需要预先知道它们 —— 这也是 §8.1 要求的性质。

#### run 目录布局（可复现性的落点，§2 原则 7）

```
runs/<exp_name>/<timestamp>-<git_sha>-seed<k>/
├─ config.yaml        # resolved 后的完整 config，可直接 --config 重跑
├─ metrics.jsonl      # 每行一个 flush，便于 pandas/duckdb 离线分析
├─ meta.json          # git sha + dirty flag + uv.lock hash + 硬件 + 库版本
├─ stdout.log
└─ ckpt/{latest,best}.pt
```

`metrics.jsonl` 是**一等输出**，不是 tensorboard 的附属品 —— 它让基准曲线归档（§10）和跨 run 对比不依赖任何外部服务。

#### 纪律

- **Logger 属于 L1，不 import `algos/`**（§4.7 纪律 #1）。
- **`log()` 绝不抛异常打断训练**。sink 失败（网络断、磁盘满）只警告一次并降级，训练继续。**训练跑了 6 小时因为 wandb 掉线而崩掉是不可接受的。**
- **不做全局单例**。`Logger` 显式传入 `train()`，与「无隐藏状态」原则一致；也让多 run 并行（§7.3）天然安全。
- 核心只依赖 stdlib（Console + Jsonl）；TensorBoard / wandb 在 `track` extra，延迟 import。

### 4.10 config 与 checkpoint

- `Config`：dataclass 层级 + `tyro` 风格 CLI + YAML overlay（`--config configs/ppo/mujoco.yaml --lr 1e-4`）。启动时把 **resolved 后的完整 config** 落盘。
- `Checkpoint`：**归一化器状态属于模型，必须一起存**。checkpoint = `{policy, optimizer, obs_norm, reward_norm, global_step, rng_states, config}`。少存归一化器 = 恢复后模型看到完全不同的输入分布，是极常见的静默故障。
- 评估：`evaluate()` 独立函数，自动 `norm.freeze()`（应用统计量但不更新），避免 eval rollout 污染训练统计。

---

## 5. 算法集成库

### 5.1 调研结果：近年 on-policy 算法（限 CCF-A/B 已发表）

筛选口径：**单智能体 + on-policy + 能落在 PPO 代码路径上 + 已过同行评审**。已排除多智能体（MAPPO/HAPPO/MAT）、off-policy（SAC/TD3/IMPALA）、offline、model-based、LLM 专用（GRPO/DAPO），**以及所有预印本/workshop-only 工作**（你的决定）。venue 经 dblp / OpenReview / Crossref 核对。

| algo_name | paper_name | paper_url |
|---|---|---|
| **PPG** | Phasic Policy Gradient | https://arxiv.org/abs/2009.04416 |
| **PPG Reloaded** | PPG Reloaded: An Empirical Study on What Matters in Phasic Policy Gradient | https://proceedings.mlr.press/v202/wang23aw.html |
| **DNA** | DNA: Proximal Policy Optimization with a Dual Network Architecture | https://arxiv.org/abs/2206.10027 |
| **DAAC / IDAAC** | Decoupling Value and Policy for Generalization in Reinforcement Learning | https://arxiv.org/abs/2102.10330 |
| **TR-PPO** | Truly Proximal Policy Optimization | https://arxiv.org/abs/1903.07940 |
| **SPO** | Simple Policy Optimization | https://proceedings.mlr.press/v267/xie25m.html |
| **DPO** | Discovered Policy Optimisation | https://arxiv.org/abs/2210.05639 |
| **LPO / Mirror Learning** | Mirror Learning: A Unifying Framework of Policy Optimisation | https://proceedings.mlr.press/v162/grudzien22a.html |
| **MDPO** | Mirror Descent Policy Optimization | https://openreview.net/forum?id=aBO5SvgSt1 |
| **RPO (Reflective)** | Reflective Policy Optimization | https://proceedings.mlr.press/v235/gan24b.html |
| **AGAC** | Adversarially Guided Actor-Critic | https://openreview.net/forum?id=_mQp5cr_iNy |
| **Batch-size invariance** | Batch size-invariance for policy optimization | https://arxiv.org/abs/2110.00641 |
| **PPO-RPE** | Proximal Policy Optimization with Relative Pearson Divergence | https://arxiv.org/abs/2010.03290 |
| **V-MPO** | V-MPO: On-Policy Maximum a Posteriori Policy Optimization | https://openreview.net/forum?id=SylOlp4FvH |
| **DAE** | Direct Advantage Estimation | https://arxiv.org/abs/2109.06093 |
| **RVL** | Relative Value Learning | https://openreview.net/forum?id=ulTRUwrzt9 |
| **APO** | APO: Anchored policy optimization by leveraging unsampled actions in continuous spaces | https://doi.org/10.1016/j.neunet.2026.109476 |

**venue 明细**：ICML（PPG / PPG Reloaded / LPO / Reflective-RPO / SPO）、NeurIPS（DNA / DAAC / Batch-size invariance / **DAE 2022**）、ICLR（MDPO / AGAC / V-MPO / **RVL 2026 Poster**）、UAI（TR-PPO）、ICRA（PPO-RPE）、**Neural Networks 205:109476（APO 2026）**。

三个新增条目的核实结论：
- **DAE** = NeurIPS 2022（arXiv v1 为 2021-09）。**确认是 on-policy** —— 论文明确「直接从 on-policy 数据估计」，不是 off-policy/value-based，可放心纳入。
- **RVL** = ICLR 2026 **Poster，已录用**（OpenReview 有 Camera-Ready Revision，非 submission）。其关键词直接就是 "On-Policy Actor-Critic, GAE, PPO"，并在 49 个 ALE 游戏上作为 PPO critic 的 drop-in 评估。
- **APO** = Neural Networks 卷 205、文章号 109476。⚠️ Crossref 的 `published-print` 显示 2027-01（期刊印刷期滞后），但卷号与索引年份为 2026 —— 引用时写 **"Neural Networks 205:109476, 2026"**。作者信息来自 Crossref，未能核对 PDF（付费墙）。官方代码：https://github.com/wjl-bupt/APO

**已按你的决定移除的 5 个预印本**：ESPO、Robust-RPO、COPG、TREFree、Outer-PPO。它们实现成本最低（各约 10 行），但未过同行评审 —— 若日后想加，作为 `contrib/` 或实验分支更合适，不进正选表。

**同时移除 GePPO**（原 NeurIPS 2021 条目）：它需要保留多个历史行为策略的样本做混合重要性采样 —— 形式上就是 off-policy 复用，与 §2 的「数据永远来自当前策略」不变量直接冲突。宁可放弃一个已发表算法，也不破坏那个能被到处依赖的不变量。

**DAE 的后续版本暂不做**：*Direct Advantage Estimation for Scalable and Sample-efficient Deep RL*（RLC 2026, https://arxiv.org/abs/2606.20411）扩展到 POMDP，但需要一个学习的**离散潜变量动力学模型** —— 那是 model-based，属于 §2 反目标。只做 NeurIPS 2022 的原版。

### 5.2 分类：按实现成本，不按论文年份

调研有两个改变设计的发现：

**发现一**：大多数算法**不是新的更新规则，只是替换了 surrogate 目标** → 催生 `Surrogate` 协议（§4.6）。

**发现二（DAE / RVL / GA2E 暴露的）**：还有一类算法**不动 surrogate，而是改 advantage 的估计方式**。DAE 直接回归 advantage；RVL 用状态对反对称差值重建 GAE。**两者都绕过了我原本假设固定不变的 `gae()`。** 而你的 **GA2E 用梯度对齐选 λ，需要在 advantage 计算阶段访问 policy 并做反向传播** —— 这个要求最强，直接决定了协议的形状。见 §4.7。

| 类别 | 算法 | 实现方式 | 增量成本 |
|---|---|---|---|
| **A. Surrogate 替换** | TR-PPO、SPO、DPO、MDPO、PPO-RPE、LPO、**APO** | `surrogates.py` 里一个函数 | **各约 15 行** |
| **B. Trainer 级小改** | Batch-size invariance | `ppo.py` 里几行 + config 项 | 约 10 行 |
| **C. 解耦价值网络** | PPG、PPG Reloaded、DNA、DAAC/IDAAC | 一个 aux-phase 模块 + 双网络 `Policy` | 一个模块覆盖 4 个 |
| **D. Advantage/Critic 估计** | **DAE、RVL**、**GA2E**（你现有的） | 实现 `AdvantageEstimator` 协议 | DAE/RVL 中等；**GA2E 最重**（需 policy + 反传 + 跨迭代状态） |
| **E. 需要额外网络** | AGAC | 第三个网络 + 探索奖励项 | 中 |
| **F. 重构训练循环** | V-MPO、Reflective-RPO | 独立文件 | 高，**放最后** |

**类别 A 仍是最大回报**：7 个已发表算法，各约 15 行，共享同一个协议。**在 CleanRL 里做同样的事需要 7 个复制粘贴的文件。**

**类别 D 是最有研究价值的一类**：DAE / RVL / GA2E 都直接针对样本效率。DAE 与 RVL **与类别 A 正交**，可自由组合（`--surrogate=dpo --advantage=rvl`）。**GA2E 是例外 —— 它的 λ 打分依赖当前 surrogate 的形式**（见 §4.7 修正 2），组合仍可跑但不是独立旋钮。这个组合空间是单文件框架（CleanRL）给不了的，也是 §1.3「窄化换来的复杂度预算」的最好用途：**我们能跑别人跑不了的消融** —— 前提是把耦合标注清楚，别自己误读结果。

**APO 的附带价值**：它的 UARR（Unsampled Action Ratios Regularization）针对连续动作空间的「锚定盲区」，在未采样动作上约束 π_new/π_old ≈ 1 —— 与 §6 的 MuJoCo/Isaac 路径直接相关，且实现成本只是一个正则项。

### 5.3 集成方式与纪律

1. **不做算法动物园**。每个算法必须有：与 vanilla PPO 的**同 seed 对照曲线**（至少 MinAtar + 一个 MuJoCo 任务）。**跑不出论文声称的改进就标注为「未复现」并保留结论** —— 这比静默删掉或假装成功都更有价值，也是本框架相对论文代码的实际贡献。
2. **默认值永远是 vanilla PPO**。所有变体通过 `--surrogate=dpo` 之类显式开启，不做「自动选最好的」。
3. **每个实现文件头注明 paper URL + 对应公式编号**，且实现要短到能被逐行核对。
4. **实现顺序按成本/收益**：先 A（一次性把 9 个做完），再 C（一个模块覆盖 4 个），D 和 E 按需。

---

## 6. Benchmark 适配矩阵

### 6.1 选取原则：benchmark 就是能力测试矩阵

不做「支持的环境越多越好」的军备竞赛。每个 benchmark 进来必须**压测一条不同的框架代码路径** —— 于是这张表同时就是我们的测试矩阵和 §4.7 五个扩展点的验收清单。

| Benchmark | 压测的框架能力 | Adapter | Tier |
|---|---|---|---|
| Classic control | 最小闭环、单元测试 | `GymVecAdapter` | 0 |
| **MinAtar** | 秒级 CI 回归；小 CNN | `GymVecAdapter` | **0** |
| **Atari (ALE)** | CNN encoder + frame stack + 高吞吐 | `EnvPoolAdapter` | **1** |
| **MiniGrid / BabyAI** | **dict obs + 部分可观测(RNN) + 稀疏奖励** | `GymVecAdapter` | **1** |
| **MuJoCo v5** | 连续动作 + obs/reward 归一化 + 学术可比 | `GymVecAdapter` | **1** |
| **Isaac Lab** | **GPU 零拷贝 + num_envs 上千** | `TensorEnvAdapter` | **1** |
| Brax / MJX | JAX↔torch dlpack 桥 | `TensorEnvAdapter` | 2 |
| DM Control | 经 Shimmy 转换 | `GymVecAdapter` | 2 |
| Gymnasium-Robotics | dict obs + goal-conditioned | `GymVecAdapter` | 2 |
| MetaWorld (v3) | 多任务 | `GymVecAdapter` | 2 |
| Procgen2 | 程序生成泛化（train/test split） | `GymVecAdapter` | 2 |
| Craftax | 长程 + 开放式（**纯 JAX/gymnax，需桥**） | `TensorEnvAdapter` | 2 |

**Tier 的含义（这是承诺强度，不是好坏）**：
- **Tier 0** —— 进 CI，每次 push 都跑，秒级到分钟级。MinAtar 在这里是关键：它 10×10 的小格子让完整 PPO 训练能在 CPU 上几分钟跑完，**使"每次提交都验证算法真的还能学"成为可能**，而 Atari 做不到这件事。
- **Tier 1 —— 必须有存档的多 seed 曲线**（§10）。这是对外可信度的最小集合：离散像素（Atari）、部分可观测+dict obs（MiniGrid）、连续控制（MuJoCo）、GPU 原生（Isaac）各一个，四条主代码路径全覆盖。- **Tier 2** —— best-effort，提供 adapter 与示例配置，不承诺调好的超参和曲线。

Tier 1 的四个是刻意挑的：**如果这四个都跑对了，框架的四条主要代码路径就都被验证过**。反之，只跑 MuJoCo 的框架（很多新框架的实际状态）对 dict obs 和 GPU env 是没有证据的。

### 6.2 三条 obs/action 形态，覆盖全部单智能体情形

benchmark 多样性的真实成本不在环境本身，在它们要求的**观测/动作形态**。收窄到单智能体后，这个空间是有限且可穷尽的：

| 形态 | 出现在 | 框架侧需要什么 |
|---|---|---|
| 向量 obs | MuJoCo / classic / Isaac | MLP encoder + obs 归一化 |
| 图像 obs | Atari / MinAtar / Procgen | CNN encoder + frame stack；归一化改为 `/255` |
| **dict obs** | MiniGrid（image+direction+mission）/ Robotics（obs+goal） | **`tree.py` + 多分支 encoder** |
| 离散动作 | Atari / MinAtar / MiniGrid | Categorical head |
| 连续动作 | MuJoCo / Isaac / Brax | DiagGaussian / SquashedGaussian head |
| MultiDiscrete | Procgen 等 | 多头 Categorical |
| 动作 mask | MiniGrid 变体、部分自定义 env | head 支持 `-inf` 掩码 |

**这三种 obs 形态 × 三种 action 形态就是全部**（单智能体前提下）。不需要为「以后可能有别的形态」预留 —— 有新形态时，用户自己写一个 `Policy` 即可（§4.5 层次 3），框架不必改。

### 6.3 `make_env()`：便利层，且必须有逃生舱

这里与 §4.7 纪律 #3（「传对象，不传字符串」）有直接张力，我明确一下解法。

提供一个 L3 便利函数：

```python
env = oprl.make_env("MinAtar/Breakout-v1", num_envs=64, device="cuda")
# 内部：建 gymnasium vector env → 挂标准 wrapper → 包成 EnvAdapter
```

约束三条，否则这层会腐化成 SB3 那种「配置字符串驱动一切」：
1. **`make_env` 位于 L3，L1/L2 绝不 import 它。** `train()` 只接受 `EnvAdapter` 对象。
2. **始终有等价的显式写法**，且文档里并排展示 —— 字符串路径只是省打字，不是唯一入口。
3. **它不含算法知识**，不根据 env 名字偷偷改超参。「Atari 该用哪套超参」属于 `configs/ppo/atari.yaml`，是**用户显式选择的**，不是 `make_env` 的隐藏行为。

每个 benchmark 家族的标准 wrapper 栈（frame stack、reward clip、episodic life 等）在 `envs/presets.py` 里是**可读的、可复制的显式列表**，不是硬编码的 if-else。这些 wrapper 选择本身就是「实现细节」，必须可见、可关闭。

### 6.4 依赖隔离

env 套件的原生依赖极易互斥（mujoco / isaac / jax 系），所以：

- 每个套件一个 **extra**（`oprl[minatar]`、`oprl[atari]`、`oprl[mujoco]`…），核心依赖仍然只有 3 个（§4.8 预算）。
- `import oprl` **绝不 import 任何 env 后端**；adapter 内部延迟 import，缺依赖时报错要给出准确的 `uv sync --extra xxx` 提示。
- **不用 uv workspace** —— 全员共享一个 `requires-python` 交集和一个 lockfile，而这些套件恰恰是最容易互斥的（§9.1）。
- Isaac Lab 是特例：它自带 Python 环境与安装流程，不进 extras。我们只提供 `TensorEnvAdapter` + 文档说明如何在 Isaac 的环境里 `uv pip install oprl`。

---

## 7. 性能设计：优化「拿到结论的时间」，不是优化 SPS

**目标澄清（这决定了本节的全部取舍）**：我们不研究吞吐，不追 SPS 排行榜。要优化的是**从「有个想法」到「知道它行不行」的墙钟时间**。这两者经常不是一回事 —— 一个 SPS 高 3 倍但需要 8 张卡、编译两分钟、崩了看不懂报错的框架，实际迭代速度可能更慢。

按此目标，真正的杠杆按收益/成本排序：

### 7.1 第一杠杆：样本效率与早停 —— 少跑，而不是跑得快

跑 1M 步比把 10M 步跑快 2 倍更省时间。所以优先级最高的其实是「**别把时间花在注定失败的 run 上**」：

- **超参默认值要对**（§10 的 `configs/`）。一套没调好的超参浪费的时间，远超任何 kernel 优化能省下的。**这是本框架最大的性能特性，而它是一堆 YAML 文件，不是代码。**
- **自动早停**：回报长期无改善 / KL 爆掉 / 熵塌缩 / 梯度出 NaN → 直接终止并给出诊断结论，而不是安静跑完 8 小时。
- **`--smoke` 模式**：极小 `total_steps` 跑通全链路（含 checkpoint 存取、eval、日志），秒级验证「代码没写错」，再提交长 run。**大部分浪费掉的时间是因为第 3 小时才发现某处形状写错了。**
- **优先在最小的 benchmark 上验证想法**：MinAtar 而不是 Atari（§6.1 Tier 0）。这是 benchmark 分层的另一个用途 —— 不只为了 CI，也为了你自己的迭代循环。

### 7.2 第二杠杆：消除单机上的明显浪费

不做异步、不做多机（§2 反目标），但单机上有几处「白送的」提速，且都不增加使用复杂度：

1. **buffer GPU 常驻**（§4.3）。对照 SB3 在 CPU 用 numpy 再逐 batch 搬。
2. **`TensorEnvAdapter` 零拷贝**（§4.2）。Isaac/Brax 场景下省掉整条 `torch→numpy→torch` 往返。
3. **numpy 路径用 pinned memory + `non_blocking`**，让 H2D 与计算重叠。
4. **update 阶段的 kernel launch 开销** —— 小网络 + 多 minibatch 时占比惊人（DRL 的常态就是小网络）。`torch.compile` 包住 `ppo_loss + backward + step`；因为 §4.3 保证 shape 静态，**只编译一次**。

以上 1–3 是**默认开启且无副作用**的。第 4 项 `torch.compile` **默认关闭，`--compile` 显式开启** —— 理由正是本节的目标：编译要花几十秒、报错难读、debug 体验差。**对一个还在改算法的人，默认开编译是净负收益。** 调完了再开。

### 7.3 明确降级的事项

原设计里这两项按「吞吐工程」思路排得过高，按新目标降级：

- **CUDA Graph** → 移出路线图。收益仅在 launch-bound 的极端小网络场景，代价是对控制流和形状变化极度敏感、报错几乎无法定位。**与「快速迭代」直接冲突。**
- **DDP 多卡** → 从 M5 降为可选、且明确定位为「**同一份代码在多卡上跑多组种子/超参**」，而不是「把单个 run 拆到多卡加速」。后者对 on-policy 收益有限（要么增大 batch 改变了算法语义，要么通信占比高），而前者对研究者才是真正的时间节省：**一次提交，4 张卡同时验证 4 个种子**。这甚至不需要 DDP，用 4 个独立进程更简单、更不容易出错。

### 7.4 让「时间花在哪」可见

内置 `timer.py`：每次 log 都带 `time/env_frac`、`time/fwd_frac`、`time/bwd_frac`、`time/other_frac`、`sps`。有额外开销的 estimator 自报一项（如 GA2E 的 `time/lambda_select_frac`，§4.7）—— **让昂贵的算法把成本亮在明处**。

这个功能的意义在新目标下更重要了：**它防止你把时间浪费在优化错误的地方。** 如果 90% 时间在 env stepping，那再怎么 `torch.compile` 都是白费；框架直接告诉你答案，而不是让你猜。其他框架基本都缺这个。

配套：`sps` 只作为诊断信息展示，**不作为项目的宣传指标，也不进 README 的对比表** —— 避免自己被这个数字牵着走，做出损害可用性的优化。

---

## 8. 科研工作流：指标 → 归档 → 出图

这是**科研框架**而非生产框架，所以「从 run 到论文里那张图」必须是一条被设计过的流水线，而不是每次手写 matplotlib。三个阶段各自独立可用。

### 8.1 阶段一：记录（`Logger`，§4.9）

已在 §4.9 定稿。对科研场景补两条要求：

- **`metrics.jsonl` 是唯一权威数据源**。tensorboard/wandb 是可选的可视化镜像，**不是数据存储** —— 论文里的图必须能只靠 `runs/` 目录重画出来，不依赖任何 SaaS 账号是否还在、是否还免费。
- **算法自带的诊断量必须能无摩擦进入**。GA2E 的 `ga2e/*` 有 20+ 个键（`lambda_star`、`lambda_used`、`score_bias`、`score_var`、`val_frac`、`n_segments`…），`AdvantageEstimator.compute()` 的诊断返回值直接进 `Logger` 即可，**框架不需要预先登记这些 key**。这是「新算法的独有指标零成本接入」，也是 §4.9 沿用 SB3 `record()` API 的直接收益：**你现有的 `self.logger.record("ga2e/...")` 调用一行都不用改。**

### 8.2 阶段二：收集与归档（`oprl.results`）

科研的实际单位不是「一个 run」，而是「一组 run」：N 个算法 × M 个环境 × K 个种子。所以核心操作是**把 run 目录树读成一张长表**：

```python
from oprl.results import load_runs

df = load_runs("runs/**/", metrics=["charts/episodic_return"])
# 长表（tidy format），一行一个 (run, step, metric, value)：
# run_id | algo | env | seed | step | metric | value | + config 展开的列
```

五个设计要点：

1. **依赖只有 stdlib + numpy**（读 jsonl）；pandas / polars **可选**（`df.to_pandas()`）。守住 §4.8 的依赖预算。
2. **run 的身份来自 `config.yaml`，不是目录名**。`algo`/`env`/`seed` 从落盘的 resolved config 读 —— 目录名可以改、可以乱，config 不会骗人。
3. **自动检测不可比的 run**：同一组里若 config 有除 `seed` 外的差异（有人改了 `gamma` 又忘了改实验名），`load_runs` **警告并列出差异字段**。这是科研中最常见也最难发现的错误 —— 拿着两组超参不同的曲线得出「我的算法更好」的结论。**框架应当主动拦这一刀。**
4. **step 网格不对齐是常态**（不同 `num_envs`/`rollout_len` 导致 log 点不同）。聚合前统一到公共网格并线性插值，且**必须在输出里声明做了插值** —— 静默重采样会制造不存在的差异。
5. **缓存**：`runs/.cache/` 存解析后的 parquet/npz，jsonl 未变则命中。几百个 run 的重复解析是实际瓶颈。

配套 CLI（薄壳，逻辑都在库里）：

```bash
oprl results collect "runs/**" -o results/raw.parquet
oprl results summary results/raw.parquet          # 表格：最终/最优回报 ± 95%CI
oprl results export  results/raw.parquet -f csv   # 交给任何外部工具
```

### 8.3 阶段三：绘图（`oprl.plot`）

**只做科研论文里真的会用的几种图**，每种把统计正确性做对。不做通用绘图库 —— 那是 matplotlib 的活。

| 图种 | 用途 | 统计要点 |
|---|---|---|
| **学习曲线** | 主图 | 多 seed 均值 + **bootstrap 95% CI**（不是 min-max，也不是 ±std） |
| **聚合对比条形图** | 跨环境总结 | 归一化分数（vs baseline 或 human-normalized），带 CI |
| **消融矩阵** | `surrogate × advantage` 组合（§5.2） | 热力图，每格多 seed 均值 |
| **诊断多子图** | 排查训练 | `diag/kl`、`explained_variance`、`entropy`、`time/*_frac` 共享 x 轴 |
| **算法自定义诊断** | 如 GA2E 的 λ* / score_bias 曲线 | 按 key 前缀自动分组，无需框架预知 |

三条统计纪律（**这是与「随手画图」的实质差别**）：

1. **默认 bootstrap CI，不用 ±std**。RL 的 seed 分布经常严重非正态、重尾，±std 会系统性误导。同时提供 **IQM（interquartile mean）**，它对离群 seed 稳健，是 RL 评估文献的现行建议。
2. **seed 数必须画在图上**（图注或标签里的 `n=5`）。3 个 seed 的曲线不该和 10 个 seed 的看起来一样可信。
3. **平滑必须声明**。默认**不平滑**；开启时图注写明窗口大小。「我的算法更稳」经常只是平滑窗口更大。

输出：`--format {png,pdf,svg,pgf}`，其中 **pgf/pdf 面向 LaTeX 直接 `\includegraphics`**；同时落盘该图所用的数据子集与命令行（`figure.json`），**让每张图可追溯、可重画** —— 审稿人问「这条线哪来的」时能答得出。

```bash
oprl plot curves   results/raw.parquet --y charts/episodic_return \
                   --group algo --facet env --ci bootstrap --out figs/main.pdf
oprl plot ablation results/raw.parquet --rows surrogate --cols advantage --out figs/ablation.pdf
```

### 8.4 边界（不做什么）

- **不做实验调度器 / 超参搜索**。命令行矩阵交给 shell 或 wandb sweeps；我们只管「跑完之后的事」。
- **不做 web dashboard**。tensorboard 够用，且 §7 的目标是省时间而不是造 UI。
- **不做数据库**。文件系统 + jsonl + parquet 缓存足够，且天然可 `rsync` / 打包给审稿人。
- **不替代 rliable**。严格的 RL 评估统计（performance profile、probability of improvement）导出 CSV 交给 [rliable](https://github.com/google-research/rliable) 更合适。我们保证**导出格式与它兼容**，不重复实现。
- 依赖：`plot` 需要 matplotlib，放 **`viz` extra**，核心不依赖。

---

## 9. 工程与 uv（硬要求）

用 uv，不用 conda。已确认环境：`uv 0.11.24`、`Python 3.11.6`。

### 9.1 `pyproject.toml` 要点

- 布局 `src/`，build backend **hatchling**（不用 `uv_build`：为将来可能的 C++/CUDA 扩展留后路，换 backend 在有下游用户后成本很高）。
- `requires-python = ">=3.10"`（Gymnasium 1.3 要求 ≥3.10）。
- **核心依赖只有三个**：`gymnasium>=1.3`、`numpy>=2.0`、torch（见下）。轻量是卖点，要守住。
- **torch 的 CUDA 索引问题**——uv 的标准解法是 `explicit = true`，把索引限定到只服务被显式 pin 的包，否则 uv 会拿 `download.pytorch.org` 去解析 numpy 等所有包：

```toml
[project.optional-dependencies]
# --- 计算后端（互斥，用户选一个）---
cpu   = ["torch>=2.11"]
cu128 = ["torch>=2.11"]
cu130 = ["torch>=2.11"]

# --- benchmark 套件（对应 §6.1 的 Tier）---
minatar  = ["minatar"]                        # Tier 0：进 CI，最轻
mujoco   = ["mujoco>=3.11"]                   # Tier 1
atari    = ["ale-py>=0.12.1", "envpool"]      # Tier 1（0.12.0 被 yank）
minigrid = ["minigrid>=3.1"]                  # Tier 1
# isaac 不在此列：Isaac Lab 自带环境与安装流程，见下
robotics = ["gymnasium-robotics>=1.4.2"]      # Tier 2
dmc      = ["shimmy[dm-control]", "dm-control"]  # Tier 2
metaworld = ["metaworld"]                     # Tier 2
jaxenvs  = ["brax", "mujoco-mjx", "craftax"]  # Tier 2，与 torch 走 dlpack 桥

# --- 便利聚合 ---
bench-ci = ["oprl[minatar]"]                              # CI 用
bench    = ["oprl[minatar,mujoco,atari,minigrid]"]        # Tier 0+1
track    = ["wandb", "tensorboard"]   # 可选 Sink；核心 logger 只用 stdlib（§4.9）
viz      = ["matplotlib>=3.9"]        # oprl.plot（§8.3）；核心不依赖
results  = ["pyarrow", "pandas"]      # parquet 缓存 + df.to_pandas()（§8.2）

[tool.uv]
conflicts = [[{extra="cpu"}, {extra="cu128"}, {extra="cu130"}]]

[tool.uv.sources]
torch = [
  {index="pytorch-cpu",   extra="cpu"},
  {index="pytorch-cu128", extra="cu128"},
  {index="pytorch-cu130", extra="cu130"},
]

[[tool.uv.index]]
name="pytorch-cu130"; url="https://download.pytorch.org/whl/cu130"; explicit=true
# ... cpu / cu128 同理

[dependency-groups]        # PEP 735，不发布给用户
dev = ["pytest>=8.3", "pytest-xdist", "ruff>=0.9"]
```

- **`jaxenvs` 是唯一有真实冲突风险的 extra**（jax 与 torch 争 CUDA 版本）。它是 Tier 2、best-effort，文档里要明说建议单独装在独立 venv。
- **Isaac Lab 不进 extras**：它自带 Python 环境与安装流程，我们只提供 `TensorEnvAdapter` + 文档说明如何在 Isaac 的环境里 `uv pip install oprl`。反过来把 Isaac 塞进我们的依赖树只会两边都装不上。
- 注：Atari ROM 自 `ale-py 0.9` 起随 wheel 分发，**`AutoROM` / `accept-rom-license` 已废弃**，README 里别再写老指令。
- 用户安装：`uv sync --extra cu130 --extra bench --group dev` / 笔记本 `uv sync --extra cpu --extra minatar`。
- **env 套件用 extras 而不是 uv workspace**：workspace 全员共享一个 `requires-python` 交集和一个 lockfile，而 RL 的 env 依赖（mujoco / isaac / jax 系）极易互斥，workspace 对此无解。单包 + extras 对用户也更简单（`pip install oprl[mujoco]`）。
- CI：`astral-sh/setup-uv` + `uv sync --locked`（**`--locked` 而非 `--frozen`**：前者会在 `pyproject.toml` 与 lock 漂移时报错，能抓到"改了依赖忘了 relock"）。
- 提交 `uv.lock`。Docker 分两层（先只装依赖，再装项目），`UV_LINK_MODE=copy`，`.venv` 进 `.dockerignore`。

### 9.2 质量基线

`ruff`（lint+format）；类型检查用 `mypy` 做 CI 闸门（`ty` 还在 0.0.x beta，本地用可以，别当 gate）；pytest 分 marker（`slow` / `gpu` / `mujoco`）。

---

## 10. 正确性与基准（这是可信度的来源，不是可选项）

单靠 smoke test 不足以说明一个 RL 框架是对的。四层验证：

1. **解析测试**：GAE 对暴力 for 循环参考实现逐元素比对；构造 `γ` 与 `λ` 的边界（λ=0 退化为 TD(0)，λ=1 退化为 MC）。
2. **Bootstrap 语义测试**：造一个无真终止、reward 恒 1、固定 horizon 截断的 env，正确实现必须收敛到 `V → 1/(1−γ)`；错误实现（用 `done` 塌缩）会收敛到依赖 horizon 的更小值。**这个测试直接钉死 §4.1 那个全行业 bug。**
3. **Autoreset 一致性测试**：同一个 env 在 `NEXT_STEP` / `SAME_STEP` / `DISABLED` 三种模式下，收集到的有效 transition 集合必须一致。
4. **基准回归（分层，对应 §6.1 的 Tier）**：
   - **Tier 0 进 CI，每次 push**：CartPole / Pendulum / **MinAtar (Breakout, Asterix)**，各 3 seed。MinAtar 让「完整 PPO 训练真的还能学」在 CPU 上几分钟内可验证 —— **这是把"算法退化"从人工发现变成 CI 拦截的关键**，其他框架普遍只有 smoke test。
   - **Tier 1 定期跑（nightly/release）**：MuJoCo（HalfCheetah/Walker2d/Ant/Humanoid）、Atari（Pong/Breakout/BeamRider）、MiniGrid（DoorKey/KeyCorridor）、Isaac（Ant/Anymal），各 3–5 seed。
   - 曲线存档进 `benchmarks/reference_curves/`；断言"最终回报下限 + SPS 下限"。

同时**必须交付一套调好的超参**（`configs/ppo/{classic,minatar,atari,minigrid,mujoco,isaac}.yaml`）。SB3 的 RL Zoo 和 CleanRL 的 benchmark 才是它们真正的护城河——一个没有可复现曲线的新框架，无论代码多漂亮都没人敢用。

---

## 11. 路线图

benchmark 不是「最后再适配」的收尾工作 —— 它们各自解锁一条代码路径，所以**按能力解锁顺序穿插进里程碑**：

| 阶段 | 内容 | 解锁的 benchmark | 完成标志 |
|---|---|---|---|
| **M0 骨架** | pyproject/uv、CI（含 §4.7 依赖方向检查）、`types` `tree` `config` `logger` `sinks` `timer` | — | `uv sync` 通，CI 绿；`metrics.jsonl` 可离线读 |
| **M1 正确的 PPO** | `policy` `buffer`+**`Schema`** `advantages/gae` `norm` `nets` `GymVecAdapter` + `ppo.py` + `--smoke` | classic control | §10 的 1–3 全过；CartPole/Pendulum 收敛；§4.8 预算全部达标 |
| **M2 CI 可信度** | CNN encoder、`make_env`、`presets.py`、`health.py`（早停诊断） | **MinAtar** | MinAtar 3-seed 曲线进 CI；**从此每次 push 都验证算法能学** |
| **M3 学术可比** | obs/reward 归一化定稿、连续动作 head、MuJoCo 超参、**`oprl.results` + `oprl.plot`（§8）** | **MuJoCo v5** | 曲线与 SB3/CleanRL 公开结果对得上；**论文级图可一条命令产出** |
| **M4 算法库（A 类）** | `Surrogate` 协议 + `surrogates.py`：TR-PPO / SPO / DPO / MDPO / PPO-RPE / LPO / **APO** | — | **7 个算法，每个 ~15 行**；各有 vs vanilla 同 seed 对照曲线（§5.3） |
| **M5 形态广度** | dict obs（`tree.py`）、`SequenceSampler` + 循环策略、`EnvPoolAdapter`、多离散/动作 mask | **MiniGrid、Atari** | LSTM 策略与前馈**共用同一个 `ppo.py`**；MiniGrid 稀疏奖励能解 |
| **M6 GPU 原生** | `TensorEnvAdapter`、`torch.compile`（默认关）、多进程跑多种子 | **Isaac Lab**、Brax/MJX | 零拷贝路径验证；`timer` 显示 env 占比不再是瓶颈 |
| **M7 算法库（D 类）** | `AdvantageEstimator` 协议 + **DAE**、**RVL**、**GA2E 迁移**（含 `on_epoch_start`） | — | DAE/RVL 与 A 类组合消融跑通；**GA2E 迁移后复现迁移前曲线**（3-seed 均值容差内，§4.7） |
| **M8 算法库（B/C 类）** | batch-size invariance；`ppg.py` + aux phase（覆盖 PPG/DNA/DAAC/IDAAC） | — | 对照曲线齐备 |
| **M9 长尾** | Tier 2 adapter（DM Control/Robotics/MetaWorld/Procgen2/Craftax）、AGAC、V-MPO、Reflective-RPO | Tier 2 全部 | 按需，不承诺曲线 |

**M3 是对外分水岭**：在拿出可复现的 MuJoCo 曲线之前，这个项目对外没有说服力。
**M2 是对内分水岭**：MinAtar 进 CI 之后，后续所有重构才有安全网 —— 所以它排在 MuJoCo 之前，尽管 MuJoCo 才是对外的门面。
**M4 排在形态广度之前**：7 个 surrogate 算法总共约 100 行，却是本框架**最独特的卖点**（§5.2）；而且它们只依赖 M1–M3 已完成的部分，不需要 dict obs 或 RNN。**先做投入产出比最高的。**
**M7（DAE/RVL/GA2E）值得单列一个里程碑**：它引入第六个扩展点，是全部算法里唯一会碰 `advantages/` 的一类 —— 必须在 A 类对照曲线已稳定之后再做，否则出问题分不清是谁的锅。**GA2E 迁移（`ga2e.py` 782 行 → 约 200 行）是框架价值主张的实证**，作为正式验收项。
**§8 的出图流水线提到 M3**：因为它是「拿出可复现曲线」这件事的组成部分 —— 没有归档和统计正确的图，M3 的产出无法被别人（或半年后的你自己）检验。

---

## 12. 待定决策（需要你拍板）

1. **包名**：`oprl` / `opol` / `onrl` / 其他？
2. **目标用户**：(a) 你自己做研究的自用框架 → 可以更激进、少写文档；(b) 打算开源给别人用 → M3 的基准和文档必须做扎实。这个选择影响后续所有优先级。
3. **M5/M6 的先后**：你的实际研究更需要 MiniGrid/Atari（形态广度）还是 Isaac（GPU 原生）？谁在前谁先做。
4. **是否要 `tensordict`**：我倾向不要（保轻量，自研 100 行 `tree.py`）。若你后续想和 TorchRL 生态互操作，则应该早引入。
5. **早停要多激进**（§7.1）：自动终止「看起来失败」的 run 能省大量时间，但误杀一个其实会晚熟的 run 更糟。倾向 (a) 只警告不终止，(b) 只在硬故障（NaN / KL 爆）时终止，还是 (c) 连"长期无改善"也终止？我建议 (b) 作为默认、(c) 作为可选开关。
6. **算法库的验证深度**（§5.3）：每个算法都要求 MinAtar + MuJoCo 双对照曲线，成本不低（17 个算法 × 3 seed × 2 任务）。是否降为「MinAtar 必须，MuJoCo 仅对 A 类中表现最好的 3 个 + DAE/RVL」？
7. **APO 的引用年份**（§5.1）：Crossref 的 `published-print` 是 2027-01，但卷号/索引是 2026。文档里我按 "Neural Networks 205:109476, 2026" 写 —— 若你手上有 PDF，确认一下正式年份。
8. **`Schema` 的静态类型代价**（§4.3）：声明式 buffer 换来「零 buffer 代码」，但 `buf.obs` 变成动态属性，IDE 补全变弱。我的对策是 minibatch 用 `NamedTuple`、核心字段留显式注解。你更在意**补全体验**还是**零样板**？若前者，可退回「核心字段硬编码 + 只有 extra 走 schema」的折中。
9. **GA2E 的 `epoch` 刷新模式是否要支持**（§4.7）：开销约 `19 × n_epochs` 次反传，`num_epochs=10` 时**可能超过 PPO 主更新本身**，与 §7 的目标冲突。我倾向默认 `rollout` 模式、`epoch` 标为「昂贵，仅消融用」。你实际用哪个？
10. **`buffer.py` 那 516 行 Lepski 要不要迁**（§4.7）：`ga2e.py` 的 docstring 说它已被 gradient-alignment 取代。是彻底弃用，还是保留为 `advantages/lepski.py` 作对照基线？
11. **GA2E 迁移的 numpy → torch**：现有实现纯 numpy（94 处 `np.`，0 处 `torch.`），且注释里多处强调「保证位一致」。迁到 GPU torch 后浮点累加顺序必然改变。验收口径我定为「3-seed 均值容差内重合」—— 能接受吗？若你确实依赖位一致性做对照实验，就只能保留 numpy 路径（放弃 GPU 常驻与 `torch.compile`）。
12. **是否需要 `oprl.results` 读取 PGAE 现有 run 数据**？你已有大量 SB3 格式产出（`progress.csv` / tb event）。要兼容就多写一个 reader —— 不难，但要现在就知道。
