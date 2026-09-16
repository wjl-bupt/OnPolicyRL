# 算法接入设计（v3 草案）

> 状态：**待 check** · 2026-09-15
> 本文档是算法接入层的重设计，回应两条要求：①先预留科研路线的完整链路（接口级定稿，
> 实现可后置，见 §1）；②算法接入重新设计（§2–§8）。**本文档只是设计——落地迁移清单在
> §10，待 check 通过后另起一轮执行。**
> 上游：[DESIGN_v2.md](DESIGN_v2.md)（生命周期与预算）；本文取代其 §3.5 的接口部分。

---

## 0. 设计目标与反目标

**目标**（按优先级）：

1. **接入成本恒定**：接入任何已知的 on-policy 算法变体，都不需要改框架、不需要写
   no-op 样板、不需要猜隐式钩子。
2. **表达力闭合**：下文 §7 的映射表覆盖已知算法类型；每个算法类型都能指出"接入哪一层、
   写多少行"。映射表就是验收标准——出现映射不进去的算法类型，先改设计再写代码。
3. **签名稳定**：接口对象只增不改（append-only），新信息进上下文，不破坏已有组件。
4. **错误前移**：声明（schema/能力/签名）在装配期校验，错误出现在第一步训练之前，
   而不是第 3 小时。

**反目标**：不做通用算法 DSL；不追求"任何 RL 算法都能表达"（只覆盖单智能体 on-policy
同构族，DESIGN.md §2 的窄化不变）；不引入回调注册/事件总线。

---

## 1. 科研链路全图（接口级预留，不实现）

链路六阶段，每阶段给出**入口签名 + 工件 schema**。本框架当前只有 `train` 一段是实；
其余按里程碑排期。这一节是"预留"——实现顺序可调，接口先钉死。

```
┌─ ① declare ──→ ② run ──→ ③ monitor ──→ ④ report ──→ ⑤ search ──→ ⑥ discover
   实验 YAML       runner      只汇报        三件套        驱动器        候选评估
  [R2]           [R2]        [R2]          [R3]          [R4]          [R5]
```

| 阶段 | 入口签名（预留） | 工件 |
|---|---|---|
| ① declare | `runner.expand(exp: Experiment) -> list[Job]`；`Job = {algo, env, seed, overrides, run_dir, identity: hash}` | `experiments/*.yaml`（顶级键 ≤15） |
| ② run | `runner.run(exp) -> Summary`；作业身份存在且未完成 → 自动 resume | `runs/<exp>/<arm>/<env>-seed<k>/` |
| ③ monitor | `monitor.observe(stats: dict, *, global_step, iteration) -> None`（**只汇报，不停止**）；`evaluate(policy, env_factory, episodes, norm_state) -> dict` | `anomalies.jsonl`；`eval/` 前缀指标 |
| ④ report | `report.build(exp) -> Report` | `results.json` + `curves/<env>.png` + `summary.md`；指标 = `rew_last10` / `rew_last30` / `final_eval`，跨 seed mean±std |
| ⑤ search | `space.expand(spec) -> list[point]`（choice 维网格）；`space.sample(spec, n) -> list[point]`（参数维）；`point -> overrides` 确定性可哈希 | 实验文件 `space:` 段；optuna 适配在 extra |
| ⑥ discover | `evaluate_candidates(cands: list[Candidate], protocol: Experiment) -> list[ResultCard]`；baseline 按 config_hash 共享缓存 | `ResultCard = {spec, config_hash, per_env, vs_baseline, anomalies, cost, status, run_dir}` |

**checkpoint（②的一部分）**：`{policy, opt, obs_norm, reward_norm, component_states,
rng_state, global_step, iteration, config_hash}`；验收 = 同 seed+deterministic 下续跑
与不中断 run 逐位一致。组件状态如何进 checkpoint 见 §6（组件零样板）。

算法接入层（本文档主题）服务的位置：②的内部。链路其他阶段把算法当黑箱——只认
`train(cfg, env, policy, log, components)` 与 stats dict。这是"算法接入"与"科研流程"
两层互不污染的边界。

---

## 2. 现状痛点（为什么 R1 之后还要重设计）

1. **Surrogate 签名锁死在 ratio 域**。`(ratio, logp, logp_old, adv, cfg)` 表达不了
   分布域目标——V-MPO 的 M-step 是对非参数目标分布 ψ 的 KL 约束，根本不以 ratio 为
   中心；这类算法被迫开独立训练循环文件。
2. **能力钩子是 getattr 探测的隐式契约**。`critic_loss`/`on_epoch_start` 靠属性探测，
   签名漂移到运行中才炸；每类新需求 = 新钩子名。"不堆叠"的要求换了个形式继续堆。
3. **组件的数据需求不自描述**。想多存 `probs` 要绕道 config 的 `buffer.extra`；
   rollout 期写入（`write_extra`）刚被删掉，需求本身还在。
4. **三种组件三套接口风格**，且都是窄签名：每多传一个信息就破坏一次签名。
5. **"共享循环 vs 独立循环"边界模糊**。`_common.setup` 拼装 + 独立 `train()` 的组合，
   曾产生 vmpo 那种半共享半复制的状态；逃生舱没有正当名分。

---

## 3. 重设计总览：一个形状、两个上下文、两个插槽、一个逃生舱

```
                    ┌──────────────────────────────────────────────┐
   默认循环（PPO）  │  rollout → estimate → [phases] → epochs ×    │
                    │  minibatches { evaluate ×1 → Batch →         │
                    │  policy_loss + entropy + critic_term → step }│
                    └──────────────────────────────────────────────┘
   组件插槽：  Estimator.estimate(rollout) -> Estimate          （2）
              PolicyLoss(batch) -> (loss, stats)                （1）
   逃生舱：    UpdateRule.update(rollout, policy, opt, log)      （循环不同时）
```

所有组件一个形状：`__init__(自有超参)` + 一个主方法 + **显式声明**
（`schema` / `state_vars` / `observe`，见 §5–§6）。没有 getattr 探测，没有继承要求。

**插槽轴共五个**（= §9.2 阶段上的全部插拔点）：`network`（S0）、`estimator`（S3）、
`phase`（S4/S6）、`policy_loss`（S9）、`value_loss`（S10）。其中 `value_loss` 与
`network` **沿用现有机制**（value_loss 注册 kind 与 ActorCritic 组装器，不需要新接口）；
本节详述需要新接口的三个：estimator（§5.1）、phase（§5.2）、policy_loss（§5.3）。

---

## 4. 上下文对象（append-only，签名稳定的机制保证）

### 4.1 `Rollout` —— 整段 rollout 的只读视图

```python
@dataclass(frozen=True)
class Rollout:
    buf: RolloutBuffer                  # 现有 GPU 常驻 buffer（只读约定）
    # 便捷视图（全部转发到 buf，不再让组件摸 buf 内部布局）
    masks: Masks
    values: Tensor                      # [T, N]
    bootstrap_value: Tensor             # [N]
    reward: Tensor
    segments: Callable[..., list]       # buf.segments()，时间结构
    iter_trajectories: Callable         # buf.iter_trajectories()
    # 组件声明并写入的整段字段（如 APO 的锚点、episode 边界）
    extra: dict[str, Tensor]
    # 环境侧上下文
    policy: Policy                      # GA2E 的反传需求从这里拿
    cfg: Config
```

### 4.2 `Batch` —— 一个 minibatch 的全部可用信息

```python
@dataclass(frozen=True)
class Batch:
    obs: Obs
    action: Tensor
    logp_old: Tensor                    # 采集时的 log π_old
    value_old: Tensor
    reward: Tensor
    masks: Masks                        # valid 在内，loss 自行剔除（iter 已剔除假步）
    advantages: Tensor
    returns: Tensor
    # 框架每 minibatch 调一次 policy.evaluate 后填入——loss 组件拿到即算，不再自调
    logp: Tensor
    entropy: Tensor
    value: Tensor
    # 分布域目标的支持方式（待定 #2）：惰性求值 + 每 minibatch 缓存
    def dist(self): ...                 # policy 提供 dist 能力时可用，否则明确报错
    # 编排上下文
    flat_idx: Tensor
    epoch: int
    progress: float                     # 0→1，替代 cfg._progress
    cfg: Config
    extra: dict[str, Tensor]            # 本 minibatch 切到的组件 schema 字段
```

**设计规则**：Batch 字段只增不改。组件从 Batch 取所需；框架保证"一次 evaluate"
（这同时消灭了组件各自重 evaluate 的路径分歧——APO 那类 bug 的结构性根源）。

---

## 5. 两个插槽 + 一个覆盖点

### 5.1 `Estimator.estimate(rollout) -> Estimate`

```python
@dataclass
class Estimate:
    advantages: Tensor                              # [T, N]，必填
    value_targets: Tensor                           # [T, N]，必填
    critic_term: Callable[[Batch], tuple[Tensor, dict]] | None = None
    #    ↑ 覆盖默认 critic 项（value_loss）。注意是"损失项"不是"额外优化阶段"
    phases: list[Phase] = ()
    diag: dict[str, float] = {}
```

### 5.2 `Phase` —— 用一个声明机制取代 N 个命名钩子（关键决策）

任何"在特定时机插入的额外计算"统一声明为 Phase，**时机是封闭集合**：

```python
@dataclass
class Phase:
    when: Literal["pre_epochs", "epoch_start", "pre_minibatch"]   # 封闭，新增须改框架
    fn: Callable[[PhaseCtx], dict]      # 返回 stats，进 logger；签名启动期校验

@dataclass
class PhaseCtx:
    policy: Policy; rollout: Rollout; loss: PolicyLoss | None
    cfg: Config; epoch: int
```

默认循环按 `pre_epochs → (epoch_start → minibatches × num_epochs)` 执行。映射：

| 时机 | 使用者 | 干什么 |
|---|---|---|
| `pre_epochs` | DAE | 望远镜残差 critic 优化（需要连续时间，minibatch 打乱前跑） |
| `epoch_start` | GA2E（epoch 模式） | 在 θ_k 重选 λ、重写 advantages |
| `pre_minibatch` | （APO 类） | 每 minibatch 的锚点重评估 |

**这是"钩子堆叠"的解法**：不再是每类需求一个钩子名 + getattr 探测，而是一个封闭的
时机集合 + 一个统一的声明。新增时机 = 改框架 + 给出第一个使用者（与 DESIGN_v2 §3.5
的纪律一致）。

### 5.3 `PolicyLoss(batch) -> (loss, stats)`

只拥有策略项。熵项与 critic 项由默认循环合成：

```python
loss = policy_loss(batch) - cfg.ent_coef * batch.entropy.mean() + cfg.vf_coef * v_term
```

`v_term = estimate.critic_term(batch)` 若声明，否则 `value_loss(batch.value,
batch.returns, batch.value_old, cfg)`。ratio 域与分布域目标现在**同等可表达**
（Batch 里有 logp/logp_old/dist()）。

---

## 6. 数据需求与状态：显式声明，框架代劳

```python
class MyEstimator:
    schema = {"probs": Field(("n_actions",), torch.float32, doc="...")}
    #        ↑ 框架合并进 buffer（与 config buffer.extra 冲突即报错）
    state_vars = ("_ema",)
    #        ↑ 框架快照进 checkpoint——组件不再写 state_dict/load_state_dict

    def observe(self, policy, obs) -> dict:      # 仅当声明了 schema 才可提供
        return {"probs": policy.dist(obs).probs} # rollout 期逐帧写入，唯一合法路径

    def estimate(self, rollout: Rollout) -> Estimate: ...
```

- `observe` 的存在性由"是否声明 schema"推出，装配期校验（声明了字段却不会写 =
  第一步前报错）。
- `state_vars` 的值须是 JSON 可序列化标量（float/bool/None）；张量状态属于 policy，
  不属于组件。

---

## 7. 映射验证表（设计够不够用的判据）

| 算法 | 接入层 | 用到的机制 | 预计量 |
|---|---|---|---|
| GAE | Estimator | `estimate` 纯函数 | ~20 行 |
| PPO clip | PolicyLoss | `Batch`（ratio = exp(logp−logp_old) 在组件内一行算出） | ~10 行 |
| **GA2E（你的）** | Estimator | `policy` 反传 + `state_vars` EMA + `phase(epoch_start)` 重选 λ + `iter_trajectories` | ~200 行，**零框架钩子** |
| DAE | Estimator | `schema(probs)` + `observe` + `critic_term` + `phase(pre_epochs)` | ~120 行 |
| RVL | Estimator | `critic_term` | ~60 行 |
| V-MPO | UpdateRule | EM + 对偶变量，循环自定 | ~250 行 |
| PPG | UpdateRule | 双相位循环 | ~200 行 |
| APO | PolicyLoss + schema 锚点字段 + `phase(pre_minibatch)` | 锚点重评估 | 可表达（略别扭；已删除，仅验证设计） |

结论：Estimator+PolicyLoss 覆盖"目标/估计"类研究（绝大多数）；UpdateRule 覆盖
"循环结构"类。没有映射不进去的已知类型。

---

## 8. UpdateRule 逃生舱

```python
class UpdateRule(Protocol):
    def update(self, rollout: Rollout, policy: Policy, opt, log: Logger) -> dict:
        """一次完整更新。返回 stats（'_' 前缀键为私有约定，不进 logger）。"""
```

- **PPO 自己就是 UpdateRule 的一个实现**（`StandardUpdate`，即 §3 的默认循环）。
  想改循环结构：整份复制 `StandardUpdate.update` 去改（CleanRL 式可读），契约不变，
  采集/buffer/日志/checkpoint/monitor 全部仍然由框架托管。
- 这正式化解了 v0.1 的老矛盾："共享样板（库级复用）vs 独立文件（可读性）"——
  共享的是契约与原语，**循环可以复制**，复制是显式且正当的。
- 框架对 UpdateRule 的期望：返回的 stats 含 monitor 需要的标准键
  （`diag/kl`、`loss/*`、熵等，缺省键只影响诊断丰富度，不报错）。
- **分支基座**：UpdateRule 的注册实例按算法分类组织——每个**分支**（目标机制不同的
  算法族）的 baseline 各是一条 UpdateRule，分支内变体走 §5 的插槽。分类定稿见
  [algo_taxonomy.md](algo_taxonomy.md)：B1 PPO（baseline PPO）/ B2 V-MPO
  （baseline V-MPO）/ B3 TRPO（预留）；各分支消费哪些插槽 kind 由该表固定，装配期
  据此校验（分支不消费的 kind 声明进来 = 第一步前报错）。

---

## 9. 分支 base 流程拆分（草案——插件化友好性的关键一节，待 check）

分类定下分支基座之后，问题变成：**把每条基座的流程拆到什么粒度，才能让后续变体
"插得进去、拔得出来"**。本节给出拆分方案；依据来自 [algo_taxonomy.md](algo_taxonomy.md)
的两个分支基座（PPO、V-MPO）。

### 9.1 拆分原则

1. **两级拆分，不引入第三级**：
   - **框架级阶段**（S1–S2、S7–S8、S10–S11）：与算法无关，框架拥有、不可插拔、
     各自可独立测试；
   - **分支级插槽**（S3–S4、S6、S9）：分支基座在阶段挂上默认组件，变体 = 组件替换。
2. **基座文件 = 流程清单**。打开 `ppo.py`，读到的就是它经过的阶段序列——每一行要么是
   框架阶段调用，要么是插槽组件，不允许第三种形态（否则拆分就失效了）。
3. **拒绝 pipeline-as-data**（配置里声明阶段序列再由引擎执行）：那会让流程脱离代码
   可读性，重演"配置黑箱"（DESIGN.md §2 一贯反对）。阶段序列在基座代码里显式写死，
   换组合 = 换基座文件（UpdateRule 逃生舱），不在配置里拼装。

### 9.2 标准阶段清单（PPO 的完整 anatomy：`policy_init → rollout → buffer → advantage → policy_loss/value_loss → network update → 循环`）

| # | 阶段 | 输入 → 输出 | 归属 | 插拔点 |
|---|---|---|---|---|
| S0 | **init** | seed/env → **policy 构建**（network 轴）→ optimizer / normalizer / **buffer 分配**（声明式 schema） | 框架 + **network 轴** | ✓ **network**（encoder/heads/组装器）；optimizer 形态与超参走 config |
| S1 | **rollout** | (env, policy, buf) → 填满 buf, last_obs | 框架 | ✗（策略经 Policy 协议已可换；estimator 的 `observe` 写入挂此） |
| S2 | bootstrap | buf → V(s_T) | 框架 | ✗ |
| S3 | **advantage** | Rollout → Estimate(adv, ret, critic_term, phases) | 组件 | ✓ **estimator**（其 `schema` 声明 = buffer 增量字段的唯一来源） |
| S4 | pre-phases | PhaseCtx → stats | 组件 | ✓ phase(pre_epochs) |
| S5 | epoch 循环 | n_epochs 次 {S6→S7→…} | 基座 config | 数值可调 |
| S6 | epoch_start | PhaseCtx → stats | 组件 | ✓ phase(epoch_start) |
| S7 | **buffer 切分** | buf → Batch 序列（iter_minibatches） | 框架 | ✗（未来的 SequenceSampler 挂在此处，仍属框架） |
| S8 | evaluate | Batch ← policy.evaluate ×1 | 框架 | ✗ |
| S9 | **policy_loss** | Batch → 策略项标量 | 组件 | ✓ **policy_loss** 轴 |
| S10 | **value_loss** | value, returns, value_old → critic 项标量 | 组件 | ✓ **value_loss** 轴（Estimate.critic_term 可整体覆盖） |
| S11 | **network update** | total = S9 − ent_coef·H + vf_coef·S10 → backward + grad-clip + step | 框架 | ✗（系数与裁剪阈值是 config） |
| S12 | post-iter | log / checkpoint / monitor(只汇报) | 框架 | ✗ |

**五个插槽轴**（变体的全部合法落点）：`network`（S0）/ `estimator`（S3）/
`phase`（S4、S6）/ `policy_loss`（S9）/ `value_loss`（S10）。
其余阶段框架拥有——这就是"三个变体够不够"的完整答案：不够，五轴封顶；
出现第五轴之外的变体诉求 = 先改本清单，再写代码。

### 9.3 两个基座的流程实例

**B1 PPO（= StandardUpdate）**：S0(MLP/CNN)→S1→S2→S3(GAE)→S4(无)→S5×S6(无)→S7→S8→
S9(clip)→S10(clipped value)→S11→S12。
可插拔面 = S0 network + S3 estimator + S4/S6 phase + S9 policy_loss + S10 value_loss。
**A2C = S9 换 vanilla_pg 损失**（−log π·A，无 ratio——它与 PPO clip 是同一插槽上的
两个不同目标），单 epoch/全批只是它的普通超参；**REINFORCE = vanilla_pg × MC 估计器
的组合**。

**B2 V-MPO**：S0→S1→S2→S3(GAE)→ **V1 old_dist 捕获**（分支私有阶段：需 `policy.dist`，
把 π_old 全分布存进 Rollout.extra）→ S5×S6(无)→S7→S8→ **V2 e_step**（分支自治插槽：
top-k + softmax(A/η) → ψ 权重 + η dual loss）→ **V3 m_step**（基座内核：加权 MLE +
解耦 KL 信任域 + α dual loss；**不是插槽**——约束形式是机制本体，`constraint: kl|none`
是基座 config）→ S9 缺席（无 policy_loss 概念）→S10(v_loss)→S11 变体合成
（total = m_step + η/α dual + vf_coef·v_loss）→S12。

V1 是新识别的分支专属阶段形态：**分支基座可以在标准阶段序列中插入自己的私有阶段**，
只要它声明清楚输入输出并复用框架阶段的原语。分支自治插槽（e_step）的注册与装配校验
随之定稿：由基座声明 `slots = {"e_step": EStep}`，框架按分支的插槽名单校验 spec
（B1 里传 e_step、B2 里传 policy_loss → 第一步前报错）。

### 9.4 拆分带来的判定规则（回答"这个新想法插在哪"）

| 想法性质 | 插入点 | 例 |
|---|---|---|
| 换网络结构/头（policy_init 期） | S0 **network 轴** | CNN/RNN encoder、DAE advantage head、PPG/DNA 双网络 |
| 换"怎么估 advantage/target" | S3 **estimator 轴** | GA2E、DAE、RVL、MC |
| 换"策略目标标量形式" | S9 **policy_loss 轴** | SPO、DPO、MDPO、vanilla_pg(A2C)… |
| 换"value 怎么回归" | S10 **value_loss 轴** | clipped / mse / huber / Estimate.critic_term 覆盖 |
| 在某个时机加一段计算 | S4/S6 **phase 轴** | DAE 望远镜 critic、GA2E epoch 重选 |
| 换"分 minibatch 的方式" | S7（框架扩展，不属算法变体） | 未来 SequenceSampler |
| 换"循环骨架本身" | 新基座文件（UpdateRule） | V-MPO 之于 PPO、PPG 若 phase 表达不了 |
| 需要第三网络 | policy 附件 + phase（不新增拆分级） | AGAC 类 |

**自检**：任何一个 on-policy 新想法都应该能在这张表里唯一落位；落不位 = 先改本节
拆分，再写代码。

## 10. 接入成本对照

| 接入什么 | R1 现状 | 本设计 |
|---|---|---|
| ratio 域策略目标 | ~10 行 | ~10 行（不变） |
| **分布域策略目标** | ❌ 只能开独立文件 | ~10–15 行（PolicyLoss + batch.dist()） |
| 新 estimator | ~6 行 + 隐式 getattr 钩子 | ~6 行 + 显式 schema/state 声明 |
| 改 critic 目标 | critic_loss 钩子 | `critic_term` 显式返回值 |
| 采集期额外存储 | config buffer.extra 绕道 | `schema` + `observe` 一等公民 |
| 跨迭代状态 | 手写 state_dict ×2 | `state_vars = ("_ema",)` 一行 |
| 新循环结构 | 独立文件 + 拼装 _common | UpdateRule 一个类 |
| 签名漂移 | 运行中炸 | 装配期 inspect 校验，第一步前报错 |

---

## 11. 落地迁移清单（**check 通过后另起一轮执行**，本文档不动代码）

1. `objectives/ppo_family.py`：`Surrogate` → `PolicyLoss(Batch)`；ClipLoss 重写 ~8 行；
   `ppo_family` 更名建议为 `objectives/policy.py`（待定 #4）。
2. `advantages/base.py`：`compute` → `estimate` + `Estimate`/`Phase`/`PhaseCtx`；
   `BaseEstimator` 缩为无状态默认；删除 capability getattr 约定。
3. `algos/ppo.py`：Batch 组装 + Phase 执行器 + 损失合成（`StandardUpdate`）；
   删除 `_progress` 直写（进 Batch.progress）。
4. `buffer.py`/`_common.py`：组件 `schema` 合并进 buffer；`observe` 接线进 `collect`。
5. `checkpoint.py`（新，R2 本来就要做）：`state_vars` 快照进 ckpt。
6. `diy/advantages/ga2e.py`：迁移到 `estimate + phases(epoch_start) + state_vars`；
   预期净删 `state_dict/load_state_dict/on_epoch_start` 样板。
7. 测试：`test_minimal_integration.py` 改写为裸 PolicyLoss/Estimator 各一方法；
   新增装配期校验测试（声明 schema 无 observe、Phase 签名错 → 第一步前报错）；
   GA2E 数值回归不变。
8. 预算更新（DESIGN_v2 §9）：接入成本表换成 §9 的新对照。

## 12. 待你 check 时拍板的问题

1. **PolicyLoss 形态**：`__call__(batch)`（现 Protocol 风格）还是 `loss(batch)`？
   倾向 `__call__`（组件即函数）。
2. **分布域支持方式**：`batch.dist()` 惰性 + 要求 policy 提供可选 `dist` 能力
   （ActorCritic 已有），还是把 dist 塞进 `evaluate` 返回值（改 Policy 协议）？
   倾向前者（Policy 协议保持 4 方法不动）。
3. **Phase 时机集合**：`pre_epochs / epoch_start / pre_minibatch` 三值够吗？
   要不要 `post_epochs`（如 PPG 的 aux phase —— 但那是 UpdateRule 场景，倾向不加）。
4. **组件 kind 命名**：`policy_loss` 是否更名 `objective`？倾向不改（避免无谓 churn）。
5. **Estimate 必填项**：advantages + value_targets 必填、其余可选——确认。
6. **UpdateRule 的 stats 契约**：仅约定标准键缺失不报错（诊断降级）——确认。
