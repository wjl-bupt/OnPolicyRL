# On-Policy 算法分类（分支 × baseline 基座）

> 状态：草案 v0.3（**未定稿**——分支 base 流程拆分见 [algo_design.md §9](algo_design.md)，定稿在拆分 check 之后）· 2026-09-16
> 分类判据不变：**分支 = 目标函数核心机制不同**（需要各自的 UpdateRule 基线实现）；
> **变体 = 同一机制内的组件替换/增补**（走基座的插槽）。
> 已定决策（2026-09-16）：
> ① **A2C 属 B1 分支，身份是独立的 vanilla_pg 策略损失**（作者纠正）：它与 PPO clip 是
>   同一插槽上两个不同的目标，**不是 PPO 的 config 退化**。旧框架曾以 `clip=∞ + 单
>   epoch` 的 config 复现其梯度——那是实现复用驱动的等价（仅在 ratio=1 首遍成立），
>   不是分类学结论，本分类不继承；
> ② **V-MPO 独立分支 + e_step 分支自治插槽**，信任域形式为基座 config（受托决定）；
> ③ **AWR 不立分支**——它是 V-MPO 机制的固定温度退化，可表达为 B2 分支变体（如需）；
> ④ **组相对方法（RLOO/GRPO）整体不考虑**（作者定）；
> ⑤ **DAE / RVL 列入 B1 变体的 estimator 轴**（作者点名；机制上跨分支可用，见 §4.1）。
>
> **调研核实状态**（诚实声明）：本次补充调研发生在检索工具不可用的会话中（无法联网
> 核对）。条目分两级：`已核对` = DESIGN.md §5 的 2026-08-19 调研（dblp/OpenReview 核过
> venue）；`待核实` = 凭知识库补入，**进正选前必须补核 venue**，否则留待定池。
> 预印本一律不进正选，标注排除。

---

## 1. 分类判据

判据挂在 **algo_design §9 的流程骨架**上，不挂在"目标里有没有某个量"上：

1. 换掉它之后，**阶段序列骨架不动**（S1–S11 的结构不变，损失合成仍是标准形），
   只是 S3/S9 上的组件换了、或经 phase 插槽增补了计算 → **同分支变体**；
2. 换掉它之后，**骨架本身变了**——需要插入分支私有阶段、或 S9 的损失合成方式
   不再是标准形（policy_loss + 熵 + critic 项）→ **新分支**；
3. 需要第三个网络/额外网络件与机制无关 → 跨分支附件，不构成分支。

反例钉死：A2C 不是新分支——它与 PPO 共享同一骨架，只是 S9 换成 vanilla_pg 目标
（−log π·A，无 ratio）；DAE/RVL 不是分支（S3 组件）；PPG 不是分支（phase 增补）；
V-MPO 是分支（插入 V1–V3 私有阶段，损失合成不走标准形）。

---

## 2. 分支总表

| 分支 | 核心机制 | **baseline（基座）** | 基座暴露的插槽 | 优先级 |
|---|---|---|---|---|
| **B1 PPO 系** | 标准 on-policy 更新骨架（S1–S11）；S9 目标函数主流为 ratio 信任域 | **PPO** | policy_loss、estimator、phase、value_loss | **高**（主基座，StandardUpdate 参照实现） |
| **B2 V-MPO 系** | EM：非参数目标分布 ψ + 学习式 KL 约束 | **V-MPO** | estimator、**e_step**（分支自治）、value_loss | **中**（验证分支自治插槽机制） |
| **B3 TRPO 系** | 二阶自然梯度 | TRPO | estimator、value_loss | **预留**，不排期 |

已排除：组相对/LLM-RL 方法（RLOO、GRPO 及其族）——作者决定整体不考虑；
TRPO 系延用 v0.1 决定（机制装置只服务自身，B1 的 surrogate 变体已能拿到
trust-region 收益）。

---

## 3. 分支明细

### 3.1 B1 PPO 系（baseline：PPO）

机制：策略更新受信任域约束（PPO 用 ratio clip）。循环 = policy_init → rollout →
buffer → advantage → policy_loss/value_loss → update → 循环（algo_design §9.2 的
S0–S12）。B1 变体落在 **五个插槽轴**上（network / estimator / policy_loss /
value_loss / phase）；其中**已发表算法变体**集中在三个轴——(a) policy_loss、
(b) estimator、(c) loop。A2C 在 (a) 轴以 vanilla_pg 身份与 PPO 并列：**同一个循环
骨架、同一个插槽，不同的目标函数**。

**(a) policy_loss 变体**（S9 插槽——策略目标函数的替换）

| 变体 | 目标函数 | venue | 状态 |
|---|---|---|---|
| **A2C（vanilla_pg）** | −log π(a\|s)·A——无 ratio、无信任域的策略梯度 | Sutton & Barto 教材标准形 | 论文自明 |
| PPO clip | −min(r·A, clip(r)·A)，r = π_new/π_old | Schulman 2017 | **baseline** |
| TR-PPO | 出界时负 rollback 拉回，替代硬 clip | UAI 2019 | `已核对` |
| SPO | 去 clip，advantage 加权二次惩罚 | ICML 2025 | `已核对` |
| DPO | 元学习发现的 drift，tanh 软化（闭式） | NeurIPS 2022 | `已核对` |
| MDPO | 镜像下降：线性项 + 退火 reverse-KL | ICLR 2022 | `已核对` |
| PPO-RPE | 相对 Pearson 散度正则（对称约束） | ICRA 2021 | `已核对` |
| APO | clip + 未采样动作锚定（UARR） | Neural Networks 2026 | `已核对` |
| ppo_kl | clip + 显式 k3 KL 惩罚（GA2E 对照用） | Schulman 2017 eq.8 | 论文自明 |
| Dual-clip PPO | 负优势样本的 clip 再加下界 c | AAAI 2021（MOBA 游戏） | `待核实` |

REINFORCE = A2C 损失（vanilla_pg）× MC 估计器的**组合**，不是独立变体——组合空间
（policy_loss × estimator）由插槽自由组合，不设专门的分类条目。

**(b) estimator 变体**（S3 插槽——advantage/critic 估计的替换，循环不动）

| 变体 | 机制一句话 | venue | 状态 |
|---|---|---|---|
| **DAE** | 直接回归 advantage（per-action 头），value 经望远镜残差与之自洽 | NeurIPS 2022 | `已核对`（v0.1 调研） |
| **RVL** | critic 预测状态对反对称差值 Δ(sᵢ,sⱼ)，重建 GAE | ICLR 2026 Poster | `已核对`（v0.1 调研） |
| GA2E | 梯度对齐选 λ（**本项目研究主线**，diy 组件） | — | 自研 |
| V-trace | 重要性采样截断校正 | ICML 2018 | **排除**——off-policy 校正，违反"数据来自当前策略"不变量（DESIGN §2） |

**(c) loop 变体**（phase / 派生 UpdateRule；机制仍是 ratio 信任域，不入新分支）

| 变体 | 增补 | venue | 状态 |
|---|---|---|---|
| PPG | aux phase：策略/价值解耦交替训练 | ICML 2020 | `已核对` |
| PPG Reloaded | aux 相位实证修正 | ICML 2022 | `已核对` |
| DAAC / IDAAC | 解耦价值与策略 + 分布正则 | NeurIPS 2021 | `已核对` |
| DNA | 双网络架构 | NeurIPS 2022 | `已核对` |
| Batch-size invariance | 批不变性修正 | NeurIPS 2021 | `已核对` |
| Reflective-RPO | 反思项 | ICML 2024 | `已核对` |

**谱系注记**：Engstrom et al., "Implementation Matters in Deep RL"（ICLR 2020，`已核对`）
研究的是**代码实现**层面的相似与陷阱（实现细节足以让 VPG 变得像 TRPO）——它说明的
恰恰是：实现可以共享，**目标函数的身份不能混淆**。分类按目标函数走；实现层面的共享
由同一循环骨架（StandardUpdate）体现。

### 3.2 B2 V-MPO 系（baseline：V-MPO）

机制：无 ratio。E-step 按优势选 top-k 样本构造 ψ ∝ exp(A/η)；M-step 在 KL(π_old‖π_new)
约束下加权极大似然；η 与 KL 对偶变量**学习得到**。

| 角色 | 说明 | 状态 |
|---|---|---|
| 家族起源 | MPO（Abdolmaleki et al., **NeurIPS 2018**，`已核对`）——off-policy 原版，用 Q 与 E-step 对偶；V-MPO = 其 on-policy 化（V 代替 Q，消去 E-step 对偶），框架因 off-policy 反目标不收原版 | 注记 |
| **baseline** | V-MPO（ICLR 2020）；连续动作需 μ/σ 解耦 KL（旧实现要点，重建沿用） | `已核对` |
| e_step 插槽 | top-k 比例、温度、加权方式的替换点（分支自治，见 §5.3） | 设计项 |
| trust-region config | `constraint: kl \| none` 是基座配置而非插槽——约束形式是机制本体 | 设计项 |

**AWR 是什么（回应作者问题）**：Advantage-Weighted Regression（Peng et al., 2019,
arXiv 1910.00177，`已核对`——知名方法）。更新 = 加权极大似然，权重 ∝ exp(A/β)：
把"往高优势动作增加概率"变成加权监督学习步。它与 V-MPO 同属一个机制家族——
V-MPO 的加权 MLE 在**固定温度、全样本、无 KL 约束**下的形态。不立分支：若日后需要，
AWR = B2 基座 + `e_step: 全样本固定温度` + `constraint: none`。

**V-MPO 系补充变体（本次调研，均 `待核实`——补核 venue 前不入正选）**：

| 候选 | 机制一句话 | 说明 |
|---|---|---|
| DoC（Dichotomy of Control） | MPO 式策略改进 + action-free 奖励分解，on-policy | ICLR 2022？核实后定 |
| 正则化 MPO 变体族 | MPO 的 KL/正则系数改造 | 多为分布式/场景特定实现，逐条核对 |
| Vine / VAPO 等 LLM 时代方法 | 轨迹级/值感知的优势 | **预印本排除**（且 LLM 语境，与作者决策④同族） |

**影响力注记**：V-MPO 是 DeepMind 大规模训练的常用策略改进步（NGU / Agent57,
`已核对`），其稳定性口碑是本分支存在的实证理由——尽管公开的单机小 batch 复现极少，
这恰是本框架可以补的空档。

### 3.3 B3 TRPO 系（预留，不排期）

分类完整性保留；理由同 v0.1（§2 已注）。若未来做"估计器在二阶方向下的对照"再启用。

---

## 4. 跨分支组件轴

### 4.1 估计器 × 分支兼容性

| 估计器 | B1 PPO | B2 V-MPO | 备注 |
|---|---|---|---|
| GAE | ✓ | ✓ | 默认 |
| MC（λ=1） | ✓ | ✓ | B1 的 REINFORCE 端点 |
| GA2E（你的） | ✓ | 待验证 | 与 surrogate 耦合；B2 无 clip，λ 打分语义不同 |
| DAE | ✗ 离散限定 | ✗ 离散限定 | 需完整分布头 |
| RVL | ✓ | ✓ | critic 项替换，机制无关 |

### 4.2 网络件 / 附件轴

encoder/RNN/双网络（PPG/DNA）与分支正交，走 `network` 配置；第三网络类附件
（AGAC 等）挂靠分支循环 + 一个 phase，不构成分支，不进首批。

---

## 5. 与接入设计（algo_design v3）的对接

### 5.1 分支 = UpdateRule baseline

```
algos/
├─ ppo.py          # B1 baseline：PPO（StandardUpdate，参照实现）
└─ vmpo.py         # B2 baseline：V-MPO
```

`oprl algos` 列分支基座；分支内变体不注册 algo，是基座插槽上的组件。
config 一分支一文件：`config/ppo.yaml`、`config/vmpo.yaml`。

### 5.2 分支 × 插槽可用性（装配期校验依据）

| kind | B1 PPO | B2 V-MPO | 装配期行为 |
|---|---|---|---|
| policy_loss | ✓ | **✗** | 分支不消费的 kind 被声明 → 第一步前报错 |
| advantage | ✓ | ✓ | |
| phase | ✓ | ✓ | B2 的 loop 变体走 e_step |
| value_loss | ✓ | ✓ | |
| e_step | ✗ | ✓ | 分支自治 kind（§5.3） |

### 5.3 分支自治插槽（B2 的 e_step）

e_step 由 B2 基座**声明并注册**（`register_kind("e_step")` 的分支级用法），不在全局
固定 KINDS 里；全局 KINDS 保持封闭六种 + 分支注册的自治 kind。机制细节随流程拆分
（algo_design §9）一起定稿。

---

## 6. 实现优先级建议

1. **B1 baseline（PPO）**——StandardUpdate 是 v3 设计的参照实现，先做；
   A2C（vanilla_pg 组件 + 自身 config）与 REINFORCE（vanilla_pg × MC 组合）在基座
   全绿后按需接入，均为普通插槽件。
2. **B2 baseline（V-MPO）**——第二，验证分支自治插槽。
3. B3 预留。变体在 baseline 全绿后按研究需要逐个加，每个走 algo_design §7 行数预算。

## 7. 待定项（随流程拆分 check 一并收口）

1. e_step 自治 kind 的注册与装配校验细节（§5.3 → algo_design §9）。
2. B1 的 loop 变体（PPG 类）是 phase 组件还是派生 UpdateRule——待流程拆分定稿后判。
3. 组相对估计器的分组 schema——随 GRPO/RLOO 排除而**搁置**，仅当未来重启该方向再议。
4. **批量补核 `待核实` 条目**（Dual-clip PPO 的 venue、DoC、正则化 MPO 族）——检索工具
   可用时执行；补核通过才进正选，否则留待定池（§头部核实状态声明）。
