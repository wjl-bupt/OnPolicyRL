# oprl v2 —— 面向科研流程的 on-policy RL 框架设计

> 状态：设计草案 v2.2 · 2026-09-15
> v2.0 初稿后同日的修订（作者反馈）：
> ① **算法层重推导 + 瘦身**：src 只保留最小基线核心，其余算法处置见（§3）；
> ② **健康监控只汇报、永不停止实验**（§5.2）；结果产出收敛为三件套：
>    `results.json` + 基础曲线图 + 汇总表（last 30% / 10% 均值 + 最终评估）（§6）；
> ③ 非基线实现**全部删除**（不迁 diy/）——git 历史即归档；旧实验数据在另一台机器上，
>    无任何旧格式兼容负担（§10）；
> ④ **算法接入成本是硬约束**：必需方法只有 1 个，写入预算表并有验收测试（§3.5）。
>
> 取代 [DESIGN.md](DESIGN.md)（v0.1）的定位与路线图。v0.1 中与本文不冲突的原语设计
> （§4.1 masks、§4.2 adapter、§4.3 schema、§4.5 policy、§4.9 logger）继续有效，不再重复。
>
> 方向决定（2026-09-15）：auto 四项全选（生命周期 / 健康管理 / 空间搜索 / 算法发现）；
> 先设计后动码；库 API 与 CLI 并重（notebook 科研，终端复现）。
>
> **R1 已执行**（见 §12）：src 4 889 → 3 866 行（−1 023，全仓 −2 255 含测试），
> 115 个测试全绿（含 CartPole 真实学习验证）。

---

## 0. 定位重述

> **一个自动化的 on-policy RL 算法科研流程框架**：
> 声明实验 → 自动运行与监控 → 自动聚合出图 → 可信结论；
> 算法（surrogate × advantage × 超参组合）是一个**可枚举、可搜索、可自动发现**的空间。

核心资产 =「**实验生命周期 + 算法空间**」。正确性原语降级为地基。

与 v2.0 相比新增一条推论：**内置算法面 = 基线面**。src 里只有 PPO(clip) + GAE 这一最小
核心；"算法空间"由 diy/ 组件填充，经由 `{from: file.py:Class}` 接入——内置越少，
空间越干净，发现闭环（§8）的候选形态也越纯。

框架优化的科研闭环：

```
   ┌───────────────────────────────────────────────────────────────┐
   │ 1 声明    experiments/*.yaml : baseline + (arms|space) × envs × seeds │
   │ 2 运行    编排 + 断点续跑 + 异常监控（只汇报，不停止）           │
   │ 3 产出    results.json + 奖励曲线图 + 汇总排名表                │
   └───────────────────────────────────────────────────────────────┘
        ↑↓ 外部大脑（搜索器 / 元学习 / LLM）只经由两个契约接入：
           §7 空间采样契约 与 §8 候选评估契约。框架永远不做大脑。
```

**轻量化是约束不是代价**：核心运行时依赖不变（torch + numpy + gymnasium + pyyaml）；
pandas / pyarrow / matplotlib / optuna 全部进 extras。

---

## 1. 审计：v0.1 哪些判断成立、哪些不成立

### 1.1 成立（保留，不动）

| v0.1 设计 | 现状 | v2 裁决 |
|---|---|---|
| §4.1 Masks 三元组，不存 done | 已实现且有测试钉死 | 地基，不动 |
| §4.2 EnvAdapter 张量契约 | GymVecAdapter / TensorEnvAdapter | 不动 |
| §4.3 声明式 Schema buffer | Field + Op，零 buffer 代码 | 不动 |
| §4.5 Policy 窄协议，零继承 | 4 方法，GA2E 验证过零框架改动接入 | 不动 |
| §4.9 Logger：jsonl 权威源 + 前缀聚合 | 已实现 | 结果产出直接建在它上面 |
| registry + `{from: file.py:Class}` | 6 种 kind | 不动，是算法空间的接入机制 |

### 1.2 不成立 / 缺位

1. **科研的实际单位被放错了**：配置单位应是"一组对照"（baseline + arms × envs × seeds
   + 评估协议），不是"算法 × 环境族 preset"。
2. **生命周期 2–3 环节缺位**：无 runner / checkpoint / 监控 / 结果产出。
3. **§2 原则 5（无回调）过度延伸**：对算法实现是对的纪律，对编排与监控是错的。
   收窄为：「算法训练循环内无隐藏行为；编排层经**唯一显式信号点**接入，接触点 ≤ 2 行」。
4. **行数预算已破**：v0.1 定 L1 < 2000，实测 ~3.5k。预算按实测重定（§9）。
5. **算法层繁杂**：为个别算法铺设的机制过多（详见 §3.1）。

### 1.3 现状实测（2026-09-15，`wc -l`）

| 层 | 内容 | 行数 |
|---|---|---|
| L1 扁平原语 | types/schema/buffer/rollout/config/metrics/logger/norm/tree/registry/seeding | 1 641 |
| L1 包 | envs/nets/advantages/objectives | 1 858 |
| L2 | algos（ppo/vmpo/_common/base） | 819 |
| L3 | cli + experiment | 571 |
| **合计** | | **4 889** |
| 测试 | architecture + unit + algos（167 fast + 13 slow，全绿） | 2 052 |

---

## 2. 架构 v2

```
┌──────────────────────────────────────────────────────────────────┐
│ L4  工作流:  experiment 文件 schema · runner(编排/续跑) · health   │
│              (仅汇报) · report(results.json/曲线/汇总表) · space  │
│              · discovery 契约                                     │
├──────────────────────────────────────────────────────────────────┤
│ L3  entrypoints: CLI（train/run/report/space/...）                │
│                  每条命令 = L4 同名函数的薄壳                      │
├──────────────────────────────────────────────────────────────────┤
│ L2  algos/:  ppo（唯一内置训练循环）                              │
├──────────────────────────────────────────────────────────────────┤
│ L1  primitives:  现有原语（不动）+ evaluate.py + checkpoint.py     │
└──────────────────────────────────────────────────────────────────┘
```

**依赖纪律**（由 `test_architecture.py` 扩展强制）：

1. L1 绝不 import L2/L3/L4（不变）。
2. L4 只依赖 L1；触达 L2 的唯一位置是 `get_algo()`（编排必然要解析算法名，与 CLI 同理）。
3. **L2 对 L4 的接触点 ≤ 2 行**：循环内 `if monitor: monitor.observe(stats, ...)`；
   monitor 由 runner 构造，持 policy/opt/norms/env_factory 引用——异常监控、周期与
   最终评估、checkpoint 全部在 monitor 内部，训练循环对它们一无所知。
   自定义算法不接收 monitor 也能跑（runner 按 `inspect.signature` 决定是否传参）。

---

## 3. 算法层：重推导与瘦身（v2.1 新增）

### 3.1 繁杂的根源诊断

不是算法数量，是**为个别算法铺设的机制**：

- APO 独占 2 个框架钩子（`prepare` / `on_rollout_end`），且历史上出过 2 个正确性 bug；
- DAE 独占 3 件套（`iteration_loss` / `write_extra` / `resolve_fields`）+ advantage head，
  而自测显示它并不优于 GAE（CartPole 3 seed：206±42 vs 223±25）；
- V-MPO 是 301 行的独立训练循环；
- 8 个内置 surrogate 中 7 个从未被实际实验使用。

科研框架的内置面应该等于**基线面**。我们的研究形态是"ours（GA2E）+ baselines"，
基线就是 PPO(clip)+GAE。其余一切都是**可接入的候选**——这恰好就是发现闭环需要的形态，
两个目标在此汇合。

### 3.2 处置表（R1 已执行：全部删除，git 历史即归档）

| 现有 | 处置 | 理由 |
|---|---|---|
| `ppo.py` 训练循环 | **保留 + 裁剪**（228 → 202 行） | 唯一内置训练循环 |
| `ClipSurrogate` | 保留 | 基线 |
| `a2c` alias | R1 曾保留 alias；**分类 v0.3 已废除该身份**——A2C 是 vanilla_pg 策略损失组件 + 自身 config，不是 PPO 的 config 退化（见 [algo_taxonomy.md](algo_taxonomy.md) §3.1） | 代码已清空，随重建落实 |
| `GAE` / `vtrace_free_mc` | 保留 | 基线估计器 |
| Surrogate / AdvantageEstimator 协议 + registry | 保留 | 空间的接入机制 |
| `tr_ppo` `spo` `dpo` `mdpo` `ppo_rpe` `ppo_kl` `apo` | **删除** | 从未被实验使用；需要时从 git 历史恢复为 `{from: ...}` 组件 |
| `dae` | **删除** | 自测不优于 GAE（206±42 vs 223±25）；随删 `extra_policy_outputs`、`ActorCritic.advantage_head`、`iteration_loss`/`write_extra`/`resolve_fields` 能力 |
| `vmpo` | **删除** | 独立训练循环无当前用途；`config/vmpo.yaml`、`config/experiments.yaml`（旧 sweep）同删 |
| `diy/surrogates/spo*` | **删除** | 其存在意义（与内建交叉验证）随内建 spo 消失 |

净效果（实测）：**src −1 023 行**；L2 + objectives + advantages 1 870 → 854 行。

### 3.3 保留部分的纪律

1. 文件头引论文 + 公式编号；
2. 核心数学 ≤ 30 行，可逐行对照论文；
3. 钩子面收敛并逐一"有名有姓"（见 §3.5 的能力清单）；未来需要新能力时，
   **与它的第一个使用者一起引入**，不预留。

### 3.4 config 面

`PPOConfig` 的 19 个字段保留——它们是「PPO 37 implementation details」的显式化，不裁。
算法私有超参在组件 `__init__`（fix.md #5），不回迁。

### 3.5 接入成本预算（硬约束，作者要求④；接口设计已由 [algo_design.md](algo_design.md) v3 取代并细化）

**接入一个算法允许的全部仪式**：

| 接入什么 | 必须实现 | 全部仪式 | 验收测试 |
|---|---|---|---|
| surrogate | 1 个类、1 个 `__call__` | **~10 行** | `test_minimal_integration.py` |
| estimator | 1 个类、1 个 `compute` | **~6 行** | 同上 |
| 训练循环（algo） | 1 个 `train()` 函数 | 1 个函数 | `oprl train ./f.py:train` |
| network | `out_dim` + `forward` | 1 个类 | examples |

三条实现纪律：

1. **必需成员只有 1 个**；其余是**可选能力**，框架在调用点用 `getattr` 探测，
   缺席即跳过——裸类不写任何 no-op 样板。测试断言裸类上不存在 stub 方法。
2. **能力清单封闭**，每个能力都有具名使用者：`critic_loss`（改 critic 目标的 estimator）、
   `on_epoch_start` + `state_dict`/`load_state_dict`（GA2E 的 epoch 模式与 EMA λ）。
   新能力必须与第一个使用者一起引入，不预留。
3. **引用即接入**：`{from: ./my.py:Class}` 无需注册；注册只是给内置件的便利。
   组件私有超参走 `__init__`，由 spec dict 供给。

---

## 4. 实验文件：科研的一等公民

### 4.1 形态

```yaml
# experiments/ga2e_ablation.yaml —— "ours + baselines + N seeds" 的一份声明
meta:
  name: ga2e-ablation
  note: λ 选择模式消融

baseline:                      # 必填；无对照 runner 拒绝跑
  algo: ppo
  config: classic

arms:                          # 每个变体一个覆盖 dict，显式命名
  - name: ga2e-rollout
    advantage: {from: ./diy/advantages/ga2e.py:GA2E, refresh: rollout}
  - name: ga2e-epoch
    advantage: {from: ./diy/advantages/ga2e.py:GA2E, refresh: epoch}

envs: [CartPole-v1]
seeds: [1, 2, 3, 4, 5]
overrides: {total_steps: 100000}

eval: {every: 10000, episodes: 10}   # 训练结束必做最终评估；eval.every 未设时只做这一次
health: default                      # §5.2 的规则集；只汇报
```

### 4.2 语义与规则

1. **展开**：jobs = (baseline ∪ arms) × envs × seeds；每 job 一份 resolved config 落盘
   `runs/<exp>/<arm>/<env>-seed<k>/config.yaml`。
2. **作业身份 = `hash(实验名, arm 覆盖, env, seed, resolved config)`**；
   重跑实验 = 只跑身份变化或未完成的作业（**增量续跑**）。
3. **可比性由结构保证**：同组 config 差异 ⊄ 声明覆盖 → 聚合时报错并列出差异字段。
4. **实验文件顶级键 ≤ 15**；算法超参只能出现在 baseline/arms/overrides 内（架构测试检查）。

`config/<algo>.yaml` preset 降级为手动单跑的便利（待定 #4）。

---

## 5. 运行层

### 5.1 checkpoint / 断点续跑（`oprl/checkpoint.py`，L1，~120 行）

```
ckpt = {policy, opt, obs_norm, reward_norm,
        estimator.state_dict(),      # GA2E 的 EMA λ 在这里——漏了它，续跑即换算法
        rng_state, global_step, iteration, config_hash}
```

保存频率 = eval 频率；`oprl run` 自动 resume 未完成 job；`oprl train --resume` 手动续跑。
验收：同 seed + deterministic 下，续跑与不中断 run 的 metrics.jsonl **逐位一致**（进测试）。

### 5.2 健康监控 = 只汇报，永不停止（v2.1 改）

监控规则（NaN/Inf、KL 爆、熵塌缩、长期无改善、SPS 骤降）不变，**动作只有一种：记录异常事件**。

- 三路输出：console 即时警告、`run_dir/anomalies.jsonl`、汇总进报告（§6）；
  通道实现由 Logger 承担——`log.event(kind, **payload)`，见
  [logger_design.md](logger_design.md) §6；
- 事件 schema：`{iteration, global_step, rule, value, threshold, detail}`；
- 训练循环接触点 1 行：`if monitor: monitor.observe(stats, global_step=..., iteration=...)`。

**裁决理由**：早停会截断数据分布、破坏跨 run 可比性；对异常 run 的处置（剔除/重跑/忽略）
是**看数据的人**在分析阶段做的决策，不是运行阶段自动做的。框架把异常显性化，把决策留给人。

### 5.3 自动评估（`oprl/evaluate.py`，L1，~80 行）

`evaluate(policy, env_factory, episodes, norm_state)` —— 冻结 normalizer、独立 env 实例
（不复用训练 RNG）、指标以 `eval/` 前缀写进同一个 metrics.jsonl。训练结束**必做**最终评估。

---

## 6. 结果产出：三件套（v2.1 收敛，取代 v2.0 §5+§6）

实验完成后，`oprl report <exp>` 一条命令产出：

```
runs/<exp>/report/
├─ results.json       机器可读，一切下游分析的唯一输入
├─ curves/<env>.png   每环境一张：各 arm 奖励曲线（均值±std 阴影，n= 标注，默认不平滑）
└─ summary.md         人读：排名表 + 异常汇总
```

### 6.1 指标定义（先每 run 计算，再跨 seed 聚合）

| 指标 | 定义 |
|---|---|
| `rew_last10` | `rollout/ep_rew_mean` 在**总步数最后 10%** 区间的均值 |
| `rew_last30` | 同上，最后 30% |
| `final_eval` | 训练结束、冻结 normalizer 的评估回报（`eval/ep_rew_mean`） |
| `auc` | （可选）曲线下面积 / 总步数 |

跨 seed 聚合：**mean ± std**（n 一并输出；bootstrap CI 备选不默认）。

### 6.2 results.json 形态

```json
{"experiment": "ga2e-ablation",
 "arms": [{"name": "ppo", "kind": "baseline", "env": "CartPole-v1",
   "seeds": [1, 2, 3, 4, 5],
   "per_seed": [{"seed": 1, "rew_last10": 198.2, "rew_last30": 197.1,
                 "final_eval": 200.0, "anomalies": 0, "run_dir": "..."}],
   "agg": {"rew_last10": {"mean": 198.2, "std": 1.3},
           "rew_last30": {"mean": 197.0, "std": 1.1},
           "final_eval": {"mean": 199.5, "std": 0.9}}}],
 "ranking": {"rew_last10": ["ga2e-rollout", "ppo", "ga2e-epoch"]}}
```

### 6.3 summary.md

每 env 一张表：`arm | rew_last10 | rew_last30 | final_eval | n | 异常数`，
每列最优加粗并给排名；表后列各 run 的异常事件计数与类型。

### 6.4 内部机制（从 v2.0 §5 压缩保留）

run 身份来自落盘 config（非目录名）、可比性结构检查、step 网格插值必须声明、
`runs/.cache/` 缓存——四条不变，作为生成 results.json 的内部机制。
**移出核心**：IQM、rliable 导出、消融热力图、diagnostics 多面板——需要时按需再加，
不预先造图种。

---

## 7. 算法空间 schema（`oprl/space.py`，~200 行）

实验文件 `space:` 段（与 `arms:` 二选一）：

```yaml
space:
  surrogate: [ppo, spo, dpo, mdpo]      # choice 维 → 可枚举 → 网格展开
  advantage: [gae, dae]                 # 组件名或 {from: ...} 引用
  lr: {log: [1.0e-5, 1.0e-3]}           # 参数维 → 不可枚举 → 交给驱动器采样
exclude:
  - {advantage: dae, act: continuous}   # 已知非法组合，采样时硬过滤
  - {advantage: ga2e, note: "与 surrogate 耦合（v0.1 §4.7 修正 2）——消融结论不可分离解读"}
```

- **choice 维**有限可枚举；全 choice = 网格 = arms 的糖，`oprl run` 直接跑。
- **参数维**不可枚举；`oprl run --driver random|optuna --trials N` 或 `--point '{...}'` 单点。
- **点 → config 解析确定性且可哈希**：同一 Driver 下同点同 config 同作业身份。
  框架只做"点的执行器"，搜索策略 100% 在外部（optuna 在 `search` extra）。
- exclude 与耦合注记是数据：采样硬过滤；耦合注记渲染进报告（防止误读消融）。
- space 声明的键 ∈ {组件 kind 名} ∪ {Config 字段名}，拼错立刻报错（架构测试）。

---

## 8. 算法发现契约（最小化）

**框架不做大脑**——候选生成（元学习 / 程序搜索 / LLM 提议）一律在外部。三个接口：

1. **候选 ≡ 组件 + 点**：一个 `{from: ...}` 组件（满足 6 种协议之一）+ 一个 space 点。
   §3 瘦身后这一致性更强：内置面已经是"最小核心 + diy 候选"，发现闭环的候选与
   人的基线走**同一条接入路径**——没有特殊通道。
2. **评估契约** `evaluate_candidates(candidates, protocol) -> list[ResultCard]`：
   protocol = 实验文件的 envs/seeds/eval/analysis 段；**baseline 按 config_hash 共享缓存**
   （N 个候选不重跑 baseline）。ResultCard：

   ```
   { spec, config_hash,
     per_env: {rew_last10, rew_last30, final_eval, mean, std},
     vs_baseline, anomalies_summary,
     cost: {wall_time, env_steps},
     status: learned | not_learned | diverged | failed | import_error,
     run_dir }
   ```

   `diverged` 等判定**在分析阶段**依据 anomalies + 曲线给出，运行期不判定、不停止（§5.2）。
3. **故障契约**：候选的一切失败都变成结构化 status，绝不中断循环。
   `inprocess`（默认，异常围栏）/ `subprocess`（超时+资源上限，用于不可信生成代码，
   文档明示风险）。

---

## 9. 轻量化预算 v2.2（R1 后实测更新）

| 指标 | v0.1 预算 | R1 后实测 | **v2.2 预算** |
|---|---|---|---|
| 核心运行时依赖 | 3 | 4（+pyyaml） | **4 封顶** |
| src/oprl 总行数 | — | **3 866**（4 889 − 1 023） | **≤ 5 600**（L4 新增 +~1 700） |
| L2+objectives+advantages | — | **854**（1 870 − 1 016） | ≤ 900，冻结 |
| 接入成本（§3.5） | — | surrogate ~10 行 / estimator ~6 行 | **必需方法 ≤ 1 个**，测试钉死 |
| L4 新增：report(results+plot) | — | 0 | ≤ 550（其中 plot/curves ≤ 150） |
| L4 新增：experiment+runner | 571（现） | 571 | 重写后 ≤ 650 |
| L4 新增：space / health | — | 0 | ≤ 200 / ≤ 120 |
| L1 原语 | <2 000 ❌ | 3 499 | ≤ 3 800（+evaluate/checkpoint 后**冻结**） |
| 训练循环对 L4 接触点 | 0 | 0 | **≤ 2 行** |
| 实验文件顶级键 | — | — | ≤ 15 |
| CLI 命令数 | 5 | 5 | **≤ 7**（+run/report/space，−sweep） |
| 读懂一个算法跳转文件数 | ≤3 ✓ | 3 | 不变 |
| 依赖新增 | — | — | 全 optional：`results[pandas,pyarrow]` `viz[matplotlib]` `search[optuna]` |

---

## 10. 迁移账（作者决策③：无旧数据/旧格式兼容负担，git 历史即归档）

| 对象 | v2.2 处置 | 状态 |
|---|---|---|
| L1 全部原语、`ActorCritic`、diy/ga2e | 原样保留 | ✓ |
| `ppo.py` | 裁剪 APO 钩子与 iteration_loss 块（228→202）+ R2 加 `monitor=None` | ✓ 裁剪完成 |
| 7 个 surrogate + DAE + V-MPO + diy/spo | **删除**（git 历史可恢复） | ✓ |
| 估计器协议 | 收敛：必需 `compute`，能力 `critic_loss`/`on_epoch_start`/`state_dict`，getattr 探测 | ✓ |
| `config/vmpo.yaml`、`config/experiments.yaml` | 删除 | ✓ |
| `experiment.py` | R2 重写为 runner（保留 GPUScheduler；+身份/增量续跑/monitor 接线/report 触发） | 待 R2 |
| `cli.py` | R2/R3 +run/report/space，−sweep | 待 R2 |
| 新增 | `results.py` `plot.py` `space.py` `health.py` `checkpoint.py` `evaluate.py` | 待 R2–R5 |
| 架构测试 | +L4 不 import algos、space 键白名单、实验文件字段白名单、resume 逐位一致、行数断言 | 待 R2 |

---

## 11. 纪律裁决（v0.1 → v2.1 逐条）

| v0.1 纪律 | v2.1 裁决 | 理由 |
|---|---|---|
| §2 原则 5 无回调 | 收窄：算法循环无隐藏行为；编排经唯一信号点 `monitor.observe()`（≤2 行） | 健康监控/评估/存档是编排职责；None 即关闭 |
| §7.1 自动早停 | **否决**：监控只汇报，永不停止 | 早停截断数据、破坏可比性；处置权留给分析阶段的人（§5.2） |
| §5.3 "不做算法动物园" | 强化为**内置面 = 基线面**：非基线一律删除，需要时从 git 历史以 `{from: ...}` 组件复活 | 内置越少，空间越干净，候选接入路径唯一（§3.1） |
| §8.4 反目标"不做超参搜索框架" | 修正：不做搜索**算法**，做**空间声明与点的执行** | 空间 schema 是数据结构；它是发现闭环的必要接口 |
| 配置 dataclass 唯一来源 | 维持；实验文件是编排配置，与算法超参字段空间不相交 | v0.1 踩过的坑（surrogate 超参曾塞满 PPOConfig） |

---

## 12. 路线图 v2.1

| 阶段 | 内容 | 完成标志 |
|---|---|---|
| **R1 算法层瘦身** ✅ | §3 处置表执行：非基线全部删除、ppo.py 裁剪、协议收敛到"必需 1 方法" | **已达成**：src −1 023 行（全仓 −2 255）；115 测试全绿（含 CartPole 真实学习验证）；新增 `test_minimal_integration.py` 验收 |
| **R2a 算法接口重构** | 按 [algo_design.md](algo_design.md) §10 迁移清单执行：Batch/Rollout 上下文、Estimate+Phase 插槽、UpdateRule 逃生舱、state_vars 状态快照 | 映射表七类算法落位；test_minimal_integration 改写后仍绿；GA2E 数值回归不变；装配期校验生效 |
| **R2 实验+运行** | 实验文件 schema、runner（展开/身份/增量续跑）、checkpoint/evaluate、monitor（仅汇报） | GA2E 消融文件跑通；kill 后续跑逐位一致；NaN 注入产生异常事件且训练不中断 |
| **R3 结果产出** | `oprl report`：results.json + curves + summary.md（last10/30、final_eval、排名） | 一条命令三件套齐全；可比性违规被拦截 |
| **R4 空间+搜索** | space schema、random driver、optuna adapter（extra）、exclude/耦合注记 | 20 trial 搜索跑通；同点跨 driver 同 config |
| **R5 发现契约** | ResultCard、候选围栏、subprocess 隔离档、baseline 共享缓存 | 10 个合成候选（含 3 个故意坏的）全部产出结构化 ResultCard |

R1 已完成：瘦身后的协议面就是 R5 发现契约的候选形态，地基就位。
**R2a（算法接口重构，见 [algo_design.md](algo_design.md)）排在 R2 之前**——科研链路
①–⑥ 的接口已在该文档 §1 定稿预留，实现顺序可调，但算法接入面必须先定型，
否则 R2–R5 的每一环都会绑死旧接口。
R2–R3 是分水岭：**"一条命令从实验文件到三件套"打通之前，v2 只是设想**。

---

## 13. 待定决策

1. **发现大脑形态**（LLM 提议 / 元学习 / 程序搜索）：契约兼容三者；影响 §8.3 沙箱默认档。
2. **实验文件位置**：顶层 `experiments/`（倾向）还是 `config/experiments/`。
3. **baseline 共享缓存失效**：config_hash 变即重跑（倾向）。
4. **preset 去留**：降级为手动便利保留（倾向）。
5. **最终评估的策略形态**：离散 argmax / 连续取均值（确定性，倾向、可配）还是采样。
6. **MinAtar CI tier**（v0.1 M2 遗留，与本次重定位正交）：仍建议做。

已裁决（原待定项）：非基线算法**全部删除**（2026-09-15，作者决策①）；旧实验数据
**重新来过**，无兼容负担（②）。
