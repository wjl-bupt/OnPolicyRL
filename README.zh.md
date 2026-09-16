# oprl

面向**科研流程**的轻量单智能体 **on-policy** RL 框架：声明实验 → 运行与监控 → 产出
机器可读指标、曲线与对比表。算法侧是可插拔空间，内置面 = 基线面（PPO clip + GAE）。

## 当前状态：**设计阶段，代码已清空**

经作者决定，全部代码（`src/`、`tests/`、`config/`、`diy/`、`examples/`）已于
2026-09-16 删除：框架将按下列设计文档**重建，代码由作者本人编写**。本仓库当前**只含
文档**；旧实现可在 git 历史（`d26abdf` 及更早）中查阅。

## 文档

| 文档 | 职责 |
|---|---|
| [`doc/algo_taxonomy.md`](doc/algo_taxonomy.md) | **算法分类**——按更新机制分分支（B1 PPO · B2 V-MPO · B3 TRPO 预留），每分支以 baseline 为基座；变体清单；分支×插槽可用性矩阵 |
| [`doc/logger_design.md`](doc/logger_design.md) | **日志/指标记录设计**——参考 SB3：前缀自动聚合、`metrics.jsonl` 权威源、`event()` 异常通道（只汇报）、`history()` 滑窗；验收测试清单 |
| [`doc/algo_design.md`](doc/algo_design.md) | **算法接入设计（v3）**——统一组件形状、`Batch`/`Rollout` 上下文、`Estimate`+`Phase` 插槽、`UpdateRule` 逃生舱；已知算法类型的映射验证表；落地迁移清单（§10） |
| [`doc/DESIGN_v2.md`](doc/DESIGN_v2.md) | 框架 v2——"实验生命周期 + 算法空间"定位、科研链路全图（declare → run → monitor → report → search → discover，接口级预留）、预算 |
| [`doc/DESIGN.md`](doc/DESIGN.md) | v0.1 设计论证——原语层决策（masks 三元组、声明式 schema、Policy 协议、Logger），仍然有效 |
| [`doc/fix.md`](doc/fix.md) | 历史架构修正记录（v0.1 时期） |

## 重建入口

1. **算法层**——按 `doc/algo_design.md` §4–§8 的上下文/插槽/逃生舱实现，走 §10 迁移清单。
2. **生命周期层**——`doc/DESIGN_v2.md` §4–§8：实验文件 → runner + monitor（只汇报）+
   checkpoint → 报告三件套（`results.json` + 曲线 + last-30%/10% 与最终评估汇总表）→
   空间 schema → 发现契约。
3. **路线图**——`doc/DESIGN_v2.md` §12：R1（瘦身）已定稿；R2a（接口重构）→ R2（实验+运行）
   → R3（报告）→ R4（空间）→ R5（发现）。

## 重建已知陷阱（来自旧实现，git 历史可查）

- **MinAtar 环境需要手动注册**：gymnasium 1.x 不自动加载第三方 entry point，
  `gym.make("MinAtar/...")` 会 `NamespaceNotFound`，需在 make_env 里延迟调用
  `minatar.gym.register_envs()`（旧 `memory/minatar-env-fixes.md` @ `d26abdf`）。
- **MinAtar 观测是 channels-last `[H,W,C]` bool**：CNN 期望 `[C,H,W]`，需要
  `channels_first` preset 选项 + transpose wrapper；且 bool obs 不能做 `/255` 归一化。
