# 日志、结果与曲线

## 任务输出

| 文件 | 内容/用途 |
|---|---|
| config.json | 任务展开配置、算法配置、实际 device 与身份 |
| environment.json | 实际后端/provider 与 wrapper provenance |
| metrics.jsonl | 按 step 记录的指标，适合绘图 |
| run.log / error.log | 运行信息、异常堆栈 |
| anomalies.jsonl / anomaly.log | 异常事件，包括评估或 checkpoint 失败 |
| model_pools/model_step-*.pt | 按环境步数命名的 checkpoint |
| DONE | 训练入口完成标记，不证明评估/保存均成功 |

实验根目录保留 experiment.snapshot.*；结束后 runner 尝试生成 results.json、summary.md 和报告图。
报告生成失败不影响已完成训练，因此需要单独检查输出。
固定 commit 运行额外保存 manifest.json、config.snapshot.*、config.runtime.json、launch.log。

## 指标定义

- step：累计环境 transition 计数，包含 num_envs 的推进；不是 optimizer step 或单环境时刻。
- rollout/ep_rew_mean：最近完成 episode 的原始回报均值。默认 logger 窗口 100；不是 rollout 内即时 reward 均值，也不是 eval。
- eval/ep_rew_mean：独立评估环境完成 episode 的原始回报均值；默认确定性策略。
- eval/n_episodes：实际完成数量，必须与请求数量一起检查；安全上限可使评估提前结束。
- loss/*、train/*：优化与诊断量，具体算法含义不同，不能将不同算法同名 loss 直接比较。
- perf/sps：当前采样数量除以记录的 fwd/env/bwd timer 总和。通常不含完整评估、保存及所有估计器开销，**不等于端到端吞吐量**。

报告端到端 SPS 时使用实际完成步数 / 对应任务墙钟时间，明确是否包含 JIT、评估和保存。
不要把多个并发任务的 SPS 相加后与单任务墙钟吞吐量混比。

JSONL 中 slash 指标通常按组保存，例如 rollout.ep_rew_mean、eval.ep_rew_mean。
同一个 step 可能有多条日志；按指标逐项去重，不能简单保留最后整行而丢失该 step 的 eval。

## 多 seed 汇总

先确认各 seed 的算法/网络/环境设置、预算、代码版本和评估口径相同。
完整组使用共同 step 上的均值与样本标准差，标准差不是置信区间。
缺失或未完成 seed 应列出数量，不能补零、外推到预算终点或冒充完整 5-seed 结果。
训练曲线是滚动统计，若再平滑需公开窗口；评估曲线与训练曲线分别展示。
不同环境回报量纲不同，不直接平均原始分数；使用独立纵轴时应明确标注。
通用自动报告的聚合规则需查看 [report.py](../src/workflow/report.py)，发表结果前不要只依赖默认排名。

## 已有 campaign 绘图脚本

```bash
bash scripts/plot_saved.sh
# 指向另一个相同日志/状态结构的 campaign
CAMPAIGN=runs/<batch>/campaign_<id> OUTPUT=runs/<batch>/report bash scripts/plot_saved.sh
```

该脚本面向已有 PPO/GA2E MinAtar campaign：读取 status.json 任务清单和各任务 metrics.jsonl，
输出 train/eval PNG、PDF、curves.csv、status.json。它包含该历史批次的说明和绘图范围，
**不是任意环境、任意算法的通用报告入口**；普通 launch.sh 输出不保证具有 campaign/status.json。
改用于另一批实验时必须同步修改图注、算法、预算范围和任务结构，不能沿用旧 sticky/评估说明。

脚本去重后绘制：完整 5-seed 组使用共同记录点；部分结果逐 seed 展示，无外推。
旧 JAX MinAtar batch 的 Breakout 最终评估实际只完成 40 episodes/seed，
巨大 eval 回报受该批次环境设置影响；图中已注明，不能当作标准新基准成绩。
