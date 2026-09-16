# 日志 / 指标记录设计（参考 SB3）

> 状态：**待 check** · 2026-09-16
> 定位：训练循环 S12 阶段（algo_design §9.2）的输出基础设施；`metrics.jsonl` 是报告
> 三件套（DESIGN_v2 §6）与结果层的一切下游分析的**唯一数据源**。
> 建议模块位置（随作者重建布局）：`src/common/logger.py` + `src/common/metrics.py`。

---

## 1. 与 SB3 Logger 的对照：照搬什么、改什么、放弃什么

SB3（`stable_baselines3/common/logger.py`）的核心构件：`Logger(folder, output_formats)`
+ `record / record_mean / dump(step)` + `KVWriter` 抽象（`write(values)/close()`）
+ 内置 Human / CSV / JSON / TensorBoard 四种 format + `configure_logger` 便利函数 +
`read_csv` 工具。

| SB3 做法 | 处置 | 理由 |
|---|---|---|
| `record(key, value)` **覆盖**语义、`record_mean` 才累加 | **改**：`record` 按 key 前缀自动选聚合方式 | 用错即静默丢数据——minibatch 级指标忘用 record_mean 时只剩最后一个值，这是最难发现的实验错误之一（DESIGN §4.9 论证过） |
| CSV / JSON / Human / TensorBoard 四 format | **改**：内置 Console + **Jsonl**，TB 进 extras | jsonl 逐行 flush、逐行可 `json.loads`，是报告层的权威源；SB3 的 CSV 在 key 集中途变化时错列——我们不养这个坑（需要 csv 时用 pandas 离线转） |
| `self.logger` 挂在算法实例上（全局态） | **改**：显式传参 `log: Logger` | 无隐藏状态；同进程多 run 天然安全（DESIGN §2 原则 5） |
| `record / record_mean / dump(step)` 三个签名 | **照搬** | 与 SB3 兼容意味着旧调用零改动迁移 |
| KVWriter 抽象（`write`/`close`） | **照搬**，改名 `Sink`，两方法 | 后端可插拔的最小面 |
| HumanOutputFormat 对齐表格 | **照搬** | 终端可读性 |
| `_cumulative` 均值内部态 | **照搬思想**，但由前缀表驱动 | 不需要用户记住选哪个方法 |
| KVSequenceWriter（视频/图像媒体） | **延后**（待定 #3） | v2 报告只有曲线图（plot 模块生成，不经 logger）；需要时给 Sink 加第三方法，append-only 不破坏已有实现 |
| `configure_logger` / `read_csv` | **放弃** | 前者是便利层重复；后者被 jsonl+pandas 取代 |

---

## 2. 三层结构

```
L1  Sink 协议            write(metrics, step) / close()          —— 2 方法，可插拔后端
L2  内置 Sink            ConsoleSink（对齐表）/ JsonlSink（权威源）
                          TensorBoardSink（extras，延迟 import）
L3  Logger               聚合（前缀表）/ 路由 / 事件通道 / 历史    —— 训练循环只认它
```

---

## 3. API 定稿

```python
class Sink(Protocol):
    def write(self, metrics: dict[str, float], step: int) -> None: ...
    def close(self) -> None: ...


class Logger:
    def __init__(self, run_dir: Path | str | None = None,
                 sinks: list[Sink] | None = None, window: int = 100):
        """sinks 缺省：ConsoleSink +（有 run_dir 时）JsonlSink(run_dir/'metrics.jsonl')。

    # ---- SB3 兼容 API（签名一致）----
    def record(self, key: str, value) -> None:
        """按前缀表自动聚合；不需要记 record vs record_mean。"""
    def record_mean(self, key: str, value) -> None:   # 兼容保留：显式 mean
    def dump(self, step: int) -> None:
        """聚合 → 全部 sink；清空累积态；同时把聚合值写入历史窗口。"""

    # ---- 消息通道（带颜色，作者要求①）----
    def debug / info / success / warning / error(self, msg: str, **kv) -> None:
        """`[oprl][LEVEL] msg | k=v ...`；success/info→stdout，warning/error→stderr
        （加粗）；级别过滤 min_level；颜色自动探测：TTY + 非 dumb TERM，
        NO_COLOR 关闭、FORCE_COLOR 强制（无第三方依赖，纯 ANSI）。"""

    # ---- 便利 ----
    def add(self, **kv) -> None:                      # record 的批量形式
    def add_episode(self, ret: float, length: int) -> None:
        """未归一化回报——必须在 RewardNormalizer 之外记录，否则曲线无意义。"""

    # ---- v2 新增 ----
    def event(self, kind: str, **payload) -> None:
        """异常事件通道（monitor 专用，§6）。console 单行 + anomalies.jsonl。
        绝不停止训练——只汇报（DESIGN_v2 §5.2 裁决）。"""
    def history(self, key: str, n: int | None = None) -> list[float]:
        """最近 n 个已 dump 的（聚合后）值。monitor 的滑窗检查从这里消费。"""
    def close(self) -> None
```

---

## 4. 聚合策略（前缀表——Logger 自研价值的落点）

| 前缀 | 聚合 | 例 | 依据 |
|---|---|---|---|
| `train/` `loss/` `grad/` | mean | `loss/policy`、`grad/norm` | 每 minibatch 多次产生，逐个上报噪声大 |
| `diag/` | mean | `diag/kl`、`diag/clipfrac` | 同上 |
| `time/` | sum → dump 时派生 `time/*_frac` | `time/env` | 耗时归因要的是占比（和恒为 1） |
| `perf/` | last | `perf/sps` | 诊断信息取最新即可 |
| `rollout/` `charts/` | 滑窗 mean（window） | `rollout/ep_rew_mean` | SB3 同名语义 |
| **`eval/`** | **last** | `eval/ep_rew_mean` | 评估是低频事件，mean 会把两次评估抹平成假曲线 |
| 其他（算法自有，如 `ga2e/`） | mean | `ga2e/lambda_used` | **算法自有指标零注册成本** |

---

## 5. `metrics.jsonl` 契约（report 的唯一上游）

- 每行 = 一次 dump：`{"step": <int>, "wall_time": <float>, "<key>": <float>, ...}`；
- **写后即 flush**——进程被 kill 不丢已 dump 的行（长 run 中断续跑的数据完整性）；
- key 集可随训练进程变化（早期没有 `eval/*`），读取方按行解析、按 key 聚合，不做列对齐——
  这正是放弃 CSV 的原因；
- run 目录布局（v2 全景）：

```
runs/<exp>/<arm>/<env>-seed<k>/
├─ config.yaml          # resolved config，可重跑
├─ metrics.jsonl        # 本文档：权威数据源
├─ anomalies.jsonl      # 本文档 §6：异常事件
├─ meta.json            # git sha / 依赖锁 hash / 库版本（待定 #4，倾向恢复）
├─ ckpt/latest.pt       # checkpoint（含组件 state_vars）
└─ report/              # R3 生成：results.json + curves/ + summary.md
```

---

## 6. 事件通道 `event()`（monitor 的输出路径）

- 行 schema：`{"wall_time", "iteration", "global_step", "kind", **payload}`；
- 三路输出：console 单行 `[oprl][<kind>] ...` + `anomalies.jsonl` 追加 + 报告汇总（R3 读该文件）；
- **monitor 不自开文件**——异常通道统一由 Logger 拥有，monitor 只调 `log.event(...)`；
- 语义约束：event 改变不了训练（无 stop 返回值），处置决策在分析阶段（DESIGN_v2 §5.2）。

---

## 7. 失败隔离（沿用旧纪律，v0.1 实测有效）

- 每个 sink 独立 try/except：失败 → 警告**一次** + 该 sink 永久禁用，训练继续；
- 理由：6 小时 run 因 wandb/磁盘满崩掉不可接受；丢指标永远好过丢训练；
- `history()` 依赖的内存态与 sink 无关，sink 全挂仍可用（monitor 不受影响）。

---

## 8. `metrics.py`（同批实现的三个小件）

```python
class Timer:
    """上下文管理器计时，acc 累加，drain() 输出 {f"time/{name}": 秒}。S12 消费。"""

def explained_variance(y_pred, y_true) -> float:
    """≤0 说明 critic 没在学——被低估的诊断量，dump 前由算法显式算入 diag/。"""

def format_table(metrics, step=None, title=None, style="sb3") -> str:
    """纯文本表格（无 ANSI，可测试）；ConsoleSink 在外层上色。
    作者要求②——指标以表格打印展示。四种风格（**默认 sb3**，作者选定）：
    `sb3`（复刻 SB3：虚线框、按 `前缀/` 分组头、子键缩进四格、值左对齐）/
    `box`（unicode 框·平铺排序·右对齐）/ `tree`（unicode 框内做 sb3 分组，最紧凑）/
    `minimal`（无边框两列，高频 dump 用）。ConsoleSink(style=...) 与
    Logger(table_style=...) 贯通；颜色逻辑与风格正交。"""
```

---

## 9. 验收测试清单（作者实现时逐条可测）

1. 前缀表逐条聚合正确（7 行各一测）；
2. `dump` 后累积态清空；连续两次 dump 同 key 不重复计数；
3. `metrics.jsonl` 逐行 `json.loads` 成功；模拟 kill（写后不 close）不丢行；
4. sink 注入异常 → 训练继续、该 sink 禁用、警告恰一次；
5. `event()` 落 anomalies.jsonl + console，且训练循环无感知、无停止语义；
6. `history()` 的 maxlen 与顺序正确；
7. `eval/` 前缀 last 语义（两次评估不平均）；
8. `add_episode` 的原始值（未归一化）进入 `rollout/ep_rew_mean` 滑窗；
9. `time/*_frac` 派生值之和 ≈ 1；
10. 无 `run_dir` 时仅 ConsoleSink 可用，一切 API 照常；
11. 消息级别过滤（`level="info"` 时 debug 不打印）；warning/error 走 stderr；
12. `NO_COLOR` 下输出无 ANSI 转义，`FORCE_COLOR=1` 强制上色；
13. `format_table` 对 nan/inf/bool/int 的格式化与排序稳定性。

## 10. 待定

1. **CSV sink**：默认不做（pandas 离线转换）；若想要 SB3 式 `progress.csv` 兼容再加。
2. **TensorBoard / wandb sink** 进 extras 的时机（建议 R3 报告层落地后）。
3. **媒体通道**（视频/回放）：需要时给 Sink 加 `write_media`（append-only）。
4. **meta.json**（git sha / uv.lock hash）：倾向随重建恢复——复现价值/成本比极高。
