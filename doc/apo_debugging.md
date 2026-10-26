# APO 调试与接口修复

APO 支持连续 Box 和离散 Discrete 动作；离散时要求 `1 <= ua_nums <= |A|-1`。
连续调试：VS Code 选择 `APO: Debug (CPU / InvertedPendulum)` 后按 F5。
新增指标和离散采样语义见 [审稿补充实现](review_extensions.md)。
断点可放在 `APO._sample_policy`、`APO._make_batch_` 和 `APOPolicyLoss.__call__`。
需要在框架捕获异常前暂停时，勾选 Raised Exceptions。

## 本次故障原因

1. buffer 只声明了存储字段，没有通过 `extra_samples` 声明 minibatch 字段；
   同时写入 `uact`，读取时却访问 `unused_actions`。直接报错为
   `AttributeError: 'RolloutSamples' object has no attribute 'unused_actions'`。
2. 执行动作 `[B,D]` 与未执行动作 `[B,K,D]` 直接拼接，维度不一致。
   现在先扩为 `[B,1,D]`，沿候选动作轴拼接；计算分布概率时转为 `[K+1,B,D]`，
   保证每个状态只对应自己的候选动作。log-prob 沿动作维 D 求和，结果为 `[B,K+1]`。
3. 原自定义 collector 没有写入 `next_obs/next_value`，NativeLoop 的 GAE 却读取这些字段，
   即使绕过异常也会得到错误的 bootstrap。现在使用公共 collector，只扩展采样返回的 extras。
4. APO 损失使用默认构造参数，忽略 epsilon 及裁剪调度；现在传入配置值。
5. `ua_nums` 缺少类型注解，不是 dataclass 字段，无法正常通过配置修改；现在为 `ua_nums: int = 2`。
6. 原 `act()` 返回多个候选动作，破坏标准 policy 接口。现在保持标准 act/evaluate，
   将多动作采样与计算分别放在 act_with_anchors/evaluate_anchors。

## 修改未执行动作数

系列 YAML 中：

```yaml
algorithms:
  apo:
    train:
      ua_nums: 3
      uarr_coef: -1.0
```

或在 launch.json 的 args 中加入 `"--set", "algorithms.apo.train.ua_nums=3"`。
K 个未执行动作和 1 个执行动作都由 rollout 时的策略采样；仅执行动作进入环境。
修复保留现有 APO 正则项公式，不代表完成论文/官方实现的一致性核验。

`uarr_coef` 控制正则项的权重，默认 `-1.0`：

- `uarr_coef >= 0`：固定权重，`uarr_loss = mean(0.5 * uarr_coef * (ratio - 1)^2)`；设为 0 关闭该正则项。
- `uarr_coef < 0`：动态权重，`uarr_loss = mean(0.5 * old_prob * (ratio - 1)^2)`。
  负数仅用于选择模式，其绝对值不参与计算；`old_prob = exp(old_log_prob)`，对每个状态的每个未执行动作分别计算。

其中 `ratio = exp(new_log_prob - old_log_prob)`。连续动作下 old_prob 是联合概率密度。
launch.json 可加入 `"--set", "algorithms.apo.train.uarr_coef=0.1"` 使用固定系数。
日志 `apo/uarr_coef` 记录实际权重的均值，`apo/uarr_loss` 记录加权后的正则损失。

针对性测试位于 `tests/test_apo.py`：覆盖 K=1/3、多维动作、旧概率与新概率一致性、
截断与终止 bootstrap、两个连续 rollout 的参数更新，以及配置校验。
