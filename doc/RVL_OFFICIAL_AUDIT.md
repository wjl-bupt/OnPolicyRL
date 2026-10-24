# RVL 官方代码核对

官方仓库：[Hauf3n/relative-value-learning](https://github.com/Hauf3n/relative-value-learning)，固定提交 `98ff50b8df27705ba08c36f64f79cb0a6c970975`。

结论：旧 Posterior-GAE 版本和上一版 OnPolicyRL 都不能视为官方实现的等价迁移。已修正原生实现的一步目标分支、R-GAE 边界、损失和采样，并加入官方函数数值对照。

|核对项|官方行为|本次处理|
|---|---|---|
|一步目标|单边 terminal 使用相对差和 reward 修正；双 terminal 显式置零|按官方四种分支实现，随机输入覆盖全部组合数值一致|
|R-GAE 边界|rollout 最后一步继续累计 successor 相对差|移除 scalar bootstrap，采用同一相对值坐标系；无终止轨迹的完整 GAE 与官方对照一致|
|训练损失|策略损失、entropy、relative-value loss；没有 absolute-value loss|删除额外 scalar-value 回归头和损失|
|相对值头|默认 bias-free linear；可选 odd MLP|默认改为 linear，并保留 mlp_symmetric 选项|
|encoder 梯度|target 编码可 no_grad；预测分支必须重新编码并反传|原生预测分支保持梯度，未采用旧仓库 detached encoding 的预测路径|
|pair 采样|全 rollout + 同轨迹混合，排除自身|替换仅在 minibatch 内 roll 配对的简化实现，使用全局/同段混合|
|优势归一化|默认 batch 级|改为整个有效 rollout 上归一化|

**更正此前判断**：官方 `target_1.py` 本身就将双终止目标置零。之前仅凭 reward 差公式把它判成实现错误并改成 reward 差，不足以代表官方代码。当前保留官方启发式，并明确区分“匹配官方代码”与“理论上所有终止语义下最合适的目标”。

主要依据：

- [一步目标](https://github.com/Hauf3n/relative-value-learning/blob/98ff50b8df27705ba08c36f64f79cb0a6c970975/rv/loss/target_1.py)
- [R-GAE](https://github.com/Hauf3n/relative-value-learning/blob/98ff50b8df27705ba08c36f64f79cb0a6c970975/rv/loss/gae.py) 与 [offset 初始化](https://github.com/Hauf3n/relative-value-learning/blob/98ff50b8df27705ba08c36f64f79cb0a6c970975/rv/loss/value_init.py)
- [更新损失](https://github.com/Hauf3n/relative-value-learning/blob/98ff50b8df27705ba08c36f64f79cb0a6c970975/rv/loss/update.py)、[相对值预测](https://github.com/Hauf3n/relative-value-learning/blob/98ff50b8df27705ba08c36f64f79cb0a6c970975/rv/loss/relative_value_loss.py)、[采样](https://github.com/Hauf3n/relative-value-learning/blob/98ff50b8df27705ba08c36f64f79cb0a6c970975/rv/loss/target_sampling.py)

## 验证与保留差异

全量 **153 passed**。新增测试直接执行固定版本的官方函数作为 oracle；参考文件只放在 tests/reference/，保留 MIT LICENSE，不作为训练依赖。

本次核对并实现的是 **1-step 分支**。官方默认实验采用 n-step、多个预生成 target 集及其调度；这些尚未实现，所以不能称为官方默认完整实验复现。原生版本使用自己的模型、mask 协议和每 minibatch 重新采样；保留 truncation bootstrap、跳过 autoreset dummy 的框架语义。官方论文的 Atari 基准也尚未重跑；连续动作仅为本框架扩展，不能当作官方验证结果。
