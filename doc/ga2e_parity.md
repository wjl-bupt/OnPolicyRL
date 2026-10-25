# GA2E 原仓库一致性审计

核验日期：2026-10-25。此前的原生 GA2E 存在影响训练的实现差异；本次已修正。
这里的“一致”指给定相同网络、权重、rollout、优化器、超参数及随机状态时的 GA2E 算法行为。
不能据此宣称不同环境后端、网络架构、设备或超参数下的学习曲线逐点一致。

## 参照版本

- 原文件：`/root/Posterior-GAE/src/algo/ga2e/ga2e.py`，未修改。
- 原仓库 commit：`17ff77b73de240974d42a1e3c3046da16e708c8c`。
- 文件 SHA256：`86e2b4950fa1701572467945ed0631de8c3c984bd7e0462c2332ce6375d02ec1`。
- 原更新器依赖：Stable-Baselines3 2.7.0；原源码使用标准 RolloutBuffer，不是同目录旧的 Lepski buffer。
- 原生实现：[ga2e.py](../src/onpolicy/ga2e.py)。生产代码不导入 SB3，也不复制原仓库作为 vendor。

## 对齐内容

| 项目 | 现在保留的原实现行为 |
|---|---|
| TD/GAE | NumPy float32 反向扫描；截断奖励补 terminal value，真正终止不 bootstrap |
| 尾部 value | 每次 lambda 搜索重新计算 rollout 尾部 value；中间步仍使用采样时的 value |
| 数据划分 | 按环境内 episode 段划分，搜索样本按时间优先排列；任一折少于 8 样本时共用全量 |
| 搜索 | 保留一级/二级网格、round 精度、缓存及并列分数的首项选择规则 |
| 梯度 | 验证梯度不标准化 advantage；训练折先按 NumPy 总体标准差标准化，再分块计算 |
| 对齐得分 | 先归一化验证梯度，再计算训练梯度 cosine；支持原方差惩罚与折内梯度 |
| EMA | 使用 float64 buffer 保存 Python float 精度的原始 EMA，再对使用值裁剪 |
| rollout 模式 | 一次 vanilla PG 搜索；更新时环境优先 flatten + NumPy permutation |
| epoch 模式 | 初始化搜索一次，再在每个 epoch（包含 epoch 0）搜索；更新时为时间优先顺序 |
| epoch 目标 | 用开始时的 delta/continuation 计算各轮 advantage；returns 固定在初始化的 lambda 或 value_ref_lambda |
| value_ref_lambda | 至少 0.9；rollout 模式仅在与 lambda_used 相差大于 1e-6 时覆盖 returns |
| minibatch 标准化 | PyTorch 样本标准差；只有多于一个样本时才标准化 |
| value loss | 无 0.5 缩放的 MSE；启用 value clipping 时直接拟合裁剪后的预测，不取两种 MSE 的最大值 |
| SPO/KL | rollout 模式 SPO 优先于显式 KL；epoch 模式按原源码不启用这两个分支 |
| 提前停止 | KL 大于 1.5 × target_kl 时，在该 minibatch optimizer.step 之前停止剩余更新 |

原源码注释与执行逻辑不一致的地方以实际代码为准。例如 epoch 0 的第二次搜索和 EMA 更新没有省略。
NextStep 环境的无效 reset 行是本框架的适配扩展；原 SB3 没有这种行，数值 oracle 使用全部有效的 SameStep 等价数据。

## 参数映射

| 原接口 | 本框架配置 |
|---|---|
| n_steps | train.rollout_len |
| n_epochs | train.ppo_epoch |
| batch_size | algorithms.ga2e.train.batch_size；要对照时显式填写 |
| learning_rate | model.lr；如退火还须对齐 anneal_lr、lr_min 和总步数 |
| clip_range | train.epsilon；如退火还须对齐 clip_schedule、clip_range_min |
| clip_range_vf=None | train.use_value_clipped=false |
| clip_range_vf 数值 | train.use_value_clipped=true + train.vf_epsilon |
| optimizer epsilon | model.adam_eps |
| lambda/EMA/search 参数 | algorithms.ga2e.train 下的同名字段 |

还必须对齐 gamma、vf_coef、ent_coef、normalize_advantage、max_grad_norm、target_kl、
kl_coef、use_spo、value_ref_lambda、网络共享方式和梯度分块大小。
原仓库默认值、框架默认值、系列 YAML 覆盖值不是同一概念，不能只设置 algo=ga2e 就认为整场实验配置相同。
GA2E 的 value_loss_scale 固定为 1.0；grad_chunk_size 的类默认值已恢复为 0。
框架 train/updates 统计 rollout 次数，原 SB3 _n_updates 统计 epoch，不能直接比较这两个日志计数。

## 数值验证

[test_ga2e_reference.py](../tests/test_ga2e_reference.py) 动态加载未修改的外部源码，并检查上面的文件哈希。
测试适配器只转换 policy 接口与 buffer 布局，搜索和训练参照计算直接执行原源码。

45 项测试覆盖 rollout/epoch、离散/连续动作、分块梯度、方差惩罚、单级/二级搜索、
value_ref_lambda、value clipping、SPO、显式 KL、提前停止、关闭优势标准化、退化划分、零梯度并列与 EMA。
在原环境 PyTorch 2.9.1 和当前框架 PyTorch 2.13 的 CPU 环境分别运行，两次均为 45 passed。

- delta、continuation、索引与随机排列状态：精确相等。
- 单独搜索梯度：逐位相等。
- 44 组完整更新后的网络参数、Adam 状态：逐位相等（atol=rtol=0）。
- 搜索分数容差为 atol=2e-7、rtol=2e-6；优势/returns 容差为 atol=rtol=2e-6。
- EMA 误差检查为 1e-14；lambda 与搜索次数也逐项检查。

这里未做 GPU 逐位一致性认证，未启动新的训练实验，也没有重新认证旧实验结果。
测试可在包含 pytest、原 SB3 2.7.0 和框架依赖的独立审计环境中运行：

```bash
cd /root/OnPolicyRL
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  python -m pytest tests/test_ga2e_reference.py -q
```

可用 GA2E_REFERENCE_SOURCE 指定参照文件；文件哈希仍须匹配。缺少 SB3/外部源码时测试会 skip，skip 不代表通过原实现对照。
SB3 仅用于测试，不应添加为框架的运行依赖。

框架非 slow 回归结果：234 passed、45 skipped、2 deselected。45 个 skipped 即上面的外部源码
oracle：生产环境没有安装 SB3，因此在单独加载参照依赖的审计进程中运行并通过，不能把这次 skip 算成通过。
另补充了策略类隔离回归：导入 APO 等算法不得改变 GA2E 的网络构造；自定义策略通过
`build_policy(..., policy_cls=...)` 显式传入，避免全局注册污染其他算法。

## 旧结果与输出隔离

本次实现标识为 `posterior-17ff77b-v1`。系列任务的 GA2E 输出组哈希加入实现标识，
防止旧结果的 DONE 或 checkpoint 被当成修正后的实验继续使用。旧结果未删除。
此前 GA2E 曲线应标记为旧实现，不能作为“与原仓库一致”的结果引用；后续重新实验时再比较性能。
