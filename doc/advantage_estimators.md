# AAE 与 PAE 使用及实现约定

两者均使用原生 PyTorch 网络、buffer、PPO 更新与日志，支持 Discrete 分类策略和扁平 Box 对角高斯策略。
不依赖 SB3，不恢复已删除的探索算法 pgae。Diffusion 动作头仍由 DPPO 使用。

论文来源：

- Chen 等，Adaptive Advantage Estimation for Actor-Critic Algorithms，IJCNN 2021，DOI：10.1109/IJCNN52387.2021.9534005。
- Jin 等，Partial Advantage Estimator for Proximal Policy Optimization，IEEE Transactions on Games，DOI：10.1109/TG.2024.3408298。

## 配置方式

继续使用既有环境系列文件；Gymnasium 和 JAX 各用原来的配置。不新增每算法一份配置。
在系列文件内选择 algo，并将专属参数放入 algorithms 下：

```yaml
algo: [ppo, aae, pae]
train:
  rollout_len: 512
  gae_lambda: 0.95
algorithms:
  aae:
    train:
      aae_variant: 1
      accuracy_metric: ev
      kl_threshold: 0.01
      kl_scale: 1.0
  pae:
    train:
      partial_steps: 256
```

AAE 自行计算 actor/critic 的 lambda，不使用公共 gae_lambda；PAE 默认使用公共 gae_lambda；开启 V-trace 后使用独立的 vtrace_lambda。
模型选择、设备、种子数等参数继续遵循 experiment.md。下面的命令只预览，不启动训练：

```bash
PYTHONPATH=src .venv-jax/bin/python -m oprl.cli configs/classic_control_gymnasium.yaml \
  --set algo=aae --set algorithms.aae.train.aae_variant=2 \
  --set algorithms.aae.train.accuracy_metric=r2 --set 'sweep={}' --plan

PYTHONPATH=src .venv-jax/bin/python -m oprl.cli configs/mujoco_gymnasium.yaml \
  --set algo=pae --set train.rollout_len=512 \
  --set algorithms.pae.train.partial_steps=256 --plan
```

需要实际训练时移除 --plan；本次只做小规模运行验证，不启动正式性能比较。

## AAE：四种组合与独立 value target

| 参数 | 默认值 | 说明 |
|---|---|---|
| aae_variant | 1 | 1 或 2；对应论文两种 actor lambda 公式 |
| accuracy_metric | ev | ev 或 r2；与 variant 组合得到四种方法 |
| kl_threshold | 0.01 | 论文 b_k，低于该阈值不惩罚准确度；数值是工程默认 |
| kl_scale | 1.0 | 论文 c_k，超阈值 KL 的惩罚尺度；数值是工程默认 |
| indicator_eps | 1e-8 | target 方差退化的判定及分母保护 |

计算使用联合动作分布：分类策略用 Categorical.entropy，连续策略用各动作维的高斯熵之和。
KL 使用解析 KL(old || new)，连续维度求和，再对有效状态求均值，不用 PPO 的抽样近似 KL 替代。
离散熵非负，Conv 的可达范围与连续微分熵不同；这是相同公式用于两类分布的结果，没有额外做熵归一化。

```text
Conv = sigmoid(-mean_joint_entropy)
alpha = kl_scale * max(previous_kl - kl_threshold, 0)
Acc = clip(1 - alpha, 0, 1) * clip(previous_regression, 0, 1)
lambda_actor_1 = (1 - 0.5 * Acc) * Conv
lambda_actor_2 = 1 - Acc * (1 - 0.5 * Conv)
lambda_critic  = 1 - 0.5 * Acc * Conv
```

actor 使用 GAE(lambda_actor)；critic 使用 GAE(lambda_critic) + 采样时 V。
不把 actor advantage 直接用于 critic target。lambda、优势和回归目标均不反向传播。

论文 Algorithm 1 没有完整规定准确度、KL 与新 target 的计算时序，当前明确采用以下约定：

1. 当前 rollout 采集完成，读取上一轮更新后保存的回归指标及 KL，结合当前熵计算 lambda。
2. 本轮所有 minibatch/epoch 共用固定的两个 lambda 和回归目标。
3. 更新完成后，以当前 critic 对本轮固定 target 计算 R2 或 EV，并在本轮状态上计算更新前后解析 KL，留给下一轮。
4. 首轮 regression=0、KL=0，因此 Acc=0。target 方差不大于 indicator_eps 时 regression=0，并记录退化标记。

R2 = 1 - mean((target - V)^2) / var(target)；EV 将分子改为 var(target - V)，均使用总体方差。
指标保存在 policy buffer 中，随 checkpoint 恢复。原文式（17）积分缺少负号，采用其后给出的标准熵含义。
这里的时序和数值保护是明确记录的工程选择，不声称逐位复现作者未公开的实现。

日志包括 aae/lambda_actor、lambda_critic、entropy、convergence、accuracy、previous_regression、
previous_kl、regression_after_update、kl_after_update、target_variance_degenerate。

## PAE：窗口、消费掩码及尾部保留

| 参数 | 默认值 | 说明 |
|---|---|---|
| rollout_len | 沿用 PPO 配置 | 估计窗口长度 T |
| partial_steps | null | null 解析为 max(1, T//2)；显式值须为 1..T，表示训练前缀 K |
| gae_lambda | 沿用 PPO 配置 | 窗口内 GAE 的固定 lambda |

先用完整窗口计算 GAE，再选择训练样本。未完成片段只使用 [0:K)，将 [K:T) 的未消费数据保留至下一轮。
稳态下每轮只补 K 个新环境步；第一轮需要 T 步。若尾部已无待训练样本，下一轮重新采集完整窗口。
已终止或时间限制结束的片段允许提前消费完；真实终止切断 bootstrap，时间限制仍用 final observation bootstrap。

保留窗口的 transition-valid 与 pending-consumption 两个概念：某些已消费行可作为窗口上下文保留，
但不会再次进入训练集合。actor、critic、entropy 共同使用 eligible mask。
GAE 计算时仍使用原始 valid/terminated/truncated，不将“本轮不训练”误当作轨迹终止。

重组窗口时重新估计 target 用的 value，但保留采样时的 old_value（供 value clipping）和 old_log_prob。
保留动作的行为概率绝不由当前策略覆盖。跨轮次窗口可能混合策略版本；采用 PPO 行为概率比率，
默认不添加 V-trace 修正，不声称严格的同策略优势估计；可选扩展见下一节。

原文伪代码保留区间的边界及正文 N(T-epsilon) 样本数有冲突；当前统一为前 K 行训练、后 T-K 行保留，
每个有效 transition 最多消费一次。默认 GAE 分支在 partial_steps=T 时退化为相同配置的原生 PPO。
预算结束时不强制训练仍待后续上下文的尾部；日志和 checkpoint 保留其数量和内容。

论文实验关闭了 advantage normalization 和 value clipping。框架默认保持可配置；需要该实验口径时设置：

```yaml
algorithms:
  pae:
    train:
      normalize_advantage: false
      use_value_clipped: false
```

该设置本身不代表复现论文全部超参数、网络或环境版本。

日志 pae/new_env_steps 仅记录新发生的环境交互；pae/train_samples 记录本轮 eligible 样本数；
pae/carried_pending 和 pae/pending_tail 记录待消费数量。perf/sps 的分子也只包含新环境步，
不重复计算保留数据。与 PPO 比较时应同时查看环境步数、训练样本数和实际耗时。

## PAE 可选 V-trace 分支（默认关闭）

这是 PPO/V-trace 混合扩展，不是原论文 PAE 或原版 IMPALA。V-trace 的依据是
[IMPALA 第 4 节](https://proceedings.mlr.press/v80/espeholt18a/espeholt18a.pdf)。
截断重要性权重用来控制方差，不能保证完全消除离策略偏差或提高实验性能。

在既有环境系列配置中修改：

```yaml
algo: pae
algorithms:
  pae:
    train:
      use_vtrace: false       # 改为 true 开启；默认 false，保留原来的 GAE
      vtrace_rho_clip: 1.0    # value TD 修正的重要性权重上限
      vtrace_c_clip: 1.0      # 多步 trace 的权重上限，必须 <= rho_clip
      vtrace_lambda: 1.0      # trace 衰减，范围 [0, 1]
```

两个 clip 参数必须是有限正数；关闭时 gae_lambda 生效，开启时 gae_lambda 不参与估计。
CLI 可用 `--set algorithms.pae.train.use_vtrace=true` 开启。

本轮更新前固定参考策略 pi_ref=pi_k，在完整时间窗口上重新计算 value 和动作 log probability。
mu 是每条 transition 实际采样策略；保留的 old_log_prob 不覆盖，窗口可以包含多个行为策略版本。
对连续动作使用联合动作 log probability，对离散动作使用对应动作的 log probability。

```text
w_t = exp(log pi_ref(a_t|s_t) - log mu(a_t|s_t))
rho_t = min(rho_clip, w_t)
c_t = vtrace_lambda * min(c_clip, w_t)
delta_t = r_t + gamma * V(s_next) - V(s_t)
e_t = rho_t * delta_t + gamma * c_t * e_next
value_target_t = V(s_t) + e_t
actor_advantage_t = delta_t + gamma * e_next
```

以上省略边界掩码：真实终止不 bootstrap；时间截断用 final observation 的 value bootstrap，
但切断后续 trace；窗口末端和 invalid 行也切断 trace，不将新 episode 接入旧 episode。
先完成完整窗口估计，再应用原有 PAE 训练前缀及 pending 掩码。

actor 使用未额外乘当前步 rho 的 q-V，因为 PPO loss 已包含 `pi_theta / mu` 比率；
不把 value_target-V 直接当成 actor advantage，以免重复加权。这个选择是混合算法的实现约定。
当 lambda<1 时，该 actor advantage 不等同于相同 lambda 的普通 GAE。
所有 target/advantage 均停止梯度，并在本轮 PPO epochs 中保持固定。
采样时 old_value 仍供 value clipping 使用；old_log_prob 始终供 PPO ratio 使用。

新增日志：`pae/vtrace_rho_mean`、`pae/vtrace_c_mean`、`pae/vtrace_rho_clipfrac`，
均在完整窗口有效行上统计；c_mean 已包含 vtrace_lambda。

## checkpoint 边界

AAE 指标随 policy.state_dict 保存。PAE 的 carry、pending、last_obs 保存到 checkpoint.algorithm_state。
合成环境测试支持精确恢复后继续拼接；但是当前生产环境 wrapper 的 state_dict 只保存归一化统计，
没有保存完整模拟器、wrapper history 和随机状态。因此正常 resume 会恢复模型/优化器/进度，
丢弃无法安全衔接的 PAE 尾部，并记录 pae/resume_discarded_pending 和 warning，随后重置环境。
这不是精确的轨迹级断点续训。只有显式保证完整状态恢复的适配器才可声明 supports_exact_resume。

## 验证

tests/test_aae_pae.py 覆盖 AAE 四种组合及两类动作、公式/目标/解析 KL、PAE 跨轮次样本记账、
真实终止与截断、invalid reset 行、checkpoint 安全恢复，以及 K=T 与原生 PPO 逐位一致性。
短运行的通过只证明接口、更新、评估与保存流程可执行，不代表论文性能已复现。

2026-10-26 验证记录：19 项针对性测试通过；完整非 slow 回归 262 passed、45 skipped、2 deselected。
45 项跳过的是需要外部 SB3/原仓库的 GA2E oracle，与新增方法无关。
CartPole-v1 和 InvertedPendulum-v5 各运行 AAE 四种组合及 PAE，共 10 个 64 环境步的短任务，
全部训练完成且生成最终评估与 checkpoint；未启动长时间正式实验。
服务器验证记录在 `.integration-staging/aae-pae-validation/`，短运行输出在 `runs/debug/aae-pae-validation/`。

V-trace 扩展验证（2026-10-26）：28 项针对性测试通过，覆盖手算 target、避免 actor 重复加权、
trace 衰减、极端概率比、边界切断，以及开关两种状态下的离散/连续跨轮次记账。
完整非 slow 回归为 271 passed、45 skipped、2 deselected；跳过原因仍为上述外部 SB3 依赖。
开启 V-trace 的 CartPole-v1 和 InvertedPendulum-v5 各完成 64 环境步短运行并保存 checkpoint。
验证记录在 `.integration-staging/pae-vtrace-validation/`，输出在 `runs/debug/pae-vtrace-validation/`。
