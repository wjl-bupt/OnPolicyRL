# 算法与参数

入口通过 onpolicy/*.py 自动发现，不需要每个算法一个目录。配置用小写名称。
所有算法使用本仓库环境、网络、buffer 与 checkpoint；不是 SB3 套壳。

| algo | 当前实现摘要 | 动作空间/限制 |
|---|---|---|
| ppo | clipped policy objective + GAE | Discrete / 扁平连续 Box；分类或高斯头 |
| ga2e | 用训练/验证梯度对齐选择 GAE lambda，支持 EMA | 同 PPO；可能显著增加梯度计算 |
| spo | 用软二次惩罚替换 PPO hard clipping | 同 PPO |
| rvl | 成对相对价值学习与 R-GAE；不训练绝对价值头 | 同 PPO；需至少两个有效 transition |
| dae | 轨迹回归学习 advantage，再执行策略改进 | 仅离散动作 |
| vmpo | 加权似然、温度与 KL 对偶约束 | 分类或对角高斯头 |
| dppo | 去噪链条件概率的逐步 PPO 更新 | 仅 Diffusion 连续动作头 |

MLP、SimBa、BroNet、SpectralNet、因果 Transformer 可作为观测骨干。
Diffusion 是动作头，当前只与 DPPO 配合；不是一个能无条件替换所有 policy.dist 的分布。

## 修改方式

公共 train 只放所有所选算法都接受的字段；算法专属设置放到：

```yaml
algorithms:
  ppo:
    train:
      gae_lambda: 0.95
  ga2e:
    train:
      lambda_val: 0.99
      lambda_refresh: epoch
```

下面列的是配置类默认值，系列 YAML 可以覆盖它们。不要把“接受字段”误认为“更新器实际使用字段”：
继承 PPOConfig 的算法可以有自己的损失与优势流程，具体以各模块为准。

## GA2E

| train 参数 | 默认 | 用途 |
|---|---|---|
| lambda_min | 0.1 | 搜索下界 |
| lambda_val | 0.97 | 验证参考 lambda；MinAtar 系列配置覆盖为 0.99 |
| level2_points | 10 | 二级细搜索分辨率 |
| val_every | 5 | 轨迹段划分验证集的间隔，至少 2 |
| ema_beta | 0.9 | 所选 lambda 的 EMA，范围 [0,1) |
| two_level | true | 是否执行二级搜索 |
| lambda_refresh | rollout | rollout 或 epoch |
| grad_chunk_size | 0 | 搜索梯度不分块；正数指定分块大小，系列 YAML 可覆盖 |
| var_penalty_beta / n_var_folds | 0 / 3 | 梯度对齐分数的方差惩罚及训练折数量；beta>0 时启用 |
| use_spo | false | rollout 模式采用原实现的 SPO 分支，优先于 kl_coef；epoch 模式不使用此分支 |
| batch_size | null | 显式指定每个 minibatch 的样本数；null 时按有效样本数 / num_mini_batch 向下取整 |
| value_loss_scale | 1.0 | 为匹配原实现，必须为 1；value 系数由 vf_coef 控制 |
| value_ref_lambda | null | 可选 value return 参考 lambda；当前实现有至少 0.9 的下限 |
| parity_version | posterior-17ff77b-v1 | 实现版本标识，不应手动修改；用于区分旧实现的输出组 |

搜索分组过小时会退化为共用有效样本；日志 ga2e/split_degenerate 标记此情况。
已对照原仓库 commit `17ff77b73de240974d42a1e3c3046da16e708c8c` 的实际执行行为修正，
包括 epoch 0 再次选 lambda、两种 minibatch 顺序及未缩放的 value MSE。
原源码数值对照、参数映射和验证边界见 [GA2E 一致性审计](ga2e_parity.md)。这不等同于论文性能复现。

## SPO / RVL / DAE

SPO 没有新的配置类字段，epsilon 控制软惩罚强度，复用裁剪范围调度。

| 算法 | 参数及默认值 | 作用 |
|---|---|---|
| RVL | rv_coef=1.25、rv_clip=0.15 | 相对价值损失权重与裁剪，rv_clip 可为 null |
| RVL | optimal_offset=true | 轨迹相对值偏移处理 |
| RVL | value_head=linear | linear 或 mlp_symmetric 的相对价值头 |
| RVL | same_episode_probability=0.33 | 同 episode 配对概率 |
| DAE | critic_epochs=4 | advantage/value 轨迹回归轮数 |
| DAE | full_action=true | 使用完整动作优势策略目标 |
| DAE | dae_lambda=0.95 | advantage 序列平滑 |
| DAE | normalize_advantage=false | 与 PPO 缺省值不同 |

RVL 源码记录核对 [官方实现](https://github.com/Hauf3n/relative-value-learning) 的 98ff50b8 版本，
保留单步分支与终止目标启发式，包括两端均 terminal 时目标为零。这个说明不等于覆盖官方全部变体。

## V-MPO

| 参数 | 默认 | 用途 |
|---|---|---|
| epsilon_eta | 0.1 | 温度约束 |
| epsilon_kl | 0.01 | 离散 KL 约束 |
| epsilon_mu | 0.01 | 高斯均值 KL 约束 |
| epsilon_sigma | 0.00001 | 高斯尺度 KL 约束 |
| dual_lr | 0.001 | 对偶参数初始学习率 |

这些是 V-MPO 约束，不能用 PPO epsilon 的意义解释。对偶变量随 checkpoint 保存。

## DPPO

增加 gamma_denoising=0.99，要求 ent_coef=0；其余 rollout、GAE、PPO 裁剪参数沿用公共配置。
按环境步计算优势，再按去噪步降权。记录完整链，不使用最终动作的伪高斯 likelihood。
模型参数和论文差异见 [网络](networks.md)。

## 源码定位与来源

[ppo.py](../src/onpolicy/ppo.py)、[ga2e.py](../src/onpolicy/ga2e.py)、[spo.py](../src/onpolicy/spo.py)、
[rvl.py](../src/onpolicy/rvl.py)、[dae.py](../src/onpolicy/dae.py)、[vmpo.py](../src/onpolicy/vmpo.py)、[dppo.py](../src/onpolicy/dppo.py)。

PPO 可参照 [原论文](https://arxiv.org/abs/1707.06347)，DPPO 和新增骨干来源见网络页。
GA2E/SPO/DAE 的论文与移植来源映射仍需独立维护完整审计记录，本页不补写未经核对的论文编号。
不保留旧探索算法 BGAE/TGAE/PGAE，也没有 vendor/posterior 依赖路径。
