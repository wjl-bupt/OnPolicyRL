# 网络模块

统一骨干入口：`model.architecture: mlp | simba | bronet | spectralnet | causal_transformer`。
网络是本仓库原生 PyTorch 模块，不依赖 SB3，也不复制外部训练框架。
常规分类/高斯策略的 act/evaluate/predict_value/dist 协议保持不变；DAE、RVL 等辅助头复用同一骨干。Diffusion 使用下文说明的专用链协议。

| 选择 | 输入与残差结构 | 依据 |
|---|---|---|
| mlp | 保持原有 MLP/CNN 行为与参数名 | 原有实现 |
| simba | 线性嵌入；pre-LN、4 倍扩展、ReLU、线性收缩、残差；末端 LN | SimBa Figure 3，官方 ResidualBlock |
| bronet | Linear-LN-ReLU 嵌入；Linear-LN-ReLU-Linear-LN 残差 | 官方 BroNet |
| spectralnet | Linear-ReLU 嵌入；pre-LN、两层谱归一化线性层、中间 ReLU、残差 | SimBa Figure 20(c) |

参数：hidden_dim/num_blocks 为共同默认值，actor_hidden_dim/critic_hidden_dim、
actor_num_blocks/critic_num_blocks 可分别覆盖；共享向量骨干要求两端设置相同。
expansion 默认为 4，用于 SimBa 和 Transformer FFN；norm_eps 默认 1e-6。
现有 net_arch/image_net_arch 仅用于 mlp。图像继续经过原有 CNN，随后接独立
actor/critic 新骨干；这属于我们的 CNN+骨干扩展，不是论文的向量观测复现实验。

SimBa 的 RSNorm 在 version 2 配置编译时自动启用已有观测 wrapper，统计只在采样时更新，
评估冻结并复制训练统计，checkpoint 保存 env_state。不会在 forward 或梯度 minibatch 中更新。
显式 model.rsnorm=false 可做消融；与 wrapper 的相反设置会报错。BroNet/SpectralNet 不默认
启用 RSNorm，如做论文的统一归一化比较，给所有网络设 model.rsnorm=true。
直接 Python build_policy 只构建网络：调用者负责传入已按所需协议归一化的观测。
旧版非 family 配置也需要显式配置观测 wrapper。

SpectralNet 使用精确最大奇异值进行可微权重归一化，forward 不维护可变 power-iteration
状态，便于 GA2E 的 functional_call/vmap/grad。这比幂迭代近似更昂贵，尚未进行吞吐量实验。
实现遵循 Figure 20(c)，不宣称复现 Bjorck 的完整算法或官方数值实现。
SimBa 使用 Kaiming normal 初始化（官方 JAX He initializer 分布细节不同），BroNet 使用
正交 sqrt(2) 初始化。输出头和优化器仍遵循本框架配置；没有暗中更换 PPO 的优化器。
选择骨干不等于复现论文的完整训练配方，尤其 AdamW/weight decay、训练步数和任务设置。

来源（核对于 2026-10-25）：

- [SimBa 论文](https://arxiv.org/abs/2410.09754)，Figure 3 / Figure 20 / Table 10。
- [SimBa 官方 layers](https://github.com/SonyResearch/simba/blob/7d0358b1f1fa3404fc5ea0ffedf9dc1424b49a45/scale_rl/networks/layers.py)，核对 pre-LN 与 4 倍扩展。
- [BRO 官方 common](https://github.com/naumix/BiggerRegularizedOptimistic/blob/5870a13d4bf4f66eeddfc291a59df6d2009de600/jaxrl/networks/common.py)，核对 LN/ReLU 位置及残差相加后没有额外 ReLU。

运行方法见 [实验管理](experiment.md)。已验证协议、梯度、保存恢复和配置展开；尚未验证网络性能。



## 因果 Transformer 与 Diffusion 策略

## 统一配置：观测骨干和动作头是两个独立轴

| 配置 | 含义 | 更新器 |
|---|---|---|
| model.architecture: causal_transformer | 有限历史窗口的因果注意力骨干 | PPO / GA2E / SPO / RVL / DAE / V-MPO |
| model.action_head: auto | 按动作空间选分类或高斯分布 | 原有算法 |
| model.action_head: diffusion | 条件 DDPM 连续动作生成 | DPPO |
| causal_transformer + diffusion | 以历史窗口特征作为去噪条件 | DPPO |

Diffusion 也可以与 mlp、simba、bronet、spectralnet 组合。
`algo: dppo` 默认解析为 Diffusion 动作头。普通 PPO/GA2E 不支持该动作头，
因为最终动作的边缘密度不能用高斯密度替代；配置编译和运行入口均有检查。
所有实现都是本仓库原生 PyTorch，不引入第三方 RL 框架。

## 研究依据与实现边界

- [GTrXL: Stabilizing Transformers for Reinforcement Learning](https://arxiv.org/abs/1910.06764)：采用 pre-LN、GRU 式残差门控和门控偏置初始化；`transformer_gated=false` 可切换普通残差。
- [Transformer-XL + PPO 参考](https://docs.cleanrl.dev/rl-algorithms/ppo-trxl/)：作为在线 PPO 使用记忆的参考，不复制其训练框架。
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)：条件动作去噪策略的研究基础。
- [DPPO](https://arxiv.org/abs/2409.00588)：采用环境时间和去噪时间两层 MDP、逐步似然比裁剪及去噪折扣；单步动作也有论文 Appendix C.3 的实验依据。
- 核对 [DPPO 官方实现](https://github.com/irom-princeton/dppo/tree/cc7234ad7ff39a8f32de3af903606723a16f0648/model/diffusion)，特别是 diffusion.py 的余弦 DDPM 系数与 diffusion_ppo.py 的逐步 PPO 更新。

这里实现的是研究支持的原生适配，尚未验证论文性能复现：

1. Transformer 是固定窗口重计算，使用绝对正弦位置编码，不是带相对位置与跨段缓存的完整 Transformer-XL/GTrXL。历史仅含观测，不含历史动作或 reward-to-go，也不是 Decision Transformer。
2. Diffusion 当前支持有限边界的扁平连续 Box 动作，预测并执行单步动作（Tp=Ta=1）。采用条件残差 MLP 预测噪声；尚未实现动作 chunk、DDIM、UNet 去噪器或冻结早期去噪步。
3. 本实现对动作维度的 log-prob 求和，使用一致的采样/概率计算噪声下限，保持条件高斯密度的数学含义。官方代码还包含 log-prob 截断、维度平均、不同噪声下限、逐步 clip schedule 等实验技巧，这些并未默认照搬。
4. 提供 `DiffusionPolicy.denoising_loss` 用于后续有监督预训练；尚未接入离线数据加载流程，也不直接兼容官方预训练权重。

## 历史窗口与训练协议

`context_length` 默认 16；family 编译器令公共 wrapper 的 `frame_stack` 与之相同。
JAX 与 Gymnasium 都在同一个 tensor wrapper 构造窗口。向量按特征维拼接，图像按通道拼接，
进入 Transformer 后恢复时间轴；图像先逐帧经过原生 CNN。

完整窗口随 obs/next_obs 写入 rollout。打乱 minibatch 时，每条样本仍带自己的历史，
不会跨样本做注意力；因果 mask 禁止访问未来 token。没有 dropout 或隐式可变 KV 状态，
因此采样和更新的输入语义一致，GA2E 可反复计算梯度。

新 episode 的历史用首帧重复填充；终止/截断时，bootstrap 使用旧 episode 的 final-observation 窗口，
下一局使用重置后的窗口。checkpoint 恢复后环境重新 reset 并建立新窗口，
不会宣称精确恢复模拟器和上一局历史。窗口长度会增加 rollout 存储与注意力计算量。

配置参数：

```yaml
model:
  architecture: causal_transformer
  context_length: 16
  hidden_dim: 128
  num_blocks: 2
  attention_heads: 4
  transformer_gated: true
  action_head: auto
```

支持 actor_hidden_dim/critic_hidden_dim 和 actor_num_blocks/critic_num_blocks 分别覆盖。
宽度必须被 attention_heads 整除；共享骨干要求两端尺寸相同。
如果环境配置已有 frame_stack（例如 Atari 的 4），需要显式设为同一窗口长度。

## Diffusion 的概率与更新协议

每个环境步保存 K+1 个去噪状态以及 K 个旧条件 log-prob。DPPO 重新计算相同条件下的
K 个新 log-prob，分别求 PPO ratio；不会用整条链的 ratio 替代逐步 ratio。
环境步优势使用 GAE，早期去噪步通过 gamma_denoising 进一步降权。
critic 仍预测环境状态价值；所有去噪步骤都训练。

动作坐标由公共 wrapper 的 `action_mode: rescale` 统一到 [-1,1]。
最终采样动作可能超出该区间，wrapper 执行裁剪再映射；rollout 保存裁剪前动作和链，
概率计算也使用裁剪前样本，避免把裁剪后的点误当成高斯样本。

`act()` 返回的标量 log-prob 是整条链的条件 log-prob 之和，仅用于协议输出，
不是最终动作的边缘 log-prob。DPPO 实际更新读取单独保存的逐步 log-prob。
`dist()` 和普通 `evaluate(obs, action)` 明确拒绝调用，防止错误的概率语义。

确定性 eval 采用零初始 latent 和每一步条件均值，不消耗全局 RNG；它是一个固定的去噪路径，
不等价于最终动作分布的均值或 mode。随机评估仍可调用原有 `deterministic=false` 入口。
固定噪声下限的条件熵不能代表最终动作熵，因此当前 DPPO 要求 ent_coef=0。

```yaml
algo: dppo
train:
  gae_lambda: 0.95
  gamma_denoising: 0.99
  ent_coef: 0.0
model:
  architecture: mlp  # 也可以是 simba 或 causal_transformer
  action_head: diffusion
  diffusion_steps: 10
  diffusion_hidden_dim: 256
  diffusion_blocks: 2
  diffusion_time_dim: 32
  diffusion_min_std: 0.1
  diffusion_clip_denoised: 1.0
```

## 使用现有脚本

继续使用每个环境系列的一份配置，不增加每模型一份配置。以下命令只预览，不启动训练：

```bash
CONFIG=configs/minatar_jax.yaml ALGOS='[ppo, ga2e]' \
ARCH=causal_transformer CONTEXT_LENGTH=8 \
bash scripts/launch.sh --set model.hidden_dim=128 --set model.attention_heads=4

CONFIG=configs/mujoco_jax.yaml ALGOS=dppo ACTION_HEAD=diffusion DIFFUSION_STEPS=10 \
bash scripts/launch.sh --set train.ent_coef=0.0

CONFIG=configs/mujoco_gymnasium.yaml ALGOS=dppo ARCH=causal_transformer \
CONTEXT_LENGTH=8 ACTION_HEAD=diffusion bash scripts/launch.sh --set train.ent_coef=0.0
```

以后人工决定启动时，在脚本后的第一个位置加 `--run`。支持原有 COMMIT 固定代码、
本地未提交配置与 `--set` 参数快照。指定 commit 必须包含这些新模块。

## 验证

覆盖因果 mask、历史梯度、episode 重置与 terminal 窗口、六种原有算法的策略接口、
GA2E 历史窗口更新、DDPM 条件高斯密度、采样/更新 log-prob 一致性、DPPO 合成 rollout
更新、图像条件输入、checkpoint state_dict 恢复、确定性 eval 的 RNG 隔离和配置兼容性。
测试不等于性能实验；本轮未启动任何 benchmark 批次。
