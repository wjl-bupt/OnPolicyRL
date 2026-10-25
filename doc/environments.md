# 环境后端与 wrapper

系列配置入口都在 configs/，同一后端文件内选择算法、环境与 seed，不另建算法专用文件。
“已接入”描述接口；不等于所有任务/图像模式都做过性能验证。

| 系列 | Gymnasium 配置 | JAX 配置/实现 |
|---|---|---|
| MinAtar | [minatar_gymnasium.yaml](../configs/minatar_gymnasium.yaml) | [minatar_jax.yaml](../configs/minatar_jax.yaml)，Gymnax |
| MuJoCo | [mujoco_gymnasium.yaml](../configs/mujoco_gymnasium.yaml) | [mujoco_jax.yaml](../configs/mujoco_jax.yaml)，Brax/MJX |
| Classic Control | [classic_control_gymnasium.yaml](../configs/classic_control_gymnasium.yaml) | [classic_control_jax.yaml](../configs/classic_control_jax.yaml)，Gymnax |
| Atari | [atari_gymnasium.yaml](../configs/atari_gymnasium.yaml)，ALE | [atari_jax.yaml](../configs/atari_jax.yaml)，jaxtari；真实资源验证未完成 |
| DMControl | [dmcontrol_gymnasium.yaml](../configs/dmcontrol_gymnasium.yaml)，dm-control | [dmcontrol_jax.yaml](../configs/dmcontrol_jax.yaml)，MuJoCo Playground |
| MiniGrid | [minigrid_gymnasium.yaml](../configs/minigrid_gymnasium.yaml) | [minigrid_jax.yaml](../configs/minigrid_jax.yaml)，XMiniGrid |
| Craftax | [craftax_gymnasium.yaml](../configs/craftax_gymnasium.yaml)，Gymnasium 接口桥接 | [craftax_jax.yaml](../configs/craftax_jax.yaml)，Craftax |
| Box2D | [box2d_gymnasium.yaml](../configs/box2d_gymnasium.yaml) | 当前未提供 JAX 系列配置 |
| Toy Text | [toy_text_gymnasium.yaml](../configs/toy_text_gymnasium.yaml) | 当前未提供 JAX 系列配置 |

例如 MuJoCo 两端使用 InvertedPendulum-v5 与 MJX/inverted_pendulum，不能只换 backend 保留环境 ID。
具体可用 ID 以配置和 provider 支持为准，不代表整个上游环境集合都已接入。

## 安装与设备

pyproject.toml 提供 jax/dmcontrol/playground/craftax/xminigrid/jaxtari/box2d 等 extra。
Atari 需要对应后端资源；ALE ROM 与 jaxtari 资源不能假定互换。
Box2D 安装可能需要 SWIG，dm-control/Playground 涉及 MuJoCo 与渲染系统依赖。

train.device/--device 控制 PyTorch 网络、buffer 和 tensor wrapper；environment.jax_platform 控制 JAX 仿真。
服务器目前 JAX 以 CPU 运行；网络可以在 CUDA 上。不要把“JAX 后端”理解为默认全 GPU pipeline。

## 统一 tensor wrapper

入口为 environment.wrappers；未指定的选项使用以下默认值。

| 参数 | 默认 | 语义 |
|---|---|---|
| normalize_observation | false | 按观测运行均值/方差标准化 |
| observation_clip | null | 归一化后的观测裁剪；不开归一化时该项不单独裁剪 |
| normalize_reward | false | 用折扣回报标准差缩放 reward，不减均值 |
| gamma | 0.99 | reward 归一化累计回报的折扣；独立于 train.gamma，需人工对齐 |
| reward_sign | false | 训练 reward 取符号 |
| reward_clip | null | 训练 reward 对称裁剪 |
| epsilon | 1e-8 | 归一化稳定常数 |
| action_mode | clip | clip/rescale/none；rescale 将 [-1,1] 映射至有限 Box 边界 |
| frame_stack | 1 | 拼接历史观测；向量拼特征、图像拼通道 |

训练时更新统计；评估复制并冻结训练统计，保留原始 reward；日志 episode return 也使用原始回报。
当前 reward 处理顺序为归一化 → sign → clip，不能当作可任意重排的列表。
自然终止停止 bootstrap，时间截断使用真实 final observation bootstrap；无效 autoreset transition 排除。
历史在新 episode 重置，旧 episode 的终止窗口用于 bootstrap。详见 [网络](networks.md)。

## 各系列差异

- MinAtar：新 family 配置默认 sticky_action_prob=0.1；difficulty_ramping、动作集合和 episode 上限也需对齐。PPO 的 lambda 在 algorithms.ppo.train.gae_lambda=0.95。历史 JAX 批次没有额外 sticky wrapper，不应与新配置混作同组。
- Atari：预处理、frame skip、灰度/缩放及 frame stack 属于基准定义的一部分；按各端配置逐项核对，不能仅凭同名游戏宣称等价。
- MuJoCo/DMControl：动作缩放、观测归一化、时间上限与动作重复会改变任务；JAX Playground 当前主要为状态输入，非完整视觉模式复现。
- MiniGrid：原生接口将符号观测、方向和 mission 编码为数值输入；不是任意 Dict 直接交给策略。XMiniGrid 与 MiniGrid 不应默认逐步动态一致。
- Craftax：Gymnasium 配置只是另一种接口，底层仍是 Craftax JAX 仿真，不是独立的 CPU Python 实现。
- Toy Text：离散观测通过数值编码适配策略；不要把离散状态编号当连续物理量。

最终实际 provider、请求选项、wrapper 会写入每个任务的 environment.json，比较实验时检查该文件。
相关源码：[factory.py](../src/common/env/factory.py)、[transforms.py](../src/common/env/transforms.py)。
