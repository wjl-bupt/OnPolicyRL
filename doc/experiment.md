# 实验管理与参数修改

本页是人工启动实验的操作手册。所有未标“实际启动”的命令仅预览。
请从仓库根目录执行。先阅读 [快速开始](quickstart.md)，字段协议见 [配置](configuration.md)。

## 1. 修改入口与优先级

三种入口：直接编辑 configs/<系列>_<后端>.yaml；编辑 scripts/launch.sh 顶部参数区；使用环境变量和 --set 临时覆盖。
优先推荐把长期设置写在系列 YAML，单次变化用 --set。无需为每个算法/网络复制一份 YAML。

| 脚本变量 | 修改内容 | 脚本当前缺省行为 |
|---|---|---|
| CONFIG | 选择文件 | configs/minatar_jax.yaml |
| ALGOS | algo | [ppo, ga2e]，会覆盖文件 |
| ENVS | envs | 空值保留文件列表 |
| RUNS | runs | 5，会覆盖文件 |
| TOTAL_STEPS | train.total_steps | 10000000，会覆盖公共字段 |
| DEVICE | CLI --device | cuda:0，实际运行时优先于配置 |
| ARCH | model.architecture | 空值保留文件 |
| ACTION_HEAD | model.action_head | 空值保留文件 |
| CONTEXT_LENGTH | model.context_length | 空值保留文件 |
| DIFFUSION_STEPS | model.diffusion_steps | 空值保留文件 |
| PYTHON | Python 解释器 | 仓库 .venv-jax/bin/python |
| COMMIT | 固定代码版本 | 空值使用当前工作区源码 |

脚本使用 `${变量:-默认值}`，因此 ALGOS/RUNS/TOTAL_STEPS/DEVICE 设为空字符串不会取消其默认覆盖。
如果想完全使用配置文件参数，请用原生 CLI 的 --plan，而不是假设脚本不覆盖。
脚本默认设置 OMP_NUM_THREADS/OPENBLAS_NUM_THREADS 为 1，可通过同名环境变量调整。

配置生成与任务合并是两层覆盖：文件 → 脚本 --set → 用户追加 --set；
随后公共段 → 算法段 → 环境段 → sweep。最后追加公共参数也不能压过任务展开时的专属段或扫描。

## 2. 算法、环境与种子

```bash
CONFIG=configs/minatar_jax.yaml ALGOS='[ppo, ga2e]' ENVS='[MinAtar/Asterix-v0, MinAtar/Seaquest-v0]' RUNS=5 bash scripts/launch.sh
```

预期 2 × 2 × 5 = 20 个任务（此配置无额外 sweep）。单算法写 ALGOS=ppo；单环境仍写 YAML 列表。
`runs: 5` 使用 seed 0、1、2、3、4。当前 family 协议没有任意 seed 列表入口，
不要用 train.seed 替代：seed 由任务编排注入。runs 不是 num_envs，也不是 GPU/进程数量。
普通 runner 顺序执行展开任务，不因 runs=5 自动并行。

切换后端应切换 CONFIG，并使用该文件支持的 ID；不要仅把 backend 改成 jax 而保留不兼容的环境名称。

## 3. 训练规模和批次

| 路径（train. 下） | 含义与修改注意事项 |
|---|---|
| total_steps | 每个任务的目标环境 transition 数；非所有 seed 合计 |
| num_envs | 每次采样同时推进的环境数量，非评估数量 |
| rollout_len | 每个环境每次 update 前采样的长度 |
| num_mini_batch | 一个 epoch 划分的 minibatch 数量，不是 batch_size |
| ppo_epoch | 同一 rollout 重复更新的 epoch 数；DAE 另有 critic_epochs |
| log_interval | 每多少次 update 输出训练日志；必须正整数 |

rollout 大小 B=num_envs×rollout_len。标准完整有效 rollout 每批约 B/num_mini_batch 个样本；
无效 autoreset transition 会被排除，批次可能不等大。不要把批次数设得超过有效样本数。
无恢复时实际采样步数通常为 ceil(total_steps/B)×B。增加 epoch 不增加环境步数，但增加训练计算。
Transformer 的历史窗口与 Diffusion 去噪链会额外占用内存。

## 4. 通用算法参数

以下是路径与语义，不是所有算法都实际使用全部继承字段；专属行为以算法页和实现为准。

| train. 参数 | 修改说明 |
|---|---|
| gamma | 环境时间折扣，范围 [0,1] |
| gae_lambda | GAE 参数，范围 [0,1]；PPO 比较使用 0.95，勿与 GA2E lambda_val 混淆 |
| adv_estimator | PPO 配置保留字段；当前 NativeLoop 使用原生 GAE，不应靠修改此字段宣称切换估计器 |
| epsilon | PPO 策略 ratio 裁剪范围；SPO 中是软惩罚尺度，不是优化器 epsilon |
| clip_schedule / clip_range_min | constant 或 linear 以及裁剪下限 |
| use_value_clipped / vf_epsilon | 是否裁剪 value 更新及其绝对值范围；并非策略 ratio 裁剪 |
| vf_coef / value_loss_scale | value 损失权重与内部尺度，可能共同缩放；不是所有算法都使用相同 value 目标 |
| ent_coef | 熵奖励权重；DPPO 当前必须为 0 |
| kl_coef / target_kl | PPO 的显式 KL 惩罚、单次 update 提前结束阈值；不等于停止整场实验 |
| max_grad_norm | 梯度范数裁剪 |
| normalize_advantage | 优势标准化；DAE 默认 false，算法内部处理可能不同 |
| anneal_lr / lr_min | 学习率线性退火与下限 |

修改 PPO 专属参数的正确示例：

```bash
ALGOS=ppo bash scripts/launch.sh   --set algorithms.ppo.train.gae_lambda=0.95   --set algorithms.ppo.train.epsilon=0.15
```

只改 train.gae_lambda 可能被 algorithms.ppo.train.gae_lambda 覆盖。PPO 的 0.95 不是所有算法的统一 lambda。
GA2E、RVL、DAE、V-MPO、DPPO 的专属参数表见 [算法页](algorithms.md)，应写到 algorithms.<algo>.train 下。

GA2E 与原仓库对照时，要显式对齐 minibatch 样本数：
`algorithms.ga2e.train.batch_size` 对应原 SB3 的 `batch_size`，不等于 `num_mini_batch`。
`batch_size: null` 才使用本框架按有效样本数推导的默认行为。
GA2E 固定 `value_loss_scale: 1.0`（不接受 0.5），通过 `vf_coef` 调整 value 损失权重。
`grad_chunk_size: 0` 表示不分块；系列文件可能另有覆盖，数值复核时也要对齐。
修正后的 GA2E 输出组包含实现版本，避免沿用旧结果的 DONE/checkpoint；不要手动把旧 checkpoint 放进新组。
完整映射和原实现特有的 epoch 行为见 [一致性审计](ga2e_parity.md)。

## 5. 学习率与优化器

学习率路径是 **model.lr**，不是 train.lr；Adam epsilon 为 model.adam_eps。
退火路径为 train.anneal_lr / train.lr_min。V-MPO 的 train.dual_lr 是对偶参数初始学习率。
当前配置中的 optimizer_cls / activation 是 Python 类型接口，不支持用字符串 "AdamW" / "ReLU" 自动解析。
不要将论文的 AdamW 或 weight decay 当作已经由某个网络名称自动启用；YAML 可用字段以实际编译与构建为准。

## 6. 网络骨干与动作头

| model. 参数 | 生效范围与说明 |
|---|---|
| architecture | mlp、simba、bronet、spectralnet、causal_transformer |
| action_head | auto（分类/高斯）或 diffusion；后者要求 algo=dppo |
| net_arch | MLP 向量隐藏层列表，例如 [128,128] |
| image_encoder | auto 或 minatar_reference |
| features_dim | CNN 输出宽度 |
| image_net_arch | 原有图像 MLP 后接层；新骨干不使用它 |
| share | 是否共享 actor/critic 骨干；普通图像路径强制共享 CNN，Transformer 图像路径可分开 |
| hidden_dim / num_blocks | 新骨干共同默认宽度/块数，非 MLP net_arch |
| actor_hidden_dim / critic_hidden_dim | 分别覆盖宽度 |
| actor_num_blocks / critic_num_blocks | 分别覆盖块数 |
| expansion | SimBa 扩展倍数；Transformer FFN 也使用此倍数；BroNet/SpectralNet 不使用 |
| norm_eps | 新骨干 LayerNorm epsilon |
| rsnorm | true/false/null；null 时 SimBa 自动要求观测归一化，其他骨干不强制 |
| ortho_init | 原有提取器与输出头初始化选项；新骨干有自己的初始化规则 |
| log_std_init | 连续高斯动作头初始 log std；不控制 Diffusion 噪声 |

Transformer 专属：context_length 默认 16，attention_heads 默认 4，transformer_gated 默认 true。
宽度必须能被头数整除；frame_stack 必须等于 context_length。已有 frame_stack 时显式改成一致值。
共享 Transformer/共享向量骨干要求 actor/critic 宽度和块数一致。

Diffusion 专属：diffusion_steps=10、diffusion_hidden_dim=256、diffusion_blocks=2、
diffusion_time_dim=32（偶数且至少 4）、diffusion_min_std=0.1、diffusion_clip_denoised=1.0。
这些是模型代码默认值，不是性能最优保证。去噪步数不是环境步数；当前仅连续扁平 Box、单步动作。
选择 diffusion 不会把 GA2E 自动改写成 DPPO；不兼容组合会报错。详见 [网络页](networks.md)。

## 7. 环境选项与设备

| 路径 | 修改说明 |
|---|---|
| environment.backend / implementation | 接口后端与实际实现；不要随意混配 |
| environment.jax_platform | JAX 仿真设备，通常 cpu；服务器当前 JAX 环境没有 CUDA plugin |
| train.device 或 DEVICE | PyTorch 网络与 tensor 设备，如 cpu/cuda:0；DEVICE 是脚本传入的最高运行时覆盖 |
| environment.max_episode_steps | episode 时间上限；截断与自然终止的 bootstrap 不同 |
| environment.sticky_action_prob | 仅 MinAtar；新配置默认 0.1；不是 Atari 的配置字段 |
| environment.difficulty_ramping | MinAtar 难度递增设置 |
| environment.use_minimal_action_set | MinAtar 动作集合，影响动作头维数 |
| environment.wrappers.* | 统一 wrapper，见环境页 |

例：环境在 CPU、网络在 GPU，不应把两者混为一个 device 参数。

```bash
DEVICE=cuda:0 bash scripts/launch.sh --set environment.jax_platform=cpu   --set environment.sticky_action_prob=0.1
```

对某个环境设置独立上限，请在 YAML 中写：

```yaml
environments:
  MinAtar/Freeway-v0:
    environment:
      max_episode_steps: 2500
```

## 8. 评估、日志与 checkpoint

| evaluation. 参数 | 单位/特殊值 |
|---|---|
| eval_interval | update 次数；0 禁用周期评估，但训练结束仍评估 |
| eval_episodes | 每次评估请求的完整 episode 总数，不是每个 eval env 的数量 |
| eval_num_envs | 评估并行环境数量，不修改训练 num_envs |
| eval_seed | 不填时 runner 使用任务 seed+10000 |
| ckpt_count | 1 仅最终；>=2 首次、中间、最终，rollout 跨越目标时可能少于指定个数 |
| ckpt_interval | ckpt_count<=0 才使用，单位 update；0 仍保留最终保存 |
| kl_threshold | 异常报告阈值，不自动停止训练 |

train.log_interval 控制日志输出。评估存在安全步数上限，应检查 eval/n_episodes 是否达到请求值。
评估/保存失败可能只记录事件，训练继续；DONE 并不保证评估和 checkpoint 成功。

## 9. 输出、停止与恢复

`run_dir` 指定实验根目录；内部按算法+组哈希/环境/seed-N 分层。name 用于未指定 run_dir 时的默认命名。
新参数比较建议用新 run_dir。resume=true 且 DONE 已存在会直接跳过；不要以为改任意字段就一定触发重跑。
没有 DONE 时尝试读取最新模型池，配置/硬件指纹匹配才恢复。
**当前指纹不匹配会清理模型池并从头训练；resume=false 同样会清理旧模型池。** 不要把实验目录当成不可变归档。
日志文件也不应假定会自动清空。保存重要结果后再做重跑，或直接使用新的目录。

当前工作区的前台脚本可以 Ctrl-C 停止；不会自动在中断瞬间保存最新状态，只能恢复已有 checkpoint。
固定 commit 使用子进程，后台任务请根据本次 manifest.json/launch.log 中的 PID 定向停止，并确认退出，
不要用宽泛的 pkill python。若存在外部 supervisor，还需先停止调度，避免新任务继续启动。
恢复包含策略、优化器、随机数和归一化统计；不包含精确环境状态，环境和历史窗口会重新 reset。

## 10. 固定 commit，保留本地配置

```bash
# 将占位符替换为真实 7～40 位十六进制 commit ID；这里只准备，不训练
COMMIT=<commit-id> bash scripts/launch.sh --prepare
# 实际启动，人工执行
COMMIT=<commit-id> bash scripts/launch.sh --run
```

不接受分支/tag。缺失对象从已有 remote 获取；源码缓存排除 configs/ 与 runs/，不 checkout/reset 当前工作区。
本地 YAML 与 --set 合成后独立快照。输出位于 runs/by_commit/<sha>/<配置哈希>/<时间标识>/，
其中 manifest.json 记录源码、配置、解释器及依赖版本；日志在 launch.log。
每次启动产生独立目录，不会因 resume=true 自动找到上次 pinned run。
使用当前解释器，不会自动安装该 commit 的历史依赖。旧 commit 必须包含所选功能与配置协议。
--prepare 只准备源码/快照，不证明目标 commit 能成功构建环境或训练。
--plan 检查当前代码，不能冒充目标 commit 的验证。

## 11. 五个可复制的参数修改示例

### A. 只修改 PPO

```bash
ALGOS=ppo RUNS=2 ENVS='[MinAtar/Asterix-v0]' bash scripts/launch.sh   --set algorithms.ppo.train.gae_lambda=0.95 --set model.lr=0.0001
```

预期 2 个任务。确认 ppo 的 algo_params.gae_lambda 和 network_params.lr。

### B. 比较 PPO 与 GA2E

```bash
ALGOS='[ppo, ga2e]' RUNS=2 ENVS='[MinAtar/Asterix-v0]' bash scripts/launch.sh   --set algorithms.ppo.train.gae_lambda=0.95   --set algorithms.ga2e.train.lambda_val=0.99
```

预期 4 个任务；不要把 lambda_val 放进公共 train 导致 PPO 收到不支持字段。

### C. 切换网络

```bash
CONFIG=configs/mujoco_jax.yaml ALGOS=ppo ARCH=simba RUNS=1 bash scripts/launch.sh --set model.actor_hidden_dim=256 --set model.actor_num_blocks=1   --set model.critic_hidden_dim=512 --set model.critic_num_blocks=2
```

预期 1 个任务，检查 normalize_observation=true。切换 Transformer/Diffusion 的组合见网络页。

### D. 修改环境和 seed 数

```bash
ALGOS=ppo ENVS='[MinAtar/Asterix-v0, MinAtar/Seaquest-v0]' RUNS=3 bash scripts/launch.sh --set environment.sticky_action_prob=0.1
```

预期 6 个任务，seed 0、1、2；并不是 3 个并行训练进程。

### E. 扫描参数，再固定代码版本

```bash
ALGOS=ppo RUNS=2 ENVS='[MinAtar/Asterix-v0]' bash scripts/launch.sh   --set 'sweep={model.lr: [0.0001, 0.0003], train.epsilon: [0.1, 0.2]}'
```

预期 8 个任务。保留同样变量与 --set，将脚本模式改为 --prepare 并提供 COMMIT，准备目标版本；
人工决定训练时改 --run。sweep 最后覆盖算法/环境专属字段。

## 12. 启动前检查

预览 jobs 数、algo/env/seed、总步数、学习率、算法专属值、wrapper 和 run_dir。
额外核对 DEVICE：--plan 的 JSON 不会完整反映 --device 的最终覆盖。
--plan 不是设备或资源可用性检查；遇到问题见 [排查](troubleshooting.md)。
