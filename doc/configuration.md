# 配置协议

新实验使用 version 2 family 格式。已有旧格式读取兼容不作为新配置范例。

```yaml
version: 2
family: classic_control
name: example
algo: [ppo, ga2e]
envs: [CartPole-v1]
runs: 2
resume: false
run_dir: runs/example
train:
  num_envs: 4
  rollout_len: 32
  num_mini_batch: 4
  ppo_epoch: 2
  total_steps: 1024
  device: cpu
model:
  architecture: mlp
  net_arch: [64, 64]
  lr: 0.0003
environment:
  backend: gymnasium
  implementation: gymnasium
  autoreset_mode: SameStep
  max_episode_steps: 500
  wrappers: {}
evaluation:
  eval_interval: 10
  eval_episodes: 2
  eval_num_envs: 1
  ckpt_count: 1
algorithms:
  ppo:
    train:
      gae_lambda: 0.95
  ga2e:
    train:
      lambda_val: 0.99
environments:
  CartPole-v1:
    environment:
      max_episode_steps: 500
sweep: {}
```

上例是完整可编译配置，展开 4 个任务；文档中其余小片段通常是需要合并进现有文件的局部配置。

## 字段与展开

| 配置段 | 编译后的内部字段 | 职责 |
|---|---|---|
| train | algo_params | 算法配置、训练规模、网络设备 |
| model | network_params | 骨干、动作头、学习率 |
| environment | env_params | 后端、任务选项、公共 wrapper |
| evaluation | monitor | 评估、保存、异常监测 |

`family` 可选 minatar/mujoco/classic_control/atari/dmcontrol/minigrid/craftax/box2d/toy_text。
`algo` 接受字符串或不重复列表，`envs` 必须为不重复非空列表，`runs` 为正整数。
每个算法、环境和扫描组合展开 seed `0..runs-1`。未指定 name 时取配置文件名；未指定 run_dir 时使用 `runs/<name>`。

## 两层覆盖优先级

第一层是生成配置：文件内容 → 脚本生成的 --set → 用户追加的 --set。同一路径最后一次赋值生效。
第二层是任务展开：公共段 → algorithms.<名称> → environments.<完整ID> → sweep。

`--set train.gae_lambda=0.9` 不能覆盖已有 `algorithms.ppo.train.gae_lambda=0.95`。
应改算法专属路径；如果 sweep 也设置同一字段，还需修改或移除该扫描轴。

`--set` 的右侧使用 YAML 解析，列表/映射要按 shell 规则引用：

```bash
PYTHONPATH=src .venv-jax/bin/python -m oprl.cli configs/minatar_jax.yaml --plan   --set 'algo=[ppo, ga2e]' --set 'envs=[MinAtar/Asterix-v0]'   --set algorithms.ppo.train.gae_lambda=0.95 --set runs=2
```

--set 会生成 `.oprl/launch_configs/<内容哈希>.json`，不改原文件；同内容复用同一快照。
--set 目前要求 version 2。删除全部扫描用 `--set 'sweep={}'`。
带点号的环境 ID 不适合逐段的 --set 路径解析，应编辑 YAML，或整体替换 environments 映射。

## 参数扫描

```yaml
sweep:
  model.lr: [0.0001, 0.0003]
  train.epsilon: [0.1, 0.2]
```

上例每个算法/环境/seed 有 4 组参数。总数为 算法数 × 环境数 × runs × 各扫描轴候选数的乘积。
重复组合会报错；扫描字段必须以 train/model/environment/evaluation 开头。
算法专属字段不要放进包含其他不兼容算法的公共 train 或 sweep。

## 自动解析与校验

MinAtar 缺省 sticky_action_prob 为 0.1；SimBa 缺省要求 RSNorm；Transformer 要求
frame_stack=context_length；DPPO 解析 Diffusion 动作头并要求动作 rescale。
显式冲突报错，不静默忽略。不支持的算法参数、网络参数和 wrapper 名称会拒绝。

--plan 展示展开配置，但不会补齐所有 dataclass 默认值，也不会把 --device 覆盖写回打印的 train.device。
真正运行时 --device 优先，任务 config.json 中的 device/algo_config 反映实际值。
核心实现见 [family.py](../src/workflow/family.py)、[overrides.py](../src/workflow/overrides.py)。
