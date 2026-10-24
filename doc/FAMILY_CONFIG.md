# 环境系列配置协议 v2

`configs/` 仅保留 `minatar.yaml`、`mujoco.yaml`、`classic_control.yaml`。
一个文件声明整个实验矩阵，不再为算法、环境、预检各维护一份配置。
算法模块负责计算；配置编译器负责将声明展开为不可混淆的 Job；训练与报告消费同一份任务定义。

```yaml
version: 2
family: minatar
algo: [ppo, ga2e]                 # 单算法也可写 algo: ppo
envs: [MinAtar/Asterix-v0, MinAtar/Freeway-v0]
runs: 5                          # seed 固定为 0..4
resume: true
train:
  total_steps: 10000000
  num_envs: 512
  rollout_len: 128
  device: cuda:0
model:
  image_encoder: minatar_reference
  lr: 0.00025
environment:
  backend: jax
  jax_platform: cpu
  sticky_action_prob: 0.1
evaluation:
  eval_episodes: 100
  eval_num_envs: 20
algorithms:
  ppo:
    train: {gae_lambda: 0.95}
  ga2e:
    train: {lambda_val: 0.99}
environments:
  MinAtar/Freeway-v0:
    environment: {max_episode_steps: 2500}
```

上面是协议示意；正式实验使用仓库配置中的完整预算和优化设置。
`algo` 和 `envs` 是选择器，`algorithms`、`environments` 是稀疏覆盖表，覆盖表本身不会增加任务。
暂时未选中的已知算法/环境覆盖可以保留，切换选择器即可复用。
合并顺序：共享段 → 算法覆盖 → 环境覆盖 → `sweep`。字典递归合并，列表整体替换。
扫描示例：`sweep: {model.lr: [0.0001, 0.00025]}`；普通段里的列表是固定值。
任务数 = 算法数 × 环境数 × runs × 扫描组合数。

编译阶段检查顶层和覆盖段拼写、所选算法参数、模型参数、评估参数、重复选择与重复任务。
MinAtar 的 sticky 参数默认 0.1 且校验范围；其他系列禁止设置该参数。
编译不创建环境，不保证某个机器已经安装该环境或具备所需设备；这些在运行时检查。
`family` 当前支持 minatar、mujoco、classic_control；环境使用完整注册 ID。

```bash
cd ~/OnPolicyRL
# 验证并查看每个任务的显式合并参数、seed、标识、输出路径，不启动训练
PYTHONPATH=src .venv-jax/bin/python -m oprl.cli configs/minatar.yaml --plan
# 执行同一任务计划（当前 runner 按顺序执行）
PYTHONPATH=src .venv-jax/bin/python -m oprl.cli configs/minatar.yaml
# 指定支持 v2 的代码提交；配置直接从当前本地文件读取，无需提交配置
PYTHONPATH=src .venv-jax/bin/python -m oprl.cli configs/minatar.yaml --commit <SHA> --detach
```

输出按算法、配置摘要、环境和 seed 隔离；修改共享参数/覆盖/扫描值会改变目录，防止把旧 DONE 当作新配置结果。
相同配置的 seeds 使用同一报告组；不同算法独立分组。框架默认参数最终写在运行时 `config.json` 的 `algo_config` 中。
固定提交运行继续记录源代码 SHA、配置快照和依赖版本。旧提交若不支持 v2，需要使用相应旧格式配置，不能让旧解析器自动理解未来协议。

历史配置逐字节备份在 `.integration-staging/configs_before_family_*`，原 CartPole 本地修改保留在备份及迁移后的共享设置中。
旧 `baseline/space` 解析仅用于兼容历史实验和测试，不作为新的配置组织方式。
历史专用生成/启动脚本随配置备份归档；已经运行的实验使用自己的冻结代码与配置，不受迁移影响。
