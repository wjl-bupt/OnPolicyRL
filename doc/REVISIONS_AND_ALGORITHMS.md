# 代码版本、实验配置与算法扩展

## MinAtar 专属默认值

Gymnasium 和 JAX 两个 MinAtar 后端现在默认 `sticky_action_prob=0.1`。
所有 `configs/minatar_compare/` 和 `configs/jax_minatar_compare/` 中的
MinAtar 配置均显式记录该字段：

```yaml
baseline:
  env_params:
    backend: jax
    jax_platform: cpu
    sticky_action_prob: 0.1
    use_minimal_action_set: false
```

该默认值仅在 MinAtar 环境中注入；MuJoCo、Atari 和经典控制不注入。
合法范围为 [0,1]，允许显式设 0 关闭，拒绝 NaN/Inf。JAX 版在执行动作
前进行 Bernoulli 采样，以指定概率重复上次实际执行的动作。首次 reset
初始动作是 0，同步自动重置保持上一执行动作，与原版 MinAtar 保持一致。
不同向量环境使用独立随机 key，训练和评估使用相同环境配置。

为保存截断时的最终观测，适配器在原始 step_env 上实现此功能，不调用会
提前自动重置的 Gymnax wrapper。测试覆盖概率 0/1、默认 0.1、种子复现、
Gymnasium 参数透传、非法值和非 MinAtar 配置隔离。

旧实验的 runs/.../campaign_*/source 和 configs 快照未改动；正在运行的
无 sticky JAX 实验仍属于原批次。此修改不意味着它们已自动变成 p=0.1。

## 固定代码 commit，配置无需提交

在仓库根目录运行（把 CODE_SHA 替换为代码提交的真实 SHA）：

```bash
PYTHONPATH=src .venv-jax/bin/python -m oprl.cli \
  configs/jax_minatar_compare/ppo_Asterix.yaml \
  --commit CODE_SHA --detach
```

支持 `--repo PATH`、`--remote origin`、`--code-cache PATH`、`--run-root PATH`、
`--device cuda:0`。`--prepare-only` 仅准备代码、配置快照和清单，不启动训练。
默认前台运行，`--detach` 脱离终端并把日志写到对应运行目录。

启动流程：

1. 接受 7–40 位十六进制 commit ID，并解析成完整 SHA；拒绝可变分支/tag。
2. 本地对象存在则直接使用；缺失时从配置的 remote fetch。短 SHA 缺失时
   先 fetch remote，再解析；服务器未公开的历史提交可能需要完整 SHA。
3. 从 git 对象展开代码到 `.oprl/code/<完整SHA>/`，跳过 configs/，不 checkout
   或清理当前工作区。缓存有文件哈希校验，发现被修改则拒绝运行。
4. 从本地配置文件读取原始字节并保存快照、SHA256；无需 git add/commit。
5. 在独立子进程中仅把所选代码目录放入 PYTHONPATH，并检查 runner 的实际
   导入路径。配置 run_dir 被迁移到独立输出目录，原值保存在 manifest 中。
6. 记录源码 SHA、配置 SHA、配置快照、解释器、依赖版本、PID、结果及退出状态。

默认输出目录：

```text
runs/by_commit/<code_sha12>/<config_sha12>/<UTC时间-随机ID>/
  manifest.json
  config.snapshot.yaml
  config.runtime.json
  launch.log
  results/
```

同一代码 commit 可以反复修改本地配置运行，每次输出独立目录，不复用旧
checkpoint。原配置的 resume 字段只作用于本次输出目录。需要真正续跑时应
明确使用已存在的运行快照，而非期待本入口自动串接不同启动的 checkpoint。

代码变更仍然需要提交一次，commit 才能指向它；本入口不会把未提交的代码
偷偷混入指定版本，也不会自动提交现有工作区。此轮实现仍位于当前工作区，
提交后才能用其新 commit ID 运行。未提交配置始终可用。

代码版本隔离不等同于依赖隔离：当前沿用启动时的 Python 环境，记录依赖
版本但不自动安装历史依赖。选定代码必须与该解释器、依赖和配置格式兼容。

## 算法目录与发现协议

```text
src/onpolicy/
  _runtime.py      # 内部公共训练循环与估计工具
  ppo.py
  ga2e.py
  spo.py
  rvl.py
  dae.py
  vmpo.py
```

默认一算法一模块，消除了 `ppo/ppo.py + __init__.py` 这类重复层级。
算法公式不变，外部仍使用 `from onpolicy.ppo import PPOConfig` 等导入。
内部旧导入 `onpolicy.runtime` 改为 `onpolicy._runtime`。

每个算法模块导出：

```python
CONFIG_CLASS = MyAlgorithmConfig

def train(cfg, env, policy, log=None, monitor=None, resume=None):
    ...

# 可选；不提供时使用框架默认 actor-critic。
def build_policy(obs, action, network, algo):
    ...
```

新增公开模块即可被自动发现，无需改注册器的算法名称/配置类映射。
共享辅助模块使用 `_` 前缀；`test_` 模块不会被发现。需要多文件时可选用
一个实现相同导出协议的包，但不强制每个算法建目录。

大量独立研究算法可以留在外部包，通过安装包的入口注册：

```toml
[project.entry-points."oprl.algorithms"]
my_algo = "my_research.integration:entry"
```

`entry()` 返回 `workflow.registry.AlgoEntry(train, Config, build_policy)`。
重名插件会报错，不允许悄悄覆盖内置算法。插件代码属于独立依赖，需要
固定插件版本；主仓库 commit 不会自动固定外部插件源码。

查看可用算法：`PYTHONPATH=src .venv-jax/bin/python -m oprl.cli --list`。
