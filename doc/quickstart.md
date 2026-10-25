# 快速开始

## 1. 选择 Python 环境

在服务器仓库根目录工作：

```bash
cd ~/OnPolicyRL
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
.venv-jax/bin/python -m oprl.cli --list
```

当前服务器已有 `.venv-jax`，其中 JAX 负责环境仿真、PyTorch 负责策略训练。名称不代表策略本身用 JAX。
新机器请单独建虚拟环境，先安装与驱动匹配的 PyTorch，再按项目依赖安装：

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m pip install pytest
# 仅需要时安装；额外依赖及版本见 pyproject.toml
python -m pip install -e '.[jax,viz]'
```

不要把项目声明的 Python 最低版本视为每个可选后端都支持该版本。当前 JAX 组合在服务器 Python 3.13 环境验证；
部分依赖还需要系统组件、ROM 或资源文件，安装步骤不是完整环境验证。不要覆盖已经工作的实验环境来尝试升级。

## 2. 预览一个最小任务

仓库已有 classic_control 配置含大训练步数、256 个 epoch 和参数扫描，不宜直接当最小任务运行。
下面显式覆盖这些字段；命令只预览：

```bash
CONFIG=configs/classic_control_gymnasium.yaml ALGOS=ppo RUNS=1 TOTAL_STEPS=1024 DEVICE=cpu ENVS='[CartPole-v1]' bash scripts/launch.sh --set sweep={} --set train.ppo_epoch=2   --set train.num_envs=4 --set train.rollout_len=32 --set train.num_mini_batch=4   --set run_dir=runs/manual_cartpole --set evaluation.eval_episodes=2
```

应看到 `jobs: 1`，seed 为 0，总目标步数为 1024。预览不创建仿真环境，也不验证 ROM、GPU 内存或数值稳定性。

## 3. 实际训练（由人工启动）

确认上一步后，使用相同命令，在 `bash scripts/launch.sh` 后立即插入 `--run`。
**裸 CLI 不带 `--plan` 就会训练，脚本不带 `--run` 则只预览。**

## 4. 查看结果

任务目录在 `runs/manual_cartpole/` 下，按算法组、环境、seed 分层。
查看 `config.json`、`environment.json`、`metrics.jsonl`、`model_pools/`、`DONE`。
不要只看 DONE：评估或保存失败可以仅记录事件而不阻止训练完成，需同时检查异常日志。
更多说明见 [实验管理](experiment.md)、[结果](results.md)。
