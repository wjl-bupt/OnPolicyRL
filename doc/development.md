# 扩展开发

## 新算法

在 src/onpolicy/<name>.py 中导出 CONFIG_CLASS 与 train；可选 build_policy(obs,action,network,algo)。
注册器自动发现模块；辅助文件以下划线开头，避免被误识别为算法。不创建新的 vendor 框架。

```python
from dataclasses import dataclass
from onpolicy.ppo import PPOConfig
from onpolicy._runtime import NativeLoop, run_algorithm

@dataclass
class ExampleConfig(PPOConfig):
    pass

class Example(NativeLoop):
    # 在这里实现该方法真正不同的估计/更新逻辑。
    pass

CONFIG_CLASS = ExampleConfig

def train(cfg, env, policy, log=None, monitor=None, resume=None):
    return run_algorithm(Example, cfg, env, policy, log, monitor, resume)
```

这只是接口示例，没有新算法创新。外部包也可以通过 oprl.algorithms entry point 提供 AlgoEntry；
名称冲突会报错。新增算法需更新发现测试，不能通过移除测试来隐藏接口错误。

## 新网络或动作头

在 architecture/ 实现纯 PyTorch 模块，接入 PolicyConfig 校验和 build_policy。
骨干只产生特征，动作头负责分布/概率/熵；同时检查 actor/critic、辅助头、图像和向量路径。
forward 不应隐式更新观测统计或历史，避免 PPO ratio 重算与 GA2E 梯度计算语义改变。

如果动作分布没有可计算的边缘 log_prob，不应伪装成普通高斯；像 DPPO 一样声明额外
采样数据、扩展 buffer 与匹配更新器。配置阶段尽早拒绝不兼容组合。

## 新环境与 wrapper

新增后端 provider，统一为 tensor adapter；提供稳定 obs/action spaces、真实 final observation、
terminated/truncated/valid、完成 episode 索引和原始回报。记录 provenance。
尽可能在公共 TensorTransforms 加入跨后端相同语义，而不是复制两套实现。
新增 wrapper 参数同步修改校验、checkpoint 状态、评估冻结规则和环境文档。
环境自身新增配置留在系列文件，通过 environments 子段表达差异。

## 配置与复现

新增字段要有默认值、类型/范围/兼容性校验；更新 --plan 与文档示例。
不要通过 eval 解析 YAML 中的 Python 类型字符串。源码固定与本地配置快照的边界应保持清楚。

## 验证

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu PYTHONPATH=src .venv-jax/bin/python -m pytest tests -q -m 'not slow'
```

按改动选择测试：概率一致性、有限梯度、因果性、episode 边界、截断 bootstrap、checkpoint roundtrip、
归一化评估冻结、未知参数拒绝和任务展开。涉及训练流程时优先小型合成环境验证一次更新，
性能实验单独规划，不把单元测试通过等同于算法有效。

文档示例变更可只检查链接、YAML 编译、脚本语法和 --plan；不要为低风险文档更改启动训练。
研究模块需记录论文、核对的官方版本、采用的组件与尚未实现的部分。
