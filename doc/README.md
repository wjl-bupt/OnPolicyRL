# oprl 使用文档

本目录面向当前原生框架，核对日期：2026-10-25。示例默认预览任务；标为“实际训练”的命令才会启动训练。

## 阅读顺序

- 实验使用者：[快速开始](quickstart.md) → [实验与参数修改](experiment.md) → [结果与曲线](results.md)。
- 自定义实验：[配置协议](configuration.md) → [环境](environments.md) → [算法](algorithms.md) → [网络](networks.md)。
- 框架开发者：[架构与协议](architecture.md) → [扩展开发](development.md)。
- 遇到问题：[排查指南](troubleshooting.md)。
- GA2E 移植核验：[原仓库一致性审计](ga2e_parity.md)。

## 当前能力与边界

| 维度 | 当前实现 |
|---|---|
| 算法 | PPO、GA2E、SPO、RVL、DAE、V-MPO、DPPO；原生 PyTorch |
| 网络 | MLP/CNN、SimBa、BroNet、SpectralNet、固定窗口因果 Transformer |
| 动作头 | 离散分类、连续对角高斯；连续动作 DDPM + DPPO |
| 环境 | Gymnasium 与 JAX 分开配置，公共 tensor wrapper；具体后端见环境页 |
| 编排 | 单系列配置，多算法/环境/seed/扫描；默认顺序执行 |
| 复现 | commit 固定源码、本地配置快照、checkpoint 与归一化统计 |
| 尚不具备 | 通用多机调度、精确模拟器断点恢复、完整 Transformer-XL 跨段缓存、Diffusion 动作 chunk/DDIM |

有配置文件不等于已完成性能验证。JAX Atari 的真实资源验证尚未完成；Craftax 的 Gymnasium 接口仍使用 JAX 仿真内核。
历史的 GA2E/PPO 曲线不能证明新增网络的性能。本文档不把短测试或接口测试称为论文复现。

## 维护约定

配置、参数默认值以当前版本源码为准；文档示例不是适用于所有任务的最佳超参数。
每个环境系列每个后端一份配置，不按算法或网络复制配置。
`recipe/` 的旧入口只链接到本目录，避免重复维护。
