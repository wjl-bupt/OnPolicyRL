# OnPolicyRL

使用自研 PyTorch 模型、采样、RolloutBuffer、算法更新和实验工作流的 on-policy 强化学习框架。

已接入 PPO、GA2E、SPO、RVL、DAE、V-MPO。运行时不依赖 SB3，也不加载 Posterior-GAE 仓库代码。

```bash
cd ~/OnPolicyRL
bash scripts/run_integrated.sh --list
bash scripts/run_integrated.sh configs/integration_ga2e.yaml
```

所有算法统一使用原有 baseline/space/envs/runs/resume 实验协议。算法目录为 src/onpolicy/<算法名>/；模型位于 src/architecture/；src/oprl/ 仅为命令入口。

详见 [架构与迁移说明](doc/INTEGRATION.md) 和 [验证记录](doc/VALIDATION.md)。示例为短训练验证，非收敛基准。


## Code revisions and algorithm extensions

See [commit-based runs, editable local configs, MinAtar sticky actions, and algorithm discovery](doc/REVISIONS_AND_ALGORITHMS.md).
