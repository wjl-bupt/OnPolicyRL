# 原生框架验证记录

本记录替代此前 SB3 包装版的验证结论。

- 环境：`~/OnPolicyRL/.venv`，`stable_baselines3` 未安装。
- 全量测试：**150 passed, 2 warnings, 27.82s**；其中新增原生测试 31 项，原有测试 119 项。两个警告为已有测试中的 tensor 构造提示。
- 覆盖：六算法训练和模型状态恢复；五种支持连续动作算法的 Pendulum 短训练；自研 CNN 及 DAE/RVL 专用头反传；SPO 公式和梯度；DAE 轨迹折扣残差、策略加权零均值；RVL 反对称性与终止目标；GA2E 梯度对照及轨迹分组；V-MPO dual 梯度方向；截断后继观测及有效 episode 统计。
- 六份 configs/integration_*.yaml 通过同一个原生入口完成：各 256 steps，退出码全部 0，failed=0。
- 每个 runs/native_{ppo,ga2e,spo,rvl,dae,vmpo} 均有 results.json、DONE、原生 model_step-64.pt 和 model_step-256.pt。
- git diff --check 通过；源 Posterior-GAE 仓库不修改，用户原 cartpole_ppo.yaml 修改不覆盖。

测试命令：

```bash
cd ~/OnPolicyRL
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest tests -q
bash scripts/run_integrated.sh configs/integration_ga2e.yaml
```

这些结果验证实现链路、核心公式与梯度，不证明长训练收敛或论文指标复现。迁移支持的算法分支与未移植研究开关见 INTEGRATION.md。旧 SB3 checkpoint 不兼容；原生恢复不保存环境内部状态，不承诺逐位复现不中断训练。


后续 RVL 官方核对及 MinAtar 适配：全量 153 passed；两个算法满尺寸 GPU 预检通过；50 个长训练任务已启动，详见 RVL_OFFICIAL_AUDIT.md 与 MINATAR_CAMPAIGN.md。
