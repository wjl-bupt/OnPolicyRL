# MinAtar PPO / GA2E 比较实验

任务已在服务器后台启动：**2 算法 × 5 环境 × 5 seeds × 10M**，共 50 次训练。

环境为 Asterix、Breakout、Freeway、Seaquest、SpaceInvaders；seed 为 0–4。只运行 PPO 和 GA2E。

## 资源与执行

- GPU 0、1、2：每卡最多一个训练任务，网络推理、优势计算与反向传播在 GPU。
- MinAtar 环境：CPU 上的 Gymnasium 同步向量环境，512 个环境/任务，不占 GPU 做仿真。
- 任务完成后自动取队列中的下一项；失败写入状态和日志，继续其他任务，不无限重试。
- 独立后台 supervisor PID：3567889。SSH 断开不影响执行。
- 源码和配置已冻结在 campaign 的 source/ 和 configs/；后续修改主仓库不改变本批实验。

## 配置映射

参考 ~/Posterior-GAE/src/core/assets/configs/PPO_minatar.yml 和 GA2E_minatar.yml：

|参数|设置|
|并行环境 / rollout|512 / 128|
|minibatch / epochs|256 / 6|
|学习率|2.5e-4 线性衰减到 0|
|clip|0.1 线性衰减到 0|
|gamma / vf_coef / entropy|0.99 / 0.5 / 0.01|
|PPO lambda|0.2，按参考文件保留|
|GA2E|lambda_val=0.99，lambda_refresh=epoch，两级网格与 EMA|
|GA2E 梯度内存控制|2048 样本分块累积；与完整 fold 梯度数值对照通过|
|网络|自研 Conv(32,3) → ReLU → Conv(32,3) → ReLU → FC256 → ReLU，再接独立 actor/critic [64,64] Tanh MLP|
|动作/观测|全动作集合；bool 0/1 不除以255；SameStep autoreset；时间上限500000|
|评估|独立评估 seed，100 episodes，每10轮及末轮|

采用原有实验 YAML 协议；没有启动源仓库/SB3 训练器。参考源码中 CNN 以外的默认 [64,64] actor/critic MLP 也已核对并配置。两算法使用一致的环境、网络、预算与评估规则。

每轮 65,536 个有效环境步骤，10M 预算向完整 rollout 上取整为 **10,027,008** 步（153 轮），与常见 on-policy 固定 rollout 预算方式一致。总预算约 500M 环境步，不含评估。

## 查看进度

```bash
cd ~/OnPolicyRL
cat runs/minatar_5seed_10m/campaign_20261023/status.json
cat runs/minatar_5seed_10m/campaign_20261023/supervisor.pid
```

- 配置：configs/minatar_compare/ppo.yaml、ga2e.yaml
- 队列与快照：runs/minatar_5seed_10m/campaign_20261023/
- 每任务控制台：campaign_20261023/logs/
- PPO 结果：runs/minatar_5seed_10m/ppo/
- GA2E 结果：runs/minatar_5seed_10m/ga2e/
- 每个训练目录包含 metrics.jsonl、model_pools、DONE 或 error.log；队列末尾调用原有 report 汇总。

已完成 PPO/GA2E 各一轮满尺寸 Seaquest GPU 预检；正式首批三个任务已产生多轮训练指标。实验尚未跑完，当前不能给出算法性能比较结论。
