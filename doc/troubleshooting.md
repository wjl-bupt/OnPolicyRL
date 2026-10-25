# 常见问题排查

| 现象 | 优先检查 |
|---|---|
| 改了 YAML 但参数没变 | 脚本 ALGOS/RUNS/TOTAL_STEPS/DEVICE 覆盖；算法/环境专属段；最后生效的 sweep |
| 预期一个任务却有多个 | 算法数、env 数、runs、sweep 乘积；已有 classic_control 文件含 epsilon 扫描 |
| unknown parameters | 拼写及所属段；算法专属参数是否误放公共 train；配置类是否真的支持该字段 |
| Transformer 配置冲突 | frame_stack 与 context_length；宽度能否整除 attention_heads；共享分支尺寸 |
| Diffusion 无法配 PPO/GA2E | 当前必须 algo=dppo，连续扁平 Box、action_mode=rescale、ent_coef=0 |
| CUDA 不可用 | 实际 Python 的 PyTorch 构建、驱动和 DEVICE；不要用 jax_platform 代替网络 device |
| JAX GPU backend 不可用 | 当前服务器 JAX CPU 安装，不能仅设平台名就获得 GPU plugin |
| Atari 文件缺失 | 当前后端所需 ROM/资源；JAX Atari 的真实验证尚未完成 |
| 仿真 provider 不匹配 | CONFIG 后端、完整 env ID 与 implementation 必须一致 |
| 显存不足 | num_envs×rollout_len、minibatch、图像特征、Transformer 窗口、Diffusion 步数；GA2E 可调 grad_chunk_size |
| 一启动就 skipped | resume=true 且目录有 DONE；使用新 run_dir，勿随意删结果标记 |
| checkpoint 不兼容 | 配置/硬件/代码指纹变化；当前会从头训练并清理旧模型池，先保留结果或使用新目录 |
| 指定 commit 找不到 | ID 是否为 7～40 位十六进制；remote 是否配置、可访问、包含该对象 |
| Source cache was modified | 不要编辑 .oprl/code/<sha>；使用新的 --code-cache 路径 |
| --prepare 成功却训练失败 | prepare 不做环境或目标源码训练校验；旧 commit 可能不支持新配置/依赖 |
| eval 很慢或次数不足 | episode 上限、策略是否长期不终止、eval_num_envs、eval/n_episodes 与安全步数上限 |
| DONE 但没有评估或模型 | 检查 eval_failed/checkpoint_failed 事件；Monitor 会记录异常但允许训练继续 |
| NaN 或梯度异常 | 原始观测/reward、归一化统计、学习率、概率比、对应算法损失；检查第一处异常而非只看最后曲线 |
| 训练结束没有报告 | report 失败不回滚训练；检查可视化依赖与原始 metrics.jsonl |

定位顺序：确认解释器 → --list → --plan → 检查 config.json/environment.json → run.log/error.log/异常事件。
--plan 不运行环境；资源和数值错误无法全部在该阶段发现。

中断实验前确认本次 PID 与调度器，避免误停其他项目。Ctrl-C 不保证立即保存 checkpoint。
历史结果比较时同时核对 sticky action、GAE lambda、episode 上限、评估次数与代码版本。
