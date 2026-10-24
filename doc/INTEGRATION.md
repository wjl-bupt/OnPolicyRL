# OnPolicyRL：原生算法架构与迁移说明

本次实现取代之前错误的 SB3 包装方案。运行时没有 stable_baselines3、sb3_contrib 或 Posterior-GAE 的导入；模型、分布头、采样、buffer、优化、日志和 checkpoint 全部由本仓库拥有。依赖 PyTorch 做张量与自动微分，Gymnasium 做环境接口。

## 一条训练链路

```text
实验 YAML / 网格 / seeds
  workflow.experiment → workflow.runner
  workflow.registry → 配置类型 + 原生模型工厂 + train 入口
    architecture：自研 MLP/CNN → Categorical / DiagGaussian → value / advantage / relative-value heads
    protocol：Policy / Field / Masks / RolloutSamples / MinibatchSpec
    memory.RolloutBuffer：[T,N] 声明式存储 + 有效样本切分 + 轨迹段
    onpolicy.runtime：collect → estimate → algorithm update → monitor
    common.Logger / workflow.Monitor → 评估、model_pools、metrics.jsonl
  workflow.report → 原有结果与报告
```

`src/oprl` 只保留安装命令入口。算法位于 `src/onpolicy/{ppo,ga2e,spo,rvl,dae,vmpo}/`，模型位于 `src/architecture/`。没有 vendor/posterior 层，也没有搬运的其他实验算法目录。`bgae/tgae/pgae` 已删除。

## 模型与状态归属

- `architecture/actor_critic.py`：已有自研 ActorCritic、CategoricalHead、DiagGaussianHead；MLP/CNN 分别在 layers.py/cnn.py。未使用任何 SB3 policy、extractor、distribution 或 optimizer。
- `architecture/auxiliary.py`：DAE 的 advantage head、RVL 的无偏置奇函数 relative head，以及算法需要的状态装配。
- `protocol/policy.py`：明确 `act/evaluate/predict_value/dist` 基础契约，以及 DAE `advantages`、RVL `relative_value` 能力。
- GA2E 的 lambda EMA 注册为模型 buffer；V-MPO dual 注册为模型参数及独立学习率 optimizer group。模型权重、辅助头、EMA、dual 和 optimizer 动量通过原生 checkpoint 一起保存。
- checkpoint 保存 Python/NumPy/Torch/CUDA 随机状态；环境内部状态和在线归一化 wrapper 状态尚未序列化，恢复训练不宣称与未中断训练逐位一致。

## 各算法迁移内容

|算法|原生实现|迁移范围|
|---|---|---|
|PPO|原有 clipped surrogate、value loss、entropy、KL 与调度|保留原生损失和配置，统一使用显式 successor 的采样循环|
|SPO|`A*r - abs(A)*(r-1)^2/(2*epsilon)`|只替换原生策略损失，不复制外部 PPO 训练器|
|GA2E|轨迹 train/validation 分组、梯度余弦对齐、两级 lambda 网格、EMA|支持 rollout/epoch 刷新及独立 value-reference lambda；搜索不写模型梯度、不更新参数|
|DAE|策略加权零均值优势头、轨迹折扣残差回归、冻结 A/V 的双向回归、lambda-DAE、策略更新|支持 sampled/full-action 更新；full-action 保留源实现“未采样 log-prob stop-gradient”语义；仅离散动作|
|RVL|反对称相对价值头、轨迹值重建、最优 offset、pairwise TD、相对值裁剪（无 scalar value 辅助损失）|实现一步目标；相对价值 encoder 保持梯度通路|
|V-MPO|top-half 优势加权似然、temperature dual、KL dual|categorical 精确 KL；Gaussian 分开约束均值和尺度；同步前馈、GAE target 版本|

参考来源为用户服务器 `~/Posterior-GAE/src/algo/{ga2e,spo,dae,rvl}` 的算法数学与行为；未拷贝其 SB3 生命周期、模型或 buffer。V-MPO 依据 https://arxiv.org/abs/1909.12238 的更新结构重新实现，不继承 PPO/SB3 外部类。

源仓库中的研究开关并非全部移植：GA2E 方差惩罚、搜索梯度分块；DAE Huber/其他实验变体；RVL n-step/lambda proxy 等不在此版本配置中，传入未知参数会明确失败。不能将源仓库 YAML 直接用于本框架。

### 有意修正的边界

1. rollout 显式保存 `next_obs/next_value`；truncated 保留 bootstrap、切断递推，terminated 切断 bootstrap；invalid dummy 不参与样本和轨迹。
2. SAME_STEP 必须使用 final observation 而非 reset observation；缺少该数据时明确报错。
3. RVL 已依据固定官方版本重新核对；此前对 terminal 目标及 scalar bootstrap 的处理被替换，详细差异和更正见 RVL_OFFICIAL_AUDIT.md。
4. RVL 预测分支保持 encoder 梯度；官方 target 分支的 no_grad 与预测分支要区分。
5. runner 在模型构造前设置 seed，恢复指纹包含算法与网络配置；失败时关闭环境。

## 使用现有实验协议

```bash
cd ~/OnPolicyRL
bash scripts/run_integrated.sh --list
bash scripts/run_integrated.sh configs/integration_ga2e.yaml
# 等价原生入口
PYTHONPATH=src .venv/bin/python -m oprl.cli configs/integration_vmpo.yaml
# 原有 runner 入口仍可用
.venv/bin/python src/workflow/runner.py configs/integration_dae.yaml
```

六份 `configs/integration_*.yaml` 现在全部使用原有 `baseline/space/envs/runs/resume` 协议；它们是 256-step CPU 冒烟配置，输出在 `runs/native_*`。`resume: true` 会跳过完成作业；正式实验可调整 name、预算、runs 与 space。旧 SB3 model.zip 不兼容新的原生 checkpoint，不做隐式转换。

## 验证边界

测试在 `~/OnPolicyRL/.venv` 运行，不依赖 Posterior-GAE 的虚拟环境。覆盖离散/连续训练、自研 CNN/MLP 反传、数学损失和梯度方向、DAE 零均值约束、RVL 反对称性及终止目标、GA2E 梯度和分组、V-MPO dual、时间截断、保存恢复，以及全部算法通过原 workflow 完成评估与 checkpoint。

未运行长预算、多 seed 基准以确认性能；未验证 Atari/MuJoCo ROM/物理任务的长期收敛。原模型的 Dict/text observation 和多维离散动作能力仍未实现，不能据 CNN 单元测试声称这些环境全部支持。
