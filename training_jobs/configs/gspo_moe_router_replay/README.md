# GSPO MoE Router Replay 对照实验

本目录说明如何在 `Qwen3-30B-A3B-Base` 上运行三组参数严格对齐的
GSPO off-policy 实验：

1. GSPO，不启用 Router Replay；
2. GSPO + R2；
3. GSPO + R3。

三个入口共享同一个 Megatron/vLLM launcher。除 Router Replay 模式外，模型、数据、
batch、学习率、GSPO clip、KL、验证和资源参数完全相同。

## 代码

```bash
git clone https://github.com/LiChengxi666/verl.git
cd verl
git fetch origin package-prefix-ripo-recipes
git checkout package-prefix-ripo-recipes
```

入口脚本为：

```text
training_jobs/scripts/moe_rl/run_gspo_off2.sh
training_jobs/scripts/moe_rl/run_gspo_r2_off2.sh
training_jobs/scripts/moe_rl/run_gspo_r3_off2.sh
```

公共 launcher 为：

```text
training_jobs/scripts/moe_rl/common/launch_qwen3_30b_a3b.sh
```

不要直接复制公共 launcher 制作新的实验。策略算法和 Router Replay 是两个独立的实验
维度，后续应通过薄入口选择对应模式。

## 环境要求

运行环境需要预装：

- Python、PyTorch、Ray 和 verl 依赖；
- Megatron-Core 和 MBridge；
- 支持 Qwen3 MoE、expert parallel 和 Router Replay 的 vLLM；
- R3 所需的 rollout routed-expert 返回功能。

R2 只在训练引擎计算 old log-prob 时记录专家索引。R3 必须由 rollout 引擎返回生成时
的专家索引，再在 old-log-prob 和 actor update 中回放。因此普通上游 vLLM 即使能够
生成，也不一定具备运行 R3 所需的 Router Replay 接口。R3 入口会在启动 Ray 和占用
GPU 前检查 `enable_return_routed_experts` 能力，缺失时直接退出。

## 模型与数据

默认使用仓库内相对路径：

```text
models/Qwen3-30B-A3B-Base/
data/data_processed/math-17k.parquet
data/data_processed/moe_eval/minpro/amc23.parquet
data/data_processed/moe_eval/minpro/aime24.parquet
data/data_processed/moe_eval/minpro/aime25.parquet
```

训练集为 processed MATH-17k，使用 0/1 correctness reward。验证集为 AMC23、AIME24
和 AIME25，每题采样 8 个回答并记录 `avg@8`。

路径不同时使用环境变量覆盖：

```bash
export MODEL_PATH=./assets/models/Qwen3-30B-A3B-Base
export TRAIN_FILE=./assets/data/math-17k.parquet
export VAL_FILES='["./assets/data/amc23.parquet","./assets/data/aime24.parquet","./assets/data/aime25.parquet"]'
```

## 默认训练设置

```text
资源                 4 节点 x 4 A100，共 16 GPU
Megatron TP / PP / EP 1 / 4 / 4
rollout TP / DP / EP  4 / 1 / 4
train batch           64 prompts
rollout.n             8
OFF_POLICY_K          2
PPO mini-batch        32 prompts
PPO epochs            1
sequential updates    2
learning rate         2e-6
GSPO clip low / high  3e-4 / 4e-4
KL coefficient        1e-3
prompt / response     2048 / 8192
训练步数               200
验证和保存频率          每 5 step
```

默认明确为：

```bash
OFF_POLICY_K=2
```

同一批 64 个 prompt 的 rollout 按 32 个 prompt 分成两个 mini-batch，每个
mini-batch 执行一次 optimizer update。`ppo_epochs=1`，且不使用跨 step rollout
buffer。

## 启动三组 off-2 实验

平台需要在每个节点设置：

```text
MLP_ROLE_INDEX
MLP_WORKER_NUM
MLP_WORKER_GPU
MLP_WORKER_0_HOST
```

所有节点必须能访问同一个仓库目录和输出目录；checkpoint、日志以及多机结束状态通过
这个共享文件系统协调。不要同时使用同一个实验名启动两个独立任务。

无 Router Replay：

```bash
bash training_jobs/scripts/moe_rl/run_gspo_off2.sh
```

R2：

```bash
bash training_jobs/scripts/moe_rl/run_gspo_r2_off2.sh
```

R3：

```bash
bash training_jobs/scripts/moe_rl/run_gspo_r3_off2.sh
```

## off-4 与 off-8

公共 launcher 支持 `OFF_POLICY_K=2/4/8`。固定 64-prompt rollout batch 时：

| 模式 | PPO mini-batch | 顺序更新次数 | 默认学习率 |
| --- | ---: | ---: | ---: |
| off-2 | 32 | 2 | `2e-6` |
| off-4 | 16 | 4 | `1.5e-6` |
| off-8 | 8 | 8 | `1e-6` |

例如运行 GSPO + R2 off-4：

```bash
OFF_POLICY_K=4 \
bash training_jobs/scripts/moe_rl/run_gspo_r2_off2.sh
```

运行 GSPO + R3 off-8：

```bash
OFF_POLICY_K=8 \
bash training_jobs/scripts/moe_rl/run_gspo_r3_off2.sh
```

`ACTOR_LR` 可显式覆盖表中的论文默认值。实验名、W&B run、checkpoint 和日志目录会
自动包含实际的 `off2/off4/off8`，不会互相覆盖。

## W&B

推荐把 API key 放在仓库之外或 `.gitignore` 覆盖的 secret 文件中：

```bash
mkdir -p .secrets
printf '%s\n' 'your-api-key' > .secrets/wandb_api_key
export WANDB_API_KEY_FILE=./.secrets/wandb_api_key
```

默认 W&B project 为 `verl_moe_router_replay`。存在可读 secret 文件时启用
`console,file,tensorboard,wandb`；否则使用 `console,file,tensorboard`。

## 输出与恢复

默认输出为：

```text
checkpoints/verl_moe_router_replay/<experiment>/
train_logs/verl_moe_router_replay/<experiment>/metrics.jsonl
train_logs/verl_moe_router_replay/<experiment>/tensorboard/
validation_generations/verl_moe_router_replay/<experiment>/
```

launcher 设置 `trainer.resume_mode=auto`。重新使用相同实验名和参数启动时，从最新完整
checkpoint 恢复；如果中断发生在某个 step 内，该 step 的 rollout 会重新生成。

## 后续扩展

新增 RL 算法时，在公共 launcher 中增加经过校验的 policy 参数分支，再增加薄入口。
实现 PR2 后，在 Router Replay 参数分支中增加 `PR2`，不要把 predictor 参数散落复制
到现有三个入口。这样可以继续形成 policy、Router Replay 和 off-policy 强度的正交
实验矩阵。
