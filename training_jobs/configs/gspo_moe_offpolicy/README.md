# GSPO MoE PR2 Off-2 Baseline

本 recipe 使用 `Qwen3-30B-A3B-Base` 运行 GSPO，并采用 PR2 off-2 更新语义：每次由
behavior policy 为 64 个 prompt 各生成 8 条 rollout，再将同一批数据按 32 个 prompt
拆成两个 mini-batch，顺序执行两次 optimizer update。它不使用跨 step rollout buffer。
这里复用的是 PR2 论文定义的 off-2 数据复用协议，不包含 PR2 的 predictive router、
route cache/replay 或 predictor KL loss。

## 代码

```bash
git clone https://github.com/LiChengxi666/verl.git
cd verl
git fetch origin package-prefix-ripo-recipes
git checkout package-prefix-ripo-recipes
```

入口脚本：

```text
training_jobs/scripts/run_gspo_qwen3_30b_a3b_offpolicy_16gpu_b64n8_r8192.sh
```

## 环境与资源

当前验证镜像为：

```text
gensi-cn-beijing.cr.volces.com/sia-thu/verl:v0
```

镜像需预装 Python、PyTorch、Ray、vLLM 和 verl 依赖。任务内不要安装包，也不要激活
仓库中的 `.venv`。

默认资源拓扑：

```text
4 节点 x 4 张 NVIDIA A100 80G（共 16 张）
actor FSDP size = 16
rollout TP / DP / EP = 4 / 1 / 4
```

平台需要为每个节点注入 `MLP_ROLE_INDEX`、`MLP_WORKER_NUM`、`MLP_WORKER_GPU` 和
`MLP_WORKER_0_HOST`。脚本会等待 Ray 注册全部 GPU 后再开始训练。

## 模型和数据

默认路径均相对于仓库根目录：

```text
models/Qwen3-30B-A3B-Base/
data/data_processed/math-17k.parquet
data/data_processed/moe_eval/minpro/amc23.parquet
data/data_processed/moe_eval/minpro/aime24.parquet
data/data_processed/moe_eval/minpro/aime25.parquet
```

模型可准备为：

```bash
mkdir -p models
huggingface-cli download Qwen/Qwen3-30B-A3B-Base \
  --local-dir models/Qwen3-30B-A3B-Base \
  --local-dir-use-symlinks False
```

路径不同时用环境变量覆盖：

```bash
export MODEL_PATH=./assets/models/Qwen3-30B-A3B-Base
export TRAIN_FILE=./assets/data/math-17k.parquet
export VAL_FILES='["./assets/data/amc23.parquet","./assets/data/aime24.parquet","./assets/data/aime25.parquet"]'
```

processed MATH 数据使用 0/1 correctness reward。入口脚本会检查模型、训练集和三个验证集。

## 训练参数

```text
train batch size = 64 prompts
rollout.n = 8
ppo mini-batch size = 32 prompts
ppo epochs = 1
sequential optimizer updates per rollout batch = 2
max prompt length = 2048
max response length = 8192
loss mode = gspo
GSPO clip low / high = 3e-4 / 4e-4
learning rate = 2e-6
reference-policy KL loss coefficient = 1e-3
validation n = 8
total training steps = 200
test frequency = 5
save frequency = 5
```

`ppo_mini_batch_size=32` 会在 VeRL 内部乘以 `rollout.n=8`。因此 512 条 response
被切成两个各 256 条 response 的 mini-batch；每个 mini-batch 单独调用一次
`optimizer.step()`。不要把 `ppo_epochs` 改成 2，否则会把整批数据再重复一轮。

## 启动

在仓库根目录运行：

```bash
bash training_jobs/scripts/run_gspo_qwen3_30b_a3b_offpolicy_16gpu_b64n8_r8192.sh
```

指定共享输出目录时：

```bash
OUTPUT_ROOT=./outputs \
EXPERIMENT_NAME=gspo_moe_pr2_off2_16gpu_b64n8_r8192 \
bash training_jobs/scripts/run_gspo_qwen3_30b_a3b_offpolicy_16gpu_b64n8_r8192.sh
```

## 输出与恢复

```text
checkpoints/verl_moe/<experiment>/
train_logs/verl_moe/<experiment>/metrics.jsonl
train_logs/verl_moe/<experiment>/tensorboard/
validation_generations/verl_moe/<experiment>/
```

`trainer.resume_mode=auto` 已开启。checkpoint 保存 actor、optimizer、scheduler、dataloader
和 global step，不保存 rollout queue。若任务在一个训练 step 中途终止，恢复时从上一个完整
checkpoint 继续并重新生成该 step 的 rollout；已经完成的 step 不会重复。

## W&B

```bash
export WANDB_API_KEY=...
```

或：

```bash
mkdir -p .secrets
printf '%s\n' 'your-api-key' > .secrets/wandb_api_key
export WANDB_API_KEY_FILE=./.secrets/wandb_api_key
```

默认 project 为 `verl_moe`。未提供 key 时自动使用 `console,file,tensorboard`，不会因
W&B 登录失败退出。外场 recipe 不包含内部队列、挂载或个人绝对路径。
