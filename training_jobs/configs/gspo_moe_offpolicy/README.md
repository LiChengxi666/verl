# GSPO MoE Off-policy Baseline

本 recipe 用于在单机 8 张 A100 80G 上训练 `Qwen3-30B-A3B-Base`。训练算法为
GSPO，并使用固定延迟为 2 的 rollout buffer 构造 off-policy 数据。验证集为
AMC23、AIME24 和 AIME25，每题采样 8 个回答。

## 代码

```bash
git clone https://github.com/LiChengxi666/verl.git
cd verl
git fetch origin package-prefix-ripo-recipes
git checkout package-prefix-ripo-recipes
```

主要相关文件：

```text
verl/trainer/ppo/rollout_buffer.py
verl/trainer/ppo/ray_trainer.py
verl/trainer/config/ppo_trainer.yaml
training_jobs/scripts/run_gspo_qwen3_30b_a3b_offpolicy_8gpu_200.sh
```

## 环境与资源

当前验证过的镜像：

```text
gensi-cn-beijing.cr.volces.com/sia-thu/verl:v0
```

镜像已包含 Python、PyTorch、Ray、vLLM 和 verl 所需依赖。任务内不要安装包，也
不要激活仓库中的 `.venv`。

资源要求：

```text
1 节点 x 8 张 NVIDIA A100 80G
```

脚本默认使用 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`，并检查 Ray 集群是否
注册到 8 张 GPU。

## 模型和数据

默认路径相对于仓库根目录：

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

如果文件放在仓库内的其他相对路径，使用环境变量覆盖：

```bash
export MODEL_PATH=./assets/models/Qwen3-30B-A3B-Base
export TRAIN_FILE=./assets/data/math-17k.parquet
export VAL_FILES='["./assets/data/amc23.parquet","./assets/data/aime24.parquet","./assets/data/aime25.parquet"]'
```

processed MATH 数据使用 0/1 correctness reward。验证 parquet 需要包含
`data_source`、`prompt` 和 `reward_model` 字段；入口脚本会在启动 Ray 前检查。
模型目录必须包含 `config.json`；路径错误时脚本会在加载 transformers 前直接退出并
打印需要覆盖的环境变量。

## 启动

在仓库根目录运行：

```bash
bash training_jobs/scripts/run_gspo_qwen3_30b_a3b_offpolicy_8gpu_200.sh
```

如果训练输出需要写到共享存储：

```bash
OUTPUT_ROOT=./outputs \
EXPERIMENT_NAME=gspo_moe_offpolicy_n2_8gpu_200 \
bash training_jobs/scripts/run_gspo_qwen3_30b_a3b_offpolicy_8gpu_200.sh
```

外场平台只允许填写一条入口命令时，在仓库根目录使用：

```bash
OUTPUT_ROOT=./outputs bash training_jobs/scripts/run_gspo_qwen3_30b_a3b_offpolicy_8gpu_200.sh
```

## 默认训练设置

```text
model = Qwen3-30B-A3B-Base
train batch size = 512 prompts
rollout.n = 8
validation n = 8
max prompt length = 2048
max response length = 20480
loss mode = gspo
clip low/high = 0.002 / 0.002
learning rate = 1e-6
ppo mini-batch size = 32
ppo epochs = 1
rollout TP / DP / EP = 4 / 2 / 8
off-policy delay = 2
total training steps = 200
test frequency = 5
save frequency = 5
```

## 输出与恢复

默认输出：

```text
checkpoints/verl_moe/<experiment>/
train_logs/verl_moe/<experiment>/metrics.jsonl
train_logs/verl_moe/<experiment>/tensorboard/
validation_generations/verl_moe/<experiment>/
```

每个完整 checkpoint 包含 actor、dataloader 状态、两批延迟 rollout、metrics
快照和 `_SUCCESS` 标记。`trainer.resume_mode=auto` 已开启；中断后使用相同的
`EXPERIMENT_NAME`、`CKPT_ROOT` 和其他训练参数重启，会恢复 rollout queue，第一
个恢复后的有效更新继续保持 policy lag 2。默认保留全部 checkpoint。

## W&B

设置以下任意一种方式即可启用 W&B：

```bash
export WANDB_API_KEY=...
```

或：

```bash
mkdir -p .secrets
printf '%s\n' 'your-api-key' > .secrets/wandb_api_key
export WANDB_API_KEY_FILE=./.secrets/wandb_api_key
```

默认 project 为 `verl_moe`，run name 为 `gspo_moe_offpolicy_n2_8gpu_200`。
没有配置 key 时，脚本会使用 `console,file,tensorboard`，不会因 W&B 登录失败而退出。

外场 recipe 不包含内部平台的队列、挂载和个人路径配置。资源和存储由外场平台
单独配置，bash 入口及其所有文件参数均以仓库根目录为基准使用相对路径。
