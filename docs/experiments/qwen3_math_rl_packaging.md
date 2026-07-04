# Qwen3 Math RL Experiment Package

This note describes the minimal information needed to run the GSPO baseline and
the prefix-clip variants on another training platform. It is intentionally
platform-neutral: use the command below as the task entrypoint instead of
copying Volcengine-specific YAML files.

## Code

Use this repository/fork and the branch or commit that contains this document.

```text
Repository: git@github.com:LiChengxi666/verl.git
Branch: codex/package-prefix-ripo-recipes
```

The relevant implementation changes are in:

```text
verl/trainer/ppo/core_algos.py
verl/workers/config/actor.py
verl/trainer/config/actor/actor.yaml
verl/trainer/ppo/ray_trainer.py
verl/trainer/main_ppo_sync.py
scripts/experiments/run_qwen3_math_rl.sh
```

## Environment

Use the same container image used by the current submitted jobs:

```text
gensi-cn-beijing.cr.volces.com/sia-thu/verl:v0
```

The image already provides the working Python/conda environment. Do not install
packages or activate `.venv` inside the training task.

## Resources

Recommended resource shape:

```text
1 node x 8 NVIDIA A100 80G GPUs
```

The script starts a local Ray cluster and uses all visible GPUs. If the platform
sets `CUDA_VISIBLE_DEVICES`, the script respects it. If not, it defaults to
`0,1,2,3,4,5,6,7`.

## Data and Model Paths

Default paths expected by the script:

```text
Model: /GenSIvePFS/users/cxli/models/Qwen3-4B-Base
Train: /GenSIvePFS/users/cxli/verl/data/data_processed/math-17k.parquet
Valid: /GenSIvePFS/users/cxli/verl/data/data_processed/aime24.parquet
```

Override these with environment variables if the target platform mounts them
elsewhere:

```bash
MODEL_PATH=/path/to/Qwen3-4B-Base
TRAIN_FILE=/path/to/math-17k.parquet
VAL_FILE=/path/to/aime24.parquet
```

## One-Command Entrypoints

GSPO baseline:

```bash
cd /GenSIvePFS/users/cxli/verl && \
LOSS_MODE=gspo \
EXPERIMENT_NAME=gspo-qwen3-4b-base-processed-8gpu-300 \
bash scripts/experiments/run_qwen3_math_rl.sh
```

Average-prefix dynamic clip:

```bash
cd /GenSIvePFS/users/cxli/verl && \
LOSS_MODE=prefix_dynamic_clip \
EXPERIMENT_NAME=prefix-dynamic-qwen3-4b-base-8gpu-300 \
bash scripts/experiments/run_qwen3_math_rl.sh
```

Sqrt-schedule prefix clip:

```bash
cd /GenSIvePFS/users/cxli/verl && \
LOSS_MODE=prefix_sqrt_dynamic_clip \
EXPERIMENT_NAME=prefix-sqrt-qwen3-4b-base-8gpu-300 \
bash scripts/experiments/run_qwen3_math_rl.sh
```

RIPO-like prefix clip, paper-default delta:

```bash
cd /GenSIvePFS/users/cxli/verl && \
LOSS_MODE=prefix_ripo_clip \
RIPO_DELTA_LOW=0.05 \
RIPO_DELTA_HIGH=0.05 \
EXPERIMENT_NAME=prefix-ripo-d005-d005-qwen3-4b-base-8gpu-300 \
bash scripts/experiments/run_qwen3_math_rl.sh
```

RIPO-like delta sweep can be launched by changing only
`RIPO_DELTA_LOW/HIGH`, for example:

```text
0.02 / 0.02
0.05 / 0.04
0.05 / 0.05
0.08 / 0.08
0.05 / 0.02
0.08 / 0.02
0.08 / 0.04
```

## Default Training Parameters

Unless overridden by environment variables, the script uses:

```text
train_batch_size = 128 prompts
rollout.n = 8
validation n = 8
ppo_mini_batch_size = 16
ppo_epochs = 1
max_prompt_length = 1024
max_response_length = 16384
total_training_steps = 300
test_freq = 10
save_freq = 10
learning_rate = 1e-6
loss_agg_mode = token-mean
rollout backend = vLLM
tensor_model_parallel_size = 1
enable_chunked_prefill = true
max_num_batched_tokens = 96000
```

Prefix clip defaults:

```text
prefix_clip_first_low = 0.2
prefix_clip_first_high = 0.28
prefix_clip_final_low = 3e-4
prefix_clip_final_high = 4e-4
prefix_clip_sum_alpha = 2.0
prefix_ripo_delta_low = 0.05
prefix_ripo_delta_high = 0.05
```

## Outputs and Restart

The script writes outputs under the repository by default:

```text
checkpoints/<project>/<experiment>
train_logs/<project>/<experiment>/metrics.jsonl
train_logs/<project>/<experiment>/tensorboard
validation_generations/<project>/<experiment>
```

Checkpoint resume is enabled with `trainer.resume_mode=auto`. At every
checkpoint step, the current `metrics.jsonl` is also copied into the checkpoint
folder so metric history survives task preemption.

## W&B

W&B is optional. To enable it, set either:

```bash
WANDB_API_KEY=...
```

or:

```bash
WANDB_API_KEY_FILE=/path/to/wandb_api_key
```

Do not commit API keys to the repository or task config.

## Platform Form Fields

For platforms with a form like Seed:

```text
Task type: formal training task
Scenario: other / RL training
RL server mode: off
Image: gensi-cn-beijing.cr.volces.com/sia-thu/verl:v0
Repository mount path: /GenSIvePFS/users/cxli/verl
Entrypoint: one of the commands above
GPU resources: 1 node, 8 x A100 80G
Persistent storage: mount data, model, checkpoints, train_logs, validation_generations
```
