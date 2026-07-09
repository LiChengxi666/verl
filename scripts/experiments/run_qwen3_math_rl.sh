#!/usr/bin/env bash
set -xeuo pipefail

# Portable single-node entrypoint for the Qwen3 math RL experiments.
# This script intentionally does not depend on any platform-specific YAML.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MLP_ROLE_INDEX="${MLP_ROLE_INDEX:-0}"
export MLP_WORKER_NUM="${MLP_WORKER_NUM:-1}"
export MLP_WORKER_GPU="${MLP_WORKER_GPU:-8}"
export MLP_WORKER_0_HOST="${MLP_WORKER_0_HOST:-127.0.0.1}"

export MODEL_PATH="${MODEL_PATH:-${REPO_DIR}/models/Qwen3-4B-Base}"
export TRAIN_FILE="${TRAIN_FILE:-${REPO_DIR}/data/data_processed/math-17k.parquet}"
export VAL_FILE="${VAL_FILE:-${REPO_DIR}/data/data_processed/aime24.parquet}"

export PROJECT_NAME="${PROJECT_NAME:-verl_math_repro}"
export LOSS_MODE="${LOSS_MODE:-gspo}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-${LOSS_MODE}-qwen3-4b-base-math-8gpu}"

export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
export PPO_EPOCHS="${PPO_EPOCHS:-1}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export VAL_N="${VAL_N:-8}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-300}"
export TEST_FREQ="${TEST_FREQ:-10}"
export SAVE_FREQ="${SAVE_FREQ:-10}"
export LR="${LR:-1e-6}"

export PREFIX_CLIP_FIRST_LOW="${PREFIX_CLIP_FIRST_LOW:-0.2}"
export PREFIX_CLIP_FIRST_HIGH="${PREFIX_CLIP_FIRST_HIGH:-0.28}"
export PREFIX_CLIP_FINAL_LOW="${PREFIX_CLIP_FINAL_LOW:-3e-4}"
export PREFIX_CLIP_FINAL_HIGH="${PREFIX_CLIP_FINAL_HIGH:-4e-4}"
export PREFIX_CLIP_SUM_ALPHA="${PREFIX_CLIP_SUM_ALPHA:-2.0}"
export RIPO_DELTA_LOW="${RIPO_DELTA_LOW:-1e-5}"
export RIPO_DELTA_HIGH="${RIPO_DELTA_HIGH:-3e-5}"

export CKPT_DIR="${CKPT_DIR:-${REPO_DIR}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
export LOG_DIR="${LOG_DIR:-${REPO_DIR}/train_logs/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
export VAL_DUMP_DIR="${VAL_DUMP_DIR:-${REPO_DIR}/validation_generations/${PROJECT_NAME}/${EXPERIMENT_NAME}}"

mkdir -p "${CKPT_DIR}" "${LOG_DIR}" "${VAL_DUMP_DIR}"

export VERL_FILE_LOGGER_PATH="${VERL_FILE_LOGGER_PATH:-${LOG_DIR}/metrics.jsonl}"
export VERL_PREFIX_CLIP_DIAG_DIR="${VERL_PREFIX_CLIP_DIAG_DIR:-${LOG_DIR}/prefix_clip_diagnostics}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${LOG_DIR}/tensorboard}"
export TENSORBOARD_LOG_PATH="${TENSORBOARD_LOG_PATH:-${TENSORBOARD_DIR}}"
export WANDB_DIR="${WANDB_DIR:-${LOG_DIR}/wandb}"
mkdir -p "${TENSORBOARD_DIR}" "${WANDB_DIR}"

export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"

USER_LOGGER_BACKENDS="${LOGGER_BACKENDS:-}"
LOGGER_BACKENDS="${USER_LOGGER_BACKENDS:-[\"console\",\"file\",\"tensorboard\"]}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-}"
if [ -n "${WANDB_API_KEY:-}" ] || { [ -n "${WANDB_API_KEY_FILE}" ] && [ -r "${WANDB_API_KEY_FILE}" ]; }; then
  if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_API_KEY
    WANDB_API_KEY="$(tr -d '\r\n' < "${WANDB_API_KEY_FILE}")"
  fi
  export WANDB_MODE="${WANDB_MODE:-online}"
  export WANDB_RESUME="${WANDB_RESUME:-allow}"
  export WANDB_RUN_ID="${WANDB_RUN_ID:-${EXPERIMENT_NAME}}"
  if [ -z "${USER_LOGGER_BACKENDS}" ]; then
    LOGGER_BACKENDS="[\"console\",\"file\",\"tensorboard\",\"wandb\"]"
  fi
fi

which python
python - <<'PY'
import os
import sys
import tensordict
import verl
from transformers import AutoConfig, AutoTokenizer

model_path = os.environ["MODEL_PATH"]
config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
print("PYTHON:", sys.executable)
print("PREFIX:", sys.prefix)
print("TENSORDICT:", tensordict.__version__, tensordict.__file__)
print("VERL:", verl.__file__)
print("MODEL:", model_path, config.model_type, getattr(config, "num_hidden_layers", None), len(tok))
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
assert tensordict.__version__.split(".")[:2] >= ["0", "10"]
PY

export RAY_PORT="${RAY_PORT:-6379}"
export RAY_ADDRESS="${MLP_WORKER_0_HOST}:${RAY_PORT}"

python -m ray.scripts.scripts stop --force || true

if [ "${MLP_ROLE_INDEX}" = "0" ]; then
  python -m ray.scripts.scripts start --head \
    --node-ip-address="${MLP_WORKER_0_HOST}" \
    --port="${RAY_PORT}" \
    --num-gpus="${MLP_WORKER_GPU}" \
    --dashboard-host=0.0.0.0 \
    --disable-usage-stats

  sleep 30

  python - <<'PY'
import os
import ray

ray.init(address="auto")
resources = ray.cluster_resources()
print("RAY_RESOURCES:", resources)
expected = int(os.environ["MLP_WORKER_GPU"])
assert int(resources.get("GPU", 0)) == expected, (resources, expected)
PY

  python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=32 \
    data.truncation=error \
    data.shuffle=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.policy_loss.loss_mode="${LOSS_MODE}" \
    actor_rollout_ref.actor.policy_loss.prefix_clip_first_low="${PREFIX_CLIP_FIRST_LOW}" \
    actor_rollout_ref.actor.policy_loss.prefix_clip_first_high="${PREFIX_CLIP_FIRST_HIGH}" \
    actor_rollout_ref.actor.policy_loss.prefix_clip_final_low="${PREFIX_CLIP_FINAL_LOW}" \
    actor_rollout_ref.actor.policy_loss.prefix_clip_final_high="${PREFIX_CLIP_FINAL_HIGH}" \
    actor_rollout_ref.actor.policy_loss.prefix_clip_sum_alpha="${PREFIX_CLIP_SUM_ALPHA}" \
    actor_rollout_ref.actor.policy_loss.prefix_ripo_delta_low="${RIPO_DELTA_LOW}" \
    actor_rollout_ref.actor.policy_loss.prefix_ripo_delta_high="${RIPO_DELTA_HIGH}" \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.clip_ratio_low=0.0003 \
    actor_rollout_ref.actor.clip_ratio_high=0.0004 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr="${LR}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_epochs="${PPO_EPOCHS}" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=96000 \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.val_kwargs.n="${VAL_N}" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=65536 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=65536 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward.reward_kwargs.overlong_buffer_cfg.len=512 \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len="${MAX_RESPONSE_LENGTH}" \
    trainer.logger="${LOGGER_BACKENDS}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.nnodes="${MLP_WORKER_NUM}" \
    trainer.n_gpus_per_node="${MLP_WORKER_GPU}" \
    trainer.val_before_train=True \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.resume_mode=auto \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.max_actor_ckpt_to_keep=5 \
    trainer.max_critic_ckpt_to_keep=5 \
    trainer.validation_data_dir="${VAL_DUMP_DIR}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    '+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1="1"' \
    '+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ENABLE_V1_MULTIPROCESSING="0"'
else
  sleep 10
  python -m ray.scripts.scripts start \
    --address="${RAY_ADDRESS}" \
    --num-gpus="${MLP_WORKER_GPU}" \
    --block
fi
