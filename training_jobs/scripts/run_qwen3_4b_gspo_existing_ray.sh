#!/usr/bin/env bash
set -xeuo pipefail

# Run the aligned Qwen3-4B GSPO training recipe on an already-running Ray cluster.
# This recipe consumes one colocated 8-GPU node and never starts or stops Ray.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_DIR}"

export RAY_ADDRESS="${RAY_ADDRESS:-auto}"
export MODEL_PATH="${MODEL_PATH:-${REPO_DIR}/models/Qwen3-4B-Base}"
export TRAIN_FILE="${TRAIN_FILE:-${REPO_DIR}/data/data_processed/math-17k.parquet}"
export VAL_FILE="${VAL_FILE:-${REPO_DIR}/data/data_processed/aime24.parquet}"

export PROJECT_NAME="${PROJECT_NAME:-verl_math_repro}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-gspo-4b-8gpu}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}}"
export CKPT_DIR="${CKPT_DIR:-${OUTPUT_ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
export LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/train_logs/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
export VAL_DUMP_DIR="${VAL_DUMP_DIR:-${OUTPUT_ROOT}/validation_generations/${PROJECT_NAME}/${EXPERIMENT_NAME}}"

export WANDB_ENTITY="${WANDB_ENTITY:-licx199}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${EXPERIMENT_NAME}}"
export WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-${HOME}/.secrets/wandb_api_key}"

if [ -z "${WANDB_API_KEY:-}" ]; then
  if [ ! -r "${WANDB_API_KEY_FILE}" ]; then
    echo "Set WANDB_API_KEY or provide a readable WANDB_API_KEY_FILE: ${WANDB_API_KEY_FILE}" >&2
    exit 1
  fi
  set +x
  export WANDB_API_KEY
  WANDB_API_KEY="$(tr -d '\r\n' < "${WANDB_API_KEY_FILE}")"
  set -x
fi

mkdir -p "${CKPT_DIR}" "${LOG_DIR}" "${VAL_DUMP_DIR}"

export VERL_FILE_LOGGER_PATH="${LOG_DIR}/metrics.jsonl"
export TENSORBOARD_DIR="${LOG_DIR}/tensorboard"
export TENSORBOARD_LOG_PATH="${TENSORBOARD_DIR}"
export WANDB_DIR="${LOG_DIR}/wandb"
mkdir -p "${TENSORBOARD_DIR}" "${WANDB_DIR}"

export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"

for path in "${MODEL_PATH}" "${TRAIN_FILE}" "${VAL_FILE}"; do
  if [ ! -e "${path}" ]; then
    echo "Required path does not exist: ${path}" >&2
    exit 1
  fi
done

which python
python - <<'PY'
import os
import sys

import ray
import tensordict
import verl
from transformers import AutoConfig, AutoTokenizer

model_path = os.environ["MODEL_PATH"]
config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)

print("PYTHON:", sys.executable)
print("PREFIX:", sys.prefix)
print("TENSORDICT:", tensordict.__version__, tensordict.__file__)
print("VERL:", verl.__file__)
print("MODEL:", model_path, config.model_type, getattr(config, "num_hidden_layers", None), len(tokenizer))

assert tensordict.__version__.split(".")[:2] >= ["0", "10"]

ray.init(address=os.environ["RAY_ADDRESS"])
cluster = ray.cluster_resources()
available = ray.available_resources()
alive_nodes = [node for node in ray.nodes() if node["Alive"]]
print("RAY_CLUSTER_RESOURCES:", cluster)
print("RAY_AVAILABLE_RESOURCES:", available)
print("RAY_ALIVE_NODES:", len(alive_nodes))
assert int(cluster.get("GPU", 0)) >= 8, f"Ray cluster has fewer than 8 GPUs: {cluster}"
ray.shutdown()
PY

python - \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size=128 \
  data.max_prompt_length=1024 \
  data.max_response_length=16384 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.clip_ratio_low=0.0003 \
  actor_rollout_ref.actor.clip_ratio_high=0.0004 \
  actor_rollout_ref.actor.clip_ratio_c=10.0 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.max_num_batched_tokens=96000 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.val_kwargs.n=8 \
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
  +reward.reward_kwargs.max_resp_len=16384 \
  trainer.logger='["console","file","tensorboard","wandb"]' \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=8 \
  trainer.val_before_train=True \
  trainer.test_freq=5 \
  trainer.save_freq=5 \
  trainer.resume_mode=auto \
  trainer.default_local_dir="${CKPT_DIR}" \
  trainer.max_actor_ckpt_to_keep=5 \
  trainer.max_critic_ckpt_to_keep=5 \
  trainer.validation_data_dir="${VAL_DUMP_DIR}" \
  trainer.total_training_steps=300 <<'PY'
import os

import ray

runtime_env_names = (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "VLLM_USE_V1",
    "VLLM_ENABLE_V1_MULTIPROCESSING",
    "VERL_FILE_LOGGER_PATH",
    "TENSORBOARD_LOG_PATH",
    "WANDB_API_KEY",
    "WANDB_ENTITY",
    "WANDB_MODE",
    "WANDB_RESUME",
    "WANDB_RUN_ID",
    "WANDB_DIR",
)
runtime_env = {"env_vars": {name: os.environ[name] for name in runtime_env_names}}
ray.init(address=os.environ["RAY_ADDRESS"], runtime_env=runtime_env)

from verl.trainer.main_ppo import main

main()
PY
