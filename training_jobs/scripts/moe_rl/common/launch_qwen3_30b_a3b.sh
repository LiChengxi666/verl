#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
cd "${REPO_ROOT}"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_USE_V1=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

POLICY_LOSS_MODE="${POLICY_LOSS_MODE:-gspo}"
ROUTER_REPLAY_MODE="${ROUTER_REPLAY_MODE:-disabled}"
OFF_POLICY_K="${OFF_POLICY_K:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"

case "${POLICY_LOSS_MODE}" in
  gspo) ;;
  *)
    echo "POLICY_LOSS_MODE must currently be: gspo" >&2
    exit 2
    ;;
esac

case "${OFF_POLICY_K}" in
  2) DEFAULT_ACTOR_LR=2e-6 ;;
  4) DEFAULT_ACTOR_LR=1.5e-6 ;;
  8) DEFAULT_ACTOR_LR=1e-6 ;;
  *)
    echo "OFF_POLICY_K must be one of: 2, 4, 8" >&2
    exit 2
    ;;
esac

if (( TRAIN_BATCH_SIZE % OFF_POLICY_K != 0 )); then
  echo "TRAIN_BATCH_SIZE must be divisible by OFF_POLICY_K" >&2
  exit 2
fi
PPO_MINI_BATCH_SIZE=$((TRAIN_BATCH_SIZE / OFF_POLICY_K))
ACTOR_LR="${ACTOR_LR:-${DEFAULT_ACTOR_LR}}"

case "${ROUTER_REPLAY_MODE}" in
  disabled)
    ENABLE_ROLLOUT_ROUTING_REPLAY=False
    ROUTER_LABEL=none
    ;;
  R2)
    ENABLE_ROLLOUT_ROUTING_REPLAY=False
    ROUTER_LABEL=r2
    ;;
  R3)
    ENABLE_ROLLOUT_ROUTING_REPLAY=True
    ROUTER_LABEL=r3
    ;;
  *)
    echo "ROUTER_REPLAY_MODE must be one of: disabled, R2, R3" >&2
    exit 2
    ;;
esac

MODEL_PATH="${MODEL_PATH:-./models/Qwen3-30B-A3B-Base}"
TRAIN_FILE="${TRAIN_FILE:-./data/data_processed/math-17k.parquet}"
VAL_FILES="${VAL_FILES:-[\"./data/data_processed/moe_eval/minpro/amc23.parquet\",\"./data/data_processed/moe_eval/minpro/aime24.parquet\",\"./data/data_processed/moe_eval/minpro/aime25.parquet\"]}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
ROLLOUT_N="${ROLLOUT_N:-8}"
VAL_N="${VAL_N:-8}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-200}"
TEST_FREQ="${TEST_FREQ:-5}"
SAVE_FREQ="${SAVE_FREQ:-5}"

MLP_ROLE_INDEX="${MLP_ROLE_INDEX:-0}"
MLP_WORKER_NUM="${MLP_WORKER_NUM:-${NNODES:-4}}"
MLP_WORKER_GPU="${MLP_WORKER_GPU:-${NGPUS_PER_NODE:-4}}"
MLP_WORKER_0_HOST="${MLP_WORKER_0_HOST:-127.0.0.1}"
export MLP_ROLE_INDEX MLP_WORKER_NUM MLP_WORKER_GPU MLP_WORKER_0_HOST

ACTOR_TP="${ACTOR_TP:-1}"
ACTOR_PP="${ACTOR_PP:-4}"
ACTOR_EP="${ACTOR_EP:-4}"
ACTOR_ETP="${ACTOR_ETP:-1}"
ROLLOUT_TP="${ROLLOUT_TP:-4}"
ROLLOUT_DP="${ROLLOUT_DP:-1}"
ROLLOUT_EP="${ROLLOUT_EP:-4}"
WORLD_SIZE=$((MLP_WORKER_NUM * MLP_WORKER_GPU))

if (( WORLD_SIZE % (ACTOR_TP * ACTOR_PP) != 0 )); then
  echo "WORLD_SIZE must be divisible by ACTOR_TP * ACTOR_PP" >&2
  exit 2
fi
ACTOR_DP=$((WORLD_SIZE / (ACTOR_TP * ACTOR_PP)))
if (( ACTOR_DP % ACTOR_EP != 0 )); then
  echo "Megatron data-parallel size must be divisible by ACTOR_EP" >&2
  exit 2
fi
if (( ROLLOUT_EP != ROLLOUT_TP * ROLLOUT_DP )); then
  echo "ROLLOUT_EP must equal ROLLOUT_TP * ROLLOUT_DP" >&2
  exit 2
fi

PROJECT_NAME="${PROJECT_NAME:-verl_moe_router_replay}"
DEFAULT_EXPERIMENT_NAME="${POLICY_LOSS_MODE}_${ROUTER_LABEL}_off${OFF_POLICY_K}_qwen3_30b_a3b_b${TRAIN_BATCH_SIZE}n${ROLLOUT_N}_r${MAX_RESPONSE_LENGTH}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${DEFAULT_EXPERIMENT_NAME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-.}"
CKPT_ROOT="${CKPT_ROOT:-${OUTPUT_ROOT}/checkpoints}"
CKPT_DIR="${CKPT_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/train_logs/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
VAL_DUMP_DIR="${VAL_DUMP_DIR:-${OUTPUT_ROOT}/validation_generations/${PROJECT_NAME}/${EXPERIMENT_NAME}}"

ALGORITHM_ARGS=(
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=False
  algorithm.kl_ctrl.kl_coef=0.0
)

DATA_ARGS=(
  "data.train_files=${TRAIN_FILE}"
  "data.val_files=${VAL_FILES}"
  "data.train_batch_size=${TRAIN_BATCH_SIZE}"
  "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
  "data.max_response_length=${MAX_RESPONSE_LENGTH}"
  data.filter_overlong_prompts=True
  data.truncation=error
  data.return_raw_chat=True
)

MODEL_ARGS=(
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
)

POLICY_ARGS=(
  "actor_rollout_ref.actor.policy_loss.loss_mode=${POLICY_LOSS_MODE}"
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
  actor_rollout_ref.actor.clip_ratio_low=0.0003
  actor_rollout_ref.actor.clip_ratio_high=0.0004
  "actor_rollout_ref.actor.optim.lr=${ACTOR_LR}"
  actor_rollout_ref.actor.optim.lr_warmup_steps=10
  "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.ppo_epochs=1
  actor_rollout_ref.actor.use_dynamic_bsz=False
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=6144
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef=0.001
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff=0
)

BACKEND_ARGS=(
  actor_rollout_ref.actor.strategy=megatron
  "actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${ACTOR_TP}"
  "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${ACTOR_PP}"
  "actor_rollout_ref.actor.megatron.expert_model_parallel_size=${ACTOR_EP}"
  "actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ACTOR_ETP}"
  actor_rollout_ref.actor.megatron.param_offload=True
  actor_rollout_ref.actor.megatron.optimizer_offload=True
  actor_rollout_ref.actor.megatron.grad_offload=True
  actor_rollout_ref.actor.megatron.use_mbridge=True
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
  +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
  +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
)

ROUTER_ARGS=(
  "actor_rollout_ref.actor.megatron.router_replay.mode=${ROUTER_REPLAY_MODE}"
  "actor_rollout_ref.rollout.enable_rollout_routing_replay=${ENABLE_ROLLOUT_ROUTING_REPLAY}"
)

ROLLOUT_ARGS=(
  actor_rollout_ref.rollout.name=vllm
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}"
  "actor_rollout_ref.rollout.data_parallel_size=${ROLLOUT_DP}"
  "actor_rollout_ref.rollout.expert_parallel_size=${ROLLOUT_EP}"
  actor_rollout_ref.rollout.gpu_memory_utilization=0.50
  "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
  actor_rollout_ref.rollout.calculate_log_probs=True
  actor_rollout_ref.rollout.enable_chunked_prefill=True
  actor_rollout_ref.rollout.max_num_seqs=128
  actor_rollout_ref.rollout.max_num_batched_tokens=8192
  "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}"
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.val_kwargs.do_sample=True
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0
  actor_rollout_ref.rollout.val_kwargs.top_p=0.7
  "actor_rollout_ref.rollout.val_kwargs.n=${VAL_N}"
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=10240
)

REF_ARGS=(
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=10240
  "actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${ACTOR_TP}"
  "actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${ACTOR_PP}"
  "actor_rollout_ref.ref.megatron.expert_model_parallel_size=${ACTOR_EP}"
  "actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ACTOR_ETP}"
  actor_rollout_ref.ref.megatron.param_offload=True
  actor_rollout_ref.ref.megatron.use_mbridge=True
)

REWARD_ARGS=(
  reward.reward_manager.name=dapo
  +reward.reward_kwargs.overlong_buffer_cfg.enable=False
  +reward.reward_kwargs.overlong_buffer_cfg.len=512
  +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
  +reward.reward_kwargs.overlong_buffer_cfg.log=False
  "+reward.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH}"
)

TRAINER_ARGS=(
  trainer.rollout_buffer.enable=False
  "trainer.project_name=${PROJECT_NAME}"
  "trainer.experiment_name=${EXPERIMENT_NAME}"
  "trainer.nnodes=${MLP_WORKER_NUM}"
  "trainer.n_gpus_per_node=${MLP_WORKER_GPU}"
  trainer.val_before_train=True
  "trainer.test_freq=${TEST_FREQ}"
  "trainer.save_freq=${SAVE_FREQ}"
  trainer.resume_mode=auto
  "trainer.default_local_dir=${CKPT_DIR}"
  trainer.max_actor_ckpt_to_keep=null
  trainer.max_critic_ckpt_to_keep=null
  "trainer.validation_data_dir=${VAL_DUMP_DIR}"
  "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}"
)

SHARED_HYDRA_ARGS=(
  "${ALGORITHM_ARGS[@]}"
  "${DATA_ARGS[@]}"
  "${MODEL_ARGS[@]}"
  "${POLICY_ARGS[@]}"
  "${BACKEND_ARGS[@]}"
  "${ROLLOUT_ARGS[@]}"
  "${REF_ARGS[@]}"
  "${REWARD_ARGS[@]}"
)
ALL_HYDRA_ARGS=(
  "${SHARED_HYDRA_ARGS[@]}"
  "${ROUTER_ARGS[@]}"
  "${TRAINER_ARGS[@]}"
)

if [[ "${RECIPE_DRY_RUN:-0}" == "1" ]]; then
  export POLICY_LOSS_MODE ROUTER_REPLAY_MODE ENABLE_ROLLOUT_ROUTING_REPLAY
  export OFF_POLICY_K TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE ACTOR_LR
  export ROLLOUT_N MAX_RESPONSE_LENGTH TOTAL_TRAINING_STEPS EXPERIMENT_NAME
  printf -v SHARED_HYDRA_LINES '%s\n' "${SHARED_HYDRA_ARGS[@]}"
  printf -v ALL_HYDRA_LINES '%s\n' "${ALL_HYDRA_ARGS[@]}"
  export SHARED_HYDRA_LINES ALL_HYDRA_LINES
  python - <<'PY'
import json
import os

config = {
    "policy_loss_mode": os.environ["POLICY_LOSS_MODE"],
    "router_replay_mode": os.environ["ROUTER_REPLAY_MODE"],
    "rollout_routing_replay": os.environ["ENABLE_ROLLOUT_ROUTING_REPLAY"] == "True",
    "off_policy_k": int(os.environ["OFF_POLICY_K"]),
    "train_batch_size": int(os.environ["TRAIN_BATCH_SIZE"]),
    "mini_batch_size": int(os.environ["PPO_MINI_BATCH_SIZE"]),
    "actor_lr": os.environ["ACTOR_LR"],
    "ppo_epochs": 1,
    "rollout_n": int(os.environ["ROLLOUT_N"]),
    "max_response_length": int(os.environ["MAX_RESPONSE_LENGTH"]),
    "total_training_steps": int(os.environ["TOTAL_TRAINING_STEPS"]),
    "experiment_name": os.environ["EXPERIMENT_NAME"],
    "shared_hydra_args": os.environ["SHARED_HYDRA_LINES"].splitlines(),
    "all_hydra_args": os.environ["ALL_HYDRA_LINES"].splitlines(),
}
print("RECIPE_CONFIG_JSON=" + json.dumps(config, sort_keys=True, separators=(",", ":")))
PY
  exit 0
fi

mkdir -p "${CKPT_DIR}" "${LOG_DIR}" "${VAL_DUMP_DIR}"
checkpoint_probe="${CKPT_DIR}/.write_probe_$$"
printf 'checkpoint preflight\n' > "${checkpoint_probe}"
rm -f "${checkpoint_probe}"

export VERL_FILE_LOGGER_PATH="${LOG_DIR}/metrics.jsonl"
export TENSORBOARD_DIR="${LOG_DIR}/tensorboard"
export TENSORBOARD_LOG_PATH="${TENSORBOARD_DIR}"
export WANDB_DIR="${LOG_DIR}/wandb"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${EXPERIMENT_NAME}}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-./.secrets/wandb_api_key}"

if [[ -z "${TRAINER_LOGGER:-}" ]]; then
  if [[ -n "${WANDB_API_KEY:-}" || -r "${WANDB_API_KEY_FILE}" ]]; then
    TRAINER_LOGGER='["console","file","tensorboard","wandb"]'
  else
    TRAINER_LOGGER='["console","file","tensorboard"]'
  fi
fi
if [[ "${TRAINER_LOGGER}" == *wandb* && -z "${WANDB_API_KEY:-}" ]]; then
  if [[ ! -r "${WANDB_API_KEY_FILE}" ]]; then
    echo "W&B logging requested but WANDB_API_KEY_FILE is not readable: ${WANDB_API_KEY_FILE}" >&2
    exit 1
  fi
  set +x
  read -r WANDB_API_KEY < "${WANDB_API_KEY_FILE}"
  export WANDB_API_KEY
  set -x
fi
mkdir -p "${WANDB_DIR}"
TRAINER_ARGS+=("trainer.logger=${TRAINER_LOGGER}")

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "MODEL_PATH does not contain config.json: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "TRAIN_FILE does not exist: ${TRAIN_FILE}" >&2
  exit 1
fi
if [[ "${ROUTER_REPLAY_MODE}" == "R3" ]]; then
  python "${SCRIPT_DIR}/check_r3_vllm.py"
fi

export MODEL_PATH TRAIN_FILE VAL_FILES WORLD_SIZE ACTOR_TP ACTOR_PP ACTOR_EP
export ROLLOUT_TP ROLLOUT_DP ROLLOUT_EP
python - <<'PY'
import json
import os
from pathlib import Path

import pandas as pd
from transformers import AutoConfig

import verl
from verl.workers.config.engine import McoreEngineConfig
from verl.workers.config.rollout import RolloutConfig
from verl.workers.engine import EngineRegistry

model_path = os.environ["MODEL_PATH"]
config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
assert config.model_type == "qwen3_moe", config.model_type
assert getattr(config, "num_experts", None) == 128
assert "megatron" in EngineRegistry._engines["language_model"]

McoreEngineConfig(
    tensor_model_parallel_size=int(os.environ["ACTOR_TP"]),
    pipeline_model_parallel_size=int(os.environ["ACTOR_PP"]),
    expert_model_parallel_size=int(os.environ["ACTOR_EP"]),
)
RolloutConfig(
    name="vllm",
    tensor_model_parallel_size=int(os.environ["ROLLOUT_TP"]),
    data_parallel_size=int(os.environ["ROLLOUT_DP"]),
    expert_parallel_size=int(os.environ["ROLLOUT_EP"]),
    enable_rollout_routing_replay=os.environ["ENABLE_ROLLOUT_ROUTING_REPLAY"] == "True",
)

required_columns = {"data_source", "prompt", "reward_model"}
for path_value in json.loads(os.environ["VAL_FILES"]):
    path = Path(path_value)
    assert path.is_file(), path
    frame = pd.read_parquet(path)
    assert not frame.empty, path
    missing = required_columns.difference(frame.columns)
    assert not missing, f"{path}: missing {sorted(missing)}"

print("VERL:", verl.__file__)
print(
    "PREFLIGHT:",
    f"world_size={os.environ['WORLD_SIZE']}",
    f"actor_tp_pp_ep={os.environ['ACTOR_TP']}/{os.environ['ACTOR_PP']}/{os.environ['ACTOR_EP']}",
    f"rollout_tp_dp_ep={os.environ['ROLLOUT_TP']}/{os.environ['ROLLOUT_DP']}/{os.environ['ROLLOUT_EP']}",
    f"router_replay={os.environ['ROUTER_REPLAY_MODE']}",
)
PY

echo "RECIPE: policy=${POLICY_LOSS_MODE} router=${ROUTER_REPLAY_MODE} off=${OFF_POLICY_K}"
echo "BATCH: rollout=${TRAIN_BATCH_SIZE} update=${PPO_MINI_BATCH_SIZE} n=${ROLLOUT_N}"
echo "OPTIMIZER: lr=${ACTOR_LR} ppo_epochs=1"
echo "OUTPUT: ${CKPT_DIR}"

RAY_PORT="${RAY_PORT:-6379}"
RAY_ADDRESS="${MLP_WORKER_0_HOST}:${RAY_PORT}"
RUN_ID="${RUN_ID:-${EXPERIMENT_NAME}}"
RAY_DONE_FILE="${LOG_DIR}/.ray_done_${RUN_ID}"
RAY_DONE_TIMEOUT_SECONDS="${RAY_DONE_TIMEOUT_SECONDS:-604800}"
export RAY_PORT RAY_ADDRESS

python -m ray.scripts.scripts stop --force || true

if [[ "${MLP_ROLE_INDEX}" == "0" ]]; then
  rm -f "${RAY_DONE_FILE}" "${RAY_DONE_FILE}.tmp"
  python -m ray.scripts.scripts start --head \
    --node-ip-address="${MLP_WORKER_0_HOST}" \
    --port="${RAY_PORT}" \
    --num-gpus="${MLP_WORKER_GPU}" \
    --dashboard-host=0.0.0.0 \
    --disable-usage-stats

  python - <<'PY'
import os
import time

import ray

ray.init(address="auto")
expected = int(os.environ["MLP_WORKER_NUM"]) * int(os.environ["MLP_WORKER_GPU"])
for _ in range(120):
    resources = ray.cluster_resources()
    print("RAY_RESOURCES:", resources)
    if int(resources.get("GPU", 0)) >= expected:
        break
    time.sleep(5)
else:
    raise RuntimeError(f"Timed out waiting for {expected} GPUs")
PY

  completed_step=0
  if [[ -r "${CKPT_DIR}/latest_checkpointed_iteration.txt" ]]; then
    completed_step="$(tr -d '[:space:]' < "${CKPT_DIR}/latest_checkpointed_iteration.txt")"
  fi

  if (( completed_step >= TOTAL_TRAINING_STEPS )); then
    echo "Training already completed at step ${completed_step}; target is ${TOTAL_TRAINING_STEPS}."
    train_status=0
  else
    set +e
    python -m verl.trainer.main_ppo \
      "${ALGORITHM_ARGS[@]}" \
      "${DATA_ARGS[@]}" \
      "${MODEL_ARGS[@]}" \
      "${POLICY_ARGS[@]}" \
      "${BACKEND_ARGS[@]}" \
      "${ROUTER_ARGS[@]}" \
      "${ROLLOUT_ARGS[@]}" \
      "${REF_ARGS[@]}" \
      "${REWARD_ARGS[@]}" \
      "${TRAINER_ARGS[@]}" \
      model_engine=megatron \
      '+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1="1"' \
      '+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ENABLE_V1_MULTIPROCESSING="0"' \
      "$@"
    train_status=$?
    set -e
  fi

  printf '%s\n' "${train_status}" > "${RAY_DONE_FILE}.tmp"
  mv "${RAY_DONE_FILE}.tmp" "${RAY_DONE_FILE}"
  sleep 10
  python -m ray.scripts.scripts stop --force || true
  exit "${train_status}"
else
  ray_joined=0
  for _ in $(seq 1 120); do
    if python -m ray.scripts.scripts start \
      --address="${RAY_ADDRESS}" \
      --num-gpus="${MLP_WORKER_GPU}"; then
      ray_joined=1
      break
    fi
    python -m ray.scripts.scripts stop --force || true
    sleep 5
  done
  (( ray_joined == 1 )) || {
    echo "Timed out joining Ray head at ${RAY_ADDRESS}" >&2
    exit 1
  }

  ray_done_deadline=$((SECONDS + RAY_DONE_TIMEOUT_SECONDS))
  while [[ ! -f "${RAY_DONE_FILE}" ]]; do
    if (( SECONDS >= ray_done_deadline )); then
      echo "Timed out waiting for the head worker to finish" >&2
      python -m ray.scripts.scripts stop --force || true
      exit 1
    fi
    sleep 5
  done
  python -m ray.scripts.scripts stop --force || true
  exit 0
fi
