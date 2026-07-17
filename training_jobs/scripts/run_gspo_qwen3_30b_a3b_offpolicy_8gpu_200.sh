#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_USE_V1=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MLP_ROLE_INDEX="${MLP_ROLE_INDEX:-0}"
export MLP_WORKER_NUM="${MLP_WORKER_NUM:-1}"
export MLP_WORKER_GPU="${MLP_WORKER_GPU:-8}"
export MLP_WORKER_0_HOST="${MLP_WORKER_0_HOST:-127.0.0.1}"

export MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/models/Qwen3-30B-A3B-Base}"
export TRAIN_FILE="${TRAIN_FILE:-${REPO_ROOT}/data/data_processed/math-17k.parquet}"
export VAL_FILES="${VAL_FILES:-[\"${REPO_ROOT}/data/data_processed/moe_eval/minpro/amc23.parquet\",\"${REPO_ROOT}/data/data_processed/moe_eval/minpro/aime24.parquet\",\"${REPO_ROOT}/data/data_processed/moe_eval/minpro/aime25.parquet\"]}"
export ROLLOUT_TP="${ROLLOUT_TP:-4}"
export ROLLOUT_DP="${ROLLOUT_DP:-2}"
export ROLLOUT_EP="${ROLLOUT_EP:-8}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-20480}"
export OVERLONG_BUFFER_ENABLE="${OVERLONG_BUFFER_ENABLE:-False}"
export OVERLONG_BUFFER_LEN="${OVERLONG_BUFFER_LEN:-512}"
export VAL_N="${VAL_N:-8}"
export VAL_ONLY="${VAL_ONLY:-False}"
export MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-200}"

export PROJECT_NAME="${PROJECT_NAME:-verl_moe}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-gspo_moe_offpolicy_n2_8gpu_200}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}}"
export CKPT_ROOT="${CKPT_ROOT:-${OUTPUT_ROOT}/checkpoints}"
export CKPT_DIR=${CKPT_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}
export LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/train_logs/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
export VAL_DUMP_DIR="${VAL_DUMP_DIR:-${OUTPUT_ROOT}/validation_generations/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
export RUN_ID="${RUN_ID:-${EXPERIMENT_NAME}}"
export RAY_READY_FILE="${LOG_DIR}/.ray_ready_${RUN_ID}"
export RAY_DONE_FILE="${LOG_DIR}/.ray_done_${RUN_ID}"

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
export WANDB_ENTITY="${WANDB_ENTITY:-licx199}"
export WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-${REPO_ROOT}/.secrets/wandb_api_key}"

if [ -z "${TRAINER_LOGGER:-}" ]; then
  if [ -n "${WANDB_API_KEY:-}" ] || [ -r "${WANDB_API_KEY_FILE}" ]; then
    export TRAINER_LOGGER='["console","file","tensorboard","wandb"]'
  else
    export TRAINER_LOGGER='["console","file","tensorboard"]'
  fi
fi

if [[ "${TRAINER_LOGGER}" == *wandb* ]] && [ -z "${WANDB_API_KEY:-}" ]; then
  if [ ! -r "${WANDB_API_KEY_FILE}" ]; then
    echo "WANDB_API_KEY is not set and ${WANDB_API_KEY_FILE} is not readable." >&2
    exit 1
  fi
  set +x
  export WANDB_API_KEY
  WANDB_API_KEY="$(tr -d '\r\n' < "${WANDB_API_KEY_FILE}")"
  set -x
fi
mkdir -p "${WANDB_DIR}"

which python
python - <<'PY'
import json
import os
import sys

import pandas as pd
import tensordict
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoTokenizer

import verl
from verl.experimental.reward_loop.reward_manager.dapo import DAPORewardManager
from verl.utils.reward_score import default_compute_score
from verl.workers.config.rollout import RolloutConfig
from verl.workers.engine import EngineRegistry

model_path = os.environ["MODEL_PATH"]
config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
print("PYTHON:", sys.executable)
print("TENSORDICT:", tensordict.__version__, tensordict.__file__)
print("VERL:", verl.__file__)
print(
    "MODEL:",
    config.model_type,
    "experts=", getattr(config, "num_experts", None),
    "experts_per_tok=", getattr(config, "num_experts_per_tok", None),
    "vocab=", len(tokenizer),
)
assert config.model_type == "qwen3_moe"
assert getattr(config, "num_experts", None) == 128
assert "fsdp" in EngineRegistry._engines["language_model"]
assert int(os.environ["MLP_WORKER_NUM"]) == 1
assert int(os.environ["MLP_WORKER_GPU"]) == 8
assert int(os.environ["ROLLOUT_EP"]) == int(os.environ["ROLLOUT_TP"]) * int(os.environ["ROLLOUT_DP"])
assert int(os.environ["MAX_RESPONSE_LENGTH"]) >= int(os.environ["OVERLONG_BUFFER_LEN"])

rollout_config = RolloutConfig(
    name="vllm",
    tensor_model_parallel_size=int(os.environ["ROLLOUT_TP"]),
    data_parallel_size=int(os.environ["ROLLOUT_DP"]),
    expert_parallel_size=int(os.environ["ROLLOUT_EP"]),
)
reward_config = OmegaConf.create(
    {
        "reward": {
            "reward_kwargs": {
                "overlong_buffer_cfg": {
                    "enable": os.environ["OVERLONG_BUFFER_ENABLE"].lower() == "true",
                    "len": int(os.environ["OVERLONG_BUFFER_LEN"]),
                    "penalty_factor": 1.0,
                    "log": False,
                },
                "max_resp_len": int(os.environ["MAX_RESPONSE_LENGTH"]),
            }
        }
    }
)
DAPORewardManager(config=reward_config, tokenizer=tokenizer, compute_score=None)

required_columns = {"data_source", "prompt", "reward_model"}
for val_path in json.loads(os.environ["VAL_FILES"]):
    frame = pd.read_parquet(val_path)
    missing = required_columns.difference(frame.columns)
    assert not missing, f"{val_path} is missing columns: {sorted(missing)}"
    assert not frame.empty, f"{val_path} is empty"
    sources = frame["data_source"].unique().tolist()
    assert len(sources) == 1, f"{val_path} has multiple data sources: {sources}"
    sample = frame.iloc[0]
    reward = default_compute_score(
        sources[0],
        "",
        sample["reward_model"]["ground_truth"],
    )
    assert isinstance(reward, dict) and "score" in reward and "acc" in reward
    print("VALIDATION_CHECK:", val_path, f"rows={len(frame)}", f"source={sources[0]}")

print(
    "CONFIG_CHECK:",
    f"rollout_tp={rollout_config.tensor_model_parallel_size}",
    f"rollout_ep={rollout_config.expert_parallel_size}",
    f"max_response={os.environ['MAX_RESPONSE_LENGTH']}",
    f"val_n={os.environ['VAL_N']}",
    f"val_only={os.environ['VAL_ONLY']}",
    f"overlong_buffer={os.environ['OVERLONG_BUFFER_LEN']}",
)
PY

export RAY_PORT="${RAY_PORT:-6379}"
export RAY_ADDRESS="${MLP_WORKER_0_HOST}:${RAY_PORT}"
python -m ray.scripts.scripts stop --force || true

if [ "${MLP_ROLE_INDEX}" = "0" ]; then
  rm -f "${RAY_READY_FILE}" "${RAY_DONE_FILE}" "${RAY_DONE_FILE}.tmp"
  python -m ray.scripts.scripts start --head \
    --node-ip-address="${MLP_WORKER_0_HOST}" \
    --port="${RAY_PORT}" \
    --num-gpus="${MLP_WORKER_GPU}" \
    --dashboard-host=0.0.0.0 \
    --disable-usage-stats

  touch "${RAY_READY_FILE}"
  sleep 30

  python - <<'PY'
import os
import time

import ray

ray.init(address="auto")
expected_gpus = int(os.environ["MLP_WORKER_NUM"]) * int(os.environ["MLP_WORKER_GPU"])
for _ in range(120):
    resources = ray.cluster_resources()
    print("RAY_RESOURCES:", resources)
    if int(resources.get("GPU", 0)) >= expected_gpus:
        break
    time.sleep(5)
else:
    raise RuntimeError(f"Timed out waiting for {expected_gpus} GPUs")
PY

  completed_step=0
  if [ -r "${CKPT_DIR}/latest_checkpointed_iteration.txt" ]; then
    completed_step="$(tr -d '[:space:]' < "${CKPT_DIR}/latest_checkpointed_iteration.txt")"
  fi

  if [ "${completed_step}" -ge "${TOTAL_TRAINING_STEPS}" ]; then
    echo "Training already completed at step ${completed_step}; target is ${TOTAL_TRAINING_STEPS}."
    train_status=0
  else
    set +e
    python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=512 \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.clip_ratio_low=0.002 \
    actor_rollout_ref.actor.clip_ratio_high=0.002 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10240 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.data_parallel_size="${ROLLOUT_DP}" \
    actor_rollout_ref.rollout.expert_parallel_size="${ROLLOUT_EP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.max_num_batched_tokens=10240 \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.n="${VAL_N}" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=20480 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=4 \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable="${OVERLONG_BUFFER_ENABLE}" \
    +reward.reward_kwargs.overlong_buffer_cfg.len="${OVERLONG_BUFFER_LEN}" \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len="${MAX_RESPONSE_LENGTH}" \
    trainer.rollout_buffer.enable=True \
    trainer.rollout_buffer.delay_steps=2 \
    trainer.rollout_buffer.use_rollout_log_probs=True \
    trainer.logger="${TRAINER_LOGGER}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.nnodes="${MLP_WORKER_NUM}" \
    trainer.n_gpus_per_node="${MLP_WORKER_GPU}" \
    trainer.val_before_train=True \
    trainer.val_only="${VAL_ONLY}" \
    trainer.test_freq=5 \
    trainer.save_freq=5 \
    trainer.resume_mode=auto \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.max_actor_ckpt_to_keep=null \
    trainer.max_critic_ckpt_to_keep=null \
    trainer.validation_data_dir="${VAL_DUMP_DIR}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    '+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1="1"' \
    '+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ENABLE_V1_MULTIPROCESSING="0"'
    train_status=$?
    set -e
  fi

  printf '%s\n' "${train_status}" > "${RAY_DONE_FILE}.tmp"
  mv "${RAY_DONE_FILE}.tmp" "${RAY_DONE_FILE}"
  sleep 10
  python -m ray.scripts.scripts stop --force || true
  exit "${train_status}"
else
  for _ in $(seq 1 120); do
    [ -f "${RAY_READY_FILE}" ] && break
    sleep 5
  done
  [ -f "${RAY_READY_FILE}" ] || { echo "Timed out waiting for Ray head readiness" >&2; exit 1; }

  python -m ray.scripts.scripts start \
    --address="${RAY_ADDRESS}" \
    --num-gpus="${MLP_WORKER_GPU}"

  while [ ! -f "${RAY_DONE_FILE}" ]; do
    sleep 5
  done
  python -m ray.scripts.scripts stop --force || true
  exit 0
fi
