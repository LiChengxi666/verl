#!/usr/bin/env bash
set -xeuo pipefail

cd /GenSIvePFS/users/cxli/verl
# The verl:v0 image uses its built-in conda environment. Do not force .venv here.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MLP_ROLE_INDEX="${MLP_ROLE_INDEX:-0}"
export MLP_WORKER_NUM="${MLP_WORKER_NUM:-1}"
export MLP_WORKER_GPU="${MLP_WORKER_GPU:-8}"
export MLP_WORKER_0_HOST="${MLP_WORKER_0_HOST:-127.0.0.1}"

which python
python - <<'PY'
import os
import sys
import tensordict
import verl

print("PYTHON:", sys.executable)
print("PREFIX:", sys.prefix)
print("TENSORDICT:", tensordict.__version__, tensordict.__file__)
print("VERL:", verl.__file__)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("MLP_WORKER_GPU:", os.environ.get("MLP_WORKER_GPU"))

assert tensordict.__version__.split(".")[:2] >= ["0", "10"]
assert verl.__file__.startswith("/GenSIvePFS/users/cxli/verl/"), verl.__file__
assert int(os.environ["MLP_WORKER_GPU"]) == 8, "This script is configured for a single 8-GPU node."
PY

python - <<'PY'
from transformers import AutoConfig, AutoTokenizer

model_path = "/GenSIvePFS/users/cxli/models/Qwen3-4B-Base"
config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
print("MODEL_CHECK:", model_path, config.model_type, getattr(config, "num_hidden_layers", None), len(tok))
PY

export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_USE_V1=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export MODEL_PATH=/GenSIvePFS/users/cxli/models/Qwen3-4B-Base

export PROJECT_NAME=verl_math_repro
# RIPO-like deltas are average-token prefix KL budgets:
#   D_KL^pre(t) <= t * delta.
# They are intentionally much smaller than the old paper-scale trial values
# such as 0.02/0.05/0.08, which would be too loose under this semantics.
export RIPO_DELTA_LOW="${RIPO_DELTA_LOW:-1e-5}"
export RIPO_DELTA_HIGH="${RIPO_DELTA_HIGH:-3e-5}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-prefix-ripo-l1em5-h3em5-qwen3-4b-8gpu}"

export CKPT_DIR=/GenSIvePFS/users/cxli/verl/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}
export LOG_DIR=/GenSIvePFS/users/cxli/verl/train_logs/${PROJECT_NAME}/${EXPERIMENT_NAME}
export VAL_DUMP_DIR=/GenSIvePFS/users/cxli/verl/validation_generations/${PROJECT_NAME}/${EXPERIMENT_NAME}

mkdir -p "${CKPT_DIR}" "${LOG_DIR}" "${VAL_DUMP_DIR}"

export VERL_FILE_LOGGER_PATH="${LOG_DIR}/metrics.jsonl"
export VERL_PREFIX_CLIP_DIAG_DIR="${LOG_DIR}/prefix_clip_diagnostics"
export TENSORBOARD_DIR="${LOG_DIR}/tensorboard"
export TENSORBOARD_LOG_PATH="${TENSORBOARD_DIR}"
export WANDB_DIR="${LOG_DIR}/wandb"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${EXPERIMENT_NAME}}"
export WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-/GenSIvePFS/users/cxli/.secrets/wandb_api_key}"

if [ -z "${WANDB_API_KEY:-}" ]; then
  if [ ! -r "${WANDB_API_KEY_FILE}" ]; then
    echo "WANDB_API_KEY is not set and ${WANDB_API_KEY_FILE} is not readable." >&2
    exit 1
  fi
  export WANDB_API_KEY
  WANDB_API_KEY="$(tr -d '\r\n' < "${WANDB_API_KEY_FILE}")"
fi

mkdir -p "${WANDB_DIR}"

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
import ray

ray.init(address="auto")
print("RAY_RESOURCES:", ray.cluster_resources())
assert int(ray.cluster_resources().get("GPU", 0)) == 8


@ray.remote(num_gpus=1)
class EnvProbe:
    def check(self):
        import os
        import sys
        import torch
        import tensordict
        return {
            "python": sys.executable,
            "prefix": sys.prefix,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_cuda_available": torch.cuda.is_available(),
            "torch_cuda_device_count": torch.cuda.device_count(),
            "tensordict": tensordict.__version__,
            "tensordict_file": tensordict.__file__,
        }

probes = [EnvProbe.remote() for _ in range(8)]
infos = ray.get([probe.check.remote() for probe in probes])
print("RAY_ENV:", infos)
assert all(info["torch_cuda_available"] for info in infos)
PY

  python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=/GenSIvePFS/users/cxli/verl/data/data_processed/math-17k.parquet \
    data.val_files=/GenSIvePFS/users/cxli/verl/data/data_processed/aime24.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.policy_loss.loss_mode=prefix_ripo_clip \
    actor_rollout_ref.actor.policy_loss.prefix_clip_first_low=0.2 \
    actor_rollout_ref.actor.policy_loss.prefix_clip_first_high=0.28 \
    actor_rollout_ref.actor.policy_loss.prefix_clip_final_low=3e-4 \
    actor_rollout_ref.actor.policy_loss.prefix_clip_final_high=4e-4 \
    actor_rollout_ref.actor.policy_loss.prefix_ripo_delta_low="${RIPO_DELTA_LOW}" \
    actor_rollout_ref.actor.policy_loss.prefix_ripo_delta_high="${RIPO_DELTA_HIGH}" \
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
    trainer.nnodes="${MLP_WORKER_NUM}" \
    trainer.n_gpus_per_node="${MLP_WORKER_GPU}" \
    trainer.val_before_train=True \
    trainer.test_freq=10 \
    trainer.save_freq=10 \
    trainer.resume_mode=auto \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.max_actor_ckpt_to_keep=5 \
    trainer.max_critic_ckpt_to_keep=5 \
    trainer.validation_data_dir="${VAL_DUMP_DIR}" \
    trainer.total_training_steps=300 \
    '+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1="1"' \
    '+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ENABLE_V1_MULTIPROCESSING="0"'
else
  sleep 10
  python -m ray.scripts.scripts start \
    --address="${RAY_ADDRESS}" \
    --num-gpus="${MLP_WORKER_GPU}" \
    --block
fi
