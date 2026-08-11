#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"

REPO_ROOT="${REPO_ROOT:-/opt/tiger/offpolicyrl}"
VERL_ROOT="${REPO_ROOT}/verl"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/runtime_assets_fullthink_final/models/Qwen3-4B-SFT-fullthink-step366}"
TEACHER_TRAJECTORY_ROOT="${TEACHER_TRAJECTORY_ROOT:-hdfs://harunawl/home/byte_data_seed_wl/user/wu.hanlin/offpolicyrl/sft_rl/qwen3_4b_math17k_teacher_qwen3_32b_t07_p095_n8_schema2_s16_20260806_v1}"
VAL_DIR="${VAL_DIR:-${REPO_ROOT}/runtime_assets_fullthink_final/data/moe/evals}"
PROJECT_NAME="${PROJECT_NAME:-qwen3_4b_math17k_sft_gspo}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_4b_fullthink_sft3ep_offline_ppo_teacher32b_all139128_8x8_is10_b1024_v1}"
LOCAL_ROOT="${LOCAL_ROOT:-${REPO_ROOT}/run_state/${EXPERIMENT_NAME}}"
HDFS_CHECKPOINT_DIR="${HDFS_CHECKPOINT_DIR:-hdfs://harunawl/home/byte_data_seed_wl/user/wu.hanlin/offpolicyrl/sft_rl/checkpoints/${EXPERIMENT_NAME}}"
NNODES="${NNODES:-8}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1024}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-128}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
OFFLINE_MAX_RESPONSE_LENGTH="${OFFLINE_MAX_RESPONSE_LENGTH:-16384}"
VALIDATION_MAX_RESPONSE_LENGTH="${VALIDATION_MAX_RESPONSE_LENGTH:-30720}"

load_cluster_environment
load_wandb_environment "${REPO_ROOT}"

if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "Missing SFT model at ${MODEL_PATH}" >&2
    exit 1
fi

mkdir -p \
    "${LOCAL_ROOT}/wandb" \
    "${LOCAL_ROOT}/checkpoints" \
    "${LOCAL_ROOT}/validation" \
    "${LOCAL_ROOT}/teacher_cache" \
    "${LOCAL_ROOT}/cache/huggingface/datasets"

export WANDB_DIR="${LOCAL_ROOT}/wandb"
export HF_HOME="${LOCAL_ROOT}/cache/huggingface"
export HF_DATASETS_CACHE="${LOCAL_ROOT}/cache/huggingface/datasets"
export XDG_CACHE_HOME="${LOCAL_ROOT}/cache/xdg"
export TRANSFORMERS_CACHE="${LOCAL_ROOT}/cache/huggingface/transformers"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${EXPERIMENT_NAME}}"
export WANDB_NAME="${EXPERIMENT_NAME}"
export WANDB_RESUME=must
export WANDB_CONSOLE=off
export WANDB_CACHE_DIR="${LOCAL_ROOT}/cache/wandb"
export WANDB_DATA_DIR="${LOCAL_ROOT}/cache/wandb-data"
export WANDB_CONFIG_DIR="${LOCAL_ROOT}/cache/wandb-config"
export VERL_FILE_LOGGER_PATH="${LOCAL_ROOT}/metrics.jsonl"
export TENSORBOARD_LOG_PATH="${LOCAL_ROOT}/tensorboard"
export RAY_ADDRESS="${RAY_ADDRESS:-auto}"
export RAY_RUNTIME_ENV_IGNORE_GITIGNORE=1
export VLLM_USE_V1=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

cd "${VERL_ROOT}"
python - \
    algorithm.adv_estimator=gae \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=10.0 \
    algorithm.rollout_correction.rollout_is_batch_normalize=True \
    algorithm.rollout_correction.rollout_rs=null \
    algorithm.rollout_correction.rollout_rs_threshold=null \
    algorithm.rollout_correction.bypass_mode=True \
    algorithm.rollout_correction.loss_type=reinforce \
    data.train_files="${TEACHER_TRAJECTORY_ROOT}" \
    "data.val_files=[${VAL_DIR}/aime24.parquet,${VAL_DIR}/aime25.parquet,${VAL_DIR}/amc23.parquet,${VAL_DIR}/hmmt25.parquet]" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.dataloader_num_workers=0 \
    data.shuffle=False \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${VALIDATION_MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.offline_trajectory.enable=True \
    data.offline_trajectory.root="${TEACHER_TRAJECTORY_ROOT}" \
    data.offline_trajectory.cache_dir="${LOCAL_ROOT}/teacher_cache" \
    data.offline_trajectory.max_cached_shards=2 \
    data.offline_trajectory.pad_to_multiple="${TRAIN_BATCH_SIZE}" \
    data.offline_trajectory.max_response_length="${OFFLINE_MAX_RESPONSE_LENGTH}" \
    data.offline_trajectory.max_resp_len="${OFFLINE_MAX_RESPONSE_LENGTH}" \
    data.offline_trajectory.overlong_buffer_len=512 \
    data.offline_trajectory.overlong_penalty_factor=1.0 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.policy_loss.loss_mode=bypass_mode \
    +actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is=token \
    +actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is_threshold=10.0 \
    +actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is_batch_normalize=True \
    +actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_rs=null \
    +actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_rs_threshold=null \
    +actor_rollout_ref.actor.policy_loss.rollout_correction.bypass_mode=True \
    +actor_rollout_ref.actor.policy_loss.rollout_correction.loss_type=reinforce \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.checkpoint.remove_local_after_hdfs_save=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=96000 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=20 \
    'actor_rollout_ref.rollout.stop_token_ids=[151643,151645]' \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=65536 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=65536 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.model.path="${MODEL_PATH}" \
    critic.model.use_remove_padding=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.optim.lr=1e-5 \
    critic.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    critic.ppo_epochs=1 \
    critic.use_dynamic_bsz=True \
    critic.ppo_max_token_len_per_gpu=32768 \
    critic.forward_max_token_len_per_gpu=32768 \
    critic.cliprange_value=0.5 \
    critic.fsdp.param_offload=True \
    critic.fsdp.optimizer_offload=True \
    critic.checkpoint.remove_local_after_hdfs_save=True \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward.reward_kwargs.overlong_buffer_cfg.len=512 \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len="${VALIDATION_MAX_RESPONSE_LENGTH}" \
    trainer.balance_batch=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","file","tensorboard","wandb"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.val_before_train=True \
    trainer.test_freq=5 \
    trainer.save_freq=5 \
    trainer.resume_mode=auto \
    trainer.default_local_dir="${LOCAL_ROOT}/checkpoints" \
    trainer.default_hdfs_dir="${HDFS_CHECKPOINT_DIR}" \
    trainer.del_local_ckpt_after_load=True \
    trainer.remove_local_ckpt_after_hdfs_save=True \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.max_complete_ckpt_to_keep=1 \
    trainer.validation_data_dir="${LOCAL_ROOT}/validation" \
    trainer.total_epochs=1 "$@" <<'PY'
import os

import ray

runtime_env_names = (
    "HF_HOME",
    "HF_DATASETS_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "XDG_CACHE_HOME",
    "WANDB_API_KEY",
    "WANDB_ENTITY",
    "WANDB_MODE",
    "WANDB_BASE_URL",
    "WANDB_RUN_ID",
    "WANDB_NAME",
    "WANDB_RESUME",
    "WANDB_DIR",
    "VERL_FILE_LOGGER_PATH",
    "TENSORBOARD_LOG_PATH",
    "VLLM_USE_V1",
    "VLLM_ENABLE_V1_MULTIPROCESSING",
    "HADOOP_CONF_DIR",
    "PATH",
    "INFSEC_HADOOP_ENABLED",
    "MLX_USER",
    "MLX_USER_TOKEN",
    "SEC_IDENTITY_DIR",
    "SEC_TOKEN_PATH",
    "ZTI_TOKEN",
    "CPP_HDFS_CONF",
    "HADOOP_CLIENT_OPTS",
    "KRB5_CONFIG",
)
runtime_env = {
    "working_dir": os.getcwd(),
    "excludes": ["/.git/", "/docs/", "/tests/", "/.github/", "/examples/"],
    "env_vars": {name: os.environ[name] for name in runtime_env_names if name in os.environ},
}
for name in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy"):
    if name in os.environ:
        runtime_env["env_vars"][name] = os.environ[name]

ray.init(address=os.environ["RAY_ADDRESS"], runtime_env=runtime_env)

from verl.trainer.main_ppo import main

main()
PY

