#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Match the final 4B GSPO/prefix recipe and change only the clipping strategy.
export LOSS_MODE=prefix_exact_kl_clip
export RIPO_DELTA_LOW="${RIPO_DELTA_LOW:-1e-5}"
export RIPO_DELTA_HIGH="${RIPO_DELTA_HIGH:-3e-5}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-prefix-exact-kl-l1em5-h3em5-qwen3-4b-8gpu}"

exec bash "${SCRIPT_DIR}/run_prefix_ripo_clip_qwen3_4b_base_processed_8gpu_300.sh"
