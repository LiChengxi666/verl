#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Exact per-prefix constraint: R - 1 - log(R) <= t * delta.
# These scalar deltas are exploratory sweep defaults, not CTPO epsilons.
export LOSS_MODE=prefix_exact_kl_clip
export PREFIX_EXACT_KL_DELTA_LOW="${PREFIX_EXACT_KL_DELTA_LOW:-0.02}"
export PREFIX_EXACT_KL_DELTA_HIGH="${PREFIX_EXACT_KL_DELTA_HIGH:-0.05}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-prefix-exact-kl-l002-h005-qwen3-4b-8gpu}"

exec bash "${SCRIPT_DIR}/run_prefix_ripo_clip_qwen3_4b_base_processed_8gpu_300.sh"
