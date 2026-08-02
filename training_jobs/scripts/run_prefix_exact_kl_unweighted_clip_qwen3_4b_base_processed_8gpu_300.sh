#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Exact per-prefix constraint: R - 1 - log(R) <= t * delta.
export LOSS_MODE=prefix_exact_kl_unweighted_clip
export RIPO_DELTA_LOW="${RIPO_DELTA_LOW:-0.02}"
export RIPO_DELTA_HIGH="${RIPO_DELTA_HIGH:-0.05}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-prefix-exact-kl-unweighted-l002-h005-qwen3-4b-8gpu}"

exec bash "${SCRIPT_DIR}/run_prefix_ripo_clip_qwen3_4b_base_processed_8gpu_300.sh"
