#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Exact per-prefix constraint: R - 1 - log(R) <= t * delta.
# Match the CTPO log-space thresholds through the local expansion
# exp(z) - 1 - z = 0.5 * z^2 + O(z^3):
#   delta_low  = 0.025^2 / 2 = 3.125e-4
#   delta_high = 0.05^2  / 2 = 1.25e-3
export LOSS_MODE=prefix_exact_kl_clip
export PREFIX_EXACT_KL_DELTA_LOW="${PREFIX_EXACT_KL_DELTA_LOW:-3.125e-4}"
export PREFIX_EXACT_KL_DELTA_HIGH="${PREFIX_EXACT_KL_DELTA_HIGH:-1.25e-3}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-prefix-exact-kl-ctpo-taylor-qwen3-4b-8gpu}"

exec bash "${SCRIPT_DIR}/run_prefix_ripo_clip_qwen3_4b_base_processed_8gpu_300.sh"
