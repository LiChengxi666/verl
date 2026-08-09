#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Exact cumulative-prefix constraint: R - 1 - log(R) <= t * delta.
# The surrogate uses the geometric-average ratio R ** (1 / t); dividing the
# cumulative log bounds by t preserves exactly the same clipping decisions.
# Calibrate the budgets by exact conversion from asymmetric log-ratio bounds:
#   low:  1x CTPO, epsilon- = 0.025
#   high: 2x CTPO, epsilon+ = 0.10
export LOSS_MODE=prefix_exact_kl_clip
export PREFIX_EXACT_KL_DELTA_LOW="${PREFIX_EXACT_KL_DELTA_LOW:-3.09912028333e-4}"
export PREFIX_EXACT_KL_DELTA_HIGH="${PREFIX_EXACT_KL_DELTA_HIGH:-5.17091807565e-3}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-prefix-exact-kl-geom-ctpo1x2x-qwen3-4b-8gpu}"

exec bash "${SCRIPT_DIR}/run_prefix_ripo_clip_qwen3_4b_base_processed_8gpu_300.sh"
