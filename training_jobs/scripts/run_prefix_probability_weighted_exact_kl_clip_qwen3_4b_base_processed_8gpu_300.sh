#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Probability-weighted coordinate constraint:
# R - 1 - log(R) <= t * delta / pi_old(prefix).
# These scalar deltas are exploratory sweep defaults, not CTPO epsilons.
export LOSS_MODE=prefix_probability_weighted_exact_kl_clip
export PREFIX_EXACT_KL_DELTA_LOW="${PREFIX_EXACT_KL_DELTA_LOW:-1e-5}"
export PREFIX_EXACT_KL_DELTA_HIGH="${PREFIX_EXACT_KL_DELTA_HIGH:-3e-5}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-prefix-prob-exact-kl-l1em5-h3em5-qwen3-4b-8gpu}"

exec bash "${SCRIPT_DIR}/run_prefix_ripo_clip_qwen3_4b_base_processed_8gpu_300.sh"
