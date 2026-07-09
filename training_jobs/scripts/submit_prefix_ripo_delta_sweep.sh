#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_DIR}"

for conf in training_jobs/configs/prefix_ripo_delta_sweep/train_prefix_ripo_*.yaml; do
  echo "Submitting ${conf}"
  volc ml_task submit --conf "${conf}"
done
