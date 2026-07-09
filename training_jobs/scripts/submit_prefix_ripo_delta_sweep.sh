#!/usr/bin/env bash
set -euo pipefail

cd /GenSIvePFS/users/cxli/verl

for conf in training_jobs/configs/prefix_ripo_delta_sweep/train_prefix_ripo_*.yaml; do
  echo "Submitting ${conf}"
  volc ml_task submit --conf "${conf}"
done
