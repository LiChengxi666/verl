# Prefix RIPO-like Delta Sweep

Use these configs to sweep `prefix_ripo_clip` delta settings.

`prefix_ripo_delta_low/high` are average-token prefix KL budgets. Do not reuse
the old `0.02/0.05/0.08` trial values unless intentionally testing a very loose
setting.

Current sweep:

```text
1e-7 / 3e-7
3e-7 / 1e-7
1e-6 / 3e-6
3e-6 / 1e-6
1e-5 / 3e-5
3e-5 / 1e-5
1e-4 / 3e-4
3e-4 / 1e-4
```

Submit:

```bash
bash /GenSIvePFS/users/cxli/verl/training_jobs/scripts/submit_prefix_ripo_delta_sweep.sh
```

Each task uses the same 4B processed-data recipe and only changes
`RIPO_DELTA_LOW`, `RIPO_DELTA_HIGH`, and `EXPERIMENT_NAME`.
