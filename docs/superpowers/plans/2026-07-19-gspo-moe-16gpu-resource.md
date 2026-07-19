# GSPO MoE 16-GPU Resource Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the paper-scale Qwen3-30B-A3B GSPO off-policy recipe on four 4-GPU A100 nodes with actor state sharded across all 16 GPUs.

**Architecture:** Preserve every algorithm and data parameter. Change only the platform resource count, script resource validation, actor FSDP world size, experiment identity, and resource documentation.

**Tech Stack:** Bash, Hydra, VeRL, FSDP, Ray, Volc MLP YAML, pytest.

## Global Constraints

- Use 4 workers with 4 NVIDIA A100 80G GPUs each.
- Keep rollout TP/DP/EP at 4/1/4.
- Compute actor FSDP size from the 16-GPU training world.
- Keep batch size 512, rollout n 8, response length 20480, and delay 2 unchanged.

---

### Task 1: Lock the resource contract in tests

**Files:**
- Modify: `tests/scripts/test_moe_offpolicy_recipe.py`

- [x] Assert four workers, four GPUs per worker, dynamic actor FSDP size, and the 16-GPU experiment identity.
- [x] Run the focused test and confirm that it fails against the old 8-GPU recipe.

### Task 2: Update the runnable recipe

**Files:**
- Modify: `training_jobs/scripts/run_gspo_qwen3_30b_a3b_offpolicy_8gpu_200.sh`
- Modify: `training_jobs/configs/train_gspo_qwen3_30b_a3b_offpolicy_8gpu_200_config.yaml`
- Modify: `training_jobs/configs/gspo_moe_offpolicy/README.md`

- [x] Set the script and YAML to four 4-GPU workers.
- [x] Set actor FSDP size to the total GPU count.
- [x] Update names and documentation without changing algorithm parameters.
- [x] Run focused tests and shell/YAML validation.

### Task 3: Submit and inspect

**Files:** None.

- [x] Cancel the obsolete 8-GPU retry.
- [x] Submit the updated YAML with `volc ml_task submit`.
- [x] Confirm the new task requests four `ml.pni2.14xlarge` workers.
