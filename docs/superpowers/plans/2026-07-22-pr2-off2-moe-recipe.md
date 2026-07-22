# PR2 Off-2 MoE Recipe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the cross-step fixed-delay rollout buffer in the Qwen3-30B-A3B GSPO recipes with PR2-style off-2 sequential updates and validate the configuration with an 8-GPU smoke task.

**Architecture:** Keep the validated Ray/FSDP multi-node launcher unchanged. Express PR2 off-2 through a 64-prompt global rollout batch, a 32-prompt PPO mini-batch, and one PPO epoch; disable the cross-step rollout buffer so each generated batch is consumed immediately in exactly two optimizer updates.

**Tech Stack:** Bash, Hydra/OmegaConf, VeRL PPO trainer, Volcengine `volc ml_task`.

## Global Constraints

- Formal training uses `data.train_batch_size=64`, `actor_rollout_ref.rollout.n=8`, `actor_rollout_ref.actor.ppo_mini_batch_size=32`, and `actor_rollout_ref.actor.ppo_epochs=1`.
- `trainer.rollout_buffer.enable=False`; no rollout queue is stored in checkpoints.
- Preserve the validated launcher topology and all unrelated model, rollout, validation, logging, and checkpoint settings.
- The smoke task uses one node with eight A100 GPUs, reduced steps and response length, and automatic retry disabled.

---

### Task 1: Convert the formal and portable recipes to PR2 off-2

**Files:**
- Modify: `training_jobs/scripts/run_gspo_qwen3_30b_a3b_offpolicy_16gpu_b64n8_r8192.sh`
- Modify: `training_jobs/configs/train_gspo_qwen3_30b_a3b_offpolicy_16gpu_b64n8_r8192_config.yaml`
- Modify: `training_jobs/configs/gspo_moe_offpolicy/README.md`

**Interfaces:**
- Consumes: the existing validated distributed launcher and Hydra command.
- Produces: a recipe whose effective optimizer-step count is `64 / 32 * 1 = 2` per generated rollout batch.

- [ ] **Step 1: Change the PPO mini-batch and buffer overrides**

Set `actor_rollout_ref.actor.ppo_mini_batch_size=32`, retain `ppo_epochs=1`, and set `trainer.rollout_buffer.enable=False`. Remove delay-specific overrides that no longer apply.

- [ ] **Step 2: Update the portable run instructions**

Describe off-2 as two sequential 32-prompt updates over one 64-prompt rollout batch, and state that checkpoint recovery regenerates an interrupted outer step rather than restoring pending rollouts.

- [ ] **Step 3: Verify the effective update count statically**

Run:

```bash
python - <<'PY'
from pathlib import Path

text = Path("training_jobs/scripts/run_gspo_qwen3_30b_a3b_offpolicy_16gpu_b64n8_r8192.sh").read_text()
for expected in (
    "data.train_batch_size=64",
    "actor_rollout_ref.actor.ppo_mini_batch_size=32",
    "actor_rollout_ref.actor.ppo_epochs=1",
    "trainer.rollout_buffer.enable=False",
):
    assert expected in text, expected
assert "trainer.rollout_buffer.delay_steps=" not in text
print("PR2 off-2 formal recipe: OK")
PY
```

Expected: `PR2 off-2 formal recipe: OK`.

### Task 2: Add and submit an 8-GPU PR2 off-2 smoke task

**Files:**
- Modify: `training_jobs/scripts/run_gspo_qwen3_30b_a3b_offpolicy_smoke.sh`
- Modify: `training_jobs/configs/train_gspo_qwen3_30b_a3b_offpolicy_smoke_config.yaml`

**Interfaces:**
- Consumes: the validated single-node Ray launcher and the same PR2 off-2 update semantics as Task 1.
- Produces: a short, non-retrying 8-GPU task that reaches actor optimization and checkpoint creation.

- [ ] **Step 1: Configure the smoke workload**

Use a divisible reduced global batch and mini-batch that still gives exactly two sequential updates, one PPO epoch, short response length, and a small total-step count. Keep all unrelated MoE compatibility settings from the validated recipe.

- [ ] **Step 2: Configure one 8-GPU worker and disable retry**

Set one worker replica using the existing eight-A100 flavor and set `RetryOptions.EnableRetry: false`.

- [ ] **Step 3: Validate shell and YAML syntax**

Run:

```bash
bash -n training_jobs/scripts/run_gspo_qwen3_30b_a3b_offpolicy_16gpu_b64n8_r8192.sh
bash -n training_jobs/scripts/run_gspo_qwen3_30b_a3b_offpolicy_smoke.sh
python - <<'PY'
import yaml
for path in (
    "training_jobs/configs/train_gspo_qwen3_30b_a3b_offpolicy_smoke_config.yaml",
):
    with open(path) as f:
        yaml.safe_load(f)
print("recipe syntax: OK")
PY
```

Expected: both shell checks exit zero and Python prints `recipe syntax: OK`.

- [ ] **Step 4: Submit and inspect the smoke task**

Run:

```bash
volc ml_task submit --conf training_jobs/configs/train_gspo_qwen3_30b_a3b_offpolicy_smoke_config.yaml
```

Expected: submission returns a task ID. Inspect its logs until it either reaches two actor optimizer iterations and a completed training step, or emits a concrete error requiring correction.

### Task 3: Commit and synchronize the external branch

**Files:**
- Modify: files from Tasks 1 and 2.

**Interfaces:**
- Consumes: verified recipe changes.
- Produces: a pushed `package-prefix-ripo-recipes` branch available to the external environment.

- [ ] **Step 1: Review only intended changes**

Run:

```bash
git diff -- training_jobs/scripts/run_gspo_qwen3_30b_a3b_offpolicy_16gpu_b64n8_r8192.sh training_jobs/scripts/run_gspo_qwen3_30b_a3b_offpolicy_smoke.sh training_jobs/configs/train_gspo_qwen3_30b_a3b_offpolicy_smoke_config.yaml training_jobs/configs/gspo_moe_offpolicy/README.md
```

Expected: only PR2 off-2 semantics, smoke resource settings, and matching instructions change.

- [ ] **Step 2: Commit the recipe changes**

```bash
git add training_jobs/scripts/run_gspo_qwen3_30b_a3b_offpolicy_16gpu_b64n8_r8192.sh training_jobs/scripts/run_gspo_qwen3_30b_a3b_offpolicy_smoke.sh training_jobs/configs/train_gspo_qwen3_30b_a3b_offpolicy_smoke_config.yaml training_jobs/configs/gspo_moe_offpolicy/README.md docs/superpowers/plans/2026-07-22-pr2-off2-moe-recipe.md
git commit -m "config: switch MoE recipe to PR2 off-2"
```

- [ ] **Step 3: Push the external branch**

```bash
git push origin package-prefix-ripo-recipes
```

Expected: the remote branch advances to the new commit.
