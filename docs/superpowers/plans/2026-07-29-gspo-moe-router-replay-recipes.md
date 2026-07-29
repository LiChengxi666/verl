# GSPO MoE Router Replay Recipes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add portable, parameter-aligned GSPO off-policy entry recipes for no replay, R2, and R3 on Qwen3-30B-A3B.

**Architecture:** Three thin entry scripts declare the policy and routing modes, then delegate to one Megatron/vLLM multi-node launcher. The launcher derives off-2/off-4/off-8 update batch size and paper-default learning rate from `OFF_POLICY_K`, builds independent shared/policy/router/backend Hydra argument arrays, and performs a preflight before starting Ray.

**Tech Stack:** Bash, Hydra CLI overrides, Ray, verl MegatronEngine, vLLM Router Replay, pytest.

## Global Constraints

- Do not submit platform jobs or create platform YAML files.
- Use repository-relative defaults and allow environment-variable overrides.
- Default `OFF_POLICY_K=2`; only 2, 4, and 8 are supported.
- Keep all scientific parameters identical across disabled, R2, and R3 except routing replay.
- Keep `ppo_epochs=1` and `trainer.rollout_buffer.enable=False`.
- Use Qwen3-30B-A3B-Base, processed MATH-17k, AMC23, AIME24, and AIME25.
- Default to 4 nodes x 4 A100 GPUs, 200 steps, validation/save every 5 steps.
- Preserve unrelated dirty worktree changes.

---

### Task 1: Specify the portable experiment matrix in tests

**Files:**
- Create: `tests/scripts/test_gspo_moe_router_replay_recipes.py`

**Interfaces:**
- Consumes: the paths and invariants in `docs/superpowers/specs/2026-07-29-gspo-moe-router-replay-recipes-design.md`.
- Produces: subprocess and static assertions that define the wrapper and launcher contract.

- [ ] **Step 1: Write failing wrapper and shell-syntax tests**

Add tests that require:

```python
RECIPE_ROOT = REPO_ROOT / "training_jobs/scripts/moe_rl"
COMMON = RECIPE_ROOT / "common/launch_qwen3_30b_a3b.sh"
WRAPPERS = {
    "disabled": RECIPE_ROOT / "run_gspo_off2.sh",
    "R2": RECIPE_ROOT / "run_gspo_r2_off2.sh",
    "R3": RECIPE_ROOT / "run_gspo_r3_off2.sh",
}
```

For each wrapper, assert it contains exactly one `POLICY_LOSS_MODE=gspo`,
the expected `ROUTER_REPLAY_MODE`, and delegates to `COMMON`. Run
`bash -n` on all four scripts.

- [ ] **Step 2: Run the wrapper tests and verify they fail**

Run:

```bash
pytest -q tests/scripts/test_gspo_moe_router_replay_recipes.py
```

Expected: fail because the recipe files do not exist.

- [ ] **Step 3: Add failing dry-run matrix tests**

Invoke each wrapper with:

```python
env = {
    **os.environ,
    "RECIPE_DRY_RUN": "1",
    "OFF_POLICY_K": str(k),
}
result = subprocess.run(
    ["bash", str(wrapper)],
    cwd=REPO_ROOT,
    env=env,
    text=True,
    capture_output=True,
    check=True,
)
```

Parse `RECIPE_CONFIG_JSON=<json>` and assert:

```python
expected = {
    2: {"mini_batch_size": 32, "actor_lr": "2e-6"},
    4: {"mini_batch_size": 16, "actor_lr": "1.5e-6"},
    8: {"mini_batch_size": 8, "actor_lr": "1e-6"},
}
```

Also assert default invocation gives off-2, R2 never enables rollout replay,
R3 always enables it, all variants have identical shared arguments, and
unsupported `OFF_POLICY_K=3` exits nonzero before training.

- [ ] **Step 4: Run the dry-run tests and verify the expected failure**

Run:

```bash
pytest -q tests/scripts/test_gspo_moe_router_replay_recipes.py
```

Expected: fail because the shared launcher and dry-run output are missing.

### Task 2: Implement the shared Megatron/vLLM launcher and wrappers

**Files:**
- Create: `training_jobs/scripts/moe_rl/common/launch_qwen3_30b_a3b.sh`
- Create: `training_jobs/scripts/moe_rl/run_gspo_off2.sh`
- Create: `training_jobs/scripts/moe_rl/run_gspo_r2_off2.sh`
- Create: `training_jobs/scripts/moe_rl/run_gspo_r3_off2.sh`

**Interfaces:**
- Consumes: `POLICY_LOSS_MODE`, `ROUTER_REPLAY_MODE`, `OFF_POLICY_K`, model/data/output/resource environment variables, and additional Hydra overrides in `"$@"`.
- Produces: `RECIPE_CONFIG_JSON` under dry-run or a multi-node Ray job running `verl.trainer.main_ppo`.

- [ ] **Step 1: Add the three thin wrappers**

Each wrapper must use:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export POLICY_LOSS_MODE=gspo
export ROUTER_REPLAY_MODE=<disabled|R2|R3>
exec bash "${SCRIPT_DIR}/common/launch_qwen3_30b_a3b.sh" "$@"
```

- [ ] **Step 2: Implement validated off-policy derivation**

In the shared launcher:

```bash
OFF_POLICY_K="${OFF_POLICY_K:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"

case "${OFF_POLICY_K}" in
  2) DEFAULT_ACTOR_LR=2e-6 ;;
  4) DEFAULT_ACTOR_LR=1.5e-6 ;;
  8) DEFAULT_ACTOR_LR=1e-6 ;;
  *) echo "OFF_POLICY_K must be one of: 2, 4, 8" >&2; exit 2 ;;
esac

(( TRAIN_BATCH_SIZE % OFF_POLICY_K == 0 )) || {
  echo "TRAIN_BATCH_SIZE must be divisible by OFF_POLICY_K" >&2
  exit 2
}
PPO_MINI_BATCH_SIZE=$((TRAIN_BATCH_SIZE / OFF_POLICY_K))
ACTOR_LR="${ACTOR_LR:-${DEFAULT_ACTOR_LR}}"
```

Default experiment names must include policy, routing mode, and
`off${OFF_POLICY_K}`.

- [ ] **Step 3: Implement policy and router argument builders**

Validate `POLICY_LOSS_MODE=gspo`. Build GSPO arguments with clip
`3e-4/4e-4`, sequence aggregation, actor KL loss `1e-3`, and reward-side KL
disabled.

Map routing modes exactly:

```bash
case "${ROUTER_REPLAY_MODE}" in
  disabled)
    ENABLE_ROLLOUT_ROUTING_REPLAY=False
    ;;
  R2)
    ENABLE_ROLLOUT_ROUTING_REPLAY=False
    ;;
  R3)
    ENABLE_ROLLOUT_ROUTING_REPLAY=True
    ;;
  *)
    echo "ROUTER_REPLAY_MODE must be one of: disabled, R2, R3" >&2
    exit 2
    ;;
esac
```

Pass:

```text
actor_rollout_ref.actor.megatron.router_replay.mode
actor_rollout_ref.rollout.enable_rollout_routing_replay
```

- [ ] **Step 4: Implement shared scientific and backend arguments**

Use:

```text
train batch = 64 prompts
rollout.n = 8
ppo epochs = 1
max prompt/response = 2048/8192
validation n = 8
total steps = 200
test/save frequency = 5
Megatron TP/PP/EP = 1/4/4
vLLM rollout TP = 4
```

Set `actor.strategy=megatron`, `actor.model_engine=megatron`, MBridge for
actor/ref, router dtype FP32, recomputation, offload, and the same backend
arguments for all three variants. Keep old-log-prob recomputation enabled and
rollout log-prob calculation enabled.

- [ ] **Step 5: Implement portable preflight, logging, and Ray lifecycle**

Reuse the proven multi-node Ray head/worker lifecycle from
`training_jobs/scripts/run_gspo_qwen3_30b_a3b_offpolicy_16gpu_b64n8_r8192.sh`.
Change all defaults to repository-relative paths. Verify model config,
processed datasets, Megatron engine registration, GPU topology, and W&B key
handling before Ray starts. Configure file/TensorBoard/optional W&B logging,
checkpoint auto-resume, and validation generation dumps.

- [ ] **Step 6: Implement deterministic dry-run output**

When `RECIPE_DRY_RUN=1`, skip dependency/model/data/Ray checks and print one
JSON object prefixed with `RECIPE_CONFIG_JSON=` containing:

```text
policy_loss_mode, router_replay_mode, rollout_routing_replay,
off_policy_k, train_batch_size, mini_batch_size, actor_lr,
ppo_epochs, rollout_n, max_response_length, total_training_steps,
shared_hydra_args
```

- [ ] **Step 7: Run tests and make them pass**

Run:

```bash
pytest -q tests/scripts/test_gspo_moe_router_replay_recipes.py
```

Expected: all tests pass.

### Task 3: Add the external-environment runbook

**Files:**
- Create: `training_jobs/configs/gspo_moe_router_replay/README.md`
- Modify: `tests/scripts/test_gspo_moe_router_replay_recipes.py`

**Interfaces:**
- Consumes: launcher variables and output paths from Task 2.
- Produces: a Chinese runbook for cloning the fork and running each controlled variant.

- [ ] **Step 1: Add a failing documentation contract test**

Require the README to include:

```text
run_gspo_off2.sh
run_gspo_r2_off2.sh
run_gspo_r3_off2.sh
OFF_POLICY_K=2
OFF_POLICY_K=4
OFF_POLICY_K=8
WANDB_API_KEY_FILE
resume_mode=auto
AMC23
AIME24
AIME25
```

Assert it contains no `/GenSIvePFS/`, queue ID, registry URL, or API key.

- [ ] **Step 2: Run the documentation test and verify it fails**

Run:

```bash
pytest -q tests/scripts/test_gspo_moe_router_replay_recipes.py
```

Expected: fail because the README is missing.

- [ ] **Step 3: Write the concise Chinese runbook**

Document:

- the controlled scientific question and R2/R3 semantics;
- repository-relative model/data layout;
- required Megatron/MBridge/vLLM Router Replay support;
- default 4x4 A100 topology and configurable environment variables;
- the three off-2 commands;
- off-4/off-8 invocation and automatic mini-batch/LR mapping;
- W&B secret-file setup without committing credentials;
- logs, validation generations, checkpoints, and automatic recovery;
- how a future policy mode or PR2 branch extends the matrix.

- [ ] **Step 4: Run tests and make them pass**

Run:

```bash
pytest -q tests/scripts/test_gspo_moe_router_replay_recipes.py
```

Expected: all tests pass.

### Task 4: Verify, review, and publish the recipe changes

**Files:**
- Review all files created in Tasks 1-3.

**Interfaces:**
- Consumes: completed recipes, tests, and runbook.
- Produces: a verified commit pushed to `origin/package-prefix-ripo-recipes`.

- [ ] **Step 1: Run focused verification**

```bash
pytest -q tests/scripts/test_gspo_moe_router_replay_recipes.py
bash -n training_jobs/scripts/moe_rl/common/launch_qwen3_30b_a3b.sh
bash -n training_jobs/scripts/moe_rl/run_gspo_off2.sh
bash -n training_jobs/scripts/moe_rl/run_gspo_r2_off2.sh
bash -n training_jobs/scripts/moe_rl/run_gspo_r3_off2.sh
```

- [ ] **Step 2: Run existing neighboring recipe tests**

```bash
pytest -q tests/scripts/test_moe_offpolicy_recipe.py
```

Existing failures caused by unrelated pre-existing worktree changes must be
reported and not hidden by modifying unrelated recipes.

- [ ] **Step 3: Review the diff and portability**

```bash
git diff --check
git diff -- \
  training_jobs/scripts/moe_rl \
  training_jobs/configs/gspo_moe_router_replay \
  tests/scripts/test_gspo_moe_router_replay_recipes.py
rg -n '/GenSIvePFS|gensi-cn-beijing|q-20|WANDB_API_KEY=' \
  training_jobs/scripts/moe_rl \
  training_jobs/configs/gspo_moe_router_replay
```

Expected: no internal path, registry, queue, or embedded secret match.

- [ ] **Step 4: Commit only the scoped implementation**

```bash
git add \
  training_jobs/scripts/moe_rl \
  training_jobs/configs/gspo_moe_router_replay \
  tests/scripts/test_gspo_moe_router_replay_recipes.py \
  docs/superpowers/plans/2026-07-29-gspo-moe-router-replay-recipes.md
git commit -m "recipe: add GSPO MoE router replay matrix" \
  -m "Co-authored-by: OpenAI Codex <codex@openai.com>"
```

- [ ] **Step 5: Push the current branch**

```bash
git push origin package-prefix-ripo-recipes
```

Expected: the fork branch contains the design, plan, recipes, tests, and
external runbook after the previously pushed commits.
