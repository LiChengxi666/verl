# GSPO MoE Router Replay Recipes

## Goal

Provide three portable entry scripts for a controlled Qwen3-30B-A3B GSPO
off-2 comparison:

1. GSPO off-2 without routing replay.
2. GSPO + R2 off-2.
3. GSPO + R3 off-2.

The recipes are intended for an external environment pulled from the GitHub
fork. They must not submit platform jobs or depend on internal absolute paths.

## Structure

The recipes use one shared launcher and three thin entry scripts:

```text
training_jobs/scripts/moe_rl/
├── run_gspo_off2.sh
├── run_gspo_r2_off2.sh
├── run_gspo_r3_off2.sh
└── common/
    └── launch_qwen3_30b_a3b.sh
```

Each entry script sets only:

```text
POLICY_LOSS_MODE=gspo
ROUTER_REPLAY_MODE=disabled | R2 | R3
```

The shared launcher owns all other experiment parameters. It builds separate
argument groups for shared training settings, policy loss, routing replay, and
the Megatron/vLLM backend. Unsupported policy or routing modes fail before Ray
or training starts.

## Experiment Invariants

All three recipes use:

- Qwen3-30B-A3B-Base.
- Processed MATH-17k with 0/1 correctness reward.
- AMC23, AIME24, and AIME25 validation sets with `avg@8`.
- One behavior batch of 64 prompts and 8 responses per prompt.
- A 32-prompt PPO mini-batch and one PPO epoch, giving off-2 sequential
  optimizer updates without a cross-step rollout buffer.
- GSPO clip low/high of `3e-4/4e-4`.
- Learning rate `2e-6`.
- Reference-policy KL loss coefficient `1e-3`.
- Maximum prompt/response lengths of 2048/8192.
- 200 training steps, validation and checkpointing every 5 steps.
- A 16-GPU default topology and identical Megatron/vLLM parallel settings.
- File, TensorBoard, and optional W&B logging.
- Automatic resume from complete checkpoints.

Only routing replay differs:

| Recipe | Training-engine mode | Rollout routing return |
| --- | --- | --- |
| GSPO | `disabled` | `False` |
| GSPO + R2 | `R2` | `False` |
| GSPO + R3 | `R3` | `True` |

R2 records routes while the training engine recomputes old log probabilities
and replays them during actor updates. R3 records routes in the rollout engine
and replays them during both old-log-prob computation and actor updates.

## Portability

Defaults are relative to the repository root:

```text
models/Qwen3-30B-A3B-Base
data/data_processed/math-17k.parquet
data/data_processed/moe_eval/minpro/amc23.parquet
data/data_processed/moe_eval/minpro/aime24.parquet
data/data_processed/moe_eval/minpro/aime25.parquet
```

Environment variables may override model, data, output, W&B secret, resource,
and parallelism settings. The scripts contain no queue IDs, image registry
URLs, mounted internal paths, or embedded credentials.

## Extension Points

`POLICY_LOSS_MODE` and `ROUTER_REPLAY_MODE` are independent validated
dimensions. A future policy method adds a policy argument branch and thin
entry scripts. A future PR2 implementation adds framework support first, then
a `PR2` routing branch and entry script. The current recipes must not expose a
nonfunctional PR2 mode.

## Verification

Static recipe tests must verify:

- all three entry scripts delegate to the same launcher;
- the wrappers differ only in policy and routing mode declarations;
- shared scientific parameters match the invariants above;
- R2 cannot enable rollout routing replay;
- R3 must enable rollout routing replay;
- defaults contain no internal absolute paths;
- shell syntax is valid.

Runtime preflight checks verify the model, datasets, required Megatron engine,
router replay configuration, GPU topology, and W&B secret handling before
starting Ray.
