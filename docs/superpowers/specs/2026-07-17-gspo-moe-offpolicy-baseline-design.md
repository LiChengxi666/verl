# GSPO MoE Off-Policy Baseline Design

## Objective

Run a 200-step GSPO baseline on Qwen3-30B-A3B-Base using the validated fixed-delay off-policy path with data staleness 2.

## Training

- Dataset: processed DAPO-Math-17K.
- Train batch / mini-batch: 512 / 32 prompts.
- Responses per prompt: 8.
- Prompt / response limit: 2048 / 20480 tokens.
- Sampling: temperature 1.0 and top-p 1.0.
- GSPO clip low / high: 0.002 / 0.002.
- Loss aggregation: `seq-mean-token-mean`.
- Actor learning rate: 1e-6 with 10 warmup steps.
- Overlong buffer: disabled.
- Fixed rollout delay: 2 completed policy updates, using generation-time log probabilities as the behavior-policy anchor.
- Total training steps: 200.

## Evaluation

- Datasets: AMC23, AIME24, and AIME25.
- Eight sampled answers per problem, reported as `mean@8`.
- Validate before training and every 5 training steps.

## Runtime And Persistence

- One worker with 8 A100 GPUs; FSDP world size 8.
- Save every 5 steps under `/sia-thu/cxli/checkpoints/verl_moe/`.
- Keep metrics and validation generations on vePFS and log online to the `verl_moe` W&B project.
- Resume automatically from the latest TOS-mounted checkpoint.
- Enable platform retry for the formal preemptible task.

## Verification

Before submission, verify the TOS mount is writable, compose the Hydra configuration, run recipe tests and shell syntax checks, and confirm the submitted task requests exactly one 8-GPU worker.
