"""Run the aligned PR2 R2 and geometric-probability exact-prefix matrix.

Each mode inherits the validated off2/oversampling recipe.  The R2 arm keeps
the GSPO baseline loss and only enables training-side router replay.  The
off4/off8 arms keep router replay disabled and use the best budget from the
aligned geometric-prefix-probability exact-prefix sweep.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(os.environ.get("OFFPOLICYRL_ROOT", "/opt/tiger/offpolicyrl"))
SOURCE_RECIPE = Path(__file__).with_name("run_pr2_geometric_probability_weighted_exact_sweep.py")
PROJECT = "verl_moe_router_replay"
WANDB_GROUP = "pr2_r2_and_geomprob_exact_offpolicy_matrix_20260820"
GRPO_OFF8_WANDB_GROUP = "PR2_off8_method_comparison_over01"
LOSS_MODE = "prefix_geometric_probability_weighted_exact_kl_clip"
DELTA_LOW = 5.0e-4
DELTA_HIGH = 2.0e-3
SETTINGS = {
    "r2_off2": {"mini_batch": 32, "lr": 2.0e-6, "loss": "gspo", "router_replay": "R2"},
    "geom_off4": {"mini_batch": 16, "lr": 1.5e-6, "loss": LOSS_MODE, "router_replay": "disabled"},
    "geom_off8": {"mini_batch": 8, "lr": 1.0e-6, "loss": LOSS_MODE, "router_replay": "disabled"},
    "grpo_r2_off8": {"mini_batch": 8, "lr": 1.0e-6, "loss": "vanilla", "router_replay": "R2"},
    "grpo_r3_off8": {"mini_batch": 8, "lr": 1.0e-6, "loss": "vanilla", "router_replay": "R3"},
    "grpo_nokl_clip020_028_off8_seed43": {
        "mini_batch": 8, "lr": 1.0e-6, "loss": "vanilla", "router_replay": "disabled", "seed": 43,
    },
    "grpo_nokl_clip020_028_off8_seed44": {
        "mini_batch": 8, "lr": 1.0e-6, "loss": "vanilla", "router_replay": "disabled", "seed": 44,
    },
}


def _load_source_recipe():
    spec = importlib.util.spec_from_file_location("pr2_geomprob_exact_sweep", SOURCE_RECIPE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load source recipe: {SOURCE_RECIPE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identities(mode: str) -> tuple[str, str, str]:
    if mode not in SETTINGS:
        raise ValueError(f"unknown mode {mode!r}; choose one of {sorted(SETTINGS)}")
    names = {
        "r2_off2": "PR2_GSPO_R2_off2_oversample0p1_qwen3_30b_a3b_4x8_b64n8_r16384_lr2e-6_300",
        "geom_off4": "PR2_geomprob_exact_d5e4_2e3_off4_oversample0p1_qwen3_30b_a3b_4x8_b64n8_r16384_lr1p5e-6_300",
        "geom_off8": "PR2_geomprob_exact_d5e4_2e3_off8_oversample0p1_qwen3_30b_a3b_4x8_b64n8_r16384_lr1e-6_300",
        "grpo_r2_off8": "PR2_GRPO_R2_off8_oversample0p1_qwen3_30b_a3b_4x8_b64n8_r16384_lr1e-6_clip0p2_0p28_nokl_300",
        "grpo_r3_off8": "PR2_GRPO_R3_off8_oversample0p1_qwen3_30b_a3b_4x8_b64n8_r16384_lr1e-6_clip0p2_0p28_nokl_300",
        "grpo_nokl_clip020_028_off8_seed43": "PR2_GRPO_noKL_off8_clipL0p2_H0p28_oversample0p1_qwen3_30b_a3b_4x8_b64n8_r16384_lr1e-6_seed43_300",
        "grpo_nokl_clip020_028_off8_seed44": "PR2_GRPO_noKL_off8_clipL0p2_H0p28_oversample0p1_qwen3_30b_a3b_4x8_b64n8_r16384_lr1e-6_seed44_300",
    }
    run_id = names[mode]
    is_grpo = mode.startswith("grpo_r") or mode.startswith("grpo_nokl_")
    date_slug = "20260830" if mode.startswith("grpo_nokl_") else ("20260824" if mode.startswith("grpo_r") else "20260820")
    nokl_slug = "_nokl" if is_grpo else ""
    state_slug = f"{mode}_aligned{nokl_slug}_moe32_{date_slug}"
    alignment_slug = "aligned-nokl-over01" if is_grpo else "aligned-over01"
    hdfs_dir = (
        "hdfs://harunawl/home/byte_data_seed_wl/user/wu.hanlin/offpolicyrl/checkpoints/"
        f"moe-{mode.replace('_', '-')}-{alignment_slug}-r16384-4x8-{date_slug}"
    )
    return run_id, state_slug, hdfs_dir


def _hdfs_exists(path: str) -> bool:
    return (
        subprocess.run(
            ["hdfs", "dfs", "-test", "-e", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def _hdfs_read_text(path: str) -> str:
    return subprocess.run(
        ["hdfs", "dfs", "-cat", path],
        check=True,
        text=True,
        capture_output=True,
    ).stdout


def validate_hdfs_target(hdfs_dir: str, *, allow_resume: bool, exists=None, read_text=None) -> int | None:
    exists = exists or _hdfs_exists
    read_text = read_text or _hdfs_read_text
    if not exists(hdfs_dir):
        return None
    if not allow_resume:
        raise RuntimeError(f"Refusing to reuse existing HDFS path: {hdfs_dir}")
    latest_path = f"{hdfs_dir}/latest_checkpointed_iteration.txt"
    try:
        latest_step = int(read_text(latest_path).strip())
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise RuntimeError(f"Invalid HDFS resume marker: {latest_path}") from exc
    success_path = f"{hdfs_dir}/global_step_{latest_step}/_SUCCESS"
    if latest_step <= 0 or not exists(success_path):
        raise RuntimeError(f"Refusing incomplete checkpoint at {success_path}")
    return latest_step


def configure(base: dict, mode: str, *, state_root: Path, source) -> dict:
    setting = SETTINGS[mode]
    run_id, _, hdfs_dir = identities(mode)
    # d5e4_2e3 is the best aligned off2 sweep point and also supplies the
    # validated multi-node runtime/HDFS/W&B configuration.
    payload = source.configure(base, "d5e4_2e3", state_root=state_root, source=source._load_source_recipe())
    trainer = payload["trainer"]
    trainer["experiment_name"] = run_id
    trainer["default_hdfs_dir"] = hdfs_dir
    trainer["default_local_dir"] = str(state_root / "checkpoints")
    trainer["validation_data_dir"] = str(state_root / "validation")
    payload["actor_rollout_ref"]["_checkpoint_hdfs_dir"] = hdfs_dir
    payload["actor_rollout_ref"]["rollout"]["trace"]["experiment_name"] = run_id

    actor = payload["actor_rollout_ref"]["actor"]
    actor["ppo_mini_batch_size"] = setting["mini_batch"]
    actor["optim"]["lr"] = setting["lr"]
    payload["critic"]["ppo_mini_batch_size"] = setting["mini_batch"]
    actor["policy_loss"]["loss_mode"] = setting["loss"]
    actor["policy_loss"]["prefix_exact_kl_delta_low"] = DELTA_LOW
    actor["policy_loss"]["prefix_exact_kl_delta_high"] = DELTA_HIGH
    actor["megatron"]["router_replay"]["mode"] = setting["router_replay"]
    is_grpo = mode.startswith("grpo_r") or mode.startswith("grpo_nokl_")
    if is_grpo:
        actor["loss_agg_mode"] = "token-mean"
        actor["use_kl_loss"] = False
        actor["kl_loss_coef"] = 0.0
        actor["clip_ratio"] = 0.2
        actor["clip_ratio_low"] = 0.2
        actor["clip_ratio_high"] = 0.28
    rollout = payload["actor_rollout_ref"]["rollout"]
    rollout["enable_rollout_routing_replay"] = mode == "grpo_r3_off8"
    if is_grpo:
        payload["ray_kwargs"]["ray_init"]["runtime_env"]["env_vars"]["WANDB_RUN_GROUP"] = (
            GRPO_OFF8_WANDB_GROUP
        )
    if "seed" in setting:
        seed = setting["seed"]
        payload["data"]["seed"] = seed
        actor["data_loader_seed"] = seed
        actor["megatron"]["seed"] = seed
        payload["actor_rollout_ref"]["ref"]["megatron"]["seed"] = seed
        payload["critic"]["data_loader_seed"] = seed
        payload["critic"]["megatron"]["seed"] = seed
    return payload


def main() -> None:
    from omegaconf import OmegaConf

    if len(sys.argv) != 2 or sys.argv[1] not in SETTINGS:
        raise SystemExit(f"usage: {sys.argv[0]} {'|'.join(SETTINGS)}")
    mode = sys.argv[1]
    setting = SETTINGS[mode]
    source = _load_source_recipe()
    run_id, state_slug, hdfs_dir = identities(mode)
    state = Path("/tmp/offpolicyrl_run_state") / state_slug
    for path in (state, state / "wandb", state / "tensorboard", state / "checkpoints", state / "validation"):
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(0o777)

    base = json.loads(source._load_source_recipe().BASE_CONFIG.read_text())
    payload = configure(base, mode, state_root=state, source=source)
    source._load_source_recipe().load_environment(state, run_id)
    sys.path.insert(0, str(source.VENDOR_DIR))
    os.environ["PYTHONPATH"] = f"{source.VENDOR_DIR}:{source.CODE_ROOT}"
    is_grpo = mode.startswith("grpo_r") or mode.startswith("grpo_nokl_")
    wandb_group = GRPO_OFF8_WANDB_GROUP if is_grpo else WANDB_GROUP
    os.environ["WANDB_RUN_GROUP"] = wandb_group
    source._load_source_recipe().require_official_wandb()
    if mode == "grpo_r3_off8":
        subprocess.run(
            [sys.executable, str(Path(__file__).with_name("common") / "check_r3_vllm.py")],
            check=True,
        )
    resume_step = validate_hdfs_target(
        hdfs_dir,
        allow_resume=os.environ.get("ALLOW_EXISTING_HDFS_RESUME") == "1",
    )
    if resume_step is not None:
        print(f"Validated complete HDFS checkpoint for resume: step {resume_step}", flush=True)

    actor = payload["actor_rollout_ref"]["actor"]
    rollout = payload["actor_rollout_ref"]["rollout"]
    trainer = payload["trainer"]
    assert payload["data"]["train_batch_size"] == 64
    assert actor["ppo_mini_batch_size"] == setting["mini_batch"]
    assert payload["critic"]["ppo_mini_batch_size"] == setting["mini_batch"]
    assert actor["optim"]["lr"] == setting["lr"]
    assert actor["policy_loss"]["loss_mode"] == setting["loss"]
    assert actor["megatron"]["router_replay"]["mode"] == setting["router_replay"]
    assert rollout["enable_rollout_routing_replay"] is (mode == "grpo_r3_off8")
    if is_grpo:
        assert actor["loss_agg_mode"] == "token-mean"
        assert actor["use_kl_loss"] is False and actor["kl_loss_coef"] == 0.0
        assert actor["clip_ratio"] == actor["clip_ratio_low"] == 0.2
        assert actor["clip_ratio_high"] == 0.28
    else:
        assert actor["loss_agg_mode"] == "seq-mean-token-mean"
    assert rollout["over_sample_rate"] == 0.1 and rollout["n"] == 8
    assert trainer["project_name"] == PROJECT
    assert trainer["test_freq"] == 5 and trainer["save_freq"] == 5
    assert trainer["total_training_steps"] == 300
    assert trainer["nnodes"] == 4 and trainer["n_gpus_per_node"] == 8
    assert len(payload["data"]["val_files"]) == 4
    if "seed" in setting:
        seed = setting["seed"]
        assert payload["data"]["seed"] == seed
        assert actor["data_loader_seed"] == actor["megatron"]["seed"] == seed
        assert payload["actor_rollout_ref"]["ref"]["megatron"]["seed"] == seed
        assert payload["critic"]["data_loader_seed"] == payload["critic"]["megatron"]["seed"] == seed
    print(
        "PR2_R2_GEOM_EXACT_MATRIX_CONFIG_AUDIT",
        f"mode={mode}", f"run_id={run_id}", f"wandb_group={wandb_group}",
        f"loss={setting['loss']}", f"router_replay={setting['router_replay']}",
        f"mini_batch={setting['mini_batch']}", f"lr={setting['lr']}",
        f"delta_low={DELTA_LOW}", f"delta_high={DELTA_HIGH}",
        "oversample=0.1", "rollout_n=8", "nodes=4", "gpus_per_node=8",
        f"hdfs={hdfs_dir}", flush=True,
    )

    sys.path.insert(0, os.getcwd())
    from verl.trainer.main_ppo import run_ppo

    run_ppo(OmegaConf.create(payload))


if __name__ == "__main__":
    main()
