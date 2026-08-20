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
LOSS_MODE = "prefix_geometric_probability_weighted_exact_kl_clip"
DELTA_LOW = 5.0e-4
DELTA_HIGH = 2.0e-3
SETTINGS = {
    "r2_off2": {"mini_batch": 32, "lr": 2.0e-6, "loss": "gspo", "router_replay": "R2"},
    "geom_off4": {"mini_batch": 16, "lr": 1.5e-6, "loss": LOSS_MODE, "router_replay": "disabled"},
    "geom_off8": {"mini_batch": 8, "lr": 1.0e-6, "loss": LOSS_MODE, "router_replay": "disabled"},
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
    }
    run_id = names[mode]
    state_slug = f"{mode}_aligned_moe32_20260820"
    hdfs_dir = (
        "hdfs://harunawl/home/byte_data_seed_wl/user/wu.hanlin/offpolicyrl/checkpoints/"
        f"moe-{mode.replace('_', '-')}-aligned-over01-r16384-4x8-20260820"
    )
    return run_id, state_slug, hdfs_dir


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
    payload["actor_rollout_ref"]["rollout"]["enable_rollout_routing_replay"] = False
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
    os.environ["WANDB_RUN_GROUP"] = WANDB_GROUP
    source._load_source_recipe().require_official_wandb()
    if subprocess.run(
        ["hdfs", "dfs", "-test", "-e", hdfs_dir], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode == 0:
        raise RuntimeError(f"Refusing to reuse existing HDFS path: {hdfs_dir}")

    actor = payload["actor_rollout_ref"]["actor"]
    rollout = payload["actor_rollout_ref"]["rollout"]
    trainer = payload["trainer"]
    assert payload["data"]["train_batch_size"] == 64
    assert actor["ppo_mini_batch_size"] == setting["mini_batch"]
    assert payload["critic"]["ppo_mini_batch_size"] == setting["mini_batch"]
    assert actor["optim"]["lr"] == setting["lr"]
    assert actor["policy_loss"]["loss_mode"] == setting["loss"]
    assert actor["megatron"]["router_replay"]["mode"] == setting["router_replay"]
    assert rollout["enable_rollout_routing_replay"] is False
    assert actor["loss_agg_mode"] == "seq-mean-token-mean"
    assert rollout["over_sample_rate"] == 0.1 and rollout["n"] == 8
    assert trainer["project_name"] == PROJECT
    assert trainer["test_freq"] == 5 and trainer["save_freq"] == 5
    assert trainer["total_training_steps"] == 300
    assert trainer["nnodes"] == 4 and trainer["n_gpus_per_node"] == 8
    assert len(payload["data"]["val_files"]) == 4
    print(
        "PR2_R2_GEOM_EXACT_MATRIX_CONFIG_AUDIT",
        f"mode={mode}", f"run_id={run_id}", f"wandb_group={WANDB_GROUP}",
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
