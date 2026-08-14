"""Run oversampling-aligned PR2 GSPO baselines for off-policy 4 or 8."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(os.environ.get("OFFPOLICYRL_ROOT", "/opt/tiger/offpolicyrl"))
SOURCE_RECIPE = Path(__file__).with_name("run_pr2_probability_weighted_exact_geom_dual_over01.py")
PROJECT = "verl_moe_router_replay"
SETTINGS = {
    "off4": {"mini_batch": 16, "lr": 1.5e-6},
    "off8": {"mini_batch": 8, "lr": 1.0e-6},
}


def _load_source_recipe():
    spec = importlib.util.spec_from_file_location("pr2_pwe_geom_recipe", SOURCE_RECIPE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load source recipe: {SOURCE_RECIPE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identities(mode: str) -> tuple[str, str, str]:
    if mode not in SETTINGS:
        raise ValueError(f"unknown mode {mode!r}; choose one of {sorted(SETTINGS)}")
    suffix = mode.removeprefix("off")
    lr_slug = "1p5e-6" if mode == "off4" else "1e-6"
    run_id = f"PR2_GSPO_baseline_{mode}_oversample0p1_qwen3_30b_a3b_4x8_b64n8_r16384_lr{lr_slug}_300"
    state_slug = f"pr2_gspo_baseline_{mode}_over01_moe32_20260815"
    hdfs_dir = (
        "hdfs://harunawl/home/byte_data_seed_wl/user/wu.hanlin/offpolicyrl/checkpoints/"
        f"moe-pr2-gspo-baseline-off{suffix}-over01-r16384-4x8-20260815"
    )
    return run_id, state_slug, hdfs_dir


def configure(base: dict, mode: str, *, state_root: Path, source) -> dict:
    setting = SETTINGS[mode]
    run_id, _, hdfs_dir = identities(mode)
    # Reuse the already validated aligned-off2 runtime/HDFS/W&B setup, then
    # restore the GSPO baseline loss and the paper's off-k optimizer schedule.
    payload = source.configure(base, "0p5x", state_root=state_root, hdfs_dir=hdfs_dir, run_id=run_id)
    actor = payload["actor_rollout_ref"]["actor"]
    policy_loss = actor["policy_loss"]
    policy_loss["loss_mode"] = "gspo"
    policy_loss["prefix_exact_kl_delta_low"] = 0.02
    policy_loss["prefix_exact_kl_delta_high"] = 0.05
    actor["ppo_mini_batch_size"] = setting["mini_batch"]
    actor["optim"]["lr"] = setting["lr"]
    payload["critic"]["ppo_mini_batch_size"] = setting["mini_batch"]
    return payload


def main() -> None:
    from omegaconf import OmegaConf

    if len(sys.argv) != 2 or sys.argv[1] not in SETTINGS:
        raise SystemExit(f"usage: {sys.argv[0]} {'|'.join(SETTINGS)}")
    mode = sys.argv[1]
    source = _load_source_recipe()
    run_id, state_slug, hdfs_dir = identities(mode)
    state = Path("/tmp/offpolicyrl_run_state") / state_slug
    for path in (state, state / "wandb", state / "tensorboard", state / "checkpoints", state / "validation"):
        path.mkdir(parents=True, exist_ok=True)

    base = json.loads(source.BASE_CONFIG.read_text())
    payload = configure(base, mode, state_root=state, source=source)
    source.load_environment(state, run_id)
    source.require_official_wandb()
    if subprocess.run(
        ["hdfs", "dfs", "-test", "-e", hdfs_dir], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode == 0:
        raise RuntimeError(f"Refusing to reuse existing HDFS path: {hdfs_dir}")

    actor = payload["actor_rollout_ref"]["actor"]
    rollout = payload["actor_rollout_ref"]["rollout"]
    trainer = payload["trainer"]
    expected = SETTINGS[mode]
    assert payload["data"]["train_batch_size"] == 64
    assert actor["ppo_mini_batch_size"] == expected["mini_batch"]
    assert payload["critic"]["ppo_mini_batch_size"] == expected["mini_batch"]
    assert actor["optim"]["lr"] == expected["lr"]
    assert actor["policy_loss"]["loss_mode"] == "gspo"
    assert actor["loss_agg_mode"] == "seq-mean-token-mean"
    assert rollout["over_sample_rate"] == 0.1 and rollout["n"] == 8
    assert trainer["project_name"] == PROJECT
    assert trainer["test_freq"] == 5 and trainer["save_freq"] == 5
    assert trainer["total_training_steps"] == 300
    assert len(payload["data"]["val_files"]) == 4
    print(
        "PR2_GSPO_ALIGNED_CONFIG_AUDIT",
        f"mode={mode}", f"run_id={run_id}",
        f"train_batch=64", f"mini_batch={expected['mini_batch']}", f"lr={expected['lr']}",
        "loss=gspo", f"oversample={rollout['over_sample_rate']}", "rollout_n=8",
        f"hdfs={hdfs_dir}", flush=True,
    )

    sys.path.insert(0, os.getcwd())
    from verl.trainer.main_ppo import run_ppo

    run_ppo(OmegaConf.create(payload))


if __name__ == "__main__":
    main()
