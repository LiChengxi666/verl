"""Replicate the aligned off8 geometric-probability exact-prefix run by seed."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(os.environ.get("OFFPOLICYRL_ROOT", "/opt/tiger/offpolicyrl"))
CODE_ROOT = ROOT / "verl"
VENDOR_DIR = CODE_ROOT / ".ray_vendor"
MATRIX_RECIPE = Path(__file__).with_name("run_pr2_r2_and_geom_exact_aligned_matrix.py")
WANDB_GROUP = "pr2_best_exact_prefix_off8_seed_replication_20260826"
SEEDS = {43, 44}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load recipe: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_off8_seed(payload: dict, seed: int) -> None:
    """Apply only the validated PR2 off8 schedule and RNG seed."""
    payload["data"]["seed"] = seed
    actor_rollout_ref = payload["actor_rollout_ref"]
    actor = actor_rollout_ref["actor"]
    actor["data_loader_seed"] = seed
    actor["megatron"]["seed"] = seed
    actor["ppo_mini_batch_size"] = 8
    actor["optim"]["lr"] = 1.0e-6
    actor_rollout_ref["ref"]["megatron"]["seed"] = seed
    payload["critic"]["data_loader_seed"] = seed
    payload["critic"]["megatron"]["seed"] = seed
    payload["critic"]["ppo_mini_batch_size"] = 8


def main() -> None:
    from omegaconf import OmegaConf

    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} SEED, SEED in {sorted(SEEDS)}")
    seed = int(sys.argv[1])
    if seed not in SEEDS:
        raise SystemExit(f"seed must be one of {sorted(SEEDS)}")

    matrix = _load(MATRIX_RECIPE, "pr2_geomprob_exact_off8_matrix")
    source = matrix._load_source_recipe()
    base_source = source._load_source_recipe()
    run_id = f"pr2_geomprob_exact_d5e4_2e3_o8_over01_seed{seed}"
    state_slug = f"pr2_geomprob_exact_best_d5e4_2e3_off8_over01_seed{seed}_moe32_20260826"
    hdfs_dir = (
        "hdfs://harunawl/home/byte_data_seed_wl/user/wu.hanlin/offpolicyrl/checkpoints/"
        f"moe-pr2-geomprob-exact-best-d5e4-2e3-off8-over01-seed{seed}-r16384-4x8-20260826"
    )
    state = Path("/tmp/offpolicyrl_run_state") / state_slug
    for path in (state, state / "wandb", state / "tensorboard", state / "checkpoints", state / "validation"):
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(0o777)

    base = json.loads(base_source.BASE_CONFIG.read_text())
    payload = matrix.configure(base, "geom_off8", state_root=state, source=source)
    trainer = payload["trainer"]
    trainer["experiment_name"] = run_id
    trainer["default_hdfs_dir"] = hdfs_dir
    trainer["default_local_dir"] = str(state / "checkpoints")
    trainer["validation_data_dir"] = str(state / "validation")
    payload["actor_rollout_ref"]["_checkpoint_hdfs_dir"] = hdfs_dir
    payload["actor_rollout_ref"]["rollout"]["trace"]["experiment_name"] = run_id
    configure_off8_seed(payload, seed)

    runtime_env = payload.setdefault("ray_kwargs", {}).setdefault("ray_init", {}).setdefault("runtime_env", {})
    runtime_env.setdefault("env_vars", {})["WANDB_RUN_GROUP"] = WANDB_GROUP
    base_source.load_environment(state, run_id)
    os.environ["PYTHONPATH"] = f"{VENDOR_DIR}:{CODE_ROOT}"
    sys.path.insert(0, str(VENDOR_DIR))
    os.environ["WANDB_RUN_GROUP"] = WANDB_GROUP
    base_source.require_official_wandb()

    if subprocess.run(
        ["hdfs", "dfs", "-test", "-e", hdfs_dir], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode == 0:
        raise RuntimeError(f"Refusing to reuse existing HDFS path: {hdfs_dir}")

    actor = payload["actor_rollout_ref"]["actor"]
    rollout = payload["actor_rollout_ref"]["rollout"]
    assert payload["data"]["train_batch_size"] == 64
    assert actor["policy_loss"]["loss_mode"] == matrix.LOSS_MODE
    assert actor["policy_loss"]["prefix_exact_kl_delta_low"] == matrix.DELTA_LOW == 5.0e-4
    assert actor["policy_loss"]["prefix_exact_kl_delta_high"] == matrix.DELTA_HIGH == 2.0e-3
    assert actor["ppo_mini_batch_size"] == 8 and payload["critic"]["ppo_mini_batch_size"] == 8
    assert actor["optim"]["lr"] == 1.0e-6
    assert actor["megatron"]["router_replay"]["mode"] == "disabled"
    assert rollout["over_sample_rate"] == 0.1 and rollout["n"] == 8
    assert trainer["test_freq"] == 5 and trainer["save_freq"] == 5
    assert trainer["total_training_steps"] == 300
    assert trainer["nnodes"] == 4 and trainer["n_gpus_per_node"] == 8
    assert len(payload["data"]["val_files"]) == 4
    assert payload["data"]["seed"] == seed
    print(
        "BEST_EXACT_PREFIX_OFF8_SEED_AUDIT",
        f"seed={seed}", f"run_id={run_id}", f"wandb_group={WANDB_GROUP}",
        "delta_low=0.0005", "delta_high=0.002", "offpolicy=8", "oversample=0.1",
        "rollout_n=8", "lr=1e-6", "mini_batch=8", "nodes=4", "gpus_per_node=8",
        f"hdfs={hdfs_dir}", flush=True,
    )

    sys.path.insert(0, str(CODE_ROOT))
    from verl.trainer.main_ppo import run_ppo

    run_ppo(OmegaConf.create(payload))


if __name__ == "__main__":
    main()
