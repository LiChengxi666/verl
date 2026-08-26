"""Run the geometric-prefix-probability exact-KL budget sweep.

The training/data/validation/runtime settings are inherited from the already
validated aligned PR2 off-policy-2 recipe.  Only the policy-loss mode and its
low/high exact-KL budgets are changed.  Four explicit runs are used instead
of nested W&B sweep agents so each trainer owns exactly one W&B run.
"""

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
SOURCE_RECIPE = Path(__file__).with_name("run_pr2_probability_weighted_exact_geom_dual_over01.py")
LOSS_MODE = "prefix_geometric_probability_weighted_exact_kl_clip"
SWEEP_GROUP = "pr2_geom_prefix_probability_exact_wide_budget_sweep_20260818"

# The first geometric-probability sweep clipped 42--49% of tokens.  The first
# wide-budget follow-up established useful anchors at 1e-4 and 1e-3, while
# 1e-2 and 1e-1 were effectively unclipped.  Replace those two loose points
# with geometric midpoints; each upper budget remains 4x its lower budget.
GROUPS = {
    "d1e4_4e4": (1.0e-4, 4.0e-4),
    "d5e4_2e3": (5.0e-4, 2.0e-3),
    "d1e3_4e3": (1.0e-3, 4.0e-3),
    "d5e3_2e2": (5.0e-3, 2.0e-2),
}


def _load_source_recipe():
    spec = importlib.util.spec_from_file_location("pr2_pwe_geom_recipe", SOURCE_RECIPE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load source recipe: {SOURCE_RECIPE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identities(group: str) -> tuple[str, str, str]:
    if group not in GROUPS:
        raise ValueError(f"unknown group {group!r}; choose one of {sorted(GROUPS)}")
    run_id = f"pr2_geomprob_exact_{group}_o2_over01"
    state_slug = f"pr2_geomprob_exact_wide_{group}_off2_over01_moe32_20260818"
    hdfs_dir = (
        "hdfs://harunawl/home/byte_data_seed_wl/user/wu.hanlin/offpolicyrl/checkpoints/"
        f"moe-pr2-geomprob-exact-wide-{group}-off2-over01-r16384-4x8-20260818"
    )
    return run_id, state_slug, hdfs_dir


def configure(base: dict, group: str, *, state_root: Path, source) -> dict:
    run_id, _, hdfs_dir = identities(group)
    payload = source.configure(base, "0p5x", state_root=state_root, hdfs_dir=hdfs_dir, run_id=run_id)
    low, high = GROUPS[group]
    policy_loss = payload["actor_rollout_ref"]["actor"]["policy_loss"]
    policy_loss["loss_mode"] = LOSS_MODE
    policy_loss["prefix_exact_kl_delta_low"] = low
    policy_loss["prefix_exact_kl_delta_high"] = high
    runtime_env = payload["ray_kwargs"]["ray_init"]["runtime_env"]
    # Follow the established multi-node launch SOP: ship the current code
    # snapshot with Ray so every worker imports the same registered loss.
    runtime_env["working_dir"] = os.getcwd()
    runtime_env["excludes"] = [
        "/.git/",
        "/.venv/",
        "/docs/",
        "/tests/",
        "/.github/",
        "/examples/",
        "/.ray_vendor/",
    ]
    runtime_env["env_vars"]["PYTHONPATH"] = f".:{VENDOR_DIR}:{CODE_ROOT}"
    return payload


def main() -> None:
    from omegaconf import OmegaConf

    if len(sys.argv) != 2 or sys.argv[1] not in GROUPS:
        raise SystemExit(f"usage: {sys.argv[0]} {'|'.join(GROUPS)}")
    group = sys.argv[1]
    source = _load_source_recipe()
    run_id, state_slug, hdfs_dir = identities(group)
    state = Path("/tmp/offpolicyrl_run_state") / state_slug
    for path in (state, state / "wandb", state / "tensorboard", state / "checkpoints", state / "validation"):
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(0o777)

    base = json.loads(source.BASE_CONFIG.read_text())
    payload = configure(base, group, state_root=state, source=source)
    source.load_environment(state, run_id)
    os.environ["PYTHONPATH"] = f"{VENDOR_DIR}:{CODE_ROOT}"
    sys.path.insert(0, str(VENDOR_DIR))
    os.environ["WANDB_RUN_GROUP"] = SWEEP_GROUP
    source.require_official_wandb()

    if subprocess.run(
        ["hdfs", "dfs", "-test", "-e", hdfs_dir], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode == 0:
        raise RuntimeError(f"Refusing to reuse existing HDFS path: {hdfs_dir}")

    actor = payload["actor_rollout_ref"]["actor"]
    rollout = payload["actor_rollout_ref"]["rollout"]
    trainer = payload["trainer"]
    low, high = GROUPS[group]
    assert payload["data"]["train_batch_size"] == 64
    assert actor["policy_loss"]["loss_mode"] == LOSS_MODE
    assert actor["policy_loss"]["prefix_exact_kl_delta_low"] == low
    assert actor["policy_loss"]["prefix_exact_kl_delta_high"] == high
    assert actor["ppo_mini_batch_size"] == 32
    assert actor["optim"]["lr"] == 2e-6
    assert actor["loss_agg_mode"] == "seq-mean-token-mean"
    assert rollout["over_sample_rate"] == 0.1 and rollout["n"] == 8
    assert trainer["test_freq"] == 5 and trainer["save_freq"] == 5
    assert trainer["total_training_steps"] == 300
    assert trainer["nnodes"] == 4 and trainer["n_gpus_per_node"] == 8
    assert len(payload["data"]["val_files"]) == 4
    print(
        "GEOM_PREFIX_PROB_EXACT_SWEEP_CONFIG_AUDIT",
        f"group={group}", f"run_id={run_id}", f"wandb_group={SWEEP_GROUP}",
        f"loss={LOSS_MODE}", f"delta_low={low}", f"delta_high={high}",
        "offpolicy=2", "oversample=0.1", "rollout_n=8", "lr=2e-6",
        "nodes=4", "gpus_per_node=8", f"hdfs={hdfs_dir}", flush=True,
    )

    sys.path.insert(0, os.getcwd())
    from verl.trainer.main_ppo import run_ppo

    run_ppo(OmegaConf.create(payload))


if __name__ == "__main__":
    main()
