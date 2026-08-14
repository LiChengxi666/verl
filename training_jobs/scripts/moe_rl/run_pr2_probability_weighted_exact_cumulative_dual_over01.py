"""Run the 0.5x probability-weighted exact-prefix loss with raw prefix ratios.

This is intentionally derived from ``pr2_pwe_g_0p5x_o2_over01``.  It keeps
the aligned PR2 off-policy-2 recipe and the exact same probability-weighted
budgets, but replaces the geometric-average surrogate with the cumulative
prefix ratio and enables the exact upper-bound dual clip for negative
advantages.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(os.environ.get("OFFPOLICYRL_ROOT", "/opt/tiger/offpolicyrl"))
SOURCE_RECIPE = Path(__file__).with_name("run_pr2_probability_weighted_exact_geom_dual_over01.py")
RUN_ID = "pr2_pwe_cumulative_dual_0p5x_o2_over01"
STATE_SLUG = "pr2_prefix_prob_exact_cumulative_dual_0p5x_upper2x_off2_over01_moe32_20260815"
HDFS_DIR = (
    "hdfs://harunawl/home/byte_data_seed_wl/user/wu.hanlin/offpolicyrl/checkpoints/"
    "moe-pr2-prefix-prob-exact-cumulative-dual-0p5x-upper2x-off2-over01-r16384-4x8-20260815"
)
GROUP = "0p5x"
LOSS_MODE = "prefix_probability_weighted_exact_kl_cumulative_dual_clip"


def _load_source_recipe():
    spec = importlib.util.spec_from_file_location("pr2_pwe_geom_recipe", SOURCE_RECIPE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load source recipe: {SOURCE_RECIPE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure(base: dict, *, state_root: Path, source) -> dict:
    payload = source.configure(base, GROUP, state_root=state_root, hdfs_dir=HDFS_DIR, run_id=RUN_ID)
    payload["actor_rollout_ref"]["actor"]["policy_loss"]["loss_mode"] = LOSS_MODE
    return payload


def main() -> None:
    from omegaconf import OmegaConf

    source = _load_source_recipe()
    state = Path("/tmp/offpolicyrl_run_state") / STATE_SLUG
    for path in (state, state / "wandb", state / "tensorboard", state / "checkpoints", state / "validation"):
        path.mkdir(parents=True, exist_ok=True)

    base = json.loads(source.BASE_CONFIG.read_text())
    payload = configure(base, state_root=state, source=source)
    source.load_environment(state, RUN_ID)
    source.require_official_wandb()

    if subprocess.run(
        ["hdfs", "dfs", "-test", "-e", HDFS_DIR], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode == 0:
        raise RuntimeError(f"Refusing to reuse existing HDFS path: {HDFS_DIR}")

    actor = payload["actor_rollout_ref"]["actor"]
    policy_loss = actor["policy_loss"]
    rollout = payload["actor_rollout_ref"]["rollout"]
    trainer = payload["trainer"]
    expected_low, expected_high = source.GROUPS[GROUP]
    assert policy_loss["loss_mode"] == LOSS_MODE
    assert policy_loss["prefix_exact_kl_delta_low"] == expected_low == 7.780049e-8
    assert policy_loss["prefix_exact_kl_delta_high"] == expected_high == 6.302410000000001e-7
    assert actor["loss_agg_mode"] == "seq-mean-token-mean"
    assert rollout["over_sample_rate"] == 0.1 and rollout["n"] == 8
    assert actor["optim"]["lr"] == 2e-6
    assert trainer["test_freq"] == 5 and trainer["save_freq"] == 5
    assert trainer["total_training_steps"] == 300
    assert len(payload["data"]["val_files"]) == 4
    print(
        "PROB_WEIGHTED_EXACT_CUMULATIVE_DUAL_CONFIG_AUDIT",
        f"run_id={RUN_ID}",
        f"loss={LOSS_MODE}",
        f"delta_low={expected_low}",
        f"delta_high={expected_high}",
        f"oversample={rollout['over_sample_rate']}",
        f"rollout_n={rollout['n']}",
        f"lr={actor['optim']['lr']}",
        f"hdfs={HDFS_DIR}",
        flush=True,
    )

    sys.path.insert(0, os.getcwd())
    from verl.trainer.main_ppo import run_ppo

    run_ppo(OmegaConf.create(payload))


if __name__ == "__main__":
    main()
