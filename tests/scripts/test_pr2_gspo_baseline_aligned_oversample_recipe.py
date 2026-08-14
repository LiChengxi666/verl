import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPE = REPO_ROOT / "training_jobs/scripts/moe_rl/run_pr2_gspo_baseline_aligned_oversample.py"
SOURCE_RECIPE = REPO_ROOT / "training_jobs/scripts/moe_rl/run_pr2_probability_weighted_exact_geom_dual_over01.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mode,mini_batch,lr", [("off4", 16, 1.5e-6), ("off8", 8, 1.0e-6)])
def test_aligned_recipe_changes_only_offpolicy_schedule_and_oversampling(mode, mini_batch, lr):
    recipe = _load(RECIPE, "aligned_gspo_recipe")
    source = _load(SOURCE_RECIPE, "geometric_recipe")
    base = {
        "data": {"train_batch_size": 64, "val_files": ["amc23", "aime24", "aime25", "hmmt25"]},
        "actor_rollout_ref": {
            "_checkpoint_hdfs_dir": "old-hdfs",
            "actor": {
                "loss_agg_mode": "seq-mean-token-mean",
                "ppo_mini_batch_size": 32,
                "optim": {"lr": 2e-6},
                "policy_loss": {"loss_mode": "cum-token-cumprod-la"},
                "clip_ratio_la_power": 0.5,
                "la_clip_low": 0.025,
                "la_clip_high": 0.05,
                "la_clip_c": 0.05,
            },
            "rollout": {"n": 8, "over_sample_rate": 0.1, "trace": {"experiment_name": "old-run"}},
        },
        "critic": {"ppo_mini_batch_size": 32},
        "trainer": {
            "experiment_name": "old-run", "project_name": "verl_moe_router_replay",
            "default_hdfs_dir": "old-hdfs", "test_freq": 5, "save_freq": 5,
            "total_training_steps": 300,
        },
        "ray_kwargs": {"ray_init": {"runtime_env": {"env_vars": {}}}},
    }

    configured = recipe.configure(base, mode, state_root=Path("/tmp/test-state"), source=source)
    actor = configured["actor_rollout_ref"]["actor"]
    assert configured["data"]["train_batch_size"] == 64
    assert actor["ppo_mini_batch_size"] == mini_batch
    assert configured["critic"]["ppo_mini_batch_size"] == mini_batch
    assert actor["optim"]["lr"] == lr
    assert actor["policy_loss"]["loss_mode"] == "gspo"
    assert actor["policy_loss"]["prefix_exact_kl_delta_low"] == 0.02
    assert actor["policy_loss"]["prefix_exact_kl_delta_high"] == 0.05
    assert configured["actor_rollout_ref"]["rollout"]["over_sample_rate"] == 0.1
    assert configured["actor_rollout_ref"]["rollout"]["n"] == 8
    assert configured["trainer"]["test_freq"] == 5
    assert configured["trainer"]["save_freq"] == 5
    assert configured["trainer"]["total_training_steps"] == 300
    assert len(configured["data"]["val_files"]) == 4
