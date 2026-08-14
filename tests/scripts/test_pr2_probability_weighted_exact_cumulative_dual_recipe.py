import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPE = REPO_ROOT / "training_jobs/scripts/moe_rl/run_pr2_probability_weighted_exact_cumulative_dual_over01.py"
SOURCE_RECIPE = REPO_ROOT / "training_jobs/scripts/moe_rl/run_pr2_probability_weighted_exact_geom_dual_over01.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recipe_only_switches_surrogate_and_dual_clip_mode():
    recipe = _load(RECIPE, "cumulative_recipe")
    source = _load(SOURCE_RECIPE, "geometric_recipe")
    base = {
        "data": {"val_files": ["amc23", "aime24", "aime25", "hmmt25"]},
        "actor_rollout_ref": {
            "_checkpoint_hdfs_dir": "old-hdfs",
            "actor": {
                "loss_agg_mode": "seq-mean-token-mean",
                "optim": {"lr": 2e-6},
                "policy_loss": {},
                "clip_ratio_la_power": 0.5,
                "la_clip_low": 0.025,
                "la_clip_high": 0.05,
                "la_clip_c": 0.05,
            },
            "rollout": {
                "n": 8,
                "over_sample_rate": 0.1,
                "trace": {"experiment_name": "old-run"},
            },
        },
        "trainer": {
            "experiment_name": "old-run",
            "project_name": "verl_moe_router_replay",
            "default_hdfs_dir": "old-hdfs",
            "test_freq": 5,
            "save_freq": 5,
            "total_training_steps": 300,
        },
        "ray_kwargs": {"ray_init": {"runtime_env": {"env_vars": {}}}},
    }

    configured = recipe.configure(base, state_root=Path("/tmp/test-state"), source=source)
    actor = configured["actor_rollout_ref"]["actor"]
    policy_loss = actor["policy_loss"]
    assert policy_loss["loss_mode"] == recipe.LOSS_MODE
    assert policy_loss["prefix_exact_kl_delta_low"] == 7.780049e-8
    assert policy_loss["prefix_exact_kl_delta_high"] == 6.302410000000001e-7
    assert actor["loss_agg_mode"] == "seq-mean-token-mean"
    assert configured["actor_rollout_ref"]["rollout"]["over_sample_rate"] == 0.1
    assert configured["actor_rollout_ref"]["rollout"]["n"] == 8
    assert actor["optim"]["lr"] == 2e-6
    assert configured["trainer"]["test_freq"] == 5
    assert configured["trainer"]["save_freq"] == 5
    assert configured["trainer"]["total_training_steps"] == 300
    assert len(configured["data"]["val_files"]) == 4
