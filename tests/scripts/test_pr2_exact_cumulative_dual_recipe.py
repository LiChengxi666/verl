import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPE = REPO_ROOT / "training_jobs/scripts/moe_rl/run_pr2_exact_cumulative_dual_off2_oversample0p1.py"


def _load_recipe_module():
    spec = importlib.util.spec_from_file_location("pr2_exact_cumulative_dual_recipe", RECIPE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recipe_changes_only_exact_dual_method_and_run_identity():
    module = _load_recipe_module()
    base = {
        "data": {"val_files": ["amc23", "aime24", "aime25", "hmmt25"]},
        "actor_rollout_ref": {
            "_checkpoint_hdfs_dir": "old-hdfs",
            "actor": {
                "loss_agg_mode": "seq-mean-token-mean",
                "optim": {"lr": 2e-6},
                "policy_loss": {
                    "loss_mode": "cum-token-cumprod-la",
                    "prefix_exact_kl_delta_low": 0.000309912,
                    "prefix_exact_kl_delta_high": 0.001271096,
                },
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
        },
        "ray_kwargs": {"ray_init": {"runtime_env": {"env_vars": {}}}},
    }

    configured = module.configure_exact_cumulative_dual(base, state_root=Path("/tmp/test-state"))
    actor = configured["actor_rollout_ref"]["actor"]
    policy_loss = actor["policy_loss"]

    assert policy_loss["loss_mode"] == "prefix_exact_kl_cumulative_dual_clip"
    assert policy_loss["prefix_exact_kl_delta_low"] == 3.09912028333e-4
    assert policy_loss["prefix_exact_kl_delta_high"] == 5.17091807565e-3
    assert actor["loss_agg_mode"] == "seq-mean-token-mean"
    assert all(key not in actor for key in ("clip_ratio_la_power", "la_clip_low", "la_clip_high", "la_clip_c"))

    assert configured["actor_rollout_ref"]["rollout"]["over_sample_rate"] == 0.1
    assert configured["actor_rollout_ref"]["rollout"]["n"] == 8
    assert actor["optim"]["lr"] == 2e-6
    assert configured["data"]["val_files"] == ["amc23", "aime24", "aime25", "hmmt25"]
    assert configured["trainer"]["project_name"] == "verl_moe_router_replay"
    assert configured["trainer"]["test_freq"] == 5
    assert configured["trainer"]["save_freq"] == 5
    assert configured["trainer"]["experiment_name"] == module.RUN_ID
    assert configured["trainer"]["default_hdfs_dir"] == module.HDFS_DIR
