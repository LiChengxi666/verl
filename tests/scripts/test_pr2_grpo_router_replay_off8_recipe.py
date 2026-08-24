import importlib.util
import copy
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPE = REPO_ROOT / "training_jobs/scripts/moe_rl/run_pr2_r2_and_geom_exact_aligned_matrix.py"


def _load_recipe():
    spec = importlib.util.spec_from_file_location("pr2_grpo_router_replay_off8", RECIPE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load recipe: {RECIPE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Source:
    @staticmethod
    def _load_source_recipe():
        return None

    @staticmethod
    def configure(base, _group, **_kwargs):
        return copy.deepcopy(base)


def _base_config():
    return {
        "data": {"train_batch_size": 64, "val_files": ["a", "b", "c", "d"]},
        "actor_rollout_ref": {
            "_checkpoint_hdfs_dir": "",
            "actor": {
                "ppo_mini_batch_size": 32,
                "optim": {"lr": 2.0e-6},
                "policy_loss": {"loss_mode": "gspo"},
                "megatron": {"router_replay": {"mode": "disabled"}},
                "loss_agg_mode": "seq-mean-token-mean",
                "clip_ratio": 0.2,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.2,
            },
            "rollout": {
                "enable_rollout_routing_replay": False,
                "over_sample_rate": 0.1,
                "n": 8,
                "val_kwargs": {"n": 8},
                "trace": {"experiment_name": ""},
            },
        },
        "critic": {"ppo_mini_batch_size": 32},
        "trainer": {
            "project_name": "verl_moe_router_replay",
            "experiment_name": "",
            "default_hdfs_dir": "",
            "default_local_dir": "",
            "validation_data_dir": "",
            "test_freq": 5,
            "save_freq": 5,
            "total_training_steps": 300,
            "nnodes": 4,
            "n_gpus_per_node": 8,
        },
        "ray_kwargs": {"ray_init": {"runtime_env": {"env_vars": {}}}},
    }


@pytest.mark.parametrize(
    ("mode", "router_mode", "rollout_replay"),
    [
        ("grpo_r2_off8", "R2", False),
        ("grpo_r3_off8", "R3", True),
    ],
)
def test_grpo_router_replay_off8_matches_pr2_recipe(mode, router_mode, rollout_replay, tmp_path):
    recipe = _load_recipe()
    payload = recipe.configure(_base_config(), mode, state_root=tmp_path, source=_Source())

    actor = payload["actor_rollout_ref"]["actor"]
    rollout = payload["actor_rollout_ref"]["rollout"]
    trainer = payload["trainer"]
    assert payload["data"]["train_batch_size"] == 64
    assert actor["ppo_mini_batch_size"] == 8
    assert payload["critic"]["ppo_mini_batch_size"] == 8
    assert actor["optim"]["lr"] == 1.0e-6
    assert actor["policy_loss"]["loss_mode"] == "vanilla"
    assert actor["loss_agg_mode"] == "token-mean"
    assert actor["clip_ratio"] == 0.2
    assert actor["clip_ratio_low"] == 0.2
    assert actor["clip_ratio_high"] == 0.28
    assert actor["megatron"]["router_replay"]["mode"] == router_mode
    assert rollout["enable_rollout_routing_replay"] is rollout_replay
    assert rollout["over_sample_rate"] == 0.1
    assert rollout["n"] == rollout["val_kwargs"]["n"] == 8
    assert trainer["test_freq"] == trainer["save_freq"] == 5
    assert trainer["total_training_steps"] == 300
    assert trainer["nnodes"] == 4 and trainer["n_gpus_per_node"] == 8
    assert len(payload["data"]["val_files"]) == 4
    assert payload["ray_kwargs"]["ray_init"]["runtime_env"]["env_vars"]["WANDB_RUN_GROUP"] == (
        "PR2_off8_method_comparison_over01"
    )
