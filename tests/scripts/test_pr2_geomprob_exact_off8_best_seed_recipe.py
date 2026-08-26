import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPE = REPO_ROOT / "training_jobs/scripts/moe_rl/run_pr2_geomprob_exact_off8_best_seed.py"


def _load_recipe():
    spec = importlib.util.spec_from_file_location("pr2_geomprob_exact_off8_seed", RECIPE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("seed", [43, 44])
def test_off8_seed_recipe_only_changes_seed_and_offpolicy_schedule(seed):
    recipe = _load_recipe()
    payload = {
        "data": {"seed": 42, "train_batch_size": 64, "val_files": ["amc23", "aime24", "aime25", "hmmt25"]},
        "actor_rollout_ref": {
            "actor": {
                "data_loader_seed": 42,
                "megatron": {"seed": 42},
                "ppo_mini_batch_size": 32,
                "optim": {"lr": 2e-6},
                "policy_loss": {
                    "loss_mode": "prefix_geometric_probability_weighted_exact_kl_clip",
                    "prefix_exact_kl_delta_low": 5e-4,
                    "prefix_exact_kl_delta_high": 2e-3,
                },
            },
            "ref": {"megatron": {"seed": 42}},
            "rollout": {"n": 8, "over_sample_rate": 0.1},
        },
        "critic": {"data_loader_seed": 42, "megatron": {"seed": 42}, "ppo_mini_batch_size": 32},
        "trainer": {"nnodes": 4, "n_gpus_per_node": 8, "test_freq": 5, "save_freq": 5, "total_training_steps": 300},
    }

    recipe.configure_off8_seed(payload, seed)

    assert payload["data"]["seed"] == seed
    assert payload["actor_rollout_ref"]["actor"]["data_loader_seed"] == seed
    assert payload["actor_rollout_ref"]["actor"]["megatron"]["seed"] == seed
    assert payload["actor_rollout_ref"]["ref"]["megatron"]["seed"] == seed
    assert payload["critic"]["data_loader_seed"] == seed
    assert payload["critic"]["megatron"]["seed"] == seed
    assert payload["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] == 8
    assert payload["critic"]["ppo_mini_batch_size"] == 8
    assert payload["actor_rollout_ref"]["actor"]["optim"]["lr"] == 1e-6
    assert payload["actor_rollout_ref"]["rollout"]["over_sample_rate"] == 0.1
    assert payload["actor_rollout_ref"]["actor"]["policy_loss"]["prefix_exact_kl_delta_low"] == 5e-4
    assert payload["actor_rollout_ref"]["actor"]["policy_loss"]["prefix_exact_kl_delta_high"] == 2e-3
