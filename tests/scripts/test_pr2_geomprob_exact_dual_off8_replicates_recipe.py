import copy
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPE = REPO_ROOT / "training_jobs/scripts/moe_rl/run_pr2_geomprob_exact_dual_off8_replicates.py"


def _load_recipe():
    spec = importlib.util.spec_from_file_location("pr2_geomprob_exact_dual_off8_replicates", RECIPE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payload():
    return {
        "data": {},
        "actor_rollout_ref": {
            "actor": {
                "data_loader_seed": 42,
                "megatron": {"seed": 42},
                "policy_loss": {
                    "loss_mode": "prefix_geometric_probability_weighted_exact_kl_clip",
                    "prefix_exact_kl_delta_low": 5e-4,
                    "prefix_exact_kl_delta_high": 2e-3,
                },
            },
            "ref": {"megatron": {"seed": 42}},
        },
        "critic": {"data_loader_seed": 42, "megatron": {"seed": 42}},
    }


def test_seed42_replica_only_changes_loss_mode():
    recipe = _load_recipe()
    payload = _payload()
    expected = copy.deepcopy(payload)
    expected["actor_rollout_ref"]["actor"]["policy_loss"]["loss_mode"] = recipe.LOSS_MODE

    recipe.configure_dual_clip_replica(payload, 42)

    assert payload == expected


@pytest.mark.parametrize("seed", [43, 44])
def test_seed_replica_changes_only_loss_mode_and_existing_seed_fields(seed):
    recipe = _load_recipe()
    payload = _payload()

    recipe.configure_dual_clip_replica(payload, seed)

    assert payload["actor_rollout_ref"]["actor"]["policy_loss"]["loss_mode"] == recipe.LOSS_MODE
    assert payload["actor_rollout_ref"]["actor"]["policy_loss"]["prefix_exact_kl_delta_low"] == 5e-4
    assert payload["actor_rollout_ref"]["actor"]["policy_loss"]["prefix_exact_kl_delta_high"] == 2e-3
    assert payload["data"]["seed"] == seed
    assert payload["actor_rollout_ref"]["actor"]["data_loader_seed"] == seed
    assert payload["actor_rollout_ref"]["actor"]["megatron"]["seed"] == seed
    assert payload["actor_rollout_ref"]["ref"]["megatron"]["seed"] == seed
    assert payload["critic"]["data_loader_seed"] == seed
    assert payload["critic"]["megatron"]["seed"] == seed


def test_recipe_rejects_unplanned_seed():
    recipe = _load_recipe()

    with pytest.raises(ValueError, match="seed must be one of"):
        recipe.configure_dual_clip_replica(_payload(), 45)


def test_replicas_join_existing_off8_seed_group():
    recipe = _load_recipe()

    assert recipe.WANDB_GROUP == "pr2_best_exact_prefix_off8_seed_replication_20260826"


def test_reference_replica_audit_accepts_missing_data_seed():
    recipe = _load_recipe()
    payload = _payload()
    recipe.configure_dual_clip_replica(payload, 42)

    recipe.validate_replica_seed_fields(payload, 42, recipe.MISSING_DATA_SEED)


@pytest.mark.parametrize("seed", [43, 44])
def test_seeded_replica_audit_requires_all_existing_seed_fields(seed):
    recipe = _load_recipe()
    payload = _payload()
    recipe.configure_dual_clip_replica(payload, seed)

    recipe.validate_replica_seed_fields(payload, seed, recipe.MISSING_DATA_SEED)


def test_runtime_recipe_dependency_chain_is_loadable():
    recipe = _load_recipe()
    matrix = recipe._load(recipe.MATRIX_RECIPE, "matrix_dependency")
    sweep = matrix._load_source_recipe()

    base_recipe = sweep._load_source_recipe()

    assert base_recipe.BASE_CONFIG.name == "PR2_CTPO_off2_oversample0p1_qwen3_30b_a3b_4x8_b64n8_r16384_300.json"
