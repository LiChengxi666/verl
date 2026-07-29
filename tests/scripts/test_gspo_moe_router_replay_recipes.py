# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import importlib.util
import os
from pathlib import Path
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPE_ROOT = REPO_ROOT / "training_jobs/scripts/moe_rl"
COMMON = RECIPE_ROOT / "common/launch_qwen3_30b_a3b.sh"
R3_PREFLIGHT = RECIPE_ROOT / "common/check_r3_vllm.py"
WRAPPERS = {
    "disabled": RECIPE_ROOT / "run_gspo_off2.sh",
    "R2": RECIPE_ROOT / "run_gspo_r2_off2.sh",
    "R3": RECIPE_ROOT / "run_gspo_r3_off2.sh",
}
RUNBOOK = REPO_ROOT / "training_jobs/configs/gspo_moe_router_replay/README.md"


def _run_recipe(mode: str, **overrides: str) -> tuple[subprocess.CompletedProcess[str], dict]:
    env = {
        **os.environ,
        "RECIPE_DRY_RUN": "1",
        **overrides,
    }
    result = subprocess.run(
        ["bash", str(WRAPPERS[mode])],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    config = {}
    for line in result.stdout.splitlines():
        if line.startswith("RECIPE_CONFIG_JSON="):
            config = json.loads(line.removeprefix("RECIPE_CONFIG_JSON="))
            break
    return result, config


@pytest.mark.parametrize(
    ("mode", "wrapper_name"),
    [
        ("disabled", "run_gspo_off2.sh"),
        ("R2", "run_gspo_r2_off2.sh"),
        ("R3", "run_gspo_r3_off2.sh"),
    ],
)
def test_thin_entry_scripts_only_select_policy_and_router_mode(mode: str, wrapper_name: str):
    wrapper = RECIPE_ROOT / wrapper_name
    contents = wrapper.read_text()

    assert contents.count("export POLICY_LOSS_MODE=gspo") == 1
    assert contents.count(f"export ROUTER_REPLAY_MODE={mode}") == 1
    assert 'exec bash "${SCRIPT_DIR}/common/launch_qwen3_30b_a3b.sh" "$@"' in contents


def test_all_recipe_shell_scripts_have_valid_syntax():
    for script in [COMMON, *WRAPPERS.values()]:
        subprocess.run(["bash", "-n", str(script)], cwd=REPO_ROOT, check=True)


def test_r3_vllm_capability_check_uses_async_engine_arguments():
    spec = importlib.util.spec_from_file_location("check_r3_vllm", R3_PREFLIGHT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class SupportedArgs:
        __dataclass_fields__ = {"enable_return_routed_experts": object()}

    class UnsupportedArgs:
        __dataclass_fields__ = {}

    assert module.supports_r3_router_replay(SupportedArgs)
    assert not module.supports_r3_router_replay(UnsupportedArgs)


def test_multinode_lifecycle_does_not_reuse_stale_ready_markers():
    contents = COMMON.read_text()

    assert "RAY_READY_FILE" not in contents
    assert "Timed out joining Ray head" in contents
    assert 'RAY_DONE_TIMEOUT_SECONDS="${RAY_DONE_TIMEOUT_SECONDS:-604800}"' in contents
    assert "Timed out waiting for the head worker to finish" in contents


@pytest.mark.parametrize(
    ("off_policy_k", "mini_batch_size", "actor_lr"),
    [
        (2, 32, "2e-6"),
        (4, 16, "1.5e-6"),
        (8, 8, "1e-6"),
    ],
)
def test_off_policy_strength_derives_update_batch_and_paper_lr(
    off_policy_k: int,
    mini_batch_size: int,
    actor_lr: str,
):
    result, config = _run_recipe("disabled", OFF_POLICY_K=str(off_policy_k))

    assert result.returncode == 0, result.stderr
    assert config["off_policy_k"] == off_policy_k
    assert config["train_batch_size"] == 64
    assert config["mini_batch_size"] == mini_batch_size
    assert config["actor_lr"] == actor_lr
    assert config["ppo_epochs"] == 1


def test_off_policy_defaults_to_two():
    result, config = _run_recipe("disabled")

    assert result.returncode == 0, result.stderr
    assert config["off_policy_k"] == 2
    assert config["mini_batch_size"] == 32
    assert config["actor_lr"] == "2e-6"


def test_explicit_actor_lr_overrides_paper_default():
    result, config = _run_recipe("disabled", OFF_POLICY_K="8", ACTOR_LR="7e-7")

    assert result.returncode == 0, result.stderr
    assert config["actor_lr"] == "7e-7"


@pytest.mark.parametrize(
    ("mode", "rollout_replay"),
    [
        ("disabled", False),
        ("R2", False),
        ("R3", True),
    ],
)
def test_router_mode_controls_only_router_replay(mode: str, rollout_replay: bool):
    result, config = _run_recipe(mode)

    assert result.returncode == 0, result.stderr
    assert config["policy_loss_mode"] == "gspo"
    assert config["router_replay_mode"] == mode
    assert config["rollout_routing_replay"] is rollout_replay


def test_router_variants_keep_shared_hydra_arguments_identical():
    configs = {}
    for mode in WRAPPERS:
        result, config = _run_recipe(mode)
        assert result.returncode == 0, result.stderr
        configs[mode] = config

    assert configs["disabled"]["shared_hydra_args"] == configs["R2"]["shared_hydra_args"]
    assert configs["disabled"]["shared_hydra_args"] == configs["R3"]["shared_hydra_args"]


def test_dry_run_keeps_external_paths_relative_to_repository():
    result, config = _run_recipe("disabled")

    assert result.returncode == 0, result.stderr
    rendered = "\n".join(config["all_hydra_args"])
    assert str(REPO_ROOT) not in rendered
    assert "actor_rollout_ref.model.path=./models/Qwen3-30B-A3B-Base" in rendered
    assert "data.train_files=./data/data_processed/math-17k.parquet" in rendered


@pytest.mark.parametrize("mode", ["disabled", "R2", "R3"])
def test_hydra_accepts_each_router_recipe(mode: str):
    result, config = _run_recipe(mode)
    assert result.returncode == 0, result.stderr

    compose = subprocess.run(
        [
            "python",
            "-m",
            "verl.trainer.main_ppo",
            "--cfg",
            "job",
            "model_engine=megatron",
            *config["all_hydra_args"],
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert compose.returncode == 0, compose.stderr


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"OFF_POLICY_K": "3"}, "OFF_POLICY_K must be one of: 2, 4, 8"),
        (
            {"OFF_POLICY_K": "4", "TRAIN_BATCH_SIZE": "66"},
            "TRAIN_BATCH_SIZE must be divisible by OFF_POLICY_K",
        ),
    ],
)
def test_invalid_off_policy_configuration_fails_before_training(overrides: dict[str, str], message: str):
    result, config = _run_recipe("disabled", **overrides)

    assert result.returncode == 2
    assert not config
    assert message in result.stderr


def test_shared_launcher_contains_portable_scientific_contract():
    contents = COMMON.read_text()

    expected_fragments = [
        'MODEL_PATH="${MODEL_PATH:-./models/Qwen3-30B-A3B-Base}"',
        'TRAIN_FILE="${TRAIN_FILE:-./data/data_processed/math-17k.parquet}"',
        "data/data_processed/moe_eval/minpro/amc23.parquet",
        "data/data_processed/moe_eval/minpro/aime24.parquet",
        "data/data_processed/moe_eval/minpro/aime25.parquet",
        'OFF_POLICY_K="${OFF_POLICY_K:-2}"',
        'ROLLOUT_N="${ROLLOUT_N:-8}"',
        'VAL_N="${VAL_N:-8}"',
        'TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-200}"',
        'TEST_FREQ="${TEST_FREQ:-5}"',
        'SAVE_FREQ="${SAVE_FREQ:-5}"',
        "actor_rollout_ref.actor.policy_loss.loss_mode=${POLICY_LOSS_MODE}",
        "actor_rollout_ref.actor.clip_ratio_low=0.0003",
        "actor_rollout_ref.actor.clip_ratio_high=0.0004",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.ppo_epochs=1",
        "trainer.rollout_buffer.enable=False",
        "trainer.test_freq=${TEST_FREQ}",
        "trainer.save_freq=${SAVE_FREQ}",
        "trainer.resume_mode=auto",
        "actor_rollout_ref.actor.strategy=megatron",
        "actor_rollout_ref.actor.megatron.router_replay.mode=${ROUTER_REPLAY_MODE}",
        (
            "actor_rollout_ref.rollout.enable_rollout_routing_replay="
            "${ENABLE_ROLLOUT_ROUTING_REPLAY}"
        ),
    ]
    for fragment in expected_fragments:
        assert fragment in contents

    forbidden_fragments = [
        "/GenSIvePFS/",
        "gensi-cn-beijing",
        "ResourceQueue",
        "WANDB_API_KEY=",
    ]
    for fragment in forbidden_fragments:
        assert fragment not in contents


def test_external_runbook_documents_matrix_and_contains_no_internal_credentials():
    contents = RUNBOOK.read_text()

    expected_fragments = [
        "run_gspo_off2.sh",
        "run_gspo_r2_off2.sh",
        "run_gspo_r3_off2.sh",
        "OFF_POLICY_K=2",
        "OFF_POLICY_K=4",
        "OFF_POLICY_K=8",
        "WANDB_API_KEY_FILE",
        "trainer.resume_mode=auto",
        "AMC23",
        "AIME24",
        "AIME25",
    ]
    for fragment in expected_fragments:
        assert fragment in contents

    forbidden_fragments = [
        "/GenSIvePFS/",
        "gensi-cn-beijing",
        "ResourceQueue",
        "q-202",
        "WANDB_API_KEY=",
    ]
    for fragment in forbidden_fragments:
        assert fragment not in contents
