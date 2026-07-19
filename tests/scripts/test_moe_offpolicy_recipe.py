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

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = REPO_ROOT / "training_jobs/scripts/run_gspo_qwen3_30b_a3b_smoke.sh"
OFFPOLICY_SCRIPT = REPO_ROOT / "training_jobs/scripts/run_gspo_qwen3_30b_a3b_offpolicy_smoke.sh"
OFFPOLICY_CONFIG = REPO_ROOT / "training_jobs/configs/train_gspo_qwen3_30b_a3b_offpolicy_smoke_config.yaml"
FORMAL_SCRIPT = REPO_ROOT / "training_jobs/scripts/run_gspo_qwen3_30b_a3b_offpolicy_8gpu_200.sh"
FORMAL_CONFIG = REPO_ROOT / "training_jobs/configs/train_gspo_qwen3_30b_a3b_offpolicy_8gpu_200_config.yaml"


def _hydra_overrides(script_path: Path) -> dict[str, str]:
    overrides = {}
    in_command = False
    for raw_line in script_path.read_text().splitlines():
        line = raw_line.strip()
        if line == "python -m verl.trainer.main_ppo \\":
            in_command = True
            continue
        if not in_command:
            continue
        line = line.removesuffix(" \\")
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        overrides[key.removeprefix("+")] = value
    return overrides


def test_offpolicy_recipe_uses_two_step_behavior_policy_rollouts():
    base_script = BASE_SCRIPT.read_text()
    offpolicy_script = OFFPOLICY_SCRIPT.read_text()

    assert "trainer.rollout_buffer.enable=False" in base_script
    assert "trainer.rollout_buffer.enable=True" in offpolicy_script
    assert "trainer.rollout_buffer.delay_steps=2" in offpolicy_script
    assert "trainer.rollout_buffer.use_rollout_log_probs=True" in offpolicy_script
    assert "actor_rollout_ref.rollout.calculate_log_probs=True" in offpolicy_script
    assert "data/data_processed/math-17k.parquet" in offpolicy_script
    assert "reward.reward_manager.name=dapo" in offpolicy_script


def test_offpolicy_smoke_config_keeps_two_by_four_resources_without_retry():
    config = yaml.safe_load(OFFPOLICY_CONFIG.read_text())

    assert config["Entrypoint"].endswith("run_gspo_qwen3_30b_a3b_offpolicy_smoke.sh")
    assert config["TaskRoleSpecs"] == [
        {"RoleName": "worker", "RoleReplicas": 2, "Flavor": "ml.pni2.14xlarge"}
    ]
    assert config["RetryOptions"]["EnableRetry"] is False


def test_formal_offpolicy_recipe_matches_paper_and_storage_contract():
    script = FORMAL_SCRIPT.read_text()

    assert "/GenSIvePFS/users/cxli" not in script

    expected_fragments = [
        'MLP_WORKER_NUM="${MLP_WORKER_NUM:-4}"',
        'MLP_WORKER_GPU="${MLP_WORKER_GPU:-4}"',
        'ACTOR_FSDP_SIZE="${ACTOR_FSDP_SIZE:-$((MLP_WORKER_NUM * MLP_WORKER_GPU))}"',
        'ROLLOUT_DP="${ROLLOUT_DP:-1}"',
        'ROLLOUT_EP="${ROLLOUT_EP:-4}"',
        "data/data_processed/math-17k.parquet",
        "data/data_processed/moe_eval/minpro/amc23.parquet",
        "data/data_processed/moe_eval/minpro/aime24.parquet",
        "data/data_processed/moe_eval/minpro/aime25.parquet",
        'MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"',
        'MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-20480}"',
        'OVERLONG_BUFFER_ENABLE="${OVERLONG_BUFFER_ENABLE:-False}"',
        'VAL_N="${VAL_N:-8}"',
        'TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-200}"',
        'REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"',
        'MODEL_PATH="${MODEL_PATH:-./models/Qwen3-30B-A3B-Base}"',
        'TRAIN_FILE="${TRAIN_FILE:-./data/data_processed/math-17k.parquet}"',
        'OUTPUT_ROOT="${OUTPUT_ROOT:-.}"',
        'CKPT_ROOT="${CKPT_ROOT:-${OUTPUT_ROOT}/checkpoints}"',
        "actor_rollout_ref.actor.policy_loss.loss_mode=gspo",
        "actor_rollout_ref.actor.clip_ratio_low=0.002",
        "actor_rollout_ref.actor.clip_ratio_high=0.002",
        "actor_rollout_ref.actor.ppo_mini_batch_size=32",
        'actor_rollout_ref.actor.fsdp_config.fsdp_size="${ACTOR_FSDP_SIZE}"',
        "actor_rollout_ref.actor.optim.lr_warmup_steps=10",
        "actor_rollout_ref.rollout.n=8",
        'actor_rollout_ref.rollout.val_kwargs.n="${VAL_N}"',
        "trainer.rollout_buffer.delay_steps=2",
        "trainer.test_freq=5",
        "trainer.save_freq=5",
        'trainer.total_training_steps="${TOTAL_TRAINING_STEPS}"',
        "wandb",
        'if [ ! -f "${MODEL_PATH}/config.json" ]',
        'if [ ! -f "${TRAIN_FILE}" ]',
    ]
    for fragment in expected_fragments:
        assert fragment in script

    assert 'WANDB_ENTITY="${WANDB_ENTITY:-licx199}"' not in script


def test_formal_offpolicy_config_requests_four_by_four_a100s():
    config = yaml.safe_load(FORMAL_CONFIG.read_text())

    assert config["TaskName"] == "gspo-moe-offpolicy-n2-16gpu-200"
    assert config["TaskRoleSpecs"] == [
        {"RoleName": "worker", "RoleReplicas": 4, "Flavor": "ml.pni2.14xlarge"}
    ]
    envs = {item["Name"]: item["Value"] for item in config["Envs"]}
    assert envs["EXPERIMENT_NAME"] == "gspo_moe_offpolicy_n2_16gpu_200"
    assert envs["RUN_ID"] == "gspo_moe_offpolicy_n2_16gpu_200"


def test_formal_recipe_only_changes_documented_scale_settings_from_smoke():
    smoke = _hydra_overrides(OFFPOLICY_SCRIPT)
    formal = _hydra_overrides(FORMAL_SCRIPT)
    differences = {
        key: (smoke.get(key), formal.get(key))
        for key in smoke.keys() | formal.keys()
        if smoke.get(key) != formal.get(key)
    }

    assert differences == {
        "actor_rollout_ref.actor.fsdp_config.fsdp_size": ("8", '"${ACTOR_FSDP_SIZE}"'),
        "actor_rollout_ref.actor.optim.lr_warmup_steps": ("2", "10"),
        "actor_rollout_ref.actor.ppo_mini_batch_size": ("8", "32"),
        "data.train_batch_size": ("16", "512"),
        "trainer.max_actor_ckpt_to_keep": ("2", "null"),
        "trainer.max_critic_ckpt_to_keep": ("2", "null"),
        "trainer.save_freq": ("3", "5"),
        "trainer.test_freq": ("3", "5"),
    }


def test_formal_offpolicy_recipe_has_portable_runbook():
    readme = REPO_ROOT / "training_jobs/configs/gspo_moe_offpolicy/README.md"

    contents = readme.read_text()
    assert "package-prefix-ripo-recipes" in contents
    assert "Qwen3-30B-A3B-Base" in contents
    assert "rollout queue" in contents
    assert "policy lag 2" in contents
    assert "/path/to" not in contents
    assert "${WORKSPACE}" not in contents
