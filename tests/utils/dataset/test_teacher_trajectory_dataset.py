import gzip
import json
import shutil
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from verl.utils.dataset.teacher_trajectory_dataset import (
    TeacherTrajectoryDataset,
    consume_offline_trajectory_batch,
)
import verl.utils.dataset.teacher_trajectory_dataset as teacher_dataset_module
from verl.utils.dataset.rl_dataset import collate_fn
from verl.protocol import DataProto
from verl.trainer.main_ppo import create_rl_dataset
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class _Tokenizer:
    pad_token_id = 0


def _record(
    prompt_id: str,
    sample_index: int,
    *,
    prompt_ids: list[int],
    response_ids: list[int],
    logprobs: list[float],
    reward: float,
    finish_reason: str = "stop",
) -> dict:
    prompt = [{"role": "user", "content": f"problem {prompt_id}"}]
    return {
        "schema_version": 2,
        "prompt_id": prompt_id,
        "sample_index": sample_index,
        "sample_seed": 7 + sample_index,
        "unique_prompt_index": int(prompt_id[-1]),
        "prompt": prompt,
        "prompt_token_ids": prompt_ids,
        "response": "answer",
        "response_token_ids": response_ids,
        # Stop records intentionally omit the generated EOS here. The loader must
        # use response_token_ids, which are aligned with behavior_logprobs.
        "retokenized_response_token_ids": response_ids[:-1] if finish_reason == "stop" else response_ids,
        "behavior_logprobs": logprobs,
        "behavior_tokens": [f"t{i}" for i in range(len(response_ids))],
        "base_reward": reward,
        "is_correct": bool(reward),
        "finish_reason": finish_reason,
        "ground_truth": "42",
        "source_rows": [
            {
                "data_source": "lighteval/MATH",
                "ability": "MATH",
                "prompt": prompt,
                "reward_model": {"ground_truth": "42", "style": "rule-lighteval/MATH_v2"},
                "extra_info": {"index": f"source-{prompt_id}"},
            }
        ],
    }


def _write_fixture(root: Path, records: list[dict]) -> None:
    (root / "raw").mkdir(parents=True)
    manifest = {
        "schema_version": 2,
        "unique_prompts": 2,
        "samples_per_prompt": 2,
        "shard_prompt_count": 1,
        "sampling": {"temperature": 0.7, "top_p": 0.95, "top_k": 20, "max_tokens": 5},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    for shard_index in range(2):
        path = root / "raw" / f"part-{shard_index:05d}-of-00002.jsonl.gz"
        with gzip.open(path, "wt") as handle:
            for record in records[shard_index * 2 : (shard_index + 1) * 2]:
                handle.write(json.dumps(record) + "\n")


def _config(root: Path) -> OmegaConf:
    return OmegaConf.create(
        {
            "max_prompt_length": 4,
            "max_response_length": 5,
            "offline_trajectory": {
                "enable": True,
                "root": str(root),
                "pad_to_multiple": 3,
                "cache_dir": str(root / "cache"),
                "max_cached_shards": 2,
                "max_resp_len": 5,
                "overlong_buffer_len": 1,
                "overlong_penalty_factor": 1.0,
            },
        }
    )


@pytest.fixture
def trajectory_root(tmp_path: Path) -> Path:
    records = [
        _record(
            "prompt0",
            0,
            prompt_ids=[11, 12],
            response_ids=[21, 22, 151645],
            logprobs=[-0.1, -0.2, -0.3],
            reward=1.0,
        ),
        _record(
            "prompt0",
            1,
            prompt_ids=[11, 12],
            response_ids=[31, 151645],
            logprobs=[-0.4, -0.5],
            reward=0.0,
        ),
        _record(
            "prompt1",
            0,
            prompt_ids=[13],
            response_ids=[41, 42, 43, 44, 45],
            logprobs=[-0.6, -0.7, -0.8, -0.9, -1.0],
            reward=0.0,
            finish_reason="length",
        ),
        _record(
            "prompt1",
            1,
            prompt_ids=[13],
            response_ids=[51, 151645],
            logprobs=[-1.1, -1.2],
            reward=1.0,
        ),
    ]
    _write_fixture(tmp_path, records)
    return tmp_path


def test_stop_record_keeps_behavior_eos_and_aligned_logprob(trajectory_root: Path):
    dataset = TeacherTrajectoryDataset([str(trajectory_root)], _Tokenizer(), _config(trajectory_root))

    item = dataset[0]

    assert item["responses"].tolist() == [21, 22, 151645, 0, 0]
    assert item["response_mask"].tolist() == [1, 1, 1, 0, 0]
    assert item["rollout_log_probs"].tolist() == pytest.approx([-0.1, -0.2, -0.3, 0.0, 0.0])
    assert item["rm_scores"].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]
    assert item["prompts"].tolist() == [0, 0, 11, 12]
    assert item["__offline_trajectory__"] is True
    assert item["__offline_padding__"] is False


def test_offline_tensor_length_is_independent_from_online_validation_limit(trajectory_root: Path):
    config = _config(trajectory_root)
    config.max_response_length = 7
    config.offline_trajectory.max_response_length = 5
    dataset = TeacherTrajectoryDataset([str(trajectory_root)], _Tokenizer(), config)

    item = dataset[0]

    assert item["responses"].shape == (5,)
    assert item["rollout_log_probs"].shape == (5,)


def test_dataset_exposes_every_real_record_then_masked_padding(trajectory_root: Path):
    dataset = TeacherTrajectoryDataset([str(trajectory_root)], _Tokenizer(), _config(trajectory_root))

    assert dataset.real_num_trajectories == 4
    assert len(dataset) == 6
    assert [dataset[i]["trajectory_id"] for i in range(4)] == [
        "prompt0:0",
        "prompt0:1",
        "prompt1:0",
        "prompt1:1",
    ]
    for index in (4, 5):
        padding = dataset[index]
        assert padding["__offline_padding__"] is True
        assert torch.count_nonzero(padding["response_mask"]).item() == 0
        assert torch.count_nonzero(padding["rm_scores"]).item() == 0


def test_teacher_shard_downloads_are_physically_bounded(trajectory_root: Path, monkeypatch):
    config = _config(trajectory_root)
    config.offline_trajectory.max_cached_shards = 1
    copied_paths = []

    def fake_copy_to_local(src, cache_dir=None, **kwargs):
        del kwargs
        if src.endswith("manifest.json"):
            return src
        target = Path(cache_dir) / f"download-{len(copied_paths)}" / Path(src).name
        target.parent.mkdir(parents=True)
        shutil.copy2(src, target)
        copied_paths.append(target)
        return str(target)

    monkeypatch.setattr(teacher_dataset_module, "copy_to_local", fake_copy_to_local)
    dataset = TeacherTrajectoryDataset([str(trajectory_root)], _Tokenizer(), config)

    dataset[0]
    first_shard = copied_paths[-1]
    assert first_shard.exists()
    dataset[2]
    second_shard = copied_paths[-1]

    assert not first_shard.exists()
    assert second_shard.exists()
    assert list(Path(config.offline_trajectory.cache_dir).rglob("part-*.jsonl.gz")) == [second_shard]


def test_length_finish_gets_same_soft_overlong_penalty_as_dapo(trajectory_root: Path):
    dataset = TeacherTrajectoryDataset([str(trajectory_root)], _Tokenizer(), _config(trajectory_root))

    item = dataset[2]

    assert item["base_reward"] == 0.0
    assert item["overlong_reward"] == -1.0
    assert item["rm_scores"].tolist() == [0.0, 0.0, 0.0, 0.0, -1.0]


def test_behavior_logprob_mismatch_is_rejected(tmp_path: Path):
    records = [
        _record(
            f"prompt{i // 2}",
            i % 2,
            prompt_ids=[11],
            response_ids=[21, 151645],
            logprobs=[-0.1] if i == 0 else [-0.1, -0.2],
            reward=0.0,
        )
        for i in range(4)
    ]
    _write_fixture(tmp_path, records)
    dataset = TeacherTrajectoryDataset([str(tmp_path)], _Tokenizer(), _config(tmp_path))

    with pytest.raises(ValueError, match="behavior_logprobs.*response_token_ids"):
        dataset[0]


def test_offline_batch_is_consumed_without_generation(trajectory_root: Path):
    dataset = TeacherTrajectoryDataset([str(trajectory_root)], _Tokenizer(), _config(trajectory_root))
    batch = DataProto.from_single_dict(collate_fn([dataset[0], dataset[1]]))

    output = consume_offline_trajectory_batch(batch)

    assert output is batch
    assert output.meta_info["timing"] == {}
    assert output.meta_info["reward_extra_keys"] == [
        "acc",
        "base_reward",
        "finish_reason",
        "is_correct",
        "overlong",
        "overlong_reward",
    ]
    assert output.batch["responses"][0].tolist() == [21, 22, 151645, 0, 0]


def test_online_batch_is_not_consumed_as_offline():
    online = DataProto.from_dict(tensors={"prompts": torch.tensor([[1, 2]])})

    assert consume_offline_trajectory_batch(online) is None


def test_trainer_moves_pregenerated_tensors_out_of_prompt_batch(trajectory_root: Path):
    dataset = TeacherTrajectoryDataset([str(trajectory_root)], _Tokenizer(), _config(trajectory_root))
    prompt_batch = DataProto.from_single_dict(collate_fn([dataset[0], dataset[1]]))
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)

    rollout_batch = trainer._get_gen_batch(prompt_batch)

    assert "responses" not in prompt_batch.batch
    assert "rollout_log_probs" not in prompt_batch.batch
    assert "responses" in rollout_batch.batch
    assert "rollout_log_probs" in rollout_batch.batch
    assert "data_source" in prompt_batch.non_tensor_batch
    assert "data_source" in rollout_batch.non_tensor_batch


def test_trainer_does_not_call_rollout_manager_for_offline_batch(trajectory_root: Path):
    class _GenerationMustNotRun:
        def generate_sequences(self, batch):
            raise AssertionError("4B training rollout was called")

    dataset = TeacherTrajectoryDataset([str(trajectory_root)], _Tokenizer(), _config(trajectory_root))
    batch = DataProto.from_single_dict(collate_fn([dataset[0], dataset[1]]))
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.async_rollout_manager = _GenerationMustNotRun()

    output = trainer._generate_training_sequences(batch)

    assert output is batch
    assert output.meta_info["timing"] == {}


def test_trainer_selects_teacher_dataset_only_for_training_split(trajectory_root: Path):
    data_config = _config(trajectory_root)
    data_config.train_files = [str(trajectory_root)]
    data_config.val_files = ["unused.parquet"]
    data_config.train_batch_size = 2
    data_config.dataloader_num_workers = 0
    data_config.val_batch_size = 1
    data_config.validation_shuffle = False
    data_config.shuffle = False
    data_config.seed = 1
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {"data": data_config, "trainer": {"total_epochs": 1, "total_training_steps": None}}
    )
    trainer.tokenizer = _Tokenizer()
    trainer.processor = None
    val_dataset = [{"prompts": torch.tensor([1, 2])}]

    trainer._create_dataloader(
        train_dataset=None,
        val_dataset=val_dataset,
        collate_fn=None,
        train_sampler=None,
    )

    assert isinstance(trainer.train_dataset, TeacherTrajectoryDataset)
    assert trainer.val_dataset is val_dataset


def test_main_ppo_entry_selects_teacher_dataset_for_offline_training(trajectory_root: Path):
    data_config = _config(trajectory_root)

    train_dataset = create_rl_dataset(
        [str(trajectory_root)],
        data_config,
        _Tokenizer(),
        processor=None,
        is_train=True,
    )

    assert isinstance(train_dataset, TeacherTrajectoryDataset)
