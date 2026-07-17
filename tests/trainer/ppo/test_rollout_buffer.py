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
from pathlib import Path

import pytest
import torch

from verl import DataProto
from verl.trainer.ppo import rollout_buffer
from verl.trainer.ppo.rollout_buffer import FixedDelayRolloutBuffer, RolloutBufferCheckpointError


def make_batch(value: int) -> DataProto:
    return DataProto.from_single_dict({"value": torch.tensor([[value]])})


def test_zero_delay_returns_current_batch():
    buffer = FixedDelayRolloutBuffer(delay_steps=0)

    result = buffer.push(make_batch(3), policy_version=2, generation_step=3)

    assert result is not None
    assert result.batch.batch["value"].item() == 3
    assert result.policy_version == 2
    assert len(buffer) == 0


def test_two_step_delay_is_fifo():
    buffer = FixedDelayRolloutBuffer(delay_steps=2)

    assert buffer.push(make_batch(1), policy_version=0, generation_step=1) is None
    assert buffer.push(make_batch(2), policy_version=0, generation_step=2) is None
    first = buffer.push(make_batch(3), policy_version=0, generation_step=3)
    second = buffer.push(make_batch(4), policy_version=1, generation_step=4)

    assert first is not None and first.batch.batch["value"].item() == 1
    assert second is not None and second.batch.batch["value"].item() == 2
    assert len(buffer) == 2


def test_two_step_delay_reaches_two_policy_updates_after_warmup():
    buffer = FixedDelayRolloutBuffer(delay_steps=2)
    completed_policy_updates = [0, 0, 0, 1, 2, 3]
    observed_lags = []

    for generation_step, current_version in enumerate(completed_policy_updates, start=1):
        rollout = buffer.push(
            make_batch(generation_step),
            policy_version=current_version,
            generation_step=generation_step,
        )
        if rollout is not None:
            observed_lags.append(current_version - rollout.policy_version)

    assert observed_lags == [0, 1, 2, 2]


def test_negative_delay_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        FixedDelayRolloutBuffer(delay_steps=-1)


def _save_checkpoint(tmp_path):
    checkpoint_directory = tmp_path / "rollout_buffer"
    source = FixedDelayRolloutBuffer(delay_steps=2)
    source.push(make_batch(11), policy_version=3, generation_step=8)
    source.push(make_batch(12), policy_version=4, generation_step=9)
    source.save_to_directory(checkpoint_directory, checkpoint_step=5, generation_step=9)
    return checkpoint_directory


def _read_manifest(checkpoint_directory):
    return json.loads((checkpoint_directory / "manifest.json").read_text())


def test_checkpoint_round_trip_preserves_queue_and_generation_step(tmp_path):
    checkpoint_directory = _save_checkpoint(tmp_path)

    restored = FixedDelayRolloutBuffer(delay_steps=2)
    generation_step = restored.load_from_directory(checkpoint_directory, expected_checkpoint_step=5)

    assert generation_step == 9
    first = restored.push(make_batch(13), policy_version=5, generation_step=10)
    second = restored.push(make_batch(14), policy_version=6, generation_step=11)
    assert first is not None
    assert first.batch.batch["value"].item() == 11
    assert first.policy_version == 3
    assert first.generation_step == 8
    assert second is not None
    assert second.batch.batch["value"].item() == 12
    assert second.policy_version == 4
    assert second.generation_step == 9


def test_checkpoint_rejects_corrupted_batch(tmp_path):
    checkpoint_directory = _save_checkpoint(tmp_path)
    batch_path = checkpoint_directory / _read_manifest(checkpoint_directory)["batches"][0]["filename"]
    contents = bytearray(batch_path.read_bytes())
    contents[-1] ^= 0xFF
    batch_path.write_bytes(contents)

    restored = FixedDelayRolloutBuffer(delay_steps=2)
    with pytest.raises(RolloutBufferCheckpointError):
        restored.load_from_directory(checkpoint_directory, expected_checkpoint_step=5)


def test_checkpoint_rejects_delay_mismatch(tmp_path):
    checkpoint_directory = _save_checkpoint(tmp_path)
    manifest_path = checkpoint_directory / "manifest.json"
    manifest = _read_manifest(checkpoint_directory)
    manifest["delay_steps"] = 3
    manifest_path.write_text(json.dumps(manifest))

    restored = FixedDelayRolloutBuffer(delay_steps=2)
    with pytest.raises(RolloutBufferCheckpointError):
        restored.load_from_directory(checkpoint_directory, expected_checkpoint_step=5)


def test_checkpoint_rejects_missing_batch(tmp_path):
    checkpoint_directory = _save_checkpoint(tmp_path)
    manifest = _read_manifest(checkpoint_directory)
    (checkpoint_directory / manifest["batches"][1]["filename"]).unlink()

    restored = FixedDelayRolloutBuffer(delay_steps=2)
    with pytest.raises(RolloutBufferCheckpointError):
        restored.load_from_directory(checkpoint_directory, expected_checkpoint_step=5)


def test_checkpoint_rejects_unexpected_checkpoint_step(tmp_path):
    checkpoint_directory = _save_checkpoint(tmp_path)

    restored = FixedDelayRolloutBuffer(delay_steps=2)
    with pytest.raises(RolloutBufferCheckpointError):
        restored.load_from_directory(checkpoint_directory, expected_checkpoint_step=6)


def test_interrupted_resave_leaves_previous_checkpoint_loadable(tmp_path, monkeypatch):
    checkpoint_directory = _save_checkpoint(tmp_path)
    replacement = FixedDelayRolloutBuffer(delay_steps=2)
    replacement.push(make_batch(21), policy_version=7, generation_step=10)
    replacement.push(make_batch(22), policy_version=8, generation_step=11)
    original_replace = rollout_buffer.os.replace
    published_batch_count = 0

    def interrupt_second_batch_publish(source, destination):
        nonlocal published_batch_count
        if Path(destination).name.startswith("batch_"):
            published_batch_count += 1
            if published_batch_count == 2:
                raise OSError("simulated interruption")
        original_replace(source, destination)

    monkeypatch.setattr(rollout_buffer.os, "replace", interrupt_second_batch_publish)

    with pytest.raises(OSError, match="simulated interruption"):
        replacement.save_to_directory(checkpoint_directory, checkpoint_step=6, generation_step=11)

    restored = FixedDelayRolloutBuffer(delay_steps=2)
    assert restored.load_from_directory(checkpoint_directory, expected_checkpoint_step=5) == 9
    released = restored.push(make_batch(13), policy_version=5, generation_step=10)
    assert released is not None
    assert released.batch.batch["value"].item() == 11
    assert released.policy_version == 3
    assert released.generation_step == 8


def test_checkpoint_rejects_incomplete_fixed_delay_queue(tmp_path):
    checkpoint_directory = _save_checkpoint(tmp_path)
    manifest_path = checkpoint_directory / "manifest.json"
    manifest = _read_manifest(checkpoint_directory)
    manifest["batches"].pop()
    manifest_path.write_text(json.dumps(manifest))

    restored = FixedDelayRolloutBuffer(delay_steps=2)
    with pytest.raises(RolloutBufferCheckpointError):
        restored.load_from_directory(checkpoint_directory, expected_checkpoint_step=5)


def test_checkpoint_syncs_batches_before_manifest_publication(tmp_path, monkeypatch):
    checkpoint_directory = tmp_path / "rollout_buffer"
    source = FixedDelayRolloutBuffer(delay_steps=2)
    source.push(make_batch(11), policy_version=3, generation_step=8)
    source.push(make_batch(12), policy_version=4, generation_step=9)
    events = []
    original_replace = rollout_buffer.os.replace
    original_fsync_directory = rollout_buffer._fsync_directory

    def record_replace(source_path, destination_path):
        destination_name = Path(destination_path).name
        events.append("manifest" if destination_name == "manifest.json" else "batch")
        original_replace(source_path, destination_path)

    def record_directory_sync(directory):
        events.append("directory_sync")
        original_fsync_directory(directory)

    monkeypatch.setattr(rollout_buffer.os, "replace", record_replace)
    monkeypatch.setattr(rollout_buffer, "_fsync_directory", record_directory_sync)

    source.save_to_directory(checkpoint_directory, checkpoint_step=5, generation_step=9)

    assert events == ["batch", "batch", "directory_sync", "manifest", "directory_sync"]


def test_checkpoint_rejects_policy_versions_incompatible_with_actor_step(tmp_path):
    checkpoint_directory = _save_checkpoint(tmp_path)
    manifest_path = checkpoint_directory / "manifest.json"
    manifest = _read_manifest(checkpoint_directory)
    manifest["batches"][0]["policy_version"] = 2
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RolloutBufferCheckpointError, match="policy versions"):
        FixedDelayRolloutBuffer(delay_steps=2).load_from_directory(
            checkpoint_directory, expected_checkpoint_step=5
        )
