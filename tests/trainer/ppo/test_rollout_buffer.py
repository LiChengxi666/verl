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

import pytest
import torch

from verl import DataProto
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
    batch_path = checkpoint_directory / "batch_0.pkl"
    contents = bytearray(batch_path.read_bytes())
    contents[-1] ^= 0xFF
    batch_path.write_bytes(contents)

    restored = FixedDelayRolloutBuffer(delay_steps=2)
    with pytest.raises(RolloutBufferCheckpointError):
        restored.load_from_directory(checkpoint_directory, expected_checkpoint_step=5)


def test_checkpoint_rejects_delay_mismatch(tmp_path):
    checkpoint_directory = _save_checkpoint(tmp_path)
    manifest_path = checkpoint_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["delay_steps"] = 3
    manifest_path.write_text(json.dumps(manifest))

    restored = FixedDelayRolloutBuffer(delay_steps=2)
    with pytest.raises(RolloutBufferCheckpointError):
        restored.load_from_directory(checkpoint_directory, expected_checkpoint_step=5)


def test_checkpoint_rejects_missing_batch(tmp_path):
    checkpoint_directory = _save_checkpoint(tmp_path)
    (checkpoint_directory / "batch_1.pkl").unlink()

    restored = FixedDelayRolloutBuffer(delay_steps=2)
    with pytest.raises(RolloutBufferCheckpointError):
        restored.load_from_directory(checkpoint_directory, expected_checkpoint_step=5)


def test_checkpoint_rejects_unexpected_checkpoint_step(tmp_path):
    checkpoint_directory = _save_checkpoint(tmp_path)

    restored = FixedDelayRolloutBuffer(delay_steps=2)
    with pytest.raises(RolloutBufferCheckpointError):
        restored.load_from_directory(checkpoint_directory, expected_checkpoint_step=6)
