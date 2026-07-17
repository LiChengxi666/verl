# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, prune_complete_checkpoints
from verl.trainer.ppo.rollout_buffer import FixedDelayRolloutBuffer, RolloutBufferCheckpointError


def _batch(value: int) -> DataProto:
    return DataProto.from_single_dict({"value": torch.tensor([[value]])})


def _trainer(tmp_path: Path) -> RayPPOTrainer:
    trainer = object.__new__(RayPPOTrainer)
    trainer.global_steps = 5
    trainer.rollout_generation_step = 9
    trainer.rollout_buffer = FixedDelayRolloutBuffer(delay_steps=2)
    trainer.rollout_buffer.push(_batch(11), policy_version=3, generation_step=8)
    trainer.rollout_buffer.push(_batch(12), policy_version=4, generation_step=9)
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(default_local_dir=str(tmp_path)))
    return trainer


def test_finalize_checkpoint_publishes_buffer_and_success_before_latest(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path)
    events = []
    monkeypatch.setattr(trainer, "_save_rollout_buffer", lambda _: events.append("buffer"))
    monkeypatch.setattr(trainer, "_snapshot_metrics_file", lambda _: events.append("metrics"))
    monkeypatch.setattr(trainer, "_write_checkpoint_success_marker", lambda _: events.append("success"))
    monkeypatch.setattr(trainer, "_write_latest_checkpoint_pointer", lambda: events.append("latest"))

    trainer._finalize_checkpoint(str(tmp_path / "global_step_5"))

    assert events == ["buffer", "metrics", "success", "latest"]


def test_rollout_checkpoint_round_trip_restores_generation_and_queue(tmp_path):
    checkpoint_dir = tmp_path / "global_step_5"
    source = _trainer(tmp_path)
    source._save_rollout_buffer(str(checkpoint_dir))
    source._write_checkpoint_success_marker(str(checkpoint_dir))

    restored = _trainer(tmp_path)
    restored.rollout_buffer = FixedDelayRolloutBuffer(delay_steps=2)
    restored.rollout_generation_step = 0
    restored._validate_checkpoint_success_marker(str(checkpoint_dir))
    restored._load_rollout_buffer(str(checkpoint_dir))

    assert restored.rollout_generation_step == 9
    released = restored.rollout_buffer.push(_batch(13), policy_version=5, generation_step=10)
    assert released is not None
    assert released.policy_version == 3
    assert released.batch.batch["value"].item() == 11


def test_fixed_delay_restore_rejects_missing_success_marker(tmp_path):
    trainer = _trainer(tmp_path)

    with pytest.raises(RolloutBufferCheckpointError, match="Missing or invalid checkpoint marker"):
        trainer._validate_checkpoint_success_marker(str(tmp_path / "global_step_5"))


def test_success_marker_rejects_wrong_step(tmp_path):
    trainer = _trainer(tmp_path)
    checkpoint_dir = tmp_path / "global_step_5"
    trainer._write_checkpoint_success_marker(str(checkpoint_dir))
    trainer.global_steps = 6

    with pytest.raises(RolloutBufferCheckpointError, match="incompatible"):
        trainer._validate_checkpoint_success_marker(str(checkpoint_dir))


def test_retention_removes_only_old_complete_checkpoints(tmp_path):
    for step in (5, 10, 15):
        checkpoint_dir = tmp_path / f"global_step_{step}"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "_SUCCESS").write_text(
            f'{{"schema_version": 1, "global_step": {step}, "rollout_buffer_enabled": true}}'
        )
    incomplete = tmp_path / "global_step_7"
    incomplete.mkdir()

    removed = prune_complete_checkpoints(tmp_path, max_to_keep=2)

    assert removed == [tmp_path / "global_step_5"]
    assert not (tmp_path / "global_step_5").exists()
    assert (tmp_path / "global_step_10").exists()
    assert (tmp_path / "global_step_15").exists()
    assert incomplete.exists()
