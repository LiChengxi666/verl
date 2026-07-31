# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from argparse import Namespace

import yaml

from scripts.cluster.run_ppo_hdfs_resume import load_environment_directory, prepare_overrides, set_override


def test_set_override_replaces_plain_and_additive_forms():
    overrides = ["trainer.total_training_steps=200", "+trainer.total_training_steps=100", "data.train_batch_size=64"]

    set_override(overrides, "trainer.total_training_steps", "300")

    assert overrides == ["data.train_batch_size=64", "trainer.total_training_steps=300"]


def test_load_environment_directory_does_not_replace_current_environment(tmp_path, monkeypatch):
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    (environment_dir / "HOME").write_text("/old/node/home\n", encoding="utf-8")
    (environment_dir / "HDFS_TOKEN").write_text("token-value\n", encoding="utf-8")
    monkeypatch.setenv("HOME", "/current/home")
    monkeypatch.delenv("HDFS_TOKEN", raising=False)

    load_environment_directory(environment_dir)

    assert os.environ["HOME"] == "/current/home"
    assert os.environ["HDFS_TOKEN"] == "token-value"


def test_prepare_overrides_adds_portable_resume_paths(tmp_path):
    overrides_file = tmp_path / "overrides.yaml"
    overrides_file.write_text(
        yaml.safe_dump(["trainer.project_name=old", "trainer.max_actor_ckpt_to_keep=5"]),
        encoding="utf-8",
    )
    args = Namespace(
        overrides_file=overrides_file,
        hdfs_checkpoint_dir="hdfs://cluster/user/checkpoints/run-a",
        local_state_dir=tmp_path / "state",
        project_name="project-a",
        experiment_name="run-a",
        total_training_steps=300,
        extra_overrides=["data.train_files=/shared/math.parquet", "+custom.value=1"],
    )

    overrides = prepare_overrides(args)

    assert "trainer.project_name=project-a" in overrides
    assert "trainer.experiment_name=run-a" in overrides
    assert "trainer.default_hdfs_dir=hdfs://cluster/user/checkpoints/run-a" in overrides
    assert f"trainer.default_local_dir={(tmp_path / 'state/checkpoints').resolve()}" in overrides
    assert "trainer.del_local_ckpt_after_load=True" in overrides
    assert "trainer.max_actor_ckpt_to_keep=1" in overrides
    assert "trainer.total_training_steps=300" in overrides
    assert "data.train_files=/shared/math.parquet" in overrides
    assert "+custom.value=1" in overrides
