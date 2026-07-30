# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from verl.single_controller.ray.base import _inherited_worker_env, _merge_worker_env
from verl.utils.checkpoint.fsdp_checkpoint_manager import _resolve_hdfs_checkpoint_path
from verl.workers.engine_workers import _resolve_hdfs_actor_path


def test_explicit_hdfs_actor_path_wins(monkeypatch):
    monkeypatch.setenv("VERL_HDFS_CKPT_DIR", "hdfs://cluster/fallback")

    assert (
        _resolve_hdfs_actor_path(
            "/local/run/global_step_50/actor",
            "hdfs://cluster/explicit/global_step_50/actor",
        )
        == "hdfs://cluster/explicit/global_step_50/actor"
    )


def test_hdfs_actor_path_is_rebuilt_from_worker_environment(monkeypatch):
    monkeypatch.setenv("VERL_HDFS_CKPT_DIR", "hdfs://cluster/run")

    assert (
        _resolve_hdfs_actor_path("/local/run/global_step_50/actor", None) == "hdfs://cluster/run/global_step_50/actor"
    )


def test_local_path_in_hdfs_argument_is_rebuilt_from_worker_environment(monkeypatch):
    monkeypatch.setenv("VERL_HDFS_CKPT_DIR", "hdfs://cluster/run")

    assert (
        _resolve_hdfs_actor_path(
            "/local/run/global_step_50/actor",
            "/local/run/global_step_50/actor",
        )
        == "hdfs://cluster/run/global_step_50/actor"
    )


def test_local_hdfs_argument_is_rebuilt_from_serialized_worker_config(monkeypatch):
    monkeypatch.delenv("VERL_HDFS_CKPT_DIR", raising=False)

    assert (
        _resolve_hdfs_actor_path(
            "/local/run/global_step_50/actor",
            "/local/run/global_step_50/actor",
            "hdfs://cluster/from-worker-config",
        )
        == "hdfs://cluster/from-worker-config/global_step_50/actor"
    )


def test_fsdp_manager_rebuilds_hdfs_path_at_final_save_boundary(monkeypatch):
    monkeypatch.setenv("VERL_HDFS_CKPT_DIR", "hdfs://cluster/run")

    assert (
        _resolve_hdfs_checkpoint_path("/local/run/global_step_55/actor", None)
        == "hdfs://cluster/run/global_step_55/actor"
    )


def test_worker_env_inherits_hdfs_checkpoint_transport(monkeypatch):
    monkeypatch.setenv("VERL_HDFS_CKPT_DIR", "hdfs://cluster/run")
    monkeypatch.setenv("MLX_USER_TOKEN", "secret")
    monkeypatch.setenv("HADOOP_CONF_DIR", "/hadoop/conf")
    monkeypatch.setenv("UNRELATED_ENV", "do-not-forward")

    inherited = _inherited_worker_env()

    assert inherited["VERL_HDFS_CKPT_DIR"] == "hdfs://cluster/run"
    assert inherited["MLX_USER_TOKEN"] == "secret"
    assert inherited["HADOOP_CONF_DIR"] == "/hadoop/conf"
    assert "UNRELATED_ENV" not in inherited


def test_worker_env_allows_same_explicit_hdfs_value():
    merged = _merge_worker_env(
        {"VERL_HDFS_CKPT_DIR": "hdfs://cluster/run"},
        {"VERL_HDFS_CKPT_DIR": "hdfs://cluster/run"},
    )
    assert merged["VERL_HDFS_CKPT_DIR"] == "hdfs://cluster/run"


def test_worker_env_rejects_different_explicit_hdfs_value():
    import pytest

    with pytest.raises(ValueError, match="Cannot override protected system env"):
        _merge_worker_env(
            {"VERL_HDFS_CKPT_DIR": "hdfs://cluster/original"},
            {"VERL_HDFS_CKPT_DIR": "hdfs://cluster/different"},
        )
