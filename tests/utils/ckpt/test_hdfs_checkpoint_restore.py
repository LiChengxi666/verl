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

import tempfile
from pathlib import Path

from verl.utils.checkpoint import hdfs_checkpoint
from verl.utils.checkpoint.hdfs_checkpoint import (
    _copy_required_directory_to_local,
    _select_megatron_restore_files,
)


def test_selects_shared_metadata_and_only_current_node_shards():
    manifest = {
        "format": "megatron-node-shards-v1",
        "world_size": 32,
        "nodes": [
            {
                "hostname": "saved-0",
                "leader_rank": 0,
                "files": [".metadata", "__0_0.distcp", "common.pt", "metadata.json"],
            },
            {
                "hostname": "saved-1",
                "leader_rank": 8,
                "files": ["__8_0.distcp", "__9_0.distcp"],
            },
            {
                "hostname": "saved-2",
                "leader_rank": 16,
                "files": ["__16_0.distcp"],
            },
            {
                "hostname": "saved-3",
                "leader_rank": 24,
                "files": ["__24_0.distcp"],
            },
        ],
    }

    assert _select_megatron_restore_files(manifest, leader_rank=8) == [
        ("saved-0", ".metadata"),
        ("saved-0", "common.pt"),
        ("saved-0", "metadata.json"),
        ("saved-1", "__8_0.distcp"),
        ("saved-1", "__9_0.distcp"),
    ]


def test_rank_zero_gets_its_shards_and_shared_metadata_once():
    manifest = {
        "format": "megatron-node-shards-v1",
        "world_size": 8,
        "nodes": [
            {
                "hostname": "saved-0",
                "leader_rank": 0,
                "files": [".metadata", "__0_0.distcp", "common.pt"],
            }
        ],
    }

    assert _select_megatron_restore_files(manifest, leader_rank=0) == [
        ("saved-0", ".metadata"),
        ("saved-0", "common.pt"),
        ("saved-0", "__0_0.distcp"),
    ]


def test_shared_directory_restore_is_atomic_and_reused():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        remote = root / "remote-huggingface"
        remote.mkdir()
        (remote / "config.json").write_text("trained", encoding="utf-8")
        destination = root / "actor" / "huggingface"
        calls = []
        original_copy = hdfs_checkpoint.hdfs_io.copy

        def fake_copy(src, dst):
            calls.append((src, dst))
            import shutil

            shutil.copytree(src, dst)
            return True

        hdfs_checkpoint.hdfs_io.copy = fake_copy
        try:
            _copy_required_directory_to_local(str(remote), destination)
            _copy_required_directory_to_local(str(remote), destination)
        finally:
            hdfs_checkpoint.hdfs_io.copy = original_copy

        assert (destination / "config.json").read_text(encoding="utf-8") == "trained"
        assert len(calls) == 1


if __name__ == "__main__":
    test_selects_shared_metadata_and_only_current_node_shards()
    test_rank_zero_gets_its_shards_and_shared_metadata_once()
    test_shared_directory_restore_is_atomic_and_reused()
    print("hdfs checkpoint restore selection tests passed")
