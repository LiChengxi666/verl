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

import json
from pathlib import Path

from verl.utils.checkpoint import hdfs_checkpoint


class FakeHdfs:
    def __init__(self, remote_files):
        self.remote_files = dict(remote_files)
        self.copies = []
        self.directories = []

    def makedirs(self, path, exist_ok=False):
        self.directories.append(path)

    def copy(self, src, dst, **kwargs):
        self.copies.append((str(src), str(dst)))
        src = str(src)
        dst = str(dst)
        if src.startswith("hdfs://"):
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            Path(dst).write_bytes(self.remote_files[src])
        else:
            self.remote_files[dst] = Path(src).read_bytes()
        return True

    def exists(self, path):
        return str(path) in self.remote_files


class RejectExistingDestinationHdfs(FakeHdfs):
    def copy(self, src, dst, **kwargs):
        if str(src).startswith("hdfs://") and Path(dst).exists():
            return False
        return super().copy(src, dst, **kwargs)


def test_download_remote_metadata_uses_published_pointer(tmp_path, monkeypatch):
    remote = "hdfs://cluster/new-run"
    fake = FakeHdfs(
        {
            f"{remote}/latest_checkpointed_iteration.txt": b"15",
            f"{remote}/global_step_15/data.pt": b"loader",
            f"{remote}/global_step_15/_SUCCESS": b'{"global_step": 15}',
        }
    )
    monkeypatch.setattr(hdfs_checkpoint, "hdfs_io", fake)

    local_step, remote_step = hdfs_checkpoint.download_remote_metadata(remote, tmp_path)

    assert local_step == str(tmp_path / "global_step_15")
    assert remote_step == f"{remote}/global_step_15"
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "15"
    assert (tmp_path / "global_step_15/data.pt").read_bytes() == b"loader"


def test_download_remote_metadata_accepts_actor_only_checkpoint(tmp_path, monkeypatch):
    remote = "hdfs://cluster/recovered-run"
    fake = FakeHdfs(
        {
            f"{remote}/latest_checkpointed_iteration.txt": b"5",
            f"{remote}/global_step_5/_SUCCESS": b'{"global_step": 5}',
        }
    )
    monkeypatch.setattr(hdfs_checkpoint, "hdfs_io", fake)

    local_step, remote_step = hdfs_checkpoint.download_remote_metadata(remote, tmp_path)

    assert local_step == str(tmp_path / "global_step_5")
    assert remote_step == f"{remote}/global_step_5"
    assert (tmp_path / "global_step_5/_SUCCESS").exists()
    assert not (tmp_path / "global_step_5/data.pt").exists()


def test_download_remote_metadata_atomically_replaces_stale_local_files(tmp_path, monkeypatch):
    remote = "hdfs://cluster/resumed-run"
    local_step = tmp_path / "global_step_15"
    local_step.mkdir()
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("15-stale")
    (local_step / "_SUCCESS").write_text("stale")
    (local_step / "data.pt").write_bytes(b"stale")
    fake = RejectExistingDestinationHdfs(
        {
            f"{remote}/latest_checkpointed_iteration.txt": b"15",
            f"{remote}/global_step_15/data.pt": b"loader",
            f"{remote}/global_step_15/_SUCCESS": b'{"global_step": 15}',
        }
    )
    monkeypatch.setattr(hdfs_checkpoint, "hdfs_io", fake)

    downloaded_step, remote_step = hdfs_checkpoint.download_remote_metadata(remote, tmp_path)

    assert downloaded_step == str(local_step)
    assert remote_step == f"{remote}/global_step_15"
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "15"
    assert (local_step / "_SUCCESS").read_text() == '{"global_step": 15}'
    assert (local_step / "data.pt").read_bytes() == b"loader"


def test_publish_remote_metadata_writes_pointer_last(tmp_path, monkeypatch):
    remote = "hdfs://cluster/new-run"
    step = tmp_path / "global_step_20"
    step.mkdir()
    (step / "data.pt").write_bytes(b"loader")
    (step / "_SUCCESS").write_text('{"global_step": 20}')
    tracker = tmp_path / "latest_checkpointed_iteration.txt"
    tracker.write_text("20")
    fake = FakeHdfs({})
    monkeypatch.setattr(hdfs_checkpoint, "hdfs_io", fake)

    hdfs_checkpoint.publish_remote_metadata(remote, step, tracker)

    assert fake.copies[-1] == (
        str(tracker),
        f"{remote}/latest_checkpointed_iteration.txt",
    )


def test_verify_remote_actor_checkpoint_requires_every_rank_and_state(monkeypatch):
    remote_actor = "hdfs://cluster/new-run/global_step_20/actor"
    remote_files = {
        f"{remote_actor}/{kind}_world_size_2_rank_{rank}.pt": b"checkpoint"
        for kind in ("model", "optim", "extra_state")
        for rank in range(2)
    }
    remote_files.pop(f"{remote_actor}/optim_world_size_2_rank_1.pt")
    fake = FakeHdfs(remote_files)
    monkeypatch.setattr(hdfs_checkpoint, "hdfs_io", fake)

    try:
        hdfs_checkpoint.verify_remote_actor_checkpoint(remote_actor, world_size=2)
    except RuntimeError as error:
        assert "optim_world_size_2_rank_1.pt" in str(error)
    else:
        raise AssertionError("missing remote optimizer shard was accepted")


def test_verify_remote_actor_checkpoint_accepts_complete_checkpoint(monkeypatch):
    remote_actor = "hdfs://cluster/new-run/global_step_20/actor"
    remote_files = {
        f"{remote_actor}/{kind}_world_size_2_rank_{rank}.pt": b"checkpoint"
        for kind in ("model", "optim", "extra_state")
        for rank in range(2)
    }
    fake = FakeHdfs(remote_files)
    monkeypatch.setattr(hdfs_checkpoint, "hdfs_io", fake)

    hdfs_checkpoint.verify_remote_actor_checkpoint(remote_actor, world_size=2)


def test_verify_remote_megatron_checkpoint_requires_every_manifest_file(monkeypatch):
    remote_actor = "hdfs://cluster/moe/global_step_5/actor"
    manifest = {
        "format": "megatron-node-shards-v1",
        "world_size": 4,
        "nodes": [
            {"hostname": "node-a", "leader_rank": 0, "files": ["metadata.json", "rank0.pt"]},
            {"hostname": "node-b", "leader_rank": 2, "files": ["rank2.pt"]},
        ],
    }
    remote_files = {
        f"{remote_actor}/megatron_hdfs_manifest.json": json.dumps(manifest).encode(),
        f"{remote_actor}/node_shards/node-a/dist_ckpt/metadata.json": b"metadata",
        f"{remote_actor}/node_shards/node-a/dist_ckpt/rank0.pt": b"rank0",
    }
    fake = FakeHdfs(remote_files)
    monkeypatch.setattr(hdfs_checkpoint, "hdfs_io", fake)

    try:
        hdfs_checkpoint.verify_remote_actor_checkpoint(remote_actor, world_size=4, strategy="megatron")
    except RuntimeError as error:
        assert "node-b/dist_ckpt/rank2.pt" in str(error)
    else:
        raise AssertionError("missing Megatron node shard was accepted")


def test_restore_remote_megatron_checkpoint_downloads_only_current_node_shards(tmp_path, monkeypatch):
    remote_actor = "hdfs://cluster/moe/global_step_5/actor"
    manifest = {
        "format": "megatron-node-shards-v1",
        "world_size": 4,
        "nodes": [
            {"hostname": "node-a", "leader_rank": 0, "files": ["metadata.json", "a.pt"]},
            {"hostname": "node-b", "leader_rank": 2, "files": ["b.pt"]},
        ],
    }
    remote_files = {
        f"{remote_actor}/megatron_hdfs_manifest.json": json.dumps(manifest).encode(),
        f"{remote_actor}/node_shards/node-a/dist_ckpt/metadata.json": b"metadata",
        f"{remote_actor}/node_shards/node-a/dist_ckpt/a.pt": b"a",
        f"{remote_actor}/node_shards/node-b/dist_ckpt/b.pt": b"b",
    }
    fake = FakeHdfs(remote_files)
    monkeypatch.setattr(hdfs_checkpoint, "hdfs_io", fake)
    stale_dist = tmp_path / "actor/dist_ckpt"
    stale_dist.mkdir(parents=True)
    (stale_dist / "stale.pt").write_bytes(b"stale")

    restored = hdfs_checkpoint.restore_remote_megatron_checkpoint(
        remote_actor,
        tmp_path / "actor",
        rank=0,
        hostnames=["new-node-x", "new-node-x", "new-node-y", "new-node-y"],
    )

    assert restored is True
    assert (tmp_path / "actor/dist_ckpt/metadata.json").read_bytes() == b"metadata"
    assert (tmp_path / "actor/dist_ckpt/a.pt").read_bytes() == b"a"
    assert not (tmp_path / "actor/dist_ckpt/b.pt").exists()
    assert not (tmp_path / "actor/dist_ckpt/stale.pt").exists()


def test_nonleader_does_not_restore_remote_megatron_checkpoint(tmp_path, monkeypatch):
    fake = FakeHdfs({})
    monkeypatch.setattr(hdfs_checkpoint, "hdfs_io", fake)

    restored = hdfs_checkpoint.restore_remote_megatron_checkpoint(
        "hdfs://cluster/moe/global_step_5/actor",
        tmp_path / "actor",
        rank=1,
        hostnames=["node-a", "node-a", "node-b", "node-b"],
    )

    assert restored is False
    assert fake.copies == []


def test_build_megatron_manifest_uses_one_complete_file_list_per_node():
    manifest = hdfs_checkpoint.build_megatron_manifest(
        hostnames=["node-a", "node-a", "node-b", "node-b"],
        files_by_rank=[
            ["metadata.json", "a.pt"],
            ["metadata.json", "a.pt"],
            ["b.pt"],
            ["b.pt"],
        ],
    )

    assert manifest == {
        "format": "megatron-node-shards-v1",
        "world_size": 4,
        "nodes": [
            {
                "hostname": "node-a",
                "leader_rank": 0,
                "files": ["a.pt", "metadata.json"],
            },
            {
                "hostname": "node-b",
                "leader_rank": 2,
                "files": ["b.pt"],
            },
        ],
    }


def test_upload_megatron_node_checkpoint_uses_node_specific_paths(tmp_path, monkeypatch):
    local_dist = tmp_path / "dist_ckpt"
    local_dist.mkdir()
    (local_dist / "metadata.json").write_text("metadata")
    (local_dist / "a.pt").write_bytes(b"a")
    remote_actor = "hdfs://cluster/moe/global_step_5/actor"
    manifest = {
        "format": "megatron-node-shards-v1",
        "world_size": 2,
        "nodes": [
            {
                "hostname": "node-a",
                "leader_rank": 0,
                "files": ["a.pt", "metadata.json"],
            }
        ],
    }
    fake = FakeHdfs({})
    monkeypatch.setattr(hdfs_checkpoint, "hdfs_io", fake)

    uploaded = hdfs_checkpoint.upload_megatron_node_checkpoint(
        local_dist,
        remote_actor,
        rank=0,
        manifest=manifest,
    )

    assert uploaded is True
    assert fake.copies == [
        (
            str(local_dist / "a.pt"),
            f"{remote_actor}/node_shards/node-a/dist_ckpt/a.pt",
        ),
        (
            str(local_dist / "metadata.json"),
            f"{remote_actor}/node_shards/node-a/dist_ckpt/metadata.json",
        ),
    ]


def test_publish_megatron_manifest_writes_remote_actor_manifest(tmp_path, monkeypatch):
    remote_actor = "hdfs://cluster/moe/global_step_5/actor"
    manifest = {
        "format": "megatron-node-shards-v1",
        "world_size": 2,
        "nodes": [
            {
                "hostname": "node-a",
                "leader_rank": 0,
                "files": ["a.pt"],
            }
        ],
    }
    fake = FakeHdfs({})
    monkeypatch.setattr(hdfs_checkpoint, "hdfs_io", fake)

    hdfs_checkpoint.publish_megatron_manifest(
        tmp_path / "actor",
        remote_actor,
        manifest,
    )

    remote_manifest = f"{remote_actor}/megatron_hdfs_manifest.json"
    assert fake.copies[-1][1] == remote_manifest
    assert json.loads(fake.remote_files[remote_manifest]) == manifest
