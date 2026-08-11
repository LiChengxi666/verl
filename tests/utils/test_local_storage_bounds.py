import os
import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[2] / "verl" / "utils" / "local_storage.py"


def _local_storage():
    spec = importlib.util.spec_from_file_location("verl_local_storage", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, contents: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    return path


def test_bounded_file_cache_prunes_stale_shards_but_preserves_other_files(tmp_path):
    manifest = _write(tmp_path / "manifest.json")
    shards = [
        _write(tmp_path / f"hash-{index}" / f"part-{index:05d}.jsonl.gz", bytes([index]))
        for index in range(4)
    ]
    for index, shard in enumerate(shards):
        os.utime(shard, (index + 1, index + 1))

    cache = _local_storage().BoundedFileCache(tmp_path, pattern="part-*.jsonl.gz", max_files=2)

    assert cache.files == (shards[2], shards[3])
    assert not shards[0].exists()
    assert not shards[1].exists()
    assert manifest.exists()


def test_bounded_file_cache_evicts_the_oldest_file_when_a_new_shard_arrives(tmp_path):
    first = _write(tmp_path / "a" / "part-00000.jsonl.gz")
    second = _write(tmp_path / "b" / "part-00001.jsonl.gz")
    cache = _local_storage().BoundedFileCache(tmp_path, pattern="part-*.jsonl.gz", max_files=2)
    third = _write(tmp_path / "c" / "part-00002.jsonl.gz")

    cache.record(third)

    assert not first.exists()
    assert second.exists()
    assert third.exists()
    assert cache.files == (second, third)


def test_remove_uploaded_checkpoint_files_only_removes_explicit_files_under_root(tmp_path):
    checkpoint = tmp_path / "global_step_5" / "actor"
    model = _write(checkpoint / "model_world_size_2_rank_0.pt")
    optim = _write(checkpoint / "optim_world_size_2_rank_0.pt")
    keep = _write(checkpoint / "upload-debug.log")

    _local_storage().remove_uploaded_checkpoint_files([model, optim], root=checkpoint)

    assert not model.exists()
    assert not optim.exists()
    assert keep.exists()


def test_remove_uploaded_checkpoint_files_rejects_paths_outside_checkpoint_root(tmp_path):
    checkpoint = tmp_path / "global_step_5" / "actor"
    checkpoint.mkdir(parents=True)
    outside = _write(tmp_path / "do-not-delete.pt")

    with pytest.raises(ValueError, match="outside checkpoint root"):
        _local_storage().remove_uploaded_checkpoint_files([outside], root=checkpoint)

    assert outside.exists()


def test_remove_published_checkpoint_tree_requires_a_direct_global_step_child(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    step = checkpoint_root / "global_step_5"
    _write(step / "data.pt")

    remove_published_checkpoint_tree = _local_storage().remove_published_checkpoint_tree
    remove_published_checkpoint_tree(step, checkpoint_root=checkpoint_root)

    assert not step.exists()
    unsafe = _write(tmp_path / "global_step_6" / "data.pt").parent
    with pytest.raises(ValueError, match="direct global_step child"):
        remove_published_checkpoint_tree(unsafe, checkpoint_root=checkpoint_root)
    assert unsafe.exists()
