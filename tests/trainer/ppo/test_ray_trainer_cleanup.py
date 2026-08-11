from types import SimpleNamespace

from verl.utils import hdfs_io
from verl.utils.checkpoint import hdfs_checkpoint
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class _Iterator:
    def __init__(self):
        self.shutdown_calls = 0

    def _shutdown_workers(self):
        self.shutdown_calls += 1


class _HybridCheckpointManager:
    def __init__(self):
        self.updated_versions = []

    def update_weights(self, policy_version):
        self.updated_versions.append(policy_version)

    def wake_up_replicas(self):
        raise AssertionError("HYBRID rollout must resume through update_weights")


class _CheckpointWorkerGroup:
    def __init__(self):
        self.load_calls = []

    def load_checkpoint(self, *args, **kwargs):
        self.load_calls.append((args, kwargs))


def test_shutdown_closes_active_dataloader_workers():
    train_iterator = _Iterator()
    val_iterator = _Iterator()
    trainer = object.__new__(RayPPOTrainer)
    trainer.train_dataloader = SimpleNamespace(_iterator=train_iterator)
    trainer.val_dataloader = SimpleNamespace(_iterator=val_iterator)

    trainer.shutdown()

    assert train_iterator.shutdown_calls == 1
    assert val_iterator.shutdown_calls == 1
    assert trainer.train_dataloader._iterator is None
    assert trainer.val_dataloader._iterator is None


def test_shutdown_ignores_uninitialized_dataloaders():
    trainer = object.__new__(RayPPOTrainer)
    trainer.train_dataloader = SimpleNamespace(_iterator=None)

    trainer.shutdown()


def test_fixed_delay_warmup_resumes_hybrid_rollout_through_weight_sync():
    trainer = object.__new__(RayPPOTrainer)
    trainer.checkpoint_manager = _HybridCheckpointManager()

    trainer._resume_rollout_after_buffer_warmup(policy_version=0)

    assert trainer.checkpoint_manager.updated_versions == [0]


def test_hdfs_resume_passes_remote_paths_to_actor_and_critic(tmp_path, monkeypatch):
    local_step = tmp_path / "global_step_5"
    remote_step = "hdfs://cluster/checkpoints/run/global_step_5"
    actor_wg = _CheckpointWorkerGroup()
    critic_wg = _CheckpointWorkerGroup()
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(
            resume_mode="auto",
            default_hdfs_dir="hdfs://cluster/checkpoints/run",
            default_local_dir=str(tmp_path),
            del_local_ckpt_after_load=True,
        )
    )
    trainer.rollout_buffer = None
    trainer.use_critic = True
    trainer.actor_rollout_wg = actor_wg
    trainer.critic_wg = critic_wg
    monkeypatch.setattr(hdfs_io, "exists", lambda _: True)
    monkeypatch.setattr(
        hdfs_checkpoint,
        "download_remote_metadata",
        lambda _remote, _local: (str(local_step), remote_step),
    )

    trainer._load_checkpoint()

    assert actor_wg.load_calls == [
        ((str(local_step / "actor"), f"{remote_step}/actor", True), {})
    ]
    assert critic_wg.load_calls == [
        ((str(local_step / "critic"), f"{remote_step}/critic", True), {})
    ]
