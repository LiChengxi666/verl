from types import SimpleNamespace

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
