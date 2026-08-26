from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALLER = REPO_ROOT / "training_jobs/scripts/moe_rl/common/install_vllm012_r3_backport.sh"
PATCH = REPO_ROOT / "training_jobs/patches/vllm/v0.12.0-router-replay-r3.patch"


def test_backport_installer_is_idempotent_and_version_guarded():
    contents = INSTALLER.read_text()

    assert "4fd9d6a85c00ac0186aa9abbeff73fc2ac6c721e" in contents
    assert "git apply --check" in contents
    assert "git apply --reverse --check" in contents
    assert "check_r3_vllm.py" in contents


def test_backport_patch_contains_rollout_routing_capture_contract():
    contents = PATCH.read_text()

    assert "enable_return_routed_experts" in contents
    assert "RoutedExpertsCapturer" in contents
    assert "routed_experts" in contents
