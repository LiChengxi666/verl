import importlib.util
import os
from pathlib import Path


if script_override := os.environ.get("PR2_REVAL_SCRIPT"):
    SCRIPT = Path(script_override)
else:
    SCRIPT = Path(__file__).parents[2] / "training_jobs/scripts/moe_rl/run_pr2_paper_aligned_reval.py"


def _load_recipe():
    spec = importlib.util.spec_from_file_location("pr2_paper_aligned_reval", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_resume_is_pinned_to_requested_hdfs_step(tmp_path):
    recipe = _load_recipe()
    trainer = {}

    recipe.configure_checkpoint_resume(trainer, "hdfs://cluster/run", tmp_path, 80)

    assert trainer == {
        "resume_mode": "resume_path",
        "resume_from_path": str(tmp_path / "checkpoints/global_step_80"),
        "default_hdfs_dir": "hdfs://cluster/run",
        "default_local_dir": str(tmp_path / "checkpoints"),
    }
