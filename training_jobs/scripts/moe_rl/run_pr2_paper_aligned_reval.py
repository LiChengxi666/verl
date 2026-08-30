"""Read-only PR2 checkpoint reevaluation with a controlled validation protocol."""

from __future__ import annotations

import json
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(os.environ.get("OFFPOLICYRL_ROOT", "/opt/tiger/offpolicyrl"))
CODE_ROOT = ROOT / "verl"
VENDOR_DIR = CODE_ROOT / ".ray_vendor"
BASE_CONFIG = ROOT / "configs/PR2_CTPO_off2_oversample0p1_qwen3_30b_a3b_4x8_b64n8_r16384_300.json"
MATRIX_RECIPE = CODE_ROOT / "training_jobs/scripts/moe_rl/run_pr2_r2_and_geom_exact_aligned_matrix.py"
PROJECT = "verl_moe_router_replay"
WANDB_GROUP = "PR2_off8_paper_aligned_reval_20260830"
HDFS_BIN = "/opt/tiger/arnold/hdfs_client/hdfs"
HDFS_CONF = "/opt/tiger/arnold/hdfs_client/conf/celer_china-north5"

CHECKPOINTS = {
    "gspo": (
        "moe-pr2-gspo-baseline-off8-over01-r16384-4x8-20260815",
        300,
    ),
    "r2": (
        "moe-grpo-r2-off8-aligned-nokl-over01-r16384-4x8-20260824",
        300,
    ),
    "r3": (
        "moe-grpo-r3-off8-aligned-nokl-over01-r16384-4x8-20260824",
        300,
    ),
    "exact_dual": (
        "moe-pr2-geomprob-exact-dual-d5e4-2e3-off8-over01-seed42-r16384-4x8-20260828",
        300,
    ),
    "grpo_diag": (
        "moe-grpo-nokl-off8-clip020-028-over01-r16384-4x8-20260830",
        80,
    ),
}
HDFS_BASE = "hdfs://harunawl/home/byte_data_seed_wl/user/wu.hanlin/offpolicyrl/checkpoints"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load recipe: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_environment(state: Path, run_id: str) -> None:
    for path in Path("/home/tiger/.rh2/entrypoint_envs").glob("*"):
        try:
            os.environ[path.name] = path.read_text().strip()
        except OSError:
            pass
    os.environ["HADOOP_CONF_DIR"] = HDFS_CONF
    secret_text = (ROOT / "run_prefix_ripo_experiments.sh").read_text()
    match = re.search(r"WANDB_API_KEY=['\"]?([^'\"\s]+)", secret_text)
    if not match:
        raise RuntimeError("Could not locate the established WANDB_API_KEY")
    os.environ["WANDB_API_KEY"] = match.group(1)
    proxy = "http://sys-proxy-rd-relay.byted.org:8118"
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.setdefault(name, proxy)
    os.environ.update(
        {
            "WANDB_ENTITY": "hanlinw",
            "WANDB_PROJECT": PROJECT,
            "WANDB_RUN_ID": run_id,
            "WANDB_NAME": run_id,
            "WANDB_RUN_GROUP": WANDB_GROUP,
            "WANDB_RESUME": "allow",
            "WANDB_MODE": "online",
            "WANDB_BASE_URL": "https://api.wandb.ai",
            "WANDB_DIR": str(state / "wandb"),
            "VERL_FILE_LOGGER_PATH": str(state / "metrics.jsonl"),
            "TENSORBOARD_DIR": str(state / "tensorboard"),
            "TENSORBOARD_LOG_PATH": str(state / "tensorboard"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_USE_V1": "1",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "RAY_DEDUP_LOGS": "0",
            "RAY_ADDRESS": "auto",
            "RAY_RUNTIME_ENV_IGNORE_GITIGNORE": "1",
            "PYTHONNOUSERSITE": "1",
            "OFFPOLICYRL_ROOT": str(ROOT),
            "MOE_RUNTIME_ASSET_ROOT": "/opt/tiger/offpolicyrl/runtime_assets",
        }
    )


def hdfs_text(path: str) -> str:
    return subprocess.run(
        [HDFS_BIN, "dfs", "-cat", path], check=True, text=True, capture_output=True
    ).stdout.strip()


def hdfs_exists(path: str) -> bool:
    return subprocess.run(
        [HDFS_BIN, "dfs", "-test", "-e", path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def require_official_wandb() -> None:
    sys.path.insert(0, str(VENDOR_DIR))
    os.environ["PYTHONPATH"] = f"{VENDOR_DIR}:{CODE_ROOT}"
    import wandb

    resolved = Path(wandb.__file__).resolve()
    if not wandb.__version__ or "wandb" not in resolved.parts:
        raise RuntimeError(f"Invalid W&B SDK: version={wandb.__version__}, path={resolved}")
    print(f"WANDB_SDK_AUDIT version={wandb.__version__} path={resolved}", flush=True)


def configure_checkpoint_resume(trainer: dict, checkpoint_root: str, state: Path, step: int) -> None:
    """Pin HDFS restore to one immutable checkpoint even while training advances latest."""
    trainer.update(
        {
            "resume_mode": "resume_path",
            "resume_from_path": str(state / "checkpoints" / f"global_step_{step}"),
            "default_hdfs_dir": checkpoint_root,
            "default_local_dir": str(state / "checkpoints"),
        }
    )


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] not in CHECKPOINTS:
        choices = "|".join(CHECKPOINTS)
        raise SystemExit(f"usage: {sys.argv[0]} {choices} TOP_P")
    method = sys.argv[1]
    top_p = float(sys.argv[2])
    if top_p not in {0.7, 1.0}:
        raise SystemExit("TOP_P must be 0.7 or 1.0")

    checkpoint_slug, expected_step = CHECKPOINTS[method]
    checkpoint_root = f"{HDFS_BASE}/{checkpoint_slug}"
    top_p_slug = "0p7" if top_p == 0.7 else "1p0"
    run_id = f"reval_pr2_off8_{method}_step{expected_step}_t1_tp{top_p_slug}_n32_20260830"
    state = Path("/tmp/offpolicyrl_run_state") / run_id
    for path in (state, state / "wandb", state / "tensorboard", state / "checkpoints", state / "validation"):
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(0o777)

    load_environment(state, run_id)
    require_official_wandb()
    latest_marker = int(hdfs_text(f"{checkpoint_root}/latest_checkpointed_iteration.txt"))
    if not hdfs_exists(f"{checkpoint_root}/global_step_{expected_step}/_SUCCESS"):
        raise RuntimeError(f"Incomplete checkpoint: {checkpoint_root}/global_step_{expected_step}")

    # Build from the exact same successful recipe chain as the completed
    # off8 runs.  Besides setting the runtime topology, this removes legacy
    # actor keys that are not accepted by the current config dataclass.
    matrix = load_module(MATRIX_RECIPE, "pr2_reval_matrix_source")
    source = matrix._load_source_recipe()
    base_source = source._load_source_recipe()
    payload = matrix.configure(
        json.loads(base_source.BASE_CONFIG.read_text()),
        "geom_off8",
        state_root=state,
        source=source,
    )
    data = payload["data"]
    asset_root = Path("/opt/tiger/offpolicyrl/runtime_assets")
    data["train_files"] = str(asset_root / "data/moe/math-17k.parquet")
    data["val_files"] = [
        str(asset_root / "data/moe/evals/amc23.parquet"),
        str(asset_root / "data/moe/evals/aime24.parquet"),
        str(asset_root / "data/moe/evals/aime25.parquet"),
        str(asset_root / "data/moe/evals/hmmt25.parquet"),
    ]
    data["validation_shuffle"] = False
    payload["actor_rollout_ref"]["model"]["path"] = str(asset_root / "models/Qwen3-30B-A3B-Base")

    rollout = payload["actor_rollout_ref"]["rollout"]
    rollout["val_kwargs"].update(
        {"do_sample": True, "n": 32, "temperature": 1.0, "top_k": -1, "top_p": top_p}
    )
    rollout["trace"]["project_name"] = PROJECT
    rollout["trace"]["experiment_name"] = run_id

    trainer = payload["trainer"]
    trainer.update(
        {
            "project_name": PROJECT,
            "experiment_name": run_id,
            "nnodes": 4,
            "n_gpus_per_node": 8,
            "val_before_train": True,
            "val_only": True,
            "validation_data_dir": str(state / "validation"),
            "del_local_ckpt_after_load": True,
            "logger": ["console", "file", "tensorboard", "wandb"],
            "log_val_generations": 0,
        }
    )
    configure_checkpoint_resume(trainer, checkpoint_root, state, expected_step)
    payload["actor_rollout_ref"]["_checkpoint_hdfs_dir"] = checkpoint_root
    runtime_env = payload.setdefault("ray_kwargs", {}).setdefault("ray_init", {}).setdefault("runtime_env", {})
    runtime_env.setdefault("env_vars", {}).update(
        {
            "WANDB_RUN_GROUP": WANDB_GROUP,
            "PYTHONPATH": f"{VENDOR_DIR}:{CODE_ROOT}",
            "PYTHONNOUSERSITE": "1",
            "OFFPOLICYRL_ROOT": str(ROOT),
            "MOE_RUNTIME_ASSET_ROOT": str(asset_root),
            "VLLM_DISABLE_COMPILE_CACHE": "1",
            "VLLM_CACHE_ROOT": f"/tmp/vllm_cache_{run_id}",
        }
    )

    assert trainer["val_only"] is True and trainer["val_before_train"] is True
    assert trainer["default_hdfs_dir"] == checkpoint_root
    assert trainer["resume_mode"] == "resume_path"
    assert trainer["resume_from_path"].endswith(f"global_step_{expected_step}")
    assert rollout["val_kwargs"]["n"] == 32
    assert rollout["val_kwargs"]["temperature"] == 1.0
    assert rollout["val_kwargs"]["top_p"] == top_p
    assert len(data["val_files"]) == 4
    for legacy_key in ("clip_ratio_la_power", "la_clip_low", "la_clip_high", "la_clip_c"):
        assert legacy_key not in payload["actor_rollout_ref"]["actor"]
    print(
        "PR2_REVAL_AUDIT",
        f"method={method}",
        f"checkpoint_step={expected_step}",
        f"latest_available_step={latest_marker}",
        f"engine={rollout['name']}",
        "temperature=1.0",
        f"top_p={top_p}",
        "n_all=32",
        "report_aime_n=32",
        "report_amc_hmmt_first_n=16",
        f"checkpoint={checkpoint_root}",
        f"wandb_run={run_id}",
        f"wandb_group={WANDB_GROUP}",
        flush=True,
    )

    from omegaconf import OmegaConf
    sys.path.insert(0, str(CODE_ROOT))
    from verl.trainer.main_ppo import run_ppo

    run_ppo(OmegaConf.create(payload))


if __name__ == "__main__":
    main()
