"""Run paper-exact RIPO on the aligned PR2 off2/oversampling MoE baseline."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path("/opt/tiger/offpolicyrl")
BASE_CONFIG = ROOT / "configs/PR2_CTPO_off2_oversample0p1_qwen3_30b_a3b_4x8_b64n8_r16384_300.json"
RUN_ID = "RIPO_d005_off2_over01_Qwen3-30B-A3B_4x8_r16384_s300"
STATE_SLUG = "pr2_ripo_paper_delta0p05_off2_over01_moe32_20260811_v4"
HDFS_DIR = (
    "hdfs://harunawl/home/byte_data_seed_wl/user/wu.hanlin/offpolicyrl/checkpoints/"
    "moe-pr2-ripo-paper-delta0p05-off2-over01-r16384-4x8-20260811-v4"
)


def configure_ripo(base: dict) -> dict:
    """Change only method-bound fields plus run/checkpoint identities."""
    payload = json.loads(json.dumps(base))
    actor = payload["actor_rollout_ref"]["actor"]
    policy_loss = actor["policy_loss"]

    actor["loss_agg_mode"] = "token-mean"
    policy_loss["loss_mode"] = "ripo_clip"
    policy_loss["ripo_delta_low"] = 0.05
    policy_loss["ripo_delta_high"] = 0.05
    policy_loss["ripo_ratio_lower"] = 0.5
    policy_loss["ripo_ratio_upper"] = 10.0

    # Remove parameters belonging exclusively to the source CTPO loss. They
    # have no RIPO semantics and retaining them would make recipe audits noisy.
    for key in ("clip_ratio_la_power", "la_clip_low", "la_clip_high", "la_clip_c"):
        actor.pop(key, None)

    trainer = payload["trainer"]
    trainer["experiment_name"] = RUN_ID
    trainer["default_hdfs_dir"] = HDFS_DIR
    payload["actor_rollout_ref"]["_checkpoint_hdfs_dir"] = HDFS_DIR
    payload["actor_rollout_ref"]["rollout"]["trace"]["experiment_name"] = RUN_ID

    state = Path("/tmp/offpolicyrl_run_state") / STATE_SLUG
    for path in (state, state / "wandb", state / "tensorboard", state / "checkpoints", state / "validation"):
        path.mkdir(parents=True, exist_ok=True)
    trainer["default_local_dir"] = str(state / "checkpoints")
    trainer["validation_data_dir"] = str(state / "validation")
    trainer["max_actor_ckpt_to_keep"] = 1
    trainer["max_critic_ckpt_to_keep"] = 1
    trainer["max_complete_ckpt_to_keep"] = 1
    trainer["del_local_ckpt_after_load"] = True

    runtime_env = payload.setdefault("ray_kwargs", {}).setdefault("ray_init", {}).setdefault("runtime_env", {})
    runtime_env["working_dir"] = os.getcwd()
    runtime_env["excludes"] = ["/.git/", "/docs/", "/tests/", "/.github/", "/examples/"]
    cache_env = runtime_env.setdefault("env_vars", {})
    cache_root = f"/tmp/hf_{STATE_SLUG}"
    cache_env.update(
        {
            "HF_DATASETS_CACHE": f"{cache_root}/datasets",
            "HF_HOME": cache_root,
            "HUGGINGFACE_HUB_CACHE": f"{cache_root}/hub",
            "TRANSFORMERS_CACHE": f"{cache_root}/transformers",
            "VLLM_CACHE_ROOT": f"/tmp/vllm_cache_{STATE_SLUG}",
            "VLLM_DISABLE_COMPILE_CACHE": "1",
            # Match the known-good exact-prefix jobs: always import the
            # isolated official W&B SDK instead of ByteDance's user-site shim.
            "PYTHONPATH": ".ray_vendor",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return payload


def load_environment(state: Path) -> None:
    for path in Path("/home/tiger/.rh2/entrypoint_envs").glob("*"):
        try:
            os.environ[path.name] = path.read_text().strip()
        except OSError:
            pass

    os.environ["HADOOP_CONF_DIR"] = "/opt/tiger/arnold/hdfs_client/conf/celer_china-north5"
    secret_text = (ROOT / "run_prefix_ripo_experiments.sh").read_text()
    match = re.search(r"WANDB_API_KEY=['\"]?([^'\"\s]+)", secret_text)
    if not match:
        raise RuntimeError("Could not locate the existing WANDB_API_KEY")
    os.environ["WANDB_API_KEY"] = match.group(1)

    proxy = "http://sys-proxy-rd-relay.byted.org:8118"
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.setdefault(name, proxy)
    os.environ.update(
        {
            "WANDB_ENTITY": "hanlinw",
            "WANDB_RUN_ID": RUN_ID,
            "WANDB_NAME": RUN_ID,
            "WANDB_RESUME": "allow",
            "WANDB_MODE": "online",
            "WANDB_BASE_URL": "https://api.wandb.ai",
            "WANDB_DIR": str(state / "wandb"),
            "VERL_FILE_LOGGER_PATH": str(state / "metrics.jsonl"),
            "TENSORBOARD_LOG_PATH": str(state / "tensorboard"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_USE_V1": "1",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "RAY_DEDUP_LOGS": "0",
            "RAY_ADDRESS": "auto",
            "RAY_RUNTIME_ENV_IGNORE_GITIGNORE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )


def require_official_wandb() -> None:
    vendor = Path.cwd() / ".ray_vendor"
    if not (vendor / "wandb" / "__init__.py").is_file():
        raise RuntimeError(f"Official vendored W&B package is missing: {vendor}")
    sys.path.insert(0, str(vendor))
    os.environ["PYTHONPATH"] = f"{vendor}{os.pathsep}{os.environ.get('PYTHONPATH', '')}".rstrip(os.pathsep)

    import wandb

    resolved = Path(wandb.__file__).resolve()
    if wandb.__version__ != "0.21.1" or vendor.resolve() not in resolved.parents:
        raise RuntimeError(f"Refusing non-public W&B SDK: version={wandb.__version__}, path={resolved}")
    print(f"PUBLIC_WANDB_AUDIT version={wandb.__version__} path={resolved}", flush=True)


def main() -> None:
    from omegaconf import OmegaConf

    base = json.loads(BASE_CONFIG.read_text())
    payload = configure_ripo(base)
    state = Path(payload["trainer"]["default_local_dir"]).parent
    load_environment(state)
    require_official_wandb()

    if subprocess.run(
        ["hdfs", "dfs", "-test", "-e", HDFS_DIR],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0:
        raise RuntimeError(f"Refusing to reuse existing HDFS path: {HDFS_DIR}")

    actor = payload["actor_rollout_ref"]["actor"]
    rollout = payload["actor_rollout_ref"]["rollout"]
    print(
        "RIPO_CONFIG_AUDIT",
        f"run_id={RUN_ID}",
        f"loss={actor['policy_loss']['loss_mode']}",
        f"aggregation={actor['loss_agg_mode']}",
        "delta=0.05/0.05",
        "ratio_bounds=0.5/10.0",
        f"oversample={rollout['over_sample_rate']}",
        f"rollout_n={rollout['n']}",
        f"lr={actor['optim']['lr']}",
        f"hdfs={HDFS_DIR}",
        flush=True,
    )

    sys.path.insert(0, os.getcwd())
    from verl.trainer.main_ppo import run_ppo

    run_ppo(OmegaConf.create(payload))


if __name__ == "__main__":
    main()
