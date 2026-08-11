"""Run cumulative exact-prefix clipping with a negative-advantage dual clip."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path("/opt/tiger/offpolicyrl")
BASE_CONFIG = ROOT / "configs/PR2_CTPO_off2_oversample0p1_qwen3_30b_a3b_4x8_b64n8_r16384_300.json"
RUN_ID = "PR2_exact_prefix_cumulative_dual_upper2x_off2_oversample0p1_qwen3_30b_a3b_4x8_b64n8_r16384_300"
STATE_SLUG = "pr2_exact_prefix_cumulative_dual_upper2x_off2_over01_moe32_20260811"
HDFS_DIR = (
    "hdfs://harunawl/home/byte_data_seed_wl/user/wu.hanlin/offpolicyrl/checkpoints/"
    "moe-pr2-exact-prefix-cumulative-dual-upper2x-off2-over01-r16384-4x8-20260811"
)
DELTA_LOW = 3.09912028333e-4
DELTA_HIGH = 5.17091807565e-3


def configure_exact_cumulative_dual(base: dict, *, state_root: Path) -> dict:
    """Change only the clipping method and experiment/checkpoint identities."""
    payload = json.loads(json.dumps(base))
    actor = payload["actor_rollout_ref"]["actor"]
    policy_loss = actor["policy_loss"]

    policy_loss["loss_mode"] = "prefix_exact_kl_cumulative_dual_clip"
    policy_loss["prefix_exact_kl_delta_low"] = DELTA_LOW
    policy_loss["prefix_exact_kl_delta_high"] = DELTA_HIGH

    # These fields belong only to the CTPO source loss. The cumulative exact
    # loss derives its dual cap directly from the solved exact upper bound.
    for key in ("clip_ratio_la_power", "la_clip_low", "la_clip_high", "la_clip_c"):
        actor.pop(key, None)

    trainer = payload["trainer"]
    trainer["experiment_name"] = RUN_ID
    trainer["default_hdfs_dir"] = HDFS_DIR
    payload["actor_rollout_ref"]["_checkpoint_hdfs_dir"] = HDFS_DIR
    payload["actor_rollout_ref"]["rollout"]["trace"]["experiment_name"] = RUN_ID

    trainer["default_local_dir"] = str(state_root / "checkpoints")
    trainer["validation_data_dir"] = str(state_root / "validation")
    trainer["max_actor_ckpt_to_keep"] = 1
    trainer["max_critic_ckpt_to_keep"] = 1
    trainer["max_complete_ckpt_to_keep"] = 1
    trainer["del_local_ckpt_after_load"] = True

    runtime_env = payload.setdefault("ray_kwargs", {}).setdefault("ray_init", {}).setdefault("runtime_env", {})
    runtime_env["working_dir"] = os.getcwd()
    runtime_env["excludes"] = ["/.git/", "/docs/", "/tests/", "/.github/", "/examples/"]
    env_vars = runtime_env.setdefault("env_vars", {})
    cache_root = f"/tmp/hf_{STATE_SLUG}"
    env_vars.update(
        {
            "HF_DATASETS_CACHE": f"{cache_root}/datasets",
            "HF_HOME": cache_root,
            "HUGGINGFACE_HUB_CACHE": f"{cache_root}/hub",
            "TRANSFORMERS_CACHE": f"{cache_root}/transformers",
            "VLLM_CACHE_ROOT": f"/tmp/vllm_cache_{STATE_SLUG}",
            "VLLM_DISABLE_COMPILE_CACHE": "1",
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

    state = Path("/tmp/offpolicyrl_run_state") / STATE_SLUG
    for path in (state, state / "wandb", state / "tensorboard", state / "checkpoints", state / "validation"):
        path.mkdir(parents=True, exist_ok=True)

    base = json.loads(BASE_CONFIG.read_text())
    payload = configure_exact_cumulative_dual(base, state_root=state)
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
    trainer = payload["trainer"]
    assert rollout["over_sample_rate"] == 0.1
    assert rollout["n"] == 8
    assert actor["optim"]["lr"] == 2e-6
    assert trainer["test_freq"] == 5 and trainer["save_freq"] == 5
    assert len(payload["data"]["val_files"]) == 4
    print(
        "EXACT_CUMULATIVE_DUAL_CONFIG_AUDIT",
        f"run_id={RUN_ID}",
        f"loss={actor['policy_loss']['loss_mode']}",
        f"aggregation={actor['loss_agg_mode']}",
        f"delta_low={DELTA_LOW}",
        f"delta_high={DELTA_HIGH}",
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
