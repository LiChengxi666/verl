"""Run probability-weighted exact-prefix KL clipping with geometric surrogate.

The four variants use the historical exact-prefix low/high budget ratios, with
all budgets scaled by 1e-3 and the upper budget independently doubled.  The
loss uses the old-policy prefix probability in the exact-KL coordinate budget
and geometric-average surrogate gradients.  This recipe intentionally does
not enable a negative-advantage dual clip.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("OFFPOLICYRL_ROOT", "/opt/tiger/offpolicyrl"))
BASE_CONFIG = ROOT / "configs/PR2_CTPO_off2_oversample0p1_qwen3_30b_a3b_4x8_b64n8_r16384_300.json"
PROJECT = "verl_moe_router_replay"

SCALE = 1e-3
GROUP_SPACING = 8.0
GROUPS = {
    "0p5x": (7.780049e-5 * SCALE, 3.151205e-4 * SCALE * 2.0),
    "1x": (7.780049e-5 * SCALE * GROUP_SPACING, 3.151205e-4 * SCALE * 2.0 * GROUP_SPACING),
    "2x": (7.780049e-5 * SCALE * GROUP_SPACING**2, 3.151205e-4 * SCALE * 2.0 * GROUP_SPACING**2),
    "4x": (7.780049e-5 * SCALE * GROUP_SPACING**3, 3.151205e-4 * SCALE * 2.0 * GROUP_SPACING**3),
    "ripo_match": (1.0e-7, 2.0e-7),
}


def configure(base: dict, group: str, *, state_root: Path, hdfs_dir: str, run_id: str) -> dict:
    if group not in GROUPS:
        raise ValueError(f"unknown group {group!r}; choose one of {sorted(GROUPS)}")
    low, high = GROUPS[group]
    payload = json.loads(json.dumps(base))
    asset_root = os.environ.get("MOE_RUNTIME_ASSET_ROOT")
    if asset_root:
        payload["data"]["train_files"] = str(Path(asset_root) / "data/math-17k.parquet")
        payload["data"]["val_files"] = [
            str(Path(asset_root) / "data/evals" / Path(p).name) for p in payload["data"]["val_files"]
        ]
        payload["actor_rollout_ref"]["model"]["path"] = str(Path(asset_root) / "model/Qwen3-30B-A3B-Base")
    actor = payload["actor_rollout_ref"]["actor"]
    policy_loss = actor["policy_loss"]
    policy_loss["loss_mode"] = "prefix_probability_weighted_exact_kl_clip"
    policy_loss["prefix_exact_kl_delta_low"] = low
    policy_loss["prefix_exact_kl_delta_high"] = high
    for key in ("clip_ratio_la_power", "la_clip_low", "la_clip_high", "la_clip_c"):
        actor.pop(key, None)
    actor["loss_agg_mode"] = "seq-mean-token-mean"

    trainer = payload["trainer"]
    trainer["project_name"] = PROJECT
    trainer["experiment_name"] = run_id
    trainer["default_hdfs_dir"] = hdfs_dir
    trainer["default_local_dir"] = str(state_root / "checkpoints")
    trainer["validation_data_dir"] = str(state_root / "validation")
    trainer["max_actor_ckpt_to_keep"] = 1
    trainer["max_critic_ckpt_to_keep"] = 1
    trainer["max_complete_ckpt_to_keep"] = 1
    trainer["del_local_ckpt_after_load"] = True
    payload["actor_rollout_ref"]["_checkpoint_hdfs_dir"] = hdfs_dir
    payload["actor_rollout_ref"]["rollout"]["trace"]["experiment_name"] = run_id

    runtime_env = payload.setdefault("ray_kwargs", {}).setdefault("ray_init", {}).setdefault("runtime_env", {})
    runtime_env.pop("working_dir", None)
    runtime_env.pop("excludes", None)
    env_vars = runtime_env.setdefault("env_vars", {})
    cache_root = f"/tmp/hf_{run_id}"
    env_vars.update(
        {
            "HF_DATASETS_CACHE": f"{cache_root}/datasets",
            "HF_HOME": cache_root,
            "HUGGINGFACE_HUB_CACHE": f"{cache_root}/hub",
            "TRANSFORMERS_CACHE": f"{cache_root}/transformers",
            "VLLM_CACHE_ROOT": f"/tmp/vllm_cache_{run_id}",
            "VLLM_DISABLE_COMPILE_CACHE": "1",
            "PYTHONPATH": f"{ROOT}:{ROOT / '.ray_vendor'}",
            "PYTHONNOUSERSITE": "1",
            "OFFPOLICYRL_ROOT": str(ROOT),
            "MOE_RUNTIME_ASSET_ROOT": os.environ.get("MOE_RUNTIME_ASSET_ROOT", "/tmp/offpolicyrl-runtime"),
            "TENSORBOARD_DIR": str(state_root / "tensorboard"),
        }
    )
    return payload


def load_environment(state: Path, run_id: str) -> None:
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
            "WANDB_PROJECT": PROJECT,
            "WANDB_RUN_ID": run_id,
            "WANDB_NAME": run_id,
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
            "MOE_RUNTIME_ASSET_ROOT": os.environ.get("MOE_RUNTIME_ASSET_ROOT", "/tmp/offpolicyrl-runtime"),
        }
    )


def require_official_wandb() -> None:
    vendor = ROOT / ".ray_vendor"
    if (vendor / "wandb" / "__init__.py").is_file():
        sys.path.insert(0, str(vendor))
    pythonpath = f"{vendor}{os.pathsep}{ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
    os.environ["PYTHONPATH"] = pythonpath.rstrip(os.pathsep)
    import wandb

    resolved = Path(wandb.__file__).resolve()
    if not wandb.__version__ or "wandb" not in resolved.parts:
        raise RuntimeError(f"Invalid W&B SDK: version={wandb.__version__}, path={resolved}")
    print(f"WANDB_SDK_AUDIT version={wandb.__version__} path={resolved}", flush=True)


def main() -> None:
    from omegaconf import OmegaConf

    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} GROUP, GROUP in {sorted(GROUPS)}")
    group = sys.argv[1]
    run_id = f"pr2_pwe_g_{group}_o2_over01"
    state_slug = f"pr2_prefix_prob_exact_geom_{group}_upper2x_off2_over01_moe32_20260813"
    state = Path("/tmp/offpolicyrl_run_state") / state_slug
    hdfs_dir = (
        "hdfs://harunawl/home/byte_data_seed_wl/user/wu.hanlin/offpolicyrl/checkpoints/"
        f"moe-pr2-prefix-prob-exact-geom-{group}-upper2x-off2-over01-r16384-4x8-20260813"
    )
    for path in (state, state / "wandb", state / "tensorboard", state / "checkpoints", state / "validation"):
        path.mkdir(parents=True, exist_ok=True)
    base = json.loads(BASE_CONFIG.read_text())
    payload = configure(base, group, state_root=state, hdfs_dir=hdfs_dir, run_id=run_id)
    load_environment(state, run_id)
    require_official_wandb()
    hdfs_test = subprocess.run(
        ["hdfs", "dfs", "-test", "-e", hdfs_dir],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if hdfs_test.returncode == 0:
        raise RuntimeError(f"Refusing to reuse existing HDFS path: {hdfs_dir}")

    actor = payload["actor_rollout_ref"]["actor"]
    rollout = payload["actor_rollout_ref"]["rollout"]
    trainer = payload["trainer"]
    assert actor["policy_loss"]["loss_mode"] == "prefix_probability_weighted_exact_kl_clip"
    assert actor["loss_agg_mode"] == "seq-mean-token-mean"
    assert rollout["over_sample_rate"] == 0.1 and rollout["n"] == 8
    assert actor["optim"]["lr"] == 2e-6
    assert trainer["test_freq"] == 5 and trainer["save_freq"] == 5 and trainer["total_training_steps"] == 300
    assert len(payload["data"]["val_files"]) == 4
    print(
        "PROB_WEIGHTED_EXACT_GEOM_CONFIG_AUDIT",
        f"group={group}", f"run_id={run_id}",
        f"delta_low={GROUPS[group][0]}", f"delta_high={GROUPS[group][1]}",
        f"oversample={rollout['over_sample_rate']}", f"rollout_n={rollout['n']}",
        f"lr={actor['optim']['lr']}", f"hdfs={hdfs_dir}", flush=True,
    )
    sys.path.insert(0, os.getcwd())
    from verl.trainer.main_ppo import run_ppo

    run_ppo(OmegaConf.create(payload))


if __name__ == "__main__":
    main()
