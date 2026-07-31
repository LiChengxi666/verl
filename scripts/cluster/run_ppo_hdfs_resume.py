# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Launch a Ray PPO job from an existing multi-node HDFS checkpoint."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def override_key(override: str) -> str:
    return override.lstrip("+").split("=", 1)[0]


def set_override(overrides: list[str], key: str, value: str, *, add: bool = False) -> None:
    overrides[:] = [item for item in overrides if override_key(item) != key]
    overrides.append(f"{'+' if add else ''}{key}={value}")


def load_overrides(path: Path) -> list[str]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, list) or not all(isinstance(item, str) for item in loaded):
        raise ValueError(f"Expected a YAML list of Hydra overrides in {path}")
    return loaded


def load_environment_directory(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Environment directory does not exist: {path}")
    for env_file in path.iterdir():
        if env_file.is_file():
            os.environ.setdefault(env_file.name, env_file.read_text(encoding="utf-8").strip())


def hdfs_checkpoint_step(hdfs_dir: str) -> int | None:
    if shutil.which("hdfs") is None:
        raise RuntimeError("The hdfs executable is required for checkpoint preflight")
    marker = f"{hdfs_dir.rstrip('/')}/latest_checkpointed_iteration.txt"
    exists = subprocess.run(
        ["hdfs", "dfs", "-test", "-e", marker],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if exists.returncode != 0:
        return None
    value = subprocess.check_output(["hdfs", "dfs", "-cat", marker], text=True).strip()
    try:
        step = int(value)
    except ValueError as exc:
        raise RuntimeError(f"Invalid checkpoint iteration in {marker}: {value!r}") from exc
    success_marker = f"{hdfs_dir.rstrip('/')}/global_step_{step}/_SUCCESS"
    subprocess.run(
        ["hdfs", "dfs", "-test", "-e", success_marker],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return step


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overrides-file", type=Path, required=True)
    parser.add_argument("--hdfs-checkpoint-dir", required=True)
    parser.add_argument("--local-state-dir", type=Path, required=True)
    parser.add_argument("--project-name", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--total-training-steps", type=int)
    parser.add_argument("--set", dest="extra_overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--security-env-dir", type=Path)
    parser.add_argument("--hadoop-conf-dir", type=Path)
    parser.add_argument("--wandb-api-key-file", type=Path)
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-id")
    parser.add_argument("--wandb-resume", choices=("must", "allow", "never", "auto"), default="must")
    parser.add_argument(
        "--allow-new-run",
        action="store_true",
        help="Allow launch when the HDFS checkpoint marker does not exist.",
    )
    return parser


def prepare_overrides(args: argparse.Namespace) -> list[str]:
    overrides = load_overrides(args.overrides_file)
    local_state_dir = args.local_state_dir.resolve()
    required = {
        "trainer.project_name": args.project_name,
        "trainer.experiment_name": args.experiment_name,
        "trainer.default_hdfs_dir": args.hdfs_checkpoint_dir,
        "trainer.default_local_dir": str(local_state_dir / "checkpoints"),
        "trainer.validation_data_dir": str(local_state_dir / "validation_generations"),
        "trainer.del_local_ckpt_after_load": "True",
        "trainer.max_actor_ckpt_to_keep": "1",
        "trainer.max_critic_ckpt_to_keep": "1",
        "trainer.max_complete_ckpt_to_keep": "1",
    }
    if args.total_training_steps is not None:
        required["trainer.total_training_steps"] = str(args.total_training_steps)
    for key, value in required.items():
        set_override(overrides, key, value)
    for item in args.extra_overrides:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE for --set, got {item!r}")
        key, value = item.split("=", 1)
        add = key.startswith("+")
        set_override(overrides, key.lstrip("+"), value, add=add)
    return overrides


def main() -> None:
    args = build_parser().parse_args()
    if args.security_env_dir is not None:
        load_environment_directory(args.security_env_dir)
    if args.hadoop_conf_dir is not None:
        os.environ["HADOOP_CONF_DIR"] = str(args.hadoop_conf_dir.resolve())
    if not os.environ.get("HADOOP_CONF_DIR"):
        raise RuntimeError("Set HADOOP_CONF_DIR or pass --hadoop-conf-dir")

    step = hdfs_checkpoint_step(args.hdfs_checkpoint_dir)
    if step is None and not args.allow_new_run:
        raise RuntimeError(
            "No completed checkpoint marker was found. Pass --allow-new-run only when intentionally starting fresh."
        )

    if args.wandb_api_key_file is not None:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key_file.read_text(encoding="utf-8").strip()
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("Set WANDB_API_KEY or pass --wandb-api-key-file")

    local_state_dir = args.local_state_dir.resolve()
    for relative in ("checkpoints", "validation_generations", "wandb", "tensorboard"):
        (local_state_dir / relative).mkdir(parents=True, exist_ok=True)

    run_id = args.wandb_run_id or args.experiment_name
    os.environ.update(
        {
            "RAY_ADDRESS": os.environ.get("RAY_ADDRESS", "auto"),
            "WANDB_RUN_ID": run_id,
            "WANDB_NAME": args.experiment_name,
            "WANDB_RESUME": args.wandb_resume,
            "WANDB_DIR": str(local_state_dir / "wandb"),
            "VERL_FILE_LOGGER_PATH": str(local_state_dir / "metrics.jsonl"),
            "TENSORBOARD_LOG_PATH": str(local_state_dir / "tensorboard"),
        }
    )
    if args.wandb_entity is not None:
        os.environ["WANDB_ENTITY"] = args.wandb_entity

    overrides = prepare_overrides(args)
    status = f"checkpoint_step={step}" if step is not None else "new_run=True"
    print(
        f"Launching project={args.project_name} experiment={args.experiment_name} {status}",
        flush=True,
    )
    os.execvpe(
        sys.executable,
        [sys.executable, "-m", "verl.trainer.main_ppo", *overrides],
        os.environ,
    )


if __name__ == "__main__":
    main()
