#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export POLICY_LOSS_MODE=gspo
export ROUTER_REPLAY_MODE=R3
exec bash "${SCRIPT_DIR}/common/launch_qwen3_30b_a3b.sh" "$@"
