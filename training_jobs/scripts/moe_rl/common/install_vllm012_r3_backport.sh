#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
PATCH_FILE="${REPO_ROOT}/training_jobs/patches/vllm/v0.12.0-router-replay-r3.patch"
VLLM_ROOT="${VLLM_ROOT:-/vllm}"
EXPECTED_BASE="4fd9d6a85c00ac0186aa9abbeff73fc2ac6c721e"

if [[ ! -d "${VLLM_ROOT}/.git" ]]; then
  echo "vLLM source checkout not found: ${VLLM_ROOT}" >&2
  exit 1
fi

if [[ ! -f "${PATCH_FILE}" ]]; then
  echo "R3 vLLM backport patch not found: ${PATCH_FILE}" >&2
  exit 1
fi

# A successful reverse check means the patch is already installed. This makes
# repeated node bootstrap safe after cluster recreation.
if (cd "${VLLM_ROOT}" && git apply --reverse --check "${PATCH_FILE}") >/dev/null 2>&1; then
  echo "R3 vLLM backport already installed in ${VLLM_ROOT}"
else
  actual_base="$(git -C "${VLLM_ROOT}" rev-parse HEAD)"
  if [[ "${actual_base}" != "${EXPECTED_BASE}" ]]; then
    echo "Refusing to patch unexpected vLLM revision: ${actual_base}" >&2
    echo "Expected: ${EXPECTED_BASE}" >&2
    exit 1
  fi

  (cd "${VLLM_ROOT}" && git apply --check "${PATCH_FILE}")
  if [[ -w "${VLLM_ROOT}/vllm" ]]; then
    git -C "${VLLM_ROOT}" apply "${PATCH_FILE}"
  else
    sudo -n git -C "${VLLM_ROOT}" apply "${PATCH_FILE}"
  fi
  echo "Installed R3 vLLM backport in ${VLLM_ROOT}"
fi

python3 "${SCRIPT_DIR}/check_r3_vllm.py"
