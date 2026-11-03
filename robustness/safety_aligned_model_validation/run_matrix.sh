#!/usr/bin/env bash
RELEASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${RELEASE_SCRIPT_DIR}/../.." && pwd)"
cd "${RELEASE_SCRIPT_DIR}"
mkdir -p logs
set -euo pipefail

TASK_DIR="$(cd "${RELEASE_SCRIPT_DIR}" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPUS="${GPUS:-0,1,2,3}"

export HF_HOME="${SAFETY_HF_HOME:-${HOME}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${SAFETY_TRANSFORMERS_CACHE:-${HOME}/.cache/huggingface/hub}"
export VLLM_CACHE_ROOT="${SAFETY_VLLM_CACHE_ROOT:-${HOME}/.cache/vllm}"
export TOKENIZERS_PARALLELISM=false

exec "${PYTHON_BIN}" "${TASK_DIR}/schedule_matrix.py" \
  --python "${PYTHON_BIN}" \
  --gpus "${GPUS}" \
  "$@"
