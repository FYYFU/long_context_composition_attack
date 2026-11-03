#!/usr/bin/env bash
RELEASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${RELEASE_SCRIPT_DIR}/../.." && pwd)"
cd "${RELEASE_SCRIPT_DIR}"
mkdir -p logs
set -euo pipefail

TASK_DIR="$(cd "${RELEASE_SCRIPT_DIR}" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${1:-AI-ISL/DeepSeek-R1-Distill-Qwen-7B-SP}"
ALIAS="${2:-smoke-safepath-qwen-7b}"
GPU="${GPU:-0}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_HOME="${SAFETY_HF_HOME:-${HOME}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${SAFETY_TRANSFORMERS_CACHE:-${HOME}/.cache/huggingface/hub}"
export VLLM_CACHE_ROOT="${SAFETY_VLLM_CACHE_ROOT:-${HOME}/.cache/vllm}"
export TOKENIZERS_PARALLELISM=false

"${PYTHON_BIN}" "${TASK_DIR}/run_inference.py" \
  --model "${MODEL}" \
  --alias "${ALIAS}" \
  --levels "0,1,2,3" \
  --contexts "0k,16k" \
  --n-examples 2 \
  --batch-size 2 \
  --output-dir "${TASK_DIR}/smoke_results" \
  --overwrite
