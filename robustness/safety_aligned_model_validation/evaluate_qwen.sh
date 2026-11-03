#!/usr/bin/env bash
RELEASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${RELEASE_SCRIPT_DIR}/../.." && pwd)"
cd "${RELEASE_SCRIPT_DIR}"
mkdir -p logs
set -euo pipefail

TASK_DIR="$(cd "${RELEASE_SCRIPT_DIR}" && pwd)"
JUDGE_MODEL="${JUDGE_MODEL:-google/gemini-3.5-flash}"
JUDGE_NAME="${JUDGE_NAME:-google-gemini-3.5-flash-pre-retry}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "Evaluating the three Qwen-7B checkpoints with ${JUDGE_MODEL}."
echo "Judge output name: ${JUDGE_NAME}"
MODEL_FAMILY=qwen-7b \
JUDGE_MODEL="${JUDGE_MODEL}" \
JUDGE_NAME="${JUDGE_NAME}" \
PYTHON_BIN="${PYTHON_BIN}" \
  bash "${TASK_DIR}/evaluate_all.sh"

echo "Writing the preliminary Qwen comparison tables."
"${PYTHON_BIN}" "${TASK_DIR}/summarize.py" \
  --judge-name "${JUDGE_NAME}" \
  --family qwen-7b \
  --output-prefix qwen_safety_results_pre_retry
