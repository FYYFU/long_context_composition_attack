#!/usr/bin/env bash
RELEASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${RELEASE_SCRIPT_DIR}/../.." && pwd)"
cd "${RELEASE_SCRIPT_DIR}"
mkdir -p logs
set -euo pipefail

TASK_DIR="$(cd "${RELEASE_SCRIPT_DIR}" && pwd)"
JUDGE_MODEL="${JUDGE_MODEL:-google/gemini-3.5-flash}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "Evaluating all six checkpoints at 0k only with ${JUDGE_MODEL}."
CONTEXT_LENGTHS=0k \
JUDGE_MODEL="${JUDGE_MODEL}" \
PYTHON_BIN="${PYTHON_BIN}" \
  bash "${TASK_DIR}/evaluate_all.sh"

echo "Writing the 0k-only comparison tables."
"${PYTHON_BIN}" "${TASK_DIR}/summarize.py" \
  --judge-name "${JUDGE_MODEL//\//-}" \
  --contexts 0k \
  --output-prefix safety_results_0k

echo "0k-only formal evaluation completed."
