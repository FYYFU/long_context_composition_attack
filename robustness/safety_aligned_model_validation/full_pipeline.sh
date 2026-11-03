#!/usr/bin/env bash
RELEASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${RELEASE_SCRIPT_DIR}/../.." && pwd)"
cd "${RELEASE_SCRIPT_DIR}"
mkdir -p logs
set -euo pipefail

TASK_DIR="$(cd "${RELEASE_SCRIPT_DIR}" && pwd)"
GPUS="${GPUS:-0,1,2,3}"

echo "Stage 1/4: resume and complete the six-model generation matrix."
GPUS="${GPUS}" bash "${TASK_DIR}/run_matrix.sh"

echo "Stage 2/4: regenerate only rows truncated by the earlier 8192-token limit."
GPUS="${GPUS}" bash "${TASK_DIR}/run_matrix.sh" --retry-truncated

echo "Stage 3/4: evaluate all completed generations with the paper's judge."
bash "${TASK_DIR}/evaluate_all.sh"

echo "Stage 4/4: create CSV, Markdown, and LaTeX summary tables."
python "${TASK_DIR}/summarize.py"

echo "Full safety-aligned validation pipeline completed."
