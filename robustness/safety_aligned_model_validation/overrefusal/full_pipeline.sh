#!/usr/bin/env bash
RELEASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${RELEASE_SCRIPT_DIR}/../../.." && pwd)"
cd "${RELEASE_SCRIPT_DIR}"
mkdir -p logs
set -euo pipefail

ROOT="$(cd "${RELEASE_SCRIPT_DIR}" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPUS="${GPUS:-1,2}"

echo "Stage 1/3: generate responses for 250 safe XSTest prompts."
"${PYTHON_BIN}" "${ROOT}/schedule.py" \
  --python "${PYTHON_BIN}" \
  --gpus "${GPUS}"

echo "Stage 2/3: judge compliance versus over-refusal."
bash "${ROOT}/evaluate_all.sh"

echo "Stage 3/3: summarize paired Base/SafePATH/STAR-1 results."
python "${ROOT}/summarize.py"

echo "XSTest over-refusal pipeline completed."
