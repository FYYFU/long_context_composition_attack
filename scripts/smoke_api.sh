#!/usr/bin/env bash
# This command makes paid API calls: two target requests and up to two judge requests.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/outputs/smoke}"
PROVIDER="${PROVIDER:-google-vertex}"
JUDGE_MODEL="${JUDGE_MODEL:-google/gemini-2.0-flash-lite-001}"
: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY before running the API smoke test}"
"${PYTHON_BIN}" "${ROOT}/src/prepare_data.py"
"${PYTHON_BIN}" "${ROOT}/src/run_inference.py" \
  --model-id openai/gpt-oss-20b --model-name gpt-oss-20b \
  --samples 2 --levels 0 --contexts 0k --workers 1 \
  --provider "${PROVIDER}" --run-id smoke-2-v1 \
  --output-dir "${RUN_ROOT}/results" --checkpoint-dir "${RUN_ROOT}/checkpoints/inference"
"${PYTHON_BIN}" "${ROOT}/src/evaluate.py" \
  --models gpt-oss-20b --samples 2 --levels 0 --contexts 0k --workers 1 \
  --judge-model "${JUDGE_MODEL}" --provider "${PROVIDER}" \
  --results-dir "${RUN_ROOT}/results" --output-dir "${RUN_ROOT}/evaluations" \
  --checkpoint-dir "${RUN_ROOT}/checkpoints/judge"
