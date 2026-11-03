#!/usr/bin/env bash
RELEASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${RELEASE_SCRIPT_DIR}/../.." && pwd)"
cd "${RELEASE_SCRIPT_DIR}"
mkdir -p logs
set -euo pipefail

SCRIPT_DIR="$(cd "${RELEASE_SCRIPT_DIR}" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_FILE="${INPUT_FILE:-${WORKSPACE_ROOT}/paper/results/advbench_transformed_api_gemini-2.0-flash-001_fixed.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results_objective_aligned_first100}"
MODEL="${MODEL:-google/gemini-2.5-flash-lite}"
NUM_WORKERS="${NUM_WORKERS:-16}"
TIMEOUT="${TIMEOUT:-180}"
MAX_RETRIES="${MAX_RETRIES:-5}"
MAX_TOKENS="${MAX_TOKENS:-2048}"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "Error: OPENROUTER_API_KEY is not set." >&2
    exit 1
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_objective_aligned.py" \
    --input "${INPUT_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --model "${MODEL}" \
    --levels "1,2,3" \
    --max-samples 100 \
    --reasoning-effort "default" \
    --num-workers "${NUM_WORKERS}" \
    --timeout "${TIMEOUT}" \
    --max-retries "${MAX_RETRIES}" \
    --max-tokens "${MAX_TOKENS}" \
    "$@"
