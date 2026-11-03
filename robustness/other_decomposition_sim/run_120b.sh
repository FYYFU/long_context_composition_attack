#!/usr/bin/env bash
RELEASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${RELEASE_SCRIPT_DIR}/../.." && pwd)"
cd "${RELEASE_SCRIPT_DIR}"
mkdir -p logs
set -euo pipefail

SCRIPT_DIR="$(cd "${RELEASE_SCRIPT_DIR}" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
API_BASE_URL="${API_BASE_URL:-https://openrouter.ai/api/v1}"
MODEL_PATH="${MODEL_PATH:-openai/gpt-oss-120b}"
PROVIDER="${PROVIDER:-google-vertex}"
NUM_WORKERS="${NUM_WORKERS:-16}"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "Error: OPENROUTER_API_KEY is not set." >&2
  exit 1
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_inference.py" \
  --model-path "${MODEL_PATH}" \
  --model-output-name "gpt-oss-120b-medium" \
  --api-base-url "${API_BASE_URL}" \
  --provider "${PROVIDER}" \
  --reasoning-effort medium \
  --temperature 1 \
  --top-p 1 \
  --max-tokens 16368 \
  --num-workers "${NUM_WORKERS}" \
  "$@"
