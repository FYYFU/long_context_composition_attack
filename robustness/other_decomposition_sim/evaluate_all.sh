#!/usr/bin/env bash
RELEASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${RELEASE_SCRIPT_DIR}/../.." && pwd)"
cd "${RELEASE_SCRIPT_DIR}"
mkdir -p logs
set -euo pipefail

SCRIPT_DIR="$(cd "${RELEASE_SCRIPT_DIR}" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_WORKERS="${NUM_WORKERS:-16}"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "Error: OPENROUTER_API_KEY is not set." >&2
  exit 1
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate.py" \
  --judge-model "google/gemini-3.5-flash-lite" \
  --generators "qwen3" \
  --models "gpt-oss-20b-medium,gpt-oss-120b-medium" \
  --levels "0,1,2,3" \
  --contexts "0k,16k,64k" \
  --num-workers "${NUM_WORKERS}" \
  "$@"
