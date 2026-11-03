#!/usr/bin/env bash
RELEASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${RELEASE_SCRIPT_DIR}/../../.." && pwd)"
cd "${RELEASE_SCRIPT_DIR}"
mkdir -p logs
set -euo pipefail

ROOT="$(cd "${RELEASE_SCRIPT_DIR}" && pwd)"

MODEL_FAMILY=qwen-7b bash "${ROOT}/evaluate_all.sh"
python "${ROOT}/summarize.py" \
  --family qwen-7b \
  --output-prefix xstest_overrefusal_qwen
