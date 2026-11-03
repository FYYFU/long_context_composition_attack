#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" "${ROOT}/src/prepare_data.py"
"${PYTHON_BIN}" "${ROOT}/src/validate_tasks.py"
"${PYTHON_BIN}" "${ROOT}/src/prepare_reference.py" --verify-only
"${PYTHON_BIN}" "${ROOT}/src/verify_release.py"

echo "[complete] repository data and scripts verified"
