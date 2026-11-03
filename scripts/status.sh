#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/outputs/openrouter}"

"${PYTHON_BIN}" "${ROOT}/src/status.py" --outputs-dir "${RUN_ROOT}" "$@"
