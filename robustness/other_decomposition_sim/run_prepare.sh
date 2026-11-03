#!/usr/bin/env bash
RELEASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${RELEASE_SCRIPT_DIR}/../.." && pwd)"
cd "${RELEASE_SCRIPT_DIR}"
mkdir -p logs
set -euo pipefail

SCRIPT_DIR="$(cd "${RELEASE_SCRIPT_DIR}" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_tasks.py" "$@"
