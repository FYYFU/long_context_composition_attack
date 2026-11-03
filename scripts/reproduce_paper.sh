#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/paper100}"
"${PYTHON_BIN}" "${ROOT}/src/prepare_reference.py" --scores-only
"${PYTHON_BIN}" "${ROOT}/robustness/statistic_analysis/analyze_statistics.py" \
  --output-dir "${OUTPUT_DIR}" "$@"
"${PYTHON_BIN}" "${ROOT}/src/plot_paper_results.py" --input-dir "${OUTPUT_DIR}"

