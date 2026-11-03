#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/outputs/local-vllm-20b}"
MODEL="${MODEL:-gpt-oss-20b}"
JUDGE_MODEL="${JUDGE_MODEL:-google/gemini-2.0-flash-lite-001}"
JUDGE_PROVIDER="${JUDGE_PROVIDER:-google-vertex}"
JUDGE_WORKERS="${JUDGE_WORKERS:-8}"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is required" >&2
  exit 1
fi

"${PYTHON_BIN}" -u "${ROOT}/src/evaluate.py" \
  --models "${MODEL}" \
  --reasoning-effort medium \
  --samples 520 \
  --levels 0,1,2,3 \
  --contexts 0k,16k,64k \
  --results-dir "${RUN_ROOT}/results" \
  --output-dir "${RUN_ROOT}/evaluations" \
  --checkpoint-dir "${RUN_ROOT}/checkpoints/judge" \
  --judge-model "${JUDGE_MODEL}" \
  --provider "${JUDGE_PROVIDER}" \
  --workers "${JUDGE_WORKERS}"

"${PYTHON_BIN}" "${ROOT}/src/compare_sample_sizes.py" \
  --evaluations-dir "${RUN_ROOT}/evaluations" \
  --output-dir "${RUN_ROOT}/summary" \
  --model "${MODEL}" \
  --judge-slug "${JUDGE_MODEL//\//-}"

"${PYTHON_BIN}" "${ROOT}/src/summarize.py" \
  --models "${MODEL}" \
  --evaluations-dir "${RUN_ROOT}/evaluations" \
  --output-dir "${RUN_ROOT}/summary" \
  --judge-slug "${JUDGE_MODEL//\//-}"
