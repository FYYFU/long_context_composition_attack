#!/usr/bin/env bash
RELEASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${RELEASE_SCRIPT_DIR}/../../.." && pwd)"
cd "${RELEASE_SCRIPT_DIR}"
mkdir -p logs
set -euo pipefail

ROOT="$(cd "${RELEASE_SCRIPT_DIR}" && pwd)"
PARENT="${ROOT}/.."
EVALUATOR="$(
  cd "${ROOT}/../../../controls/system_prompt_defense/xstest"
  pwd
)/eval_xstest.py"
PYTHON_BIN="${PYTHON_BIN:-python}"
JUDGE_MODEL="${JUDGE_MODEL:-google/gemini-3.5-flash}"
MODEL_FAMILY="${MODEL_FAMILY:-}"

while IFS= read -r alias; do
  result_file="${ROOT}/results/${alias}/xstest_results.json"
  eval_file="${ROOT}/results/${alias}/xstest_eval.json"
  if [[ ! -f "${result_file}" ]]; then
    echo "Missing ${result_file}" >&2
    exit 1
  fi
  if [[ -f "${eval_file}" ]]; then
    echo "Skipping existing ${eval_file}"
    continue
  fi
  "${PYTHON_BIN}" "${EVALUATOR}" \
    --results_file "${result_file}" \
    --judge_model "${JUDGE_MODEL}" \
    --api_provider openrouter \
    --num_workers 16
done < <(
  MODEL_FAMILY="${MODEL_FAMILY}" "${PYTHON_BIN}" -c '
import json, os, pathlib
p = pathlib.Path("'"${PARENT}"'") / "models.json"
family = os.environ.get("MODEL_FAMILY", "")
for model in json.loads(p.read_text())["models"]:
    if family and model["family"] != family:
        continue
    print(model["alias"])
'
)
