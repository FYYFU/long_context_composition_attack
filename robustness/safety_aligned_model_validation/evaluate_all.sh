#!/usr/bin/env bash
RELEASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${RELEASE_SCRIPT_DIR}/../.." && pwd)"
cd "${RELEASE_SCRIPT_DIR}"
mkdir -p logs
set -euo pipefail

TASK_DIR="$(cd "${RELEASE_SCRIPT_DIR}" && pwd)"
REPO_DIR="$(cd "${TASK_DIR}/../.." && pwd)"
EVALUATOR="${REPO_DIR}/paper/eval/check_safe_gpt.py"
JUDGE_MODEL="${JUDGE_MODEL:-google/gemini-3.5-flash}"
JUDGE_NAME="${JUDGE_NAME:-${JUDGE_MODEL//\//-}}"
MODEL_FAMILY="${MODEL_FAMILY:-}"
MODEL_ALIASES="${MODEL_ALIASES:-}"
LEVELS="${LEVELS:-0,1,2,3}"
CONTEXT_LENGTHS="${CONTEXT_LENGTHS:-0k,16k}"
PYTHON_BIN="${PYTHON_BIN:-python}"

while IFS=$'\t' read -r alias model_id; do
  "${PYTHON_BIN}" "${EVALUATOR}" \
    --meta_path "${TASK_DIR}/results" \
    --judge_model "${JUDGE_MODEL}" \
    --judge_name "${JUDGE_NAME}" \
    --target_model_holder "OpenAI" \
    --reasoning_efforts "default" \
    --levels "${LEVELS}" \
    --context_lengths "${CONTEXT_LENGTHS}" \
    --model_name "${alias}" \
    --num_workers 16 \
    --timeout 180 \
    --api_provider openrouter \
    --skip_existing
done < <(
  MODEL_FAMILY="${MODEL_FAMILY}" MODEL_ALIASES="${MODEL_ALIASES}" \
  "${PYTHON_BIN}" -c '
import json, os, pathlib
p = pathlib.Path("'"${TASK_DIR}"'") / "models.json"
family = os.environ.get("MODEL_FAMILY", "")
aliases = {
    value.strip()
    for value in os.environ.get("MODEL_ALIASES", "").split(",")
    if value.strip()
}
for m in json.loads(p.read_text())["models"]:
    if family and m["family"] != family:
        continue
    if aliases and m["alias"] not in aliases:
        continue
    print("{}\t{}".format(m["alias"], m["model_id"]))
'
)
