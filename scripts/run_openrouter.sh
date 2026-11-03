#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/outputs/openrouter}"
MODELS="${MODELS:-gpt-oss-120b}"
RUN_ID="${RUN_ID:-full520-openrouter-google-vertex-v1}"
PROVIDER="${PROVIDER:-google-vertex}"
JUDGE_MODEL="${JUDGE_MODEL:-google/gemini-2.0-flash-lite-001}"
JUDGE_PROVIDER="${JUDGE_PROVIDER:-${PROVIDER}}"
RUN_JUDGE="${RUN_JUDGE:-1}"
INFERENCE_WORKERS_20B="${INFERENCE_WORKERS_20B:-1}"
INFERENCE_WORKERS_120B="${INFERENCE_WORKERS_120B:-8}"
JUDGE_WORKERS="${JUDGE_WORKERS:-8}"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is required" >&2
  exit 1
fi

mkdir -p \
  "${RUN_ROOT}/artifacts" \
  "${RUN_ROOT}/results" \
  "${RUN_ROOT}/evaluations" \
  "${RUN_ROOT}/checkpoints/inference" \
  "${RUN_ROOT}/checkpoints/judge" \
  "${RUN_ROOT}/logs" \
  "${RUN_ROOT}/summary"

exec 9> "${RUN_ROOT}/artifacts/pipeline.lock"
if ! flock -n 9; then
  echo "another pipeline already holds ${RUN_ROOT}/artifacts/pipeline.lock" >&2
  exit 1
fi
printf '%s\n' "$$" > "${RUN_ROOT}/artifacts/pipeline.pid"
trap 'rm -f "${RUN_ROOT}/artifacts/pipeline.pid"' EXIT

"${PYTHON_BIN}" "${ROOT}/src/prepare_data.py"
"${PYTHON_BIN}" "${ROOT}/src/validate_tasks.py" \
  --manifest "${RUN_ROOT}/artifacts/task_manifest.json"

model_id() {
  case "$1" in
    gpt-oss-20b) printf '%s\n' "openai/gpt-oss-20b" ;;
    gpt-oss-120b) printf '%s\n' "openai/gpt-oss-120b" ;;
    *) echo "unsupported model: $1" >&2; return 1 ;;
  esac
}

model_workers() {
  case "$1" in
    gpt-oss-20b) printf '%s\n' "${INFERENCE_WORKERS_20B}" ;;
    gpt-oss-120b) printf '%s\n' "${INFERENCE_WORKERS_120B}" ;;
    *) return 1 ;;
  esac
}

IFS=',' read -r -a model_list <<< "${MODELS}"
for model in "${model_list[@]}"; do
  model="${model//[[:space:]]/}"
  echo "[inference] ${model}, provider=${PROVIDER}"
  "${PYTHON_BIN}" -u "${ROOT}/src/run_inference.py" \
    --model-id "$(model_id "${model}")" \
    --model-name "${model}" \
    --api-base-url https://openrouter.ai/api/v1 \
    --provider "${PROVIDER}" \
    --reasoning-effort medium \
    --samples 520 \
    --levels 0,1,2,3 \
    --contexts 0k,16k,64k \
    --temperature 1.0 \
    --top-p 1.0 \
    --max-tokens 16368 \
    --run-id "${RUN_ID}" \
    --workers "$(model_workers "${model}")" \
    --output-dir "${RUN_ROOT}/results" \
    --checkpoint-dir "${RUN_ROOT}/checkpoints/inference" \
    2>&1 | tee -a "${RUN_ROOT}/logs/${model}_inference.log"

  if [[ "${RUN_JUDGE}" == "1" ]]; then
    echo "[judge] ${model}, judge=${JUDGE_MODEL}, provider=${JUDGE_PROVIDER}"
    "${PYTHON_BIN}" -u "${ROOT}/src/evaluate.py" \
      --models "${model}" \
      --reasoning-effort medium \
      --samples 520 \
      --levels 0,1,2,3 \
      --contexts 0k,16k,64k \
      --results-dir "${RUN_ROOT}/results" \
      --output-dir "${RUN_ROOT}/evaluations" \
      --checkpoint-dir "${RUN_ROOT}/checkpoints/judge" \
      --judge-model "${JUDGE_MODEL}" \
      --provider "${JUDGE_PROVIDER}" \
      --workers "${JUDGE_WORKERS}" \
      2>&1 | tee -a "${RUN_ROOT}/logs/${model}_judge.log"

    "${PYTHON_BIN}" "${ROOT}/src/compare_sample_sizes.py" \
      --evaluations-dir "${RUN_ROOT}/evaluations" \
      --output-dir "${RUN_ROOT}/summary" \
      --model "${model}" \
      --judge-slug "${JUDGE_MODEL//\//-}"
  fi
done

if [[ "${RUN_JUDGE}" == "1" ]]; then
  "${PYTHON_BIN}" "${ROOT}/src/summarize.py" \
    --models "${MODELS}" \
    --evaluations-dir "${RUN_ROOT}/evaluations" \
    --output-dir "${RUN_ROOT}/summary" \
    --judge-slug "${JUDGE_MODEL//\//-}"
fi

echo "[complete] outputs: ${RUN_ROOT}"
