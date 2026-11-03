#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
VLLM_BIN="${VLLM_BIN:-vllm}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/outputs/local-vllm-20b}"
MODEL_PATH="${MODEL_PATH:-openai/gpt-oss-20b}"
SERVED_MODEL="${SERVED_MODEL:-openai/gpt-oss-20b}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
PORT="${PORT:-8001}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-10240}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
INFERENCE_WORKERS="${INFERENCE_WORKERS:-8}"
RUN_ID="${RUN_ID:-full520-local-vllm-tp4-temp1-v1}"
RUN_JUDGE="${RUN_JUDGE:-1}"
JUDGE_MODEL="${JUDGE_MODEL:-google/gemini-2.0-flash-lite-001}"
JUDGE_PROVIDER="${JUDGE_PROVIDER:-google-vertex}"
JUDGE_WORKERS="${JUDGE_WORKERS:-8}"
API_BASE_URL="http://127.0.0.1:${PORT}/v1"
SERVER_PID=""

IFS=',' read -r -a gpu_list <<< "${GPU_IDS}"
TP_SIZE="${TP_SIZE:-${#gpu_list[@]}}"

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

stop_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}
cleanup() {
  stop_server
  rm -f \
    "${RUN_ROOT}/artifacts/pipeline.pid" \
    "${RUN_ROOT}/artifacts/vllm_server.pid"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"${PYTHON_BIN}" "${ROOT}/src/prepare_data.py"
"${PYTHON_BIN}" "${ROOT}/src/validate_tasks.py" \
  --manifest "${RUN_ROOT}/artifacts/task_manifest.json"

echo "[vllm] GPUs=${GPU_IDS}, TP=${TP_SIZE}, model=${MODEL_PATH}"
setsid env \
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
  NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" \
  VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}" \
  "${VLLM_BIN}" serve "${MODEL_PATH}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --served-model-name "${SERVED_MODEL}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --reasoning-parser openai_gptoss \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --trust-remote-code \
    --disable-log-requests \
    > "${RUN_ROOT}/logs/vllm_server.log" 2>&1 &
SERVER_PID="$!"
printf '%s\n' "${SERVER_PID}" > "${RUN_ROOT}/artifacts/vllm_server.pid"

echo "[vllm] waiting for ${API_BASE_URL}/models"
ready=0
for _ in $(seq 1 300); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "vLLM exited during startup" >&2
    tail -100 "${RUN_ROOT}/logs/vllm_server.log" >&2
    exit 1
  fi
  if curl -fsS "${API_BASE_URL}/models" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done
if [[ "${ready}" != "1" ]]; then
  echo "vLLM did not become ready within 10 minutes" >&2
  exit 1
fi

"${PYTHON_BIN}" -u "${ROOT}/src/run_inference.py" \
  --model-id "${SERVED_MODEL}" \
  --model-name gpt-oss-20b \
  --api-base-url "${API_BASE_URL}" \
  --api-key EMPTY \
  --provider "" \
  --reasoning-effort medium \
  --samples 520 \
  --levels 0,1,2,3 \
  --contexts 0k,16k,64k \
  --temperature 1.0 \
  --top-p 1.0 \
  --max-tokens 16368 \
  --run-id "${RUN_ID}" \
  --workers "${INFERENCE_WORKERS}" \
  --output-dir "${RUN_ROOT}/results" \
  --checkpoint-dir "${RUN_ROOT}/checkpoints/inference" \
  2>&1 | tee -a "${RUN_ROOT}/logs/gpt-oss-20b_inference.log"

stop_server

if [[ "${RUN_JUDGE}" == "1" ]]; then
  if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "OPENROUTER_API_KEY is required for the judge stage" >&2
    exit 1
  fi
  "${PYTHON_BIN}" -u "${ROOT}/src/evaluate.py" \
    --models gpt-oss-20b \
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
    2>&1 | tee -a "${RUN_ROOT}/logs/gpt-oss-20b_judge.log"

  "${PYTHON_BIN}" "${ROOT}/src/compare_sample_sizes.py" \
    --evaluations-dir "${RUN_ROOT}/evaluations" \
    --output-dir "${RUN_ROOT}/summary" \
    --model gpt-oss-20b \
    --judge-slug "${JUDGE_MODEL//\//-}"
  "${PYTHON_BIN}" "${ROOT}/src/summarize.py" \
    --models gpt-oss-20b \
    --evaluations-dir "${RUN_ROOT}/evaluations" \
    --output-dir "${RUN_ROOT}/summary" \
    --judge-slug "${JUDGE_MODEL//\//-}"
fi

echo "[complete] outputs: ${RUN_ROOT}"
