#!/usr/bin/env bash
RELEASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${RELEASE_SCRIPT_DIR}/../.." && pwd)"
cd "${RELEASE_SCRIPT_DIR}"
mkdir -p logs
set -euo pipefail

SCRIPT_DIR="$(cd "${RELEASE_SCRIPT_DIR}" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${1:-}"
PHASE="${2:-all}"

case "${MODEL}" in
    qwen3-next-80b-thinking)
        MODEL_NAME="qwen3-next-80b-a3b-thinking"
        REASONING_EFFORT="default"
        DEFAULT_BATCH=2
        DEFAULT_WORKERS=2
        REQUIRED_KEY="OPENROUTER_API_KEY"
        ;;
    gemini-3-flash)
        MODEL_NAME="gemini-3-flash-preview"
        REASONING_EFFORT="default"
        DEFAULT_BATCH=20
        DEFAULT_WORKERS=8
        REQUIRED_KEY="OPENROUTER_API_KEY"
        ;;
    gemini-3-pro)
        MODEL_NAME="gemini-3-pro-preview"
        REASONING_EFFORT="default"
        DEFAULT_BATCH=20
        DEFAULT_WORKERS=8
        REQUIRED_KEY="OPENROUTER_API_KEY"
        ;;
    *)
        echo "Usage: $0 {qwen3-next-80b-thinking|gemini-3-flash|gemini-3-pro} [inference|eval|all]" >&2
        exit 2
        ;;
esac

RESULT_DIR_NAME="${MODEL_NAME}-${REASONING_EFFORT}"

if [[ -z "${!REQUIRED_KEY:-}" ]]; then
    echo "Error: ${REQUIRED_KEY} is not set." >&2
    exit 1
fi

run_inference() {
    "${PYTHON_BIN}" "${SCRIPT_DIR}/run_model.py" "${MODEL}" \
        --batch-size "${BATCH_SIZE:-${DEFAULT_BATCH}}" \
        --num-workers "${NUM_WORKERS:-${DEFAULT_WORKERS}}" \
        --timeout "${TIMEOUT:-300}" \
        --max-retries "${MAX_RETRIES:-5}" \
        --max-tokens "${MAX_TOKENS:-16368}"
}

run_evaluation() {
    if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
        echo "Error: OPENROUTER_API_KEY is required by the Gemini judge." >&2
        exit 1
    fi
    "${PYTHON_BIN}" \
        "${SCRIPT_DIR}/../../controls/benign_baseline/eval_reconstruction.py" \
        --meta_path "${SCRIPT_DIR}/results" \
        --model_name "${RESULT_DIR_NAME}" \
        --reasoning_efforts "${REASONING_EFFORT}" \
        --judge_model "google/gemini-2.5-flash-lite" \
        --api_provider openrouter \
        --levels "0,1,2,3" \
        --context_lengths "0k,16k,64k" \
        --num_workers "${JUDGE_WORKERS:-8}" \
        --timeout "${JUDGE_TIMEOUT:-180}" \
        --skip_existing
}

case "${PHASE}" in
    inference) run_inference ;;
    eval) run_evaluation ;;
    all)
        run_inference
        run_evaluation
        ;;
    *)
        echo "Phase must be inference, eval, or all." >&2
        exit 2
        ;;
esac
