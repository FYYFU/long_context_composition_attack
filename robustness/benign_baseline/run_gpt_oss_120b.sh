#!/usr/bin/env bash
RELEASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${RELEASE_SCRIPT_DIR}/../.." && pwd)"
cd "${RELEASE_SCRIPT_DIR}"
mkdir -p logs
set -euo pipefail

SCRIPT_DIR="$(cd "${RELEASE_SCRIPT_DIR}" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PHASE="${1:-all}"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "Error: OPENROUTER_API_KEY is not set." >&2
    exit 1
fi

run_inference() {
    "${PYTHON_BIN}" "${SCRIPT_DIR}/run_gpt_oss_120b.py" \
        --num-workers "${NUM_WORKERS:-8}" \
        --batch-size "${BATCH_SIZE:-20}" \
        --timeout "${TIMEOUT:-300}" \
        --max-retries "${MAX_RETRIES:-5}" \
        --max-tokens "${MAX_TOKENS:-16368}"
}

run_evaluation() {
    "${PYTHON_BIN}" \
        "${SCRIPT_DIR}/../../controls/benign_baseline/eval_reconstruction.py" \
        --meta_path "${SCRIPT_DIR}/results" \
        --model_name "gpt-oss-120b" \
        --reasoning_efforts "medium" \
        --judge_model "google/gemini-2.5-flash-lite" \
        --api_provider openrouter \
        --levels "0,1,2,3" \
        --context_lengths "0k,16k,64k" \
        --num_workers "${JUDGE_WORKERS:-8}" \
        --timeout "${JUDGE_TIMEOUT:-180}" \
        --skip_existing
}

case "${PHASE}" in
    inference)
        run_inference
        ;;
    eval)
        run_evaluation
        ;;
    all)
        run_inference
        run_evaluation
        ;;
    *)
        echo "Usage: $0 [inference|eval|all]" >&2
        exit 2
        ;;
esac
