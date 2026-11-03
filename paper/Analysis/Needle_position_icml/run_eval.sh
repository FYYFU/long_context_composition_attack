#!/usr/bin/env bash
RELEASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${RELEASE_SCRIPT_DIR}/../../.." && pwd)"
cd "${RELEASE_SCRIPT_DIR}"
mkdir -p logs
# ============================================================================
# Evaluate adjacent results with Gemini sequentially to avoid rate limits.
# ============================================================================

cd ${RELEASE_ROOT}/paper/eval

# gpt-oss-120b
echo "========== Eval gpt-oss-120b =========="
python check_safe_gpt.py \
    --meta_path ../Analysis/Needle_position_icml/results/dist_adjacent \
    --judge_model google/gemini-2.0-flash-lite-001 \
    --model_name gpt-oss-120b \
    --reasoning_efforts medium \
    --levels 0,1,2,3 \
    --context_lengths 1k,4k,16k \
    --api_provider openrouter \
    --num_workers 10 \
    --skip_existing

# gpt-oss-20b
echo "========== Eval gpt-oss-20b =========="
python check_safe_gpt.py \
    --meta_path ../Analysis/Needle_position_icml/results/dist_adjacent \
    --judge_model google/gemini-2.0-flash-lite-001 \
    --model_name gpt-oss-20b \
    --reasoning_efforts medium \
    --levels 0,1,2,3 \
    --context_lengths 1k,4k,16k \
    --api_provider openrouter \
    --num_workers 10 \
    --skip_existing

echo "========== Done =========="
