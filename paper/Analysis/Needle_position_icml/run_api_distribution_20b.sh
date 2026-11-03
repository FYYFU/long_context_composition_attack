#!/usr/bin/env bash
RELEASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${RELEASE_SCRIPT_DIR}/../../.." && pwd)"
cd "${RELEASE_SCRIPT_DIR}"
mkdir -p logs
# ============================================================================
# Run the fragment distribution experiment with gpt-oss-20b (adjacent only).
# Use batch_size=5, num_workers=32, timeout=60 to avoid long-context timeouts.
# ============================================================================

echo "=========================================="
echo "Processing distribution: dist_adjacent"
echo "=========================================="

python run_api.py \
    --tasks_folder "./generated_tasks/dist_adjacent/gemini-2.0-flash-001" \
    --model_path "openai/gpt-oss-20b" \
    --output_folder "./results/dist_adjacent" \
    --api_provider "openrouter" \
    --temperature 1 \
    --top_p 1 \
    --max_tokens 8192 \
    --batch_size 5 \
    --num_workers 32 \
    --timeout 160 \
    --max_examples 100 \
    --reasoning_effort "medium" | tee ./gpt-oss-20b_dist_adjacent.log
