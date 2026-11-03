#!/usr/bin/env bash
RELEASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${RELEASE_SCRIPT_DIR}/../../.." && pwd)"
cd "${RELEASE_SCRIPT_DIR}"
mkdir -p logs
# Iterate over the four positions.
for position in "position_start" "position_early_mid" "position_late_mid" "position_end"; do
    echo "Processing position: ${position}"
    
    # gpt-oss-20b
    python run_api.py \
        --tasks_folder "./generated_tasks/${position}/gemini-2.0-flash-001" \
        --model_path "openai/gpt-oss-20b" \
        --output_folder "./results/${position}" \
        --api_provider "openrouter" \
        --temperature 1 \
        --top_p 1 \
        --max_tokens 8192 \
        --batch_size 20 \
        --num_workers 64 \
        --timeout 20 \
        --max_examples 100 \
        --reasoning_effort "medium"
done
