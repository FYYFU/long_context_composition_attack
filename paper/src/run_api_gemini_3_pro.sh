#!/usr/bin/env bash
RELEASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${RELEASE_SCRIPT_DIR}/../.." && pwd)"
cd "${RELEASE_SCRIPT_DIR}"
mkdir -p logs
for model in "google/gemini-3-pro-preview"; do
    model_name=$model
    for reasoning_effort in "default"; do
        python run_api.py \
            --tasks_folder "../generated_tasks/gemini-2.0-flash-001" \
            --model_path $model_name \
            --output_folder "../results/gemini-2.0-flash-001" \
            --api_provider "openrouter" \
            --temperature 0 \
            --top_p 1 \
            --max_tokens 16368 \
            --batch_size 50 \
            --num_workers 64 \
            --timeout 120 \
            --max_examples 100 \
            --reasoning_effort $reasoning_effort | tee ./logs/${model_name}_${reasoning_effort}.log
    done
done
