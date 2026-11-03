#!/usr/bin/env bash
RELEASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${RELEASE_SCRIPT_DIR}/../.." && pwd)"
cd "${RELEASE_SCRIPT_DIR}"
mkdir -p logs
for reasoning_effort in "default"; do
python run_api.py \
    --tasks_folder "../generated_tasks/gemini-2.0-flash-001" \
    --model_path "moonshotai/kimi-k2-thinking" \
    --output_folder "../results/gemini-2.0-flash-001" \
    --api_provider "openrouter" \
    --temperature 1 \
    --top_p 0.8 \
    --max_tokens 16368 \
    --batch_size 5 \
    --num_workers 64 \
    --timeout 120 \
    --max_examples 100 \
    --reasoning_effort $reasoning_effort | tee ./logs/kimi-k2-thinking_${reasoning_effort}.log
done
