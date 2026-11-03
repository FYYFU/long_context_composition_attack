#!/usr/bin/env bash
RELEASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${RELEASE_SCRIPT_DIR}/../.." && pwd)"
cd "${RELEASE_SCRIPT_DIR}"
mkdir -p logs
# for reasoning_effort in "low" "medium" "high"; do
# python run_gpt_oss.py \
#     --model_path "openai/gpt-oss-20b" \
#     --tasks_folder "../generated_tasks/gemini-2.0-flash-001" \
#     --output_folder "../results/gemini-2.0-flash-001/" \
#     --api_base_url "http://localhost:8001/v1" \
#     --reasoning_effort $reasoning_effort \
#     --batch_size 10 \
#     --temperature 1 \
#     --top_p 1 \
#     --max_tokens 16368 \
#     --n_examples 100

# done


for reasoning_effort in "medium"; do
python run_gpt_oss.py \
    --model_path "openai/gpt-oss-20b" \
    --tasks_folder "../generated_tasks/gemini-2.0-flash-001" \
    --output_folder "../results/gemini-2.0-flash-001/" \
    --api_base_url "http://localhost:8001/v1" \
    --reasoning_effort $reasoning_effort \
    --batch_size 10 \
    --temperature 1 \
    --top_p 1 \
    --max_tokens 16368 \
    --n_examples 100
done
