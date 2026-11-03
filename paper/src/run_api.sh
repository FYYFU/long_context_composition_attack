#!/usr/bin/env bash
RELEASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${RELEASE_SCRIPT_DIR}/../.." && pwd)"
cd "${RELEASE_SCRIPT_DIR}"
mkdir -p logs
# for reasoning_effort in "low" "medium" "high"; do
# python run_api.py \
#     --tasks_folder "../generated_tasks/gemini-2.0-flash-001" \
#     --model_path "openai/gpt-oss-20b" \
#     --output_folder "../results/gemini-2.0-flash-001" \
#     --api_provider "openrouter" \
#     --temperature 1 \
#     --top_p 1 \
#     --max_tokens 16368 \
#     --batch_size 10 \
#     --num_workers 64 \
#     --timeout 120 \
#     --max_examples 100 \
#     --reasoning_effort $reasoning_effort | tee ./logs/gpt-oss-20b_${reasoning_effort}.log
# done

# python run_api.py \
#     --tasks_folder "../generated_tasks/gemini-2.0-flash-001" \
#     --model_path "openai/gpt-oss-120b" \
#     --output_folder "../results/gemini-2.0-flash-001" \
#     --api_provider "openrouter" \
#     --temperature 1 \
#     --top_p 1 \
#     --max_tokens 16368 \
#     --batch_size 20 \
#     --num_workers 64 \
#     --timeout 120 \
#     --max_examples 100 \
#     --reasoning_effort "medium"


# python run_api.py \
#     --tasks_folder "../generated_tasks/gemini-2.0-flash-001" \
#     --model_path "xiaomi/mimo-v2-flash:free" \
#     --output_folder "../results/gemini-2.0-flash-001" \
#     --api_provider "openrouter" \
#     --temperature 0.8 \
#     --top_p 0.95 \
#     --max_tokens 16368 \
#     --batch_size 20 \
#     --use_batch True \
#     --num_workers 64 \
#     --timeout 120 \
#     --max_examples 100 \
    


###### Mini-max m2 ######
# for reasoning_effort in "default"; do
# python run_api.py \
#     --tasks_folder "../generated_tasks/gemini-2.0-flash-001" \
#     --model_path "minimax/minimax-m2" \
#     --output_folder "../results/gemini-2.0-flash-001" \
#     --api_provider "openrouter" \
#     --temperature 1 \
#     --top_p 0.95 \
#     --max_tokens 16368 \
#     --batch_size 5 \
#     --num_workers 64 \
#     --timeout 120 \
#     --max_examples 100 \
#     --reasoning_effort $reasoning_effort
# done


#### deepseek v3.2 ###
# for reasoning_effort in "default"; do
# python run_api.py \
#     --tasks_folder "../generated_tasks/gemini-2.0-flash-001" \
#     --model_path "deepseek/deepseek-v3.2" \
#     --output_folder "../results/gemini-2.0-flash-001" \
#     --api_provider "openrouter" \
#     --temperature 1 \
#     --top_p 0.95 \
#     --max_tokens 16368 \
#     --batch_size 3 \
#     --num_workers 64 \
#     --timeout 120 \
#     --max_examples 100 \
#     --reasoning_effort $reasoning_effort | tee ./logs/deepseek-v3.2_${reasoning_effort}.log
# done


### Qwen3-30B-A3B-Thinking-2507 ###
# for reasoning_effort in "default"; do
# python run_api.py \
#     --tasks_folder "../generated_tasks/gemini-2.0-flash-001" \
#     --model_path "qwen/qwen3-next-80b-a3b-thinking" \
#     --output_folder "../results/gemini-2.0-flash-001" \
#     --api_provider "openrouter" \
#     --temperature 0.6 \
#     --top_p 0.95 \
#     --max_tokens 16368 \
#     --batch_size 3 \
#     --num_workers 64 \
#     --timeout 120 \
#     --max_examples 100 \
#     --reasoning_effort $reasoning_effort
# done



### Gemini 3 Flash Preview ###
# python run_api.py \
#     --tasks_folder "../generated_tasks/gemini-2.0-flash-001" \
#     --model_path "google/gemini-3-flash-preview" \
#     --output_folder "../results/gemini-2.0-flash-001" \
#     --api_provider "openrouter" \
#     --temperature 0 \
#     --top_p 1 \
#     --max_tokens 16368 \
#     --batch_size 10 \
#     --use_batch True \
#     --num_workers 64 \
#     --timeout 120 \
#     --max_examples 100 \
#     --reasoning_effort "default"


### Claude ###
python run_api.py \
    --tasks_folder "../generated_tasks/gemini-2.0-flash-001" \
    --model_path "anthropic/claude-sonnet-4.5" \
    --output_folder "../results/gemini-2.0-flash-001" \
    --api_provider "openrouter" \
    --temperature 0 \
    --top_p 1 \
    --max_tokens 16368 \
    --batch_size 2 \
    --num_workers 64 \
    --timeout 120 \
    --max_examples 100 \
    --reasoning_effort "default"
