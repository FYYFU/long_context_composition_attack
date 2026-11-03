#!/usr/bin/env bash
RELEASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${RELEASE_SCRIPT_DIR}/../.." && pwd)"
cd "${RELEASE_SCRIPT_DIR}"
mkdir -p logs

meta_path=${RELEASE_ROOT}/paper/results/gemini-2.0-flash-001

CUDA_VISIBLE_DEVICES=6 python check_safe_harmbench.py \
    --meta_path $meta_path \
    --model_path cais/HarmBench-Llama-2-13b-cls \
    --tensor_parallel_size 1 \
    --reasoning_efforts "low,medium,high" \
    --levels "0,1,2,3,4" \
    --context_lengths "0k,1k,4k,8k,16k,32k,64k" \
    --model_name "gpt-oss-20b" \
    --judge_name "HarmBench-Llama-2-13b-cls" \
    --skip_existing
