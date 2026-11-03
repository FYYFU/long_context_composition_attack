#!/usr/bin/env bash
RELEASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${RELEASE_SCRIPT_DIR}/../.." && pwd)"
cd "${RELEASE_SCRIPT_DIR}"
mkdir -p logs
# meta_path="${RELEASE_ROOT}/paper/"

# python check_safe_gpt.py \
#     --meta_path $meta_path/results/gemini-2.0-flash-001_prompt1/ \
#     --judge_model gpt-4o \
#     --model_name gpt-oss-20b \
#     --reasoning_efforts low \
#     --levels 0,1,2,3,4 \
#     --context_lengths 0k,16k,64k \
#     --skip_existing




########### Gemini 2.0 Flash Lite 001 ###########

meta_path="${RELEASE_ROOT}/paper/"

# # model_name="gemini-3-flash-preview"
# # model_name="minimax-m2"
# # model_name="deepseek-v3.2"
# # model_name="qwen3-next-80b-a3b-thinking"
# # model_name='Qwen3-30B-A3B-Thinking-2507'
# # model_name="claude-sonnet-4.5"
# # model_name="claude-haiku-4.5"
# # model_name="gemini-2.5-flash"
# model_name="gemini-3-pro-preview"
# model_name="gpt-4o-mini"
# model_name="gpt-oss-20b"
model_name="kimi-k2-thinking"


for reasoning_efforts in "default"; do    
    python check_safe_gpt.py \
        --meta_path $meta_path/results/gemini-2.0-flash-001/ \
        --judge_model google/gemini-2.0-flash-lite-001 \
        --api_provider openrouter \
        --reasoning_efforts $reasoning_efforts \
        --model_name $model_name \
        --levels 0,1,2,3 \
        --context_lengths 0k,16k,64k \
        --skip_existing
done


####### Analysis Context length #######
# meta_path="${RELEASE_ROOT}/paper/Analysis/Context_Length/"

# model_name="gpt-oss-120b"

# for reasoning_efforts in "medium"; do    
#     python check_safe_gpt.py \
#         --meta_path $meta_path/results_relevant_contexts/ \
#         --judge_model google/gemini-2.0-flash-lite-001 \
#         --api_provider openrouter \
#         --reasoning_efforts $reasoning_efforts \
#         --model_name $model_name \
#         --levels 0,1,2,3,4 \
#         --context_lengths 0k,16k,64k \
#         --skip_existing
# done


####### Analysis Needle Language #######
# meta_path="${RELEASE_ROOT}/paper/Analysis/Needle_Language/"

# model_name="gpt-oss-20b"

# for reasoning_efforts in "medium"; do    
#     for lang in "ar" "pt"; do  
#         python check_safe_gpt.py \
#             --meta_path $meta_path/results/${lang}/ \
#             --judge_model google/gemini-2.0-flash-lite-001 \
#             --api_provider openrouter \
#             --reasoning_efforts $reasoning_efforts \
#             --model_name $model_name \
#             --levels 0,1,2,3,4 \
#             --context_lengths 0k,16k,64k \
#             --skip_existing
#     done
# done

# model_name="gpt-oss-120b"

# for reasoning_efforts in "medium"; do    
#     for lang in "ar" "pt"; do  
#         python check_safe_gpt.py \
#             --meta_path $meta_path/results/${lang}/ \
#             --judge_model google/gemini-2.0-flash-lite-001 \
#             --api_provider openrouter \
#             --reasoning_efforts $reasoning_efforts \
#             --model_name $model_name \
#             --levels 0,1,2,3,4 \
#             --context_lengths 0k,16k,64k \
#             --skip_existing
#     done
# done


####### Analysis Needle position #######
# meta_path="${RELEASE_ROOT}/paper/Analysis/Needle_Position/"

# model_name="gpt-oss-20b"

# for reasoning_efforts in "medium"; do    
#     for position in "position_start" "position_early_mid" "position_late_mid" "position_end"; do  
#         python check_safe_gpt.py \
#             --meta_path $meta_path/results/${position}/ \
#             --judge_model google/gemini-2.0-flash-lite-001 \
#             --api_provider openrouter \
#             --reasoning_efforts $reasoning_efforts \
#             --model_name $model_name \
#             --levels 0,1,2,3 \
#             --context_lengths 0k,16k,64k \
#             --skip_existing
#     done
# done

# model_name="gpt-oss-120b"

# for reasoning_efforts in "medium"; do    
#     for position in "position_start" "position_early_mid" "position_late_mid" "position_end"; do  
#         python check_safe_gpt.py \
#             --meta_path $meta_path/results/${position}/ \
#             --judge_model google/gemini-2.0-flash-lite-001 \
#             --api_provider openrouter \
#             --reasoning_efforts $reasoning_efforts \
#             --model_name $model_name \
#             --levels 0,1,2,3 \
#             --context_lengths 0k,16k,64k \
#             --skip_existing
#     done
# done
