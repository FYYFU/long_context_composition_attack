#!/usr/bin/env bash
RELEASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${RELEASE_SCRIPT_DIR}/../../.." && pwd)"
cd "${RELEASE_SCRIPT_DIR}"
mkdir -p logs
# Run multilingual NIAH experiments.
# Use gpt-oss-20b and gpt-oss-120b.

cd "${RELEASE_SCRIPT_DIR}"

# Language list.
LANGUAGES=("ar" "pt")

# API configuration.
API_PROVIDER="openrouter"
BATCH_SIZE=10
NUM_WORKERS=32
TIMEOUT=60
MAX_RETRIES=3

# Reasoning effort settings.
REASONING_EFFORT="medium"

echo "========================================"
echo "运行多语言 NIAH 实验"
echo "========================================"
echo "语言: ${LANGUAGES[*]}"
echo "========================================"

# ========================================
# Run experiments with gpt-oss-120b.
# ========================================
echo ""
echo "########################################"
echo "# 模型: openai/gpt-oss-120b"
echo "########################################"

for lang in "${LANGUAGES[@]}"; do
    echo ""
    echo "========================================"
    echo "Language: ${lang}, Model: gpt-oss-120b"
    echo "========================================"
    
    TASKS_FOLDER="./generated_tasks/lang_${lang}"
    OUTPUT_FOLDER="./results/${lang}"
    
    if [ ! -d "${TASKS_FOLDER}" ]; then
        echo "警告: 任务文件夹不存在: ${TASKS_FOLDER}"
        echo "请先运行 run_create_tasks.sh 生成任务"
        continue
    fi
    
    python run_api.py \
        --tasks_folder "${TASKS_FOLDER}" \
        --model_path "openai/gpt-oss-120b" \
        --output_folder "${OUTPUT_FOLDER}" \
        --temperature 1 \
        --top_p 1 \
        --max_tokens 16368 \
        --batch_size ${BATCH_SIZE} \
        --task_pattern "level_*.json" \
        --api_provider "${API_PROVIDER}" \
        --num_workers ${NUM_WORKERS} \
        --timeout ${TIMEOUT} \
        --max_retries ${MAX_RETRIES} \
        --reasoning_effort "${REASONING_EFFORT}"
    
    echo ""
done

echo ""
echo "========================================"
echo "实验完成!"
echo "========================================"
echo "结果保存在: ./results/"
for lang in "${LANGUAGES[@]}"; do
    echo "  - ${lang}/gpt-oss-20b-${REASONING_EFFORT}/"
    echo "  - ${lang}/gpt-oss-120b-${REASONING_EFFORT}/"
done
