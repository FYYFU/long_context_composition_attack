#!/usr/bin/env bash
RELEASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${RELEASE_SCRIPT_DIR}/../../.." && pwd)"
cd "${RELEASE_SCRIPT_DIR}"
mkdir -p logs
# Generate multi-hop reasoning tasks with per-query contexts.
# Each query uses only its own relevant contexts.

# Set the working directory.
cd "${RELEASE_SCRIPT_DIR}"

# ============ Configuration parameters ============
# Input data.
DATASET_PATH="${RELEASE_ROOT}/data/source/advbench_transformed_api_gemini-2.0-flash-001_fixed.json"
RELEVANT_CONTEXTS_DIR="./relevant_contexts"

# Output directory.
OUT_FOLDER="./generated_tasks_relevant_contexts"

# Task parameters.
LEVELS=(0 1 2 3)  # List of levels to generate.
NUMBER_OF_SAMPLES=100
RANDOM_SEED=42

# tokenizer
TOKENIZER_NAME="gpt2"

# ============ Start generation ============
echo "========================================"
echo "生成多跳推理任务 (Per-Query Contexts)"
echo "========================================"
echo "数据集: ${DATASET_PATH}"
echo "Contexts 目录: ${RELEVANT_CONTEXTS_DIR}"
echo "输出目录: ${OUT_FOLDER}"
echo "样本数: ${NUMBER_OF_SAMPLES}"
echo "Levels: ${LEVELS[*]}"
echo "========================================"

# Check whether relevant_contexts exists.
if [ ! -d "${RELEVANT_CONTEXTS_DIR}" ]; then
    echo "错误: Relevant contexts 目录不存在: ${RELEVANT_CONTEXTS_DIR}"
    echo "请先运行 run_generate.sh 生成相关 contexts"
    exit 1
fi

# Check whether the dataset exists.
if [ ! -f "${DATASET_PATH}" ]; then
    echo "错误: 数据集文件不存在: ${DATASET_PATH}"
    exit 1
fi

# Create the output directory.
mkdir -p "${OUT_FOLDER}"

# Generate tasks for each level.
for level in "${LEVELS[@]}"; do
    echo ""
    echo "========================================"
    echo "生成 Level ${level} 任务"
    echo "========================================"
    
    python create_task.py \
        --dataset_path "${DATASET_PATH}" \
        --level ${level} \
        --out_folder "${OUT_FOLDER}" \
        --number_of_samples ${NUMBER_OF_SAMPLES} \
        --tokenizer_name "${TOKENIZER_NAME}" \
        --random_seed ${RANDOM_SEED} \
        --use_per_query_contexts True \
        --relevant_contexts_dir "${RELEVANT_CONTEXTS_DIR}"
    
    if [ $? -ne 0 ]; then
        echo "错误: Level ${level} 生成失败"
        exit 1
    fi
done

echo ""
echo "========================================"
echo "完成!"
echo "========================================"
echo "所有任务已保存至: ${OUT_FOLDER}"
echo ""
echo "生成的文件列表:"
ls -la "${OUT_FOLDER}/"
