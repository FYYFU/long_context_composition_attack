#!/usr/bin/env bash
RELEASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${RELEASE_SCRIPT_DIR}/../../.." && pwd)"
cd "${RELEASE_SCRIPT_DIR}"
mkdir -p logs
# Generate multilingual NIAH tasks.
# Use Wikipedia in the matching language as background text.

cd "${RELEASE_SCRIPT_DIR}"

# Configuration.
TRANSLATED_DIR="./translated_datasets"
OUTPUT_DIR="./generated_tasks"

# Language list.
LANGUAGES=("ar" "pt")

# Level list.
LEVELS=(0 1 2 3)

# Context lengths.
MESSAGE_LENGTHS="[0, 1000, 4000, 8000, 16000, 32000, 64000]"

echo "========================================"
echo "生成多语言 NIAH 任务"
echo "========================================"
echo "语言: ${LANGUAGES[*]}"
echo "Levels: ${LEVELS[*]}"
echo "使用对应语言的 Wikipedia 作为背景文本"
echo "========================================"

for lang in "${LANGUAGES[@]}"; do
    for level in "${LEVELS[@]}"; do
        echo ""
        echo "----------------------------------------"
        echo "Language: ${lang}, Level: ${level}"
        echo "----------------------------------------"
        
        python create_task.py create \
            --dataset_path "${TRANSLATED_DIR}/advbench_translated_${lang}.json" \
            --language "${lang}" \
            --level ${level} \
            --out_folder "${OUTPUT_DIR}" \
            --number_of_samples 100 \
            --message_lengths "${MESSAGE_LENGTHS}" \
            --random_seed 42 \
            --use_native_noise True
    done
done

echo ""
echo "========================================"
echo "任务生成完成!"
echo "========================================"
echo "输出目录: ${OUTPUT_DIR}"
echo "  - lang_zh/ (中文 facts + 中文背景)"
echo "  - lang_ru/ (俄语 facts + 俄语背景)"
echo "  - lang_vi/ (越南语 facts + 越南语背景)"
echo "  - lang_bn/ (孟加拉语 facts + 孟加拉语背景)"
echo "  - lang_sw/ (斯瓦希里语 facts + 斯瓦希里语背景)"
echo "  - lang_ar/ (阿拉伯语 facts + 阿拉伯语背景)"
echo "  - lang_pt/ (葡萄牙语 facts + 葡萄牙语背景)"
