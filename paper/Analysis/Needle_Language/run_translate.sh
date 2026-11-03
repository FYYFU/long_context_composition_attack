#!/usr/bin/env bash
RELEASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${RELEASE_SCRIPT_DIR}/../../.." && pwd)"
cd "${RELEASE_SCRIPT_DIR}"
mkdir -p logs
# Translate AdvBench facts into different languages.
# Use Gemini 2.0 Flash Lite.

cd "${RELEASE_SCRIPT_DIR}"

# Input file.
INPUT_FILE="${RELEASE_ROOT}/data/source/advbench_transformed_api_gemini-2.0-flash-001_fixed.json"
OUTPUT_DIR="./translated_datasets"

# Translation model.
MODEL_NAME="google/gemini-2.0-flash-lite-001"
API_PROVIDER="openrouter"

# Language configuration.
# zh: Chinese (high-resource).
# ru: Russian (medium-resource).
# vi: Vietnamese (low-resource).

# echo "========================================"
# echo "Step 1: Translate into Chinese"
# echo "========================================"
# python translate_facts.py translate \
#     --input_file "${INPUT_FILE}" \
#     --output_dir "${OUTPUT_DIR}" \
#     --target_language "zh" \
#     --model_name "${MODEL_NAME}" \
#     --api_provider "${API_PROVIDER}" \
#     --levels "[0, 1, 2, 3]" \
#     --num_workers 32 \
#     --timeout 60 \
#     --max_samples 100

# echo ""
# echo "========================================"
# echo "Step 2: Translate into Russian"
# echo "========================================"
# python translate_facts.py translate \
#     --input_file "${INPUT_FILE}" \
#     --output_dir "${OUTPUT_DIR}" \
#     --target_language "ru" \
#     --model_name "${MODEL_NAME}" \
#     --api_provider "${API_PROVIDER}" \
#     --levels "[0, 1, 2, 3]" \
#     --num_workers 32 \
#     --max_samples 100 \
#     --timeout 60

# echo ""
# echo "========================================"
# echo "Step 3: Translate into Vietnamese"
# echo "========================================"
# python translate_facts.py translate \
#     --input_file "${INPUT_FILE}" \
#     --output_dir "${OUTPUT_DIR}" \
#     --target_language "vi" \
#     --model_name "${MODEL_NAME}" \
#     --api_provider "${API_PROVIDER}" \
#     --levels "[0, 1, 2, 3]" \
#     --num_workers 32 \
#     --max_samples 100 \
#     --timeout 60

# echo ""
# echo "========================================"
# echo "Step 4: Translate into Bengali"
# echo "========================================"
# python translate_facts.py translate \
#     --input_file "${INPUT_FILE}" \
#     --output_dir "${OUTPUT_DIR}" \
#     --target_language "bn" \
#     --model_name "${MODEL_NAME}" \
#     --api_provider "${API_PROVIDER}" \
#     --levels "[0, 1, 2, 3]" \
#     --num_workers 32 \
#     --max_samples 100 \
#     --timeout 60

# echo ""
# echo "========================================"
# echo "Step 5: Translate into Swahili"
# echo "========================================"
# python translate_facts.py translate \
#     --input_file "${INPUT_FILE}" \
#     --output_dir "${OUTPUT_DIR}" \
#     --target_language "sw" \
#     --model_name "${MODEL_NAME}" \
#     --api_provider "${API_PROVIDER}" \
#     --levels "[0, 1, 2, 3]" \
#     --num_workers 32 \
#     --max_samples 100 \
#     --timeout 60

echo ""
echo "========================================"
echo "Step 6: 翻译到阿拉伯语 (Arabic)"
echo "========================================"
python translate_facts.py translate \
    --input_file "${INPUT_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --target_language "ar" \
    --model_name "${MODEL_NAME}" \
    --api_provider "${API_PROVIDER}" \
    --levels "[0, 1, 2, 3]" \
    --num_workers 32 \
    --max_samples 100 \
    --timeout 60

echo ""
echo "========================================"
echo "Step 7: 翻译到葡萄牙语 (Portuguese)"
echo "========================================"
python translate_facts.py translate \
    --input_file "${INPUT_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --target_language "pt" \
    --model_name "${MODEL_NAME}" \
    --api_provider "${API_PROVIDER}" \
    --levels "[0, 1, 2, 3]" \
    --num_workers 32 \
    --max_samples 100 \
    --timeout 60

echo ""
echo "========================================"
echo "翻译完成!"
echo "========================================"
echo "输出目录: ${OUTPUT_DIR}"
echo "  - advbench_translated_zh.json (中文)"
echo "  - advbench_translated_ru.json (俄语)"
echo "  - advbench_translated_vi.json (越南语)"
echo "  - advbench_translated_bn.json (孟加拉语)"
echo "  - advbench_translated_sw.json (斯瓦希里语)"
echo "  - advbench_translated_ar.json (阿拉伯语)"
echo "  - advbench_translated_pt.json (葡萄牙语)"
