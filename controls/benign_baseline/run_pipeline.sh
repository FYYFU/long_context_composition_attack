#!/usr/bin/env bash
RELEASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${RELEASE_SCRIPT_DIR}/../.." && pwd)"
cd "${RELEASE_SCRIPT_DIR}"
mkdir -p logs
# Complete benign baseline pipeline.
# For the rebuttal: Evaluate reconstruction ability independently of safety.
#
# Prerequisites:
#   1. Generate benign_baseline_decomposed.json with create_benign_baseline.py.
#   2. Set the OPENROUTER_API_KEY environment variable.
#
# Usage:
#   bash run_pipeline.sh                    # Run all models.
#   bash run_pipeline.sh gpt-oss-20b        # Run only the specified model.
#   bash run_pipeline.sh gemini-3-flash     # Run only the specified model.

set -e

SCRIPT_DIR="$(cd "${RELEASE_SCRIPT_DIR}" && pwd)"
DECOMPOSED_FILE="${SCRIPT_DIR}/../benign_baseline_decomposed.json"
TASKS_DIR="${SCRIPT_DIR}/generated_tasks"
RESULTS_DIR="${SCRIPT_DIR}/results"

# ==============================
# Model list, which can be extended later.
# ==============================
declare -A MODELS
MODELS=(
    ["gpt-oss-20b"]="openai/gpt-oss-20b"
    ["gemini-3-flash"]="google/gemini-3-flash"
    # --- Add more models here. ---
    # ["gpt-oss-120b"]="openai/gpt-oss-120b"
    # ["claude-sonnet-4.5"]="anthropic/claude-sonnet-4.5"
    # ["claude-haiku-4.5"]="anthropic/claude-haiku-4.5"
    # ["gpt-5.2"]="openai/gpt-5.2"
    # ["deepseek-v3.2"]="deepseek/deepseek-v3.2"
    # ["gemini-2.5-flash"]="google/gemini-2.5-flash-preview"
)

# Run the specified model, or all models if none is specified.
if [ -n "$1" ]; then
    if [ -z "${MODELS[$1]}" ]; then
        echo "错误: 未知模型 '$1'. 可用模型: ${!MODELS[*]}"
        exit 1
    fi
    SELECTED_MODELS=("$1")
else
    SELECTED_MODELS=("${!MODELS[@]}")
fi

# ==============================
# Step 3: Generate tasks once for all models.
# ==============================
echo "============================================"
echo "Benign Baseline Pipeline"
echo "============================================"
echo "Models: ${SELECTED_MODELS[*]}"
echo "Decomposed file: ${DECOMPOSED_FILE}"
echo "============================================"

# Check whether tasks have already been generated.
TASKS_EXIST=true
for level in 0 1 2 3; do
    if [ ! -f "${TASKS_DIR}/level_${level}_0k.json" ]; then
        TASKS_EXIST=false
        break
    fi
done

if [ "$TASKS_EXIST" = false ]; then
    echo ""
    echo "=== Step 3: Creating tasks (embedding facts into PG19 haystack) ==="
    for level in 0 1 2 3; do
        echo "  Processing level ${level}..."
        python "${SCRIPT_DIR}/create_task_benign.py" \
            --dataset_path "${DECOMPOSED_FILE}" \
            --level ${level} \
            --out_folder "${TASKS_DIR}" \
            --number_of_samples 20
    done
else
    echo ""
    echo "=== Step 3: Tasks already exist, skipping ==="
fi

# ==============================
# Steps 4 and 5: Run inference and evaluation for each model.
# ==============================
for MODEL_KEY in "${SELECTED_MODELS[@]}"; do
    MODEL_PATH="${MODELS[$MODEL_KEY]}"
    MODEL_NAME=$(echo $MODEL_PATH | sed 's|.*/||')

    echo ""
    echo "============================================"
    echo "Processing model: ${MODEL_KEY} (${MODEL_PATH})"
    echo "============================================"

    # Step 4: Model inference.
    echo ""
    echo "=== Step 4: Running model inference (${MODEL_KEY}) ==="
    python "${SCRIPT_DIR}/run_benign.py" \
        --tasks_folder "${TASKS_DIR}" \
        --model_path "${MODEL_PATH}" \
        --output_folder "${RESULTS_DIR}" \
        --api_provider openrouter \
        --batch_size 100 \
        --num_workers 64 \
        --timeout 120 \
        --temperature 0.0 \
        --top_p 1.0 \
        --max_tokens 16368

    # Step 5: Evaluate reconstruction accuracy.
    echo ""
    echo "=== Step 5: Evaluating reconstruction accuracy (${MODEL_KEY}) ==="
    python "${SCRIPT_DIR}/eval_reconstruction.py" \
        --meta_path "${RESULTS_DIR}" \
        --model_name "${MODEL_NAME}" \
        --judge_model "google/gemini-2.0-flash-lite-001" \
        --api_provider openrouter \
        --levels 0,1,2,3 \
        --context_lengths 0k,16k,64k \
        --num_workers 32

    echo ""
    echo "✓ ${MODEL_KEY} complete! Results: ${RESULTS_DIR}/${MODEL_NAME}/"
done

echo ""
echo "============================================"
echo "All models complete!"
echo "============================================"
