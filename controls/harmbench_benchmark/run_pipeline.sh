#!/usr/bin/env bash
RELEASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${RELEASE_SCRIPT_DIR}/../.." && pwd)"
cd "${RELEASE_SCRIPT_DIR}"
mkdir -p logs
# Complete HarmBench benchmark pipeline.
# For the rebuttal: Evaluate multi-hop reasoning safety on HarmBench.
#
# Prerequisites:
#   1. Set the OPENROUTER_API_KEY environment variable.
#   2. pip install datasets transformers fire tqdm openai
#
# Usage:
#   bash run_pipeline.sh                       # Run the full pipeline.
#   bash run_pipeline.sh --skip-decompose      # Skip decomposition when decomposed data already exists.
#   bash run_pipeline.sh --skip-tasks          # Skip task generation when generated_tasks already exists.
#   bash run_pipeline.sh --model gpt-oss-20b   # Run only the specified model.

set -e

SCRIPT_DIR="$(cd "${RELEASE_SCRIPT_DIR}" && pwd)"
HARMBENCH_FILE="${SCRIPT_DIR}/harmbench_standard_100.json"
DECOMPOSED_FILE="${SCRIPT_DIR}/harmbench_decomposed.json"
TASKS_DIR="${SCRIPT_DIR}/generated_tasks"
RESULTS_DIR="${SCRIPT_DIR}/results"

# Use the original project's evaluation script.
EVAL_SCRIPT="${RELEASE_ROOT}/paper/eval/check_safe_gpt.py"

# Parse arguments.
SKIP_DECOMPOSE=false
SKIP_TASKS=false
SELECTED_MODEL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-decompose)
            SKIP_DECOMPOSE=true
            shift
            ;;
        --skip-tasks)
            SKIP_TASKS=true
            shift
            ;;
        --model)
            SELECTED_MODEL="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# ==============================
# Model list.
# ==============================
declare -A MODELS
MODELS=(
    ["gpt-oss-20b"]="openai/gpt-oss-20b"
    ["gpt-oss-120b"]="openai/gpt-oss-120b"
    # ["claude-sonnet-4.5"]="anthropic/claude-sonnet-4.5"
    # ["claude-haiku-4.5"]="anthropic/claude-haiku-4.5"
    # ["gemini-3-flash-preview"]="google/gemini-3-flash-preview"
    # ["deepseek-v3.2"]="deepseek/deepseek-v3.2"
    # ["kimi-k2-thinking"]="moonshotai/kimi-k2-thinking"
    # ["gpt-4o-mini"]="openai/gpt-4o-mini"
    # ["minimax-m2"]="minimax/minimax-m2"
)

if [ -n "$SELECTED_MODEL" ]; then
    if [ -z "${MODELS[$SELECTED_MODEL]}" ]; then
        echo "错误: 未知模型 '$SELECTED_MODEL'. 可用模型: ${!MODELS[*]}"
        exit 1
    fi
    MODEL_KEYS=("$SELECTED_MODEL")
else
    MODEL_KEYS=("${!MODELS[@]}")
fi

echo "============================================"
echo "HarmBench Benchmark Pipeline"
echo "============================================"
echo "Models: ${MODEL_KEYS[*]}"
echo "============================================"

# ==============================
# Step 1: Download HarmBench.
# ==============================
if [ ! -f "${HARMBENCH_FILE}" ]; then
    echo ""
    echo "=== Step 1: Downloading HarmBench standard behaviors ==="
    python "${SCRIPT_DIR}/download_harmbench.py" --max_samples 100
else
    echo ""
    echo "=== Step 1: HarmBench data exists, skipping ==="
fi

# ==============================
# Step 2: Decompose behaviors into facts at multiple levels.
# ==============================
if [ "$SKIP_DECOMPOSE" = false ] && [ ! -f "${DECOMPOSED_FILE}" ]; then
    echo ""
    echo "=== Step 2: Decomposing HarmBench behaviors into multi-level facts ==="
    python "${SCRIPT_DIR}/decompose_harmbench.py" \
        --model "google/gemini-2.0-flash-001" \
        --api_provider openrouter \
        --num_workers 5 \
        --input_file "${HARMBENCH_FILE}" \
        --output_file "${DECOMPOSED_FILE}"
else
    echo ""
    echo "=== Step 2: Decomposed data exists (or skipped), continuing ==="
fi

# ==============================
# Step 3: Generate NIAH tasks.
# ==============================
if [ "$SKIP_TASKS" = false ]; then
    TASKS_EXIST=true
    for level in 0 1 2 3; do
        if [ ! -f "${TASKS_DIR}/level_${level}_0k.json" ]; then
            TASKS_EXIST=false
            break
        fi
    done

    if [ "$TASKS_EXIST" = false ]; then
        echo ""
        echo "=== Step 3: Creating NIAH tasks (embedding facts into PG19 haystack) ==="
        for level in 0 1 2 3; do
            echo "  Processing level ${level}..."
            python "${SCRIPT_DIR}/create_task.py" \
                --dataset_path "${DECOMPOSED_FILE}" \
                --level ${level} \
                --out_folder "${TASKS_DIR}" \
                --number_of_samples 100
        done
    else
        echo ""
        echo "=== Step 3: Tasks already exist, skipping ==="
    fi
else
    echo ""
    echo "=== Step 3: Skipped (--skip-tasks) ==="
fi

# ==============================
# Steps 4 and 5: Run inference and evaluation for each model.
# ==============================
for MODEL_KEY in "${MODEL_KEYS[@]}"; do
    MODEL_PATH="${MODELS[$MODEL_KEY]}"
    MODEL_NAME=$(echo $MODEL_PATH | sed 's|.*/||')

    echo ""
    echo "============================================"
    echo "Processing model: ${MODEL_KEY} (${MODEL_PATH})"
    echo "============================================"

    # Step 4: Model inference.
    echo ""
    echo "=== Step 4: Running model inference (${MODEL_KEY}) ==="
    python "${SCRIPT_DIR}/run_harmbench.py" \
        --tasks_folder "${TASKS_DIR}" \
        --model_path "${MODEL_PATH}" \
        --output_folder "${RESULTS_DIR}" \
        --api_provider openrouter \
        --batch_size 5 \
        --num_workers 64 \
        --timeout 120 \
        --temperature 1 \
        --top_p 1.0 \
        --max_tokens 16368 \
        --max_examples 100 \
        --reasoning_effort "medium"

    # Step 5: Safety evaluation with a GPT judge.
    echo ""
    echo "=== Step 5: Safety evaluation (${MODEL_KEY}) ==="
    python "${EVAL_SCRIPT}" \
        --meta_path "${RESULTS_DIR}" \
        --judge_model "google/gemini-2.0-flash-lite-001" \
        --api_provider openrouter \
        --reasoning_efforts "medium" \
        --model_name "${MODEL_NAME}" \
        --levels 0,1,2,3 \
        --num_workers 10 \
        --context_lengths 0k,16k,64k \
        --skip_existing

    echo ""
    echo "✓ ${MODEL_KEY} complete! Results: ${RESULTS_DIR}/${MODEL_NAME}-medium/"
done

echo ""
echo "============================================"
echo "All models complete!"
echo "============================================"
