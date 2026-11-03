#!/usr/bin/env bash
RELEASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${RELEASE_SCRIPT_DIR}/../../.." && pwd)"
cd "${RELEASE_SCRIPT_DIR}"
mkdir -p logs
# ============================================================
# XSTest Over-Refusal Experiment (with Decomposition)
#
# Full pipeline: download → decompose → embed in haystack →
# run with/without system prompt → evaluate reconstruction
#
# Usage:
#   bash run_xstest.sh prepare      # Step 0: Download dataset
#   bash run_xstest.sh decompose    # Step 1: Decompose into Level 0-3
#   bash run_xstest.sh create_tasks # Step 2: Embed in haystack
#   bash run_xstest.sh inference    # Step 3: Run models (with & without sys prompt)
#   bash run_xstest.sh eval         # Step 4: Evaluate reconstruction accuracy
#   bash run_xstest.sh all          # Run all steps
# ============================================================

set -e

SCRIPT_DIR="$(cd "${RELEASE_SCRIPT_DIR}" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DECOMPOSED_FILE="$SCRIPT_DIR/xstest_decomposed.json"
TASKS_DIR="$SCRIPT_DIR/generated_tasks"
RESULTS_DIR="$SCRIPT_DIR/results"

# Import defense prompt from parent
DEFENSE_PROMPT="safety_v1"

# Format: model_path|reasoning_effort|temperature|top_p|batch_size|api_provider
MODELS=(
    "openai/gpt-oss-20b|medium|1|1|5|openrouter"
    "openai/gpt-4.1|default|0|1|10|openai"
    "openai/gpt-5.2|default|0|1|10|openai"
)

STEP=${1:-"all"}

# ============================================================
# Step 0: Download XSTest data
# ============================================================
run_prepare() {
    echo "=========================================="
    echo "Step 0: Downloading XSTest dataset"
    echo "=========================================="
    python "$SCRIPT_DIR/prepare_xstest.py"
}

# ============================================================
# Step 1: Decompose XSTest safe prompts into Level 0-3 facts
# ============================================================
run_decompose() {
    echo "=========================================="
    echo "Step 1: Decomposing XSTest safe prompts"
    echo "=========================================="

    if [ -f "$DECOMPOSED_FILE" ]; then
        echo "Decomposed file exists: $DECOMPOSED_FILE"
        echo "Delete it manually if you want to re-run."
        return
    fi

    python "$SCRIPT_DIR/decompose_xstest.py" \
        --input_file "$SCRIPT_DIR/xstest_safe.json" \
        --output_file "$DECOMPOSED_FILE" \
        --model "google/gemini-2.0-flash-001" \
        --api_provider openrouter \
        --batch_size 50 \
        --num_workers 64
}

# ============================================================
# Step 2: Create task files (embed facts in haystack)
# ============================================================
run_create_tasks() {
    echo "=========================================="
    echo "Step 2: Creating task files (embedding in haystack)"
    echo "=========================================="

    for level in 0 1 2 3; do
        echo "  Creating Level ${level} tasks..."
        python "$SCRIPT_DIR/create_task_xstest.py" \
            --dataset_path "$DECOMPOSED_FILE" \
            --level ${level} \
            --out_folder "$TASKS_DIR" \
            --number_of_samples 250 \
            --context_lengths "0,16000,64000"
    done
}

# ============================================================
# Step 3: Run inference (with & without system prompt)
# ============================================================
run_inference() {
    echo "=========================================="
    echo "Step 3: Running XSTest inference (with & without system prompt)"
    echo "=========================================="

    DEFENSE_SCRIPT="$SCRIPT_DIR/../run_defense.py"

    for model_config in "${MODELS[@]}"; do
        IFS='|' read -r MODEL_PATH REASONING_EFFORT TEMP TOP_P BATCH_SIZE API_PROVIDER <<< "$model_config"
        API_PROVIDER=${API_PROVIDER:-openrouter}
        MODEL_NAME=$(echo "$MODEL_PATH" | sed 's|.*/||')

        # --- WITHOUT system prompt (baseline) ---
        echo ""
        echo "--- $MODEL_NAME (NO defense) ---"
        python "$DEFENSE_SCRIPT" \
            --tasks_folder "$TASKS_DIR" \
            --model_path "$MODEL_PATH" \
            --output_folder "$RESULTS_DIR/no_defense" \
            --defense_prompt "none" \
            --temperature "$TEMP" \
            --top_p "$TOP_P" \
            --max_tokens 16368 \
            --batch_size "$BATCH_SIZE" \
            --api_provider "$API_PROVIDER" \
            --num_workers 20 \
            --timeout 120 \
            --reasoning_effort "$REASONING_EFFORT" \
            --levels "0,1,2,3" \
            --context_lengths "0k,16k,64k"

        # --- WITH system prompt (defense) ---
        echo ""
        echo "--- $MODEL_NAME (safety_v1 defense) ---"
        python "$DEFENSE_SCRIPT" \
            --tasks_folder "$TASKS_DIR" \
            --model_path "$MODEL_PATH" \
            --output_folder "$RESULTS_DIR/with_defense" \
            --defense_prompt "$DEFENSE_PROMPT" \
            --temperature "$TEMP" \
            --top_p "$TOP_P" \
            --max_tokens 16368 \
            --batch_size "$BATCH_SIZE" \
            --api_provider "$API_PROVIDER" \
            --num_workers 20 \
            --timeout 120 \
            --reasoning_effort "$REASONING_EFFORT" \
            --levels "0,1,2,3" \
            --context_lengths "0k,16k,64k"
    done
}

# ============================================================
# Step 4: Evaluate reconstruction accuracy
# ============================================================
run_eval() {
    echo "=========================================="
    echo "Step 4: Evaluating XSTest reconstruction accuracy"
    echo "=========================================="

    EVAL_SCRIPT="$PROJECT_ROOT/controls/benign_baseline/eval_reconstruction.py"

    for condition in "no_defense" "with_defense"; do
        COND_DIR="$RESULTS_DIR/$condition"
        if [ ! -d "$COND_DIR" ]; then
            echo "No results for $condition, skipping"
            continue
        fi

        # Find all defense_* subdirs
        for defense_dir in "$COND_DIR"/defense_*/; do
            if [ ! -d "$defense_dir" ]; then
                # For no_defense, the structure might be different
                continue
            fi

            for model_dir in "$defense_dir"*/; do
                if [ -d "$model_dir" ]; then
                    MODEL_NAME=$(basename "$model_dir")
                    echo ""
                    echo "--- Evaluating: $condition / $MODEL_NAME ---"

                    python "$EVAL_SCRIPT" \
                        --meta_path "$defense_dir" \
                        --model_name "$MODEL_NAME" \
                        --judge_model "google/gemini-2.0-flash-lite-001" \
                        --api_provider openrouter \
                        --levels "0,1,2,3" \
                        --context_lengths "0k,16k,64k" \
                        --num_workers 32 \
                        --skip_existing
                fi
            done
        done
    done
}

# ============================================================
# Execute
# ============================================================
case "$STEP" in
    prepare)      run_prepare ;;
    decompose)    run_decompose ;;
    create_tasks) run_create_tasks ;;
    inference)    run_inference ;;
    eval)         run_eval ;;
    all)
        run_prepare
        run_decompose
        run_create_tasks
        run_inference
        run_eval
        ;;
    *)
        echo "Usage: $0 {prepare|decompose|create_tasks|inference|eval|all}"
        exit 1
        ;;
esac

echo ""
echo "Done!"
