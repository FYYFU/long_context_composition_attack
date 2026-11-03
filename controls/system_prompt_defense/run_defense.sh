#!/usr/bin/env bash
RELEASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${RELEASE_SCRIPT_DIR}/../.." && pwd)"
cd "${RELEASE_SCRIPT_DIR}"
mkdir -p logs
# ============================================================
# System Prompt Defense Experiment
#
# Tests whether adding a safety system prompt can mitigate
# compositional reasoning attacks.
#
# Usage:
#   bash run_defense.sh                    # Run all steps
#   bash run_defense.sh inference           # Step 1: Run inference only
#   bash run_defense.sh eval                # Step 2: Evaluate only
#   bash run_defense.sh analyze             # Step 3: Analyze only
# ============================================================

set -e

SCRIPT_DIR="$(cd "${RELEASE_SCRIPT_DIR}" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TASKS_FOLDER="$PROJECT_ROOT/paper/generated_tasks/gemini-2.0-flash-001"
BENIGN_TASKS_FOLDER="$PROJECT_ROOT/controls/benign_baseline/generated_tasks"
BENIGN_BASELINE_RESULTS="$PROJECT_ROOT/controls/benign_baseline/results"
OUTPUT_FOLDER="$SCRIPT_DIR/results"
BASELINE_ROOT="$PROJECT_ROOT/paper/results/gemini-2.0-flash-001"

# Defense prompt variant to test
DEFENSE_PROMPT="safety_v1"

# Models to test (use same hyperparams as original experiments)
# Format: model_path|reasoning_effort|temperature|top_p|batch_size|api_provider
MODELS=(
    # "openai/gpt-oss-20b|medium|1|1|5|openrouter"
    "openai/gpt-4.1|default|0|1|10|openai"
    "openai/gpt-5.2|default|0|1|10|openai"
    # "google/gemini-2.5-flash-preview|default|0|1|5|openrouter"
    # "anthropic/claude-sonnet-4.5|default|0|1|2|openrouter"
)

STEP=${1:-"all"}

# ============================================================
# Step 1: Run inference with safety system prompt
# ============================================================
run_inference() {
    echo "=========================================="
    echo "Step 1: Running inference with system prompt defense"
    echo "=========================================="

    for model_config in "${MODELS[@]}"; do
        IFS='|' read -r MODEL_PATH REASONING_EFFORT TEMP TOP_P BATCH_SIZE API_PROVIDER <<< "$model_config"
        API_PROVIDER=${API_PROVIDER:-openrouter}

        echo ""
        echo "--- Model: $MODEL_PATH (effort=$REASONING_EFFORT, api=$API_PROVIDER) ---"

        python "$SCRIPT_DIR/run_defense.py" \
            --tasks_folder "$TASKS_FOLDER" \
            --model_path "$MODEL_PATH" \
            --output_folder "$OUTPUT_FOLDER" \
            --defense_prompt "$DEFENSE_PROMPT" \
            --temperature "$TEMP" \
            --top_p "$TOP_P" \
            --max_tokens 16368 \
            --batch_size "$BATCH_SIZE" \
            --api_provider "$API_PROVIDER" \
            --num_workers 10 \
            --timeout 120 \
            --reasoning_effort "$REASONING_EFFORT" \
            --levels "0,1,2,3" \
            --context_lengths "0k,16k,64k" \
            --max_examples 100
    done
}

# ============================================================
# Step 2: Evaluate results with judge
# ============================================================
run_eval() {
    echo "=========================================="
    echo "Step 2: Evaluating defense results"
    echo "=========================================="

    DEFENSE_DIR="$OUTPUT_FOLDER/defense_${DEFENSE_PROMPT}"

    for model_dir in "$DEFENSE_DIR"/*/; do
        if [ -d "$model_dir" ]; then
            echo ""
            echo "--- Evaluating: $(basename "$model_dir") ---"

            python "$SCRIPT_DIR/eval_defense.py" \
                --results_dir "$model_dir" \
                --judge_model "google/gemini-2.0-flash-lite-001" \
                --api_provider openrouter \
                --levels "0,1,2,3" \
                --context_lengths "0k,16k,64k" \
                --num_workers 10 \
                --skip_existing
        fi
    done
}

# ============================================================
# Step 3: Compare defense vs baseline
# ============================================================
run_analyze() {
    echo "=========================================="
    echo "Step 3: Analyzing defense vs baseline"
    echo "=========================================="

    DEFENSE_DIR="$OUTPUT_FOLDER/defense_${DEFENSE_PROMPT}"

    # Map defense model dirs to baseline dirs
    declare -A BASELINE_MAP
    BASELINE_MAP["gpt-oss-20b-medium"]="gpt-oss-20b-medium"
    BASELINE_MAP["claude-sonnet-4.5-default"]="claude-sonnet-4.5-default"
    BASELINE_MAP["gemini-2.5-flash-preview-default"]="gemini-2.5-flash-default"

    for model_dir in "$DEFENSE_DIR"/*/; do
        if [ -d "$model_dir" ]; then
            MODEL_NAME=$(basename "$model_dir")
            BASELINE_NAME="${BASELINE_MAP[$MODEL_NAME]:-$MODEL_NAME}"
            BASELINE_PATH="$BASELINE_ROOT/$BASELINE_NAME"

            if [ -d "$BASELINE_PATH" ]; then
                echo ""
                echo "--- $MODEL_NAME ---"

                python "$SCRIPT_DIR/analyze_results.py" \
                    --baseline_dir "$BASELINE_PATH" \
                    --defense_dir "$model_dir" \
                    --levels "0,1,2,3" \
                    --context_lengths "0k,16k" \
                    --output_file "$model_dir/comparison.json"
            else
                echo "Baseline not found for $MODEL_NAME at $BASELINE_PATH"
            fi
        fi
    done
}

# ============================================================
# Step 4: Run inference on BENIGN tasks with system prompt
#         (tests over-refusal / utility loss)
# ============================================================
run_inference_benign() {
    echo "=========================================="
    echo "Step 4: Running BENIGN inference with system prompt defense"
    echo "=========================================="

    for model_config in "${MODELS[@]}"; do
        IFS='|' read -r MODEL_PATH REASONING_EFFORT TEMP TOP_P BATCH_SIZE API_PROVIDER <<< "$model_config"
        API_PROVIDER=${API_PROVIDER:-openrouter}

        echo ""
        echo "--- Model: $MODEL_PATH (effort=$REASONING_EFFORT, api=$API_PROVIDER, BENIGN) ---"

        python "$SCRIPT_DIR/run_defense.py" \
            --tasks_folder "$BENIGN_TASKS_FOLDER" \
            --model_path "$MODEL_PATH" \
            --output_folder "$OUTPUT_FOLDER/benign" \
            --defense_prompt "$DEFENSE_PROMPT" \
            --temperature "$TEMP" \
            --top_p "$TOP_P" \
            --max_tokens 16368 \
            --batch_size "$BATCH_SIZE" \
            --api_provider "$API_PROVIDER" \
            --num_workers 10 \
            --timeout 120 \
            --reasoning_effort "$REASONING_EFFORT" \
            --levels "0,1,2,3" \
            --context_lengths "0k,16k,64k"
    done
}

# ============================================================
# Step 5: Evaluate BENIGN reconstruction accuracy
# ============================================================
run_eval_benign() {
    echo "=========================================="
    echo "Step 5: Evaluating BENIGN reconstruction accuracy"
    echo "=========================================="

    BENIGN_DEFENSE_DIR="$OUTPUT_FOLDER/benign/defense_${DEFENSE_PROMPT}"

    for model_dir in "$BENIGN_DEFENSE_DIR"/*/; do
        if [ -d "$model_dir" ]; then
            MODEL_NAME=$(basename "$model_dir")
            echo ""
            echo "--- Evaluating benign reconstruction: $MODEL_NAME ---"

            python "$PROJECT_ROOT/controls/benign_baseline/eval_reconstruction.py" \
                --meta_path "$BENIGN_DEFENSE_DIR" \
                --model_name "$MODEL_NAME" \
                --judge_model "google/gemini-2.0-flash-lite-001" \
                --api_provider openrouter \
                --levels "0,1,2,3" \
                --context_lengths "0k,16k,64k" \
                --num_workers 10 \
                --skip_existing
        fi
    done
}

# ============================================================
# Step 6: Compare benign results (with vs without system prompt)
# ============================================================
run_analyze_benign() {
    echo "=========================================="
    echo "Step 6: Comparing benign reconstruction (with vs without system prompt)"
    echo "=========================================="

    BENIGN_DEFENSE_DIR="$OUTPUT_FOLDER/benign/defense_${DEFENSE_PROMPT}"

    # Map defense model dirs to baseline benign dirs
    declare -A BENIGN_BASELINE_MAP
    BENIGN_BASELINE_MAP["gpt-oss-20b-medium"]="gpt-oss-20b"

    for model_dir in "$BENIGN_DEFENSE_DIR"/*/; do
        if [ -d "$model_dir" ]; then
            MODEL_NAME=$(basename "$model_dir")
            BENIGN_NAME="${BENIGN_BASELINE_MAP[$MODEL_NAME]:-$MODEL_NAME}"
            BENIGN_PATH="$BENIGN_BASELINE_RESULTS/$BENIGN_NAME"

            echo ""
            echo "--- $MODEL_NAME (BENIGN) ---"
            echo "  Baseline (no sys prompt): $BENIGN_PATH"
            echo "  Defense  (with sys prompt): $model_dir"

            if [ -d "$BENIGN_PATH" ]; then
                python "$SCRIPT_DIR/analyze_benign.py" \
                    --baseline_dir "$BENIGN_PATH" \
                    --defense_dir "$model_dir" \
                    --levels "0,1,2,3" \
                    --context_lengths "0k,16k,64k" \
                    --output_file "$model_dir/benign_comparison.json"
            else
                echo "  Benign baseline not found at $BENIGN_PATH"
            fi
        fi
    done
}

# ============================================================
# Execute
# ============================================================
case "$STEP" in
    inference)        run_inference ;;
    eval)             run_eval ;;
    analyze)          run_analyze ;;
    inference_benign) run_inference_benign ;;
    eval_benign)      run_eval_benign ;;
    analyze_benign)   run_analyze_benign ;;
    all)
        run_inference
        run_eval
        run_analyze
        run_inference_benign
        run_eval_benign
        run_analyze_benign
        ;;
    *)
        echo "Usage: $0 {inference|eval|analyze|inference_benign|eval_benign|analyze_benign|all}"
        exit 1
        ;;
esac

echo ""
echo "Done!"
