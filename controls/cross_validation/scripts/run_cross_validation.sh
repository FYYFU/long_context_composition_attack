#!/usr/bin/env bash
RELEASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${RELEASE_SCRIPT_DIR}/../../.." && pwd)"
cd "${RELEASE_SCRIPT_DIR}"
mkdir -p logs
# =============================================================
# ICML 2026 Rebuttal: Cross-validation with multiple judges
# =============================================================
# All judges use the SAME X-teaming 1-5 score prompt via
# check_safe_gpt.py, only switching the judge model.
# All called via OpenAI official API.
#
# Judge 1: gpt-5.4-nano
# Judge 2: gpt-4o-mini
#
# Models: 4 representative models from Table 1
# Context lengths: 0k, 16k, 64k (matching Table 1)
# =============================================================

EVAL_DIR="$(cd "${RELEASE_SCRIPT_DIR}/../eval" && pwd)"
RESULT_BASE="${RELEASE_ROOT}/paper/results/gemini-2.0-flash-001"

MODELS=("gpt-oss-20b:medium" "gpt-oss-120b:medium" "claude-sonnet-4.5:default" "gemini-2.5-flash:default")

# =============================================================
# Judge 1: gpt-5.4-nano
# =============================================================
echo ""
echo "############################################################"
echo "# Judge 1: gpt-5.4-nano"
echo "############################################################"

for entry in "${MODELS[@]}"; do
    model_name="${entry%%:*}"
    reasoning_effort="${entry##*:}"

    echo ""
    echo "========== [gpt-5.4-nano] ${model_name}-${reasoning_effort} =========="
    python "$EVAL_DIR/check_safe_gpt.py" \
        --meta_path "$RESULT_BASE" \
        --judge_model "gpt-5.4-nano" \
        --judge_name "gpt-5.4-nano" \
        --api_provider openai \
        --model_name "$model_name" \
        --reasoning_efforts "$reasoning_effort" \
        --levels 0,1,2,3 \
        --context_lengths 0k,16k,64k \
        --skip_existing
done

# =============================================================
# Judge 2: gpt-4o-mini
# =============================================================
echo ""
echo "############################################################"
echo "# Judge 2: gpt-4o-mini"
echo "############################################################"

for entry in "${MODELS[@]}"; do
    model_name="${entry%%:*}"
    reasoning_effort="${entry##*:}"

    echo ""
    echo "========== [gpt-4o-mini] ${model_name}-${reasoning_effort} =========="
    python "$EVAL_DIR/check_safe_gpt.py" \
        --meta_path "$RESULT_BASE" \
        --judge_model "gpt-4o-mini" \
        --judge_name "gpt-4o-mini" \
        --api_provider openai \
        --model_name "$model_name" \
        --reasoning_efforts "$reasoning_effort" \
        --levels 0,1,2,3 \
        --context_lengths 0k,16k,64k \
        --skip_existing
done

echo ""
echo "############################################################"
echo "# Cross-validation complete!"
echo "############################################################"
