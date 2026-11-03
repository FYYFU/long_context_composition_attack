#!/usr/bin/env bash
RELEASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${RELEASE_SCRIPT_DIR}/../../.." && pwd)"
cd "${RELEASE_SCRIPT_DIR}"
mkdir -p logs
# =============================================================
# ICML 2026 Rebuttal: Analyze inter-judge agreement
# Run AFTER run_cross_validation.sh completes
#
# All 3 judges use X-teaming 1-5 scores, so all pairs get
# Pearson correlation + binarized Cohen's Kappa + Agreement Rate
# =============================================================

SCRIPT_DIR="$(cd "${RELEASE_SCRIPT_DIR}" && pwd)"
RESULT_BASE="${RELEASE_ROOT}/paper/results/gemini-2.0-flash-001"
OUTPUT_DIR="${RELEASE_ROOT}/controls/cross_validation/results"

python "$SCRIPT_DIR/analyze_judge_agreement.py" \
    --result_base "$RESULT_BASE" \
    --gemini_judge_name "google-gemini-2.0-flash-lite-001" \
    --judge2_name "gpt-5.4-nano" \
    --judge3_name "gpt-4o-mini" \
    --models "gpt-oss-20b-medium,gpt-oss-120b-medium,claude-sonnet-4.5-default,gemini-2.5-flash-default" \
    --levels 0,1,2,3 \
    --context_lengths 0k,16k,64k \
    --output "$OUTPUT_DIR/judge_agreement_analysis.json"
