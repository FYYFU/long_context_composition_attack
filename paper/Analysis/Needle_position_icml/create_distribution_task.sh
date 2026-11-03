#!/usr/bin/env bash
RELEASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${RELEASE_SCRIPT_DIR}/../../.." && pwd)"
cd "${RELEASE_SCRIPT_DIR}"
mkdir -p logs
# ============================================================================
# Fragment distribution pattern experiment: Concentrated vs dispersed.
#
# Run only the adjacent condition, which is the most concentrated; reuse existing data for the other two:
#   - concentrated: Reuse the Needle_Position experiment with 25% windows.
#   - dispersed: Reuse the main experiment baseline with 100% random placement.
# ============================================================================

DATASET_PATH="${RELEASE_ROOT}/data/source/advbench_transformed_api_gemini-2.0-flash-001_fixed.json"
NOISE_DATASET="${PG19_PATH:-${RELEASE_ROOT}/data/corpus/pg19}"

# --- Adjacent: Place all facts next to one another at the same position. ---
for level in 0 1 2 3; do
    echo "Processing level ${level}, distribution: adjacent"
    python create_task.py \
        --level ${level} \
        --dataset_path ${DATASET_PATH} \
        --out_folder ./generated_tasks/dist_adjacent/gemini-2.0-flash-001/ \
        --number_of_samples 100 \
        --noise_dataset_name "${NOISE_DATASET}" \
        --task_start_pct 0.0 \
        --task_end_pct 1.0 \
        --distribution_mode adjacent \
        --message_lengths '[1000,4000,16000]'
done

echo "Adjacent distribution tasks generated!"
