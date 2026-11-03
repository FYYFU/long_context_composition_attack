#!/usr/bin/env bash
RELEASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${RELEASE_SCRIPT_DIR}/../../.." && pwd)"
cd "${RELEASE_SCRIPT_DIR}"
mkdir -p logs
# Define four position ranges.
declare -A positions
positions["start"]="0.0 0.25"
positions["early_mid"]="0.25 0.5"
positions["late_mid"]="0.5 0.75"
positions["end"]="0.75 1.0"

for level in 0 1 2 3; do
    for pos_name in start early_mid late_mid end; do
        read start_pct end_pct <<< "${positions[$pos_name]}"
        echo "Processing level ${level}, position ${pos_name} (${start_pct} - ${end_pct})"
        
        python create_task.py \
            --level ${level} \
            --dataset_path ${RELEASE_ROOT}/data/source/advbench_transformed_api_gemini-2.0-flash-001_fixed.json \
            --out_folder ./generated_tasks/position_${pos_name}/gemini-2.0-flash-001/ \
            --number_of_samples 100 \
            --noise_dataset_name "${PG19_PATH:-${RELEASE_ROOT}/data/corpus/pg19}" \
            --task_start_pct ${start_pct} \
            --task_end_pct ${end_pct}
    done
done
