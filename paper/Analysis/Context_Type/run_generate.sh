#!/usr/bin/env bash
RELEASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${RELEASE_SCRIPT_DIR}/../../.." && pwd)"
cd "${RELEASE_SCRIPT_DIR}"
mkdir -p logs
# Run script for generating AdvBench-related context corpora.
# Use the uncensored mlabonne/NeuralDaredevil-8B-abliterated model.

# Set the working directory.
cd "${RELEASE_SCRIPT_DIR}"

# Basic configuration.
MODEL_PATH="mlabonne/NeuralDaredevil-8B-abliterated"
OUTPUT_DIR="./relevant_contexts"
MAX_SAMPLES=100
NUM_PARAGRAPHS=5
NUM_RUNS=3  # Run 3 times to enrich the corpus.

# GPU configuration.
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.95

# Sampling parameters.
TEMPERATURE=0.8
TOP_P=0.95
TOP_K=50
MAX_TOKENS=4096

echo "========================================"
echo "Step 1: Generating Relevant Contexts"
echo "========================================"
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Samples: ${MAX_SAMPLES}"
echo "Paragraphs per query: ${NUM_PARAGRAPHS}"
echo "Number of runs: ${NUM_RUNS}"
echo "========================================"

python generate_relevant.py generate \
    --model_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_samples ${MAX_SAMPLES} \
    --num_paragraphs ${NUM_PARAGRAPHS} \
    --num_runs ${NUM_RUNS} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --top_k ${TOP_K} \
    --max_tokens ${MAX_TOKENS} \
    --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
    --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION} \
    --use_domain_expansion True

echo ""
echo "========================================"
echo "Step 2: Converting to HuggingFace Dataset"
echo "========================================"

# Convert to HuggingFace Dataset format for compatibility with create_task.py.
python generate_relevant.py convert \
    --input_dir "${OUTPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}/hf_dataset" \
    --mode "all"

# echo ""
# echo "========================================"
# echo "Done!"
# echo "========================================"
# echo "Generated contexts: ${OUTPUT_DIR}/all_contexts.json"
# echo "HuggingFace Dataset: ${OUTPUT_DIR}/hf_dataset/all_contexts"
# echo ""
# echo "Use relevant contexts in create_task.py:"
# echo '  python create_task.py --noise_dataset_name "./relevant_contexts/hf_dataset/all_contexts"'
