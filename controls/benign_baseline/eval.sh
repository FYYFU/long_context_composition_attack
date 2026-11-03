#!/usr/bin/env bash
RELEASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${RELEASE_SCRIPT_DIR}/../.." && pwd)"
cd "${RELEASE_SCRIPT_DIR}"
mkdir -p logs
python eval_reconstruction.py \                                         
    --meta_path ./results \                                             
    --model_name gpt-oss-20b \                                          
    --judge_model "google/gemini-2.0-flash-lite-001" \                  
    --api_provider openrouter \                                         
    --levels 0,1,2 \                                                    
    --context_lengths 0k,16k,64k \                                      
    --num_workers 16 \                                                  
    --timeout 240 \                                                     
    --skip_existing
