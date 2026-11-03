# Experiment guide

Commands below assume the repository root, Python 3.10+, and:

```bash
pip install -r requirements-experiments.txt
python src/prepare_data.py
python src/prepare_reference.py
```

Set the relevant API key (`OPENROUTER_API_KEY` or `OPENAI_API_KEY`) for API runs.
Commands that run a target model or judge incur API usage or require local GPU
resources. Historical inputs and outputs are separate from the new root
pipeline's `outputs/` directory.

| Experiment | Entry point | Input / scope |
| --- | --- | --- |
| Historical main table | `bash scripts/reproduce_paper.sh` | Bundled per-request scores; no API |
| Full 520 evaluation | `bash scripts/run_openrouter.sh` | Exact 12 task files; supported GPT-oss models |
| Original multi-model evaluation | `paper/src/run_api.py`, `paper/src/*.sh` | Original model-specific settings; use a new output folder when rerunning |
| Benign reconstruction | `bash controls/benign_baseline/run_pipeline.sh gpt-oss-20b` | Bundled exact 20-example tasks |
| Additional benign models | `bash robustness/benign_baseline/run_gpt_oss_120b.sh` or `bash robustness/benign_baseline/run_other_model.sh qwen3-next-80b-thinking` | Same benign tasks |
| HarmBench | `bash controls/harmbench_benchmark/run_pipeline.sh --skip-decompose --skip-tasks --model gpt-oss-20b` | Bundled 100-example HarmBench source, decompositions, and task files |
| Judge agreement analysis | `bash controls/cross_validation/scripts/run_analysis.sh` | Restored historical scores; no new judge calls |
| New cross-judge evaluation | `bash controls/cross_validation/scripts/run_cross_validation.sh` | Restored final responses; requires OpenAI API access |
| Objective-aligned decomposition validation | `bash robustness/decomposition_validation/run_objective_aligned.sh` | First 100 sources; isolated local, safety-cue, and global judgments |
| Alternative-decomposition robustness | `bash robustness/other_decomposition_sim/run_prepare.sh` then `run_20b.sh`, `run_120b.sh`, `evaluate_all.sh`, `summarize.sh` from that directory | Qwen fragments inserted into matched original contexts |
| Safety-aligned checkpoints | `GPUS=0,1,2,3 bash robustness/safety_aligned_model_validation/full_pipeline.sh` | Six checkpoints in `models.json`; requires a vLLM environment |
| Over-refusal of aligned checkpoints | `GPUS=0,1,2,3 bash robustness/safety_aligned_model_validation/overrefusal/full_pipeline.sh` | XSTest safe requests; requires vLLM and API evaluation |
| System-prompt defense | `controls/system_prompt_defense/` | Harmful benchmark and XSTest controls |
| Position / distribution / language / background sweeps | `paper/Analysis/` | Regenerate the selected ablation inputs before inference |

Historical judge-agreement scripts may produce output different from a reported
manuscript aggregate if a different judge combination is selected.
Their model and judge selections are explicit in the shell scripts.

## Construct new decompositions and contexts

The original full decomposition prompts and API generator are in
`paper/src/create_task.py`. Inspect its CLI before generating new samples:

```bash
python paper/src/create_task.py --help
```

For context construction, use the modified BABILong utilities and constructor:

```bash
export PG19_PATH=/absolute/path/to/a/saved/pg19/dataset
python paper/Analysis/Needle_Position/create_task.py \
  --dataset_path data/source/advbench_transformed_api_gemini-2.0-flash-001_fixed.json \
  --out_folder outputs/rebuilt-tasks \
  --noise_dataset_name "$PG19_PATH" \
  --level 1 --number_of_samples 100 \
  --message_lengths '[0,16000,64000]' --random_seed 42
```

`PG19_PATH` must point to a Hugging Face dataset saved with `save_to_disk`, with
a `text` column. The full corpus is not bundled. Obtain it from the upstream
PG19 source listed in `THIRD_PARTY_NOTICES.md`. Install NLTK sentence-tokenizer
resources for your NLTK version when rebuilding contexts. Corpus selection,
ordering, tokenizer revision, and sampling settings affect regenerated inputs;
use the archived tasks to reproduce the fixed benchmark.

The final 520-request Gemini and Qwen decomposition files are included; the
exploratory multi-agent generation pipeline and obsolete prompt variants are
not part of this release.
