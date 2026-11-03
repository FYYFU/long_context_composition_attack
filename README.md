# Long Context Composition Attack

Research code and benchmark artifacts for [**Do Reasoning LLMs Refuse What They Infer in Long Contexts?**](https://arxiv.org/abs/2602.08874).

**Accepted at NeurIPS 2026.**

The benchmark compares Direct Retrieval, Single-hop Aggregation, Chain Reasoning,
and Multi-hop Deductive Reasoning at 0k, 16k, and 64k context lengths. It includes
the fixed 520-request task pool, the historical first-100-request results across
15 models, and a resumable pipeline for new GPT-oss evaluations.

## Quick start: reproduce the paper table without an API

Use Python 3.10+ and Bash. Run commands from the repository root.

```bash
git clone https://github.com/FYYFU/long_context_composition_attack.git
cd long_context_composition_attack
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-analysis.txt
bash scripts/reproduce_paper.sh
```

This reads the bundled historical per-request scores and writes:

- `outputs/paper100/main_table.{csv,md}`: the 15-model main table and averages;
- `outputs/paper100/main_table.{png,pdf}`: a figure computed from those scores;
- `outputs/paper100/cell_safety_ratio_wilson.csv`: per-cell Wilson intervals;
- paired bootstrap comparisons and `invalid_score_audit.csv` in the same folder.

No API key or GPU is needed. The bootstrap uses 10,000 resamples with seed
20260726. Override with `bash scripts/reproduce_paper.sh --seed 42`.

Safety Ratio follows the paper's historical definition:
`SR = count(score < 5) / total records`. Failed judgments recorded as `-1`
remain in both the numerator and denominator, with their counts reported
separately. Evaluation defaults use the original
`google/gemini-2.0-flash-lite-001` judge. See
[reproducibility details](docs/REPRODUCIBILITY.md) for sample selection and the
one recorded difference between the archived scores and the manuscript table.

## Verify and prepare the benchmark

```bash
bash scripts/verify_repo.sh
python src/prepare_reference.py
```

Verification makes no model calls. It checks the fixed task hashes, all
reference archive hashes, sample alignment, Python/shell syntax, and release
files. It extracts the 12 main task files to `data/tasks/`.
`prepare_reference.py` additionally restores historical final responses,
scores, and exact benign/HarmBench task files, and creates the input layout used
by the original experiment scripts. Extracted files are ignored by Git.

The compressed release is approximately 305 MiB; allow at least 2 GiB for
extraction and analysis. No individual bundled archive exceeds 50 MiB.

## Run a new evaluation

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY='your-key'

# Two short target requests, followed by their judgments; incurs API usage.
bash scripts/smoke_api.sh

# Full 520 requests × 4 structures × 3 context lengths for one model.
MODELS=gpt-oss-20b bash scripts/run_openrouter.sh
# Or both supported models:
MODELS=gpt-oss-20b,gpt-oss-120b bash scripts/run_openrouter.sh
```

The default full-run settings are `reasoning_effort=medium`, temperature 1.0,
top-p 1.0, max output tokens 16,368, and provider `google-vertex`. The judge is
`google/gemini-2.0-flash-lite-001` at temperature 0, using the same historical
Safety Ratio definition as the paper table.
Each full model run has 6,240 target requests plus judge calls.

Outputs go to `outputs/openrouter/`. Rerunning the same command resumes matching
checkpoints. Use a different `RUN_ROOT` and `RUN_ID` when changing a protocol:

```bash
RUN_ROOT="$PWD/outputs/repeat-2" RUN_ID=repeat-2 \
  MODELS=gpt-oss-120b bash scripts/run_openrouter.sh
RUN_ROOT="$PWD/outputs/openrouter" bash scripts/status.sh
```

For local GPT-oss-20B inference, use a separate Python environment with
`requirements-vllm.txt`, then run:

```bash
GPU_IDS=0,1,2,3 MODEL_PATH=openai/gpt-oss-20b \
  bash scripts/run_local_vllm_20b.sh
```

This launcher defaults to four GPUs and uses an API judge. Set `RUN_JUDGE=0` to
run target inference only. GPU memory requirements depend on the context length,
model, and serving settings; weights are downloaded separately.

## Repository map

| Location | Contents |
| --- | --- |
| `src/`, `scripts/` | Resumable inference, judge, data preparation, validation, summaries, and offline reproduction |
| `data/source/`, `data/tasks-gz/` | Fixed decompositions and exact 520-request task files |
| `data/reference-gz/` | Historical scores/final responses and exact control tasks |
| `configs/protocols.json` | Explicit historical and rerun protocol definitions |
| `paper/src/`, `paper/eval/`, `paper/utils/` | Original decomposition prompts, multi-model launch settings, and evaluators |
| `paper/Analysis/`, `paper/vis/` | Position, language, context, reasoning-effort, and trajectory analyses; notebooks have cleared outputs |
| `controls/` | Benign reconstruction, HarmBench, judge agreement, system-prompt defense, and over-refusal experiments |
| `robustness/` | Decomposition validation, alternative decompositions, aligned-model comparisons, and statistical analysis |
| `reference/` | Compact historical statistics and supporting summaries |
| `docs/` | Protocols, experiment instructions, and file provenance |

See [the experiment guide](docs/EXPERIMENTS.md) for supplementary runs and
[the data description](data/README.md) for schemas and archive provenance.

## Citation

If you use this code or benchmark, please cite:

```bibtex
@misc{fu2026reasoningllmsrefuseinfer,
      title={Do Reasoning LLMs Refuse What They Infer in Long Contexts?},
      author={Yu Fu and Haz Sameen Shahgir and Huanli Gong and Zhipeng Wei and N. Benjamin Erichson and Yue Dong},
      year={2026},
      eprint={2602.08874},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.08874},
}
```

Citation metadata is also available in [CITATION.cff](CITATION.cff).

## License and attribution

Original project contributions are provided under the [MIT License](LICENSE).
Third-party code, prompts, and datasets retain their own terms; in particular,
the adapted BABILong utilities retain Apache-2.0 and XSTest data retains
CC-BY-4.0. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The benchmark and historical outputs contain harmful requests and model-generated
content used to measure safety behavior. They are research evaluation artifacts.
Model weights and API access are obtained separately under their providers' terms.
