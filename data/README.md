# Data and provenance

## Fixed benchmark

`source/advbench_transformed_api_gemini-2.0-flash-001_fixed.json` is the canonical
520-request decomposition source. The Qwen source is included for the
alternative-decomposition experiment. The first 100 records retain the paper's
request order.

`tasks-gz/level_{0,1,2,3}_{0k,16k,64k}.json.gz` contains the exact task files.
Each uncompressed file is a list of 520 objects with:

- `input`: the exact target-model prompt;
- `question`: the neutral synthesis query;
- `target` and `original_query`: source-task fields;
- `level`, `context_length`, and `n_facts`: construction metadata.

The source decomposition JSON files can contain exploratory higher-level fields;
only Levels 0–3 are part of the released benchmark protocol.

Run `python src/prepare_data.py` from the repository root to verify and extract
the tasks. Hashes are recorded in that script. No remote dataset download is
required to run the fixed tasks.

## Historical results and control tasks

`reference-gz/` holds compact historical final responses, judge outputs, and
exact benign/HarmBench task files. `reference_manifest.json` describes every
archive and its source transformation. Run `python src/prepare_reference.py`
to materialize all artifacts, or pass `--scores-only` for offline table analysis.
Pass `--verify-only` to verify without writing.

The archive set also includes the existing GPT-oss benign-reconstruction and
HarmBench responses and evaluator results, restored under `controls/`.

Historical generation records retain the final response and source objective;
thinking traces and repeated long input text are not included. Evaluations
retain original scores and explanations, not newly recomputed judgments.
Any historical `generated.safe` field is a keyword-based heuristic from the
original inference script; use the separate judge scores for the reported
Safety Ratio.
Scores include multiple judge models; the main-table script selects the
original Gemini judge by exact filename.

Source records and the compact generation `sample_index` use zero-based indices.
Historical evaluator `index` fields are one-based (1–100); index 1 corresponds
to source record 0. These original evaluator IDs have been preserved.

One supplementary record originally has no `score` field: GPT-oss-120B **low**
reasoning effort, Level 0, 64k, original Gemini judge, evaluator index 44.
It is retained unchanged rather than assigned an invented score. This record
is outside the main table, which uses medium reasoning effort. Supplementary
analyses should handle and report this missing value explicitly.

Third-party source data retains its upstream license. See
`../THIRD_PARTY_NOTICES.md`; the root MIT license does not replace dataset terms.
