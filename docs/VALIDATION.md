# Release preparation validation

The following checks were run on 2026-11-03 without model or judge API calls:

- Verified compressed and uncompressed SHA-256 hashes for all 12 main task
  files and aligned all 520 source records with every condition.
- Verified 650 reference archives: 324 main/alternative-judge score files,
  228 historical final-response files, 24 benign/HarmBench task files, and
  74 historical control-result files.
- Parsed 81 Python files and checked 58 shell files with `bash -n`.
- Checked the 51 migrated experiment shell entry points' directory setup from
  a different working directory.
- Ran the root inference/evaluation/data/analysis CLIs and all three
  decomposition-validation CLIs with `--help`.
- Regenerated the 15-model, 180-cell historical table, Wilson intervals, and
  paired bootstrap outputs. All seven released reference CSV files match
  within an absolute/relative numerical tolerance of `1e-10`.
- Generated and visually inspected the main-table heatmap.
- Prepared matched Qwen tasks for Levels 1–3 at 0k, 16k, and 64k (100 each).
- Recomputed cross-judge agreement from the restored historical score files.
- Scanned publishable text and decompressed JSON archives for credential-shaped
  strings and original machine-specific paths; none were detected. Cleared
  notebook outputs were checked. A separate inspection of literal API-key
  arguments found only placeholder examples.
- Checked that no publishable file exceeds 100 MiB; the largest is about
  45.54 MiB. The compressed release is about 305 MiB.
- Rechecked the original 171 retained source files against their recorded hashes;
  the original research workspace files were unchanged.
- Translated Chinese comments and docstrings in Python, shell scripts, and
  notebooks into English. A syntax-aware scan found no remaining Chinese
  annotations. The annotation-only edits left all 103 retained Python/notebook
  syntax trees unchanged after excluding docstrings, and preserved the commands
  in all 58 shell files. The later metric alignment is described below.
- Removed obsolete alternative-rubric artifacts and their evaluation/analysis
  scripts, unused model launchers, and exploratory plotting notebooks. Updated
  the archive and source manifests to contain only retained files.
- Unified the root evaluation, summary, and sample-size comparison with the
  historical `score < 5` statistic, retaining score -1 in the numerator and
  denominator. All default judge settings use `google/gemini-2.0-flash-lite-001`.
  Regression checks compare all three aggregation paths against the 180
  archived main-table cells, including their invalid counts and Wilson intervals.
- Compared the 180 model cells with the local manuscript table: 179 match
  exactly. Gemini-3-Flash / Direct Retrieval / 0k is 99% in the scores and
  100% in the manuscript; see `docs/REPRODUCIBILITY.md`.

Offline analysis was exercised with Python 3.10.19, NumPy 1.26.4, and
Matplotlib 3.10.8. Historical code was adapted for repository-relative paths;
the recorded prompts and scores were preserved.

The declared API dependency `openai==2.41.0` was installed successfully in a
separate local validation environment. The runner/judge CLI imports and client
construction were also checked against that pinned version without API calls.

Run the metric regression checks after installing `requirements.txt`:

```bash
python -m unittest discover -s tests -v
```

No GPU model inference, paid API evaluation, or complete supplementary-model
rerun was performed during preparation. Old API model identifiers may no
longer be available. Known historical data discrepancies and omitted thinking
traces are documented in `data/README.md` and `docs/REPRODUCIBILITY.md`.

The X-Teaming prompt's upstream redistribution terms still need confirmation
before public publication; see `THIRD_PARTY_NOTICES.md`.
