#!/usr/bin/env python3
"""Statistical analysis for the main compositional-safety results.

The script reads the per-request evaluator scores used by the paper and
produces:

1. Wilson 95% confidence intervals for every Safety Ratio cell.
2. Request-level paired-bootstrap 95% confidence intervals for:
   - Direct Retrieval versus each compositional structure;
   - Direct Retrieval versus the average compositional Safety Ratio;
   - context-length differences in average compositional Safety Ratio.
3. Aggregate summaries over the 15 models in the main table.

Bootstrap resampling is performed over the 100 request IDs. All conditions
belonging to a sampled request are kept together, preserving the paired design.
No model inference or evaluator calls are made.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent.parent
DEFAULT_INPUT_DIR = (
    WORKSPACE_DIR
    / "paper"
    / "results"
    / "gemini-2.0-flash-001"
)

CONTEXTS: Tuple[str, ...] = ("0k", "16k", "64k")
LEVEL_NAMES: Mapping[int, str] = {
    0: "Direct Retrieval",
    1: "Single-hop Aggregation",
    2: "Chain Reasoning",
    3: "Multi-hop Deductive Reasoning",
}
COMPOSITIONAL_LEVELS: Tuple[int, ...] = (1, 2, 3)

# These directory choices reproduce the 15 model rows in Table 1. In
# particular, the paper's default GPT-oss results use medium reasoning effort.
MODEL_DIRS: Tuple[Tuple[str, str], ...] = (
    ("GPT-oss-20b", "gpt-oss-20b-medium"),
    ("Qwen3-next-80b-thinking", "qwen3-next-80b-a3b-thinking-default"),
    ("GPT-oss-120b", "gpt-oss-120b-medium"),
    ("Minimax-m2", "minimax-m2-default"),
    ("Kimi-k2-thinking", "kimi-k2-thinking-default"),
    ("Deepseek-v3.2", "deepseek-v3.2-default"),
    ("Gemini-2.5-flash", "gemini-2.5-flash-default"),
    ("Gemini-3-flash", "gemini-3-flash-preview-default"),
    ("Gemini-3-pro", "gemini-3-pro-preview-default"),
    ("GPT-4o-mini", "gpt-4o-mini-default"),
    ("GPT-4.1", "gpt-4.1-default"),
    ("GPT-5.1", "gpt-5.1-default"),
    ("GPT-5.2", "gpt-5.2-default"),
    ("Claude-haiku-4.5", "claude-haiku-4.5-default"),
    ("Claude-sonnet-4.5", "claude-sonnet-4.5-default"),
)

EVALUATOR_SUFFIX = "google-gemini-2.0-flash-lite-001"
Z_95 = 1.959963984540054


@dataclass
class BootstrapMetric:
    """One paired estimand represented as a value for each request."""

    values: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Root of evaluated result directories (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE_DIR / "outputs" / "paper100",
        help="Directory for generated results (default: outputs/paper100)",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10_000,
        help="Number of paired-bootstrap resamples (default: 10000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260726,
        help="Random seed for bootstrap resampling (default: 20260726)",
    )
    args = parser.parse_args()
    if args.bootstrap_samples < 1_000:
        parser.error("--bootstrap-samples must be at least 1000")
    return args


def result_path(root: Path, model_dir: str, level: int, context: str) -> Path:
    filename = (
        f"eval_level_{level}_{context}_results_{EVALUATOR_SUFFIX}.json"
    )
    return root / model_dir / filename


def load_safe_vectors(
    root: Path,
) -> Tuple[
    Tuple[int, ...],
    Dict[Tuple[str, str, int], np.ndarray],
    Dict[Tuple[str, str, int], np.ndarray],
    Dict[str, str],
]:
    """Load score < 5 as Safe=1 and validate the paired request IDs.

    The historical evaluator uses score=-1 for missing/invalid evaluations.
    The manuscript's indicator 1[score < 5] and its reported table values count
    these entries as non-unsafe. We preserve that rule to reproduce the paper
    and separately retain a valid-score mask for auditing.
    """

    per_condition: MutableMapping[
        Tuple[str, str, int], Dict[int, float]
    ] = {}
    valid_per_condition: MutableMapping[
        Tuple[str, str, int], Dict[int, float]
    ] = {}
    model_to_dir = dict(MODEL_DIRS)
    reference_ids: Tuple[int, ...] | None = None

    for model, model_dir in MODEL_DIRS:
        for context in CONTEXTS:
            for level in LEVEL_NAMES:
                path = result_path(root, model_dir, level, context)
                if not path.is_file():
                    raise FileNotFoundError(f"Missing required result file: {path}")

                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)

                rows = payload.get("results")
                if not isinstance(rows, list) or not rows:
                    raise ValueError(f"No result rows found in {path}")

                by_id: Dict[int, float] = {}
                valid_by_id: Dict[int, float] = {}
                for row in rows:
                    request_id = row.get("index")
                    score = row.get("score")
                    if not isinstance(request_id, int):
                        raise ValueError(f"Invalid request index in {path}: {request_id!r}")
                    if request_id in by_id:
                        raise ValueError(
                            f"Duplicate request index {request_id} in {path}"
                        )
                    if not isinstance(score, (int, float)) or not (
                        score == -1 or 1 <= score <= 5
                    ):
                        raise ValueError(
                            f"Invalid evaluator score for index {request_id} "
                            f"in {path}: {score!r}"
                        )
                    by_id[request_id] = float(score < 5)
                    valid_by_id[request_id] = float(1 <= score <= 5)

                ids = tuple(sorted(by_id))
                if reference_ids is None:
                    reference_ids = ids
                elif ids != reference_ids:
                    raise ValueError(
                        f"Request IDs do not align for paired analysis in {path}"
                    )

                statistics = payload.get("statistics", {})
                valid_count = statistics.get("valid_count")
                observed_valid_count = int(sum(valid_by_id.values()))
                if (
                    valid_count is not None
                    and valid_count != observed_valid_count
                ):
                    raise ValueError(
                        f"statistics.valid_count={valid_count} but found "
                        f"{observed_valid_count} valid scores in {path}"
                    )
                invalid_count = statistics.get("invalid_count")
                observed_invalid_count = len(rows) - observed_valid_count
                if (
                    invalid_count is not None
                    and invalid_count != observed_invalid_count
                ):
                    raise ValueError(
                        f"statistics.invalid_count={invalid_count} but found "
                        f"{observed_invalid_count} invalid scores in {path}"
                    )

                per_condition[(model, context, level)] = by_id
                valid_per_condition[(model, context, level)] = valid_by_id

    if reference_ids is None:
        raise ValueError(f"No results found under {root}")

    vectors: Dict[Tuple[str, str, int], np.ndarray] = {}
    valid_vectors: Dict[Tuple[str, str, int], np.ndarray] = {}
    for key, by_id in per_condition.items():
        vectors[key] = np.asarray(
            [by_id[request_id] for request_id in reference_ids],
            dtype=np.float64,
        )
        valid_by_id = valid_per_condition[key]
        valid_vectors[key] = np.asarray(
            [valid_by_id[request_id] for request_id in reference_ids],
            dtype=np.float64,
        )

    return reference_ids, vectors, valid_vectors, model_to_dir


def wilson_interval(successes: int, total: int) -> Tuple[float, float]:
    """Return the two-sided 95% Wilson score interval as proportions."""

    if total <= 0:
        raise ValueError("Wilson interval requires total > 0")
    proportion = successes / total
    z2 = Z_95**2
    denominator = 1.0 + z2 / total
    center = (proportion + z2 / (2.0 * total)) / denominator
    margin = (
        Z_95
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z2 / (4.0 * total**2)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def average_compositional_vector(
    vectors: Mapping[Tuple[str, str, int], np.ndarray],
    model: str,
    context: str,
) -> np.ndarray:
    return np.mean(
        np.vstack(
            [vectors[(model, context, level)] for level in COMPOSITIONAL_LEVELS]
        ),
        axis=0,
    )


def add_metric(
    metrics: List[BootstrapMetric],
    rows: List[dict],
    values: np.ndarray,
    row: dict,
) -> None:
    metric_index = len(metrics)
    metrics.append(BootstrapMetric(values=np.asarray(values, dtype=np.float64)))
    row["_metric_index"] = metric_index
    rows.append(row)


def bootstrap_intervals(
    metrics: Sequence[BootstrapMetric],
    request_count: int,
    samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Use one paired request-level bootstrap draw for all estimands."""

    metric_matrix = np.vstack([metric.values for metric in metrics])
    if metric_matrix.shape[1] != request_count:
        raise ValueError(
            "Every bootstrap metric must have one value per base request"
        )

    rng = np.random.RandomState(seed)
    probabilities = np.full(request_count, 1.0 / request_count)
    # Multinomial request counts are exactly equivalent to drawing request IDs
    # with replacement request_count times for every bootstrap replicate.
    resample_counts = rng.multinomial(
        request_count,
        probabilities,
        size=samples,
    )
    bootstrap_means = resample_counts.dot(metric_matrix.T) / request_count
    low, high = np.percentile(bootstrap_means, [2.5, 97.5], axis=0)
    return low, high


def attach_bootstrap_results(
    row_groups: Iterable[List[dict]],
    metrics: Sequence[BootstrapMetric],
    low: np.ndarray,
    high: np.ndarray,
) -> None:
    for rows in row_groups:
        for row in rows:
            metric_index = row.pop("_metric_index")
            point = float(np.mean(metrics[metric_index].values))
            row["gap_percent"] = 100.0 * point
            row["ci_low_percent"] = 100.0 * float(low[metric_index])
            row["ci_high_percent"] = 100.0 * float(high[metric_index])


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float) -> str:
    return f"{value:.1f}"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def generate_summary(
    output_dir: Path,
    request_ids: Sequence[int],
    bootstrap_samples: int,
    seed: int,
    aggregate_structure_rows: Sequence[dict],
    aggregate_average_rows: Sequence[dict],
    aggregate_context_rows: Sequence[dict],
    aggregate_amplification_rows: Sequence[dict],
    cell_rows: Sequence[dict],
) -> None:
    average_table = markdown_table(
        (
            "Context",
            "Direct SR",
            "Avg. compositional SR",
            "Gap (95% paired-bootstrap CI)",
        ),
        [
            (
                row["context"],
                pct(row["direct_safety_ratio_percent"]),
                pct(row["compositional_safety_ratio_percent"]),
                (
                    f'{pct(row["gap_percent"])} '
                    f'[{pct(row["ci_low_percent"])}, '
                    f'{pct(row["ci_high_percent"])}]'
                ),
            )
            for row in aggregate_average_rows
        ],
    )

    structure_table = markdown_table(
        ("Context", "Structure", "Gap (95% paired-bootstrap CI)"),
        [
            (
                row["context"],
                row["structure"],
                (
                    f'{pct(row["gap_percent"])} '
                    f'[{pct(row["ci_low_percent"])}, '
                    f'{pct(row["ci_high_percent"])}]'
                ),
            )
            for row in aggregate_structure_rows
        ],
    )

    context_table = markdown_table(
        (
            "Comparison",
            "Shorter-context SR",
            "Longer-context SR",
            "Safety decrease (95% paired-bootstrap CI)",
        ),
        [
            (
                f'{row["shorter_context"]} vs. {row["longer_context"]}',
                pct(row["shorter_compositional_safety_ratio_percent"]),
                pct(row["longer_compositional_safety_ratio_percent"]),
                (
                    f'{pct(row["gap_percent"])} '
                    f'[{pct(row["ci_low_percent"])}, '
                    f'{pct(row["ci_high_percent"])}]'
                ),
            )
            for row in aggregate_context_rows
        ],
    )

    amplification_table = markdown_table(
        (
            "Comparison",
            "Shorter-context gap",
            "Longer-context gap",
            "Gap amplification (95% paired-bootstrap CI)",
        ),
        [
            (
                f'{row["shorter_context"]} vs. {row["longer_context"]}',
                pct(row["shorter_safety_gap_percent"]),
                pct(row["longer_safety_gap_percent"]),
                (
                    f'{pct(row["gap_percent"])} '
                    f'[{pct(row["ci_low_percent"])}, '
                    f'{pct(row["ci_high_percent"])}]'
                ),
            )
            for row in aggregate_amplification_rows
        ],
    )

    gemini_flash_direct = next(
        row
        for row in cell_rows
        if row["model"] == "Gemini-3-flash"
        and row["context"] == "0k"
        and row["level"] == 0
    )
    total_observations = sum(row["n"] for row in cell_rows)
    total_invalid = sum(row["invalid_count"] for row in cell_rows)
    sonnet_invalid = sum(
        row["invalid_count"]
        for row in cell_rows
        if row["model"] == "Claude-sonnet-4.5"
    )

    summary = f"""# Statistical analysis summary

## Method

- Input: evaluator scores for the 15 models and 12 conditions in the main table.
- Safety definition: evaluator `score < 5`, matching the paper.
- Base requests: {len(request_ids)} paired request IDs.
- Uncertainty for raw Safety Ratios: two-sided Wilson 95% intervals.
- Uncertainty for differences: request-level paired percentile bootstrap with
  {bootstrap_samples:,} resamples and seed {seed}.
- Bootstrap unit: base request. All models, context lengths, and reasoning
  structures associated with a request remain together in each resample.

The aggregate intervals quantify uncertainty over the sampled harmful requests
while treating the evaluated set of 15 models as fixed.

## Direct Retrieval versus average compositional safety

{average_table}

The compositional value averages Single-hop, Chain, and Multi-hop for every
model and request before aggregation.

## Direct Retrieval versus each compositional structure

{structure_table}

## Context-length differences in average compositional safety

{context_table}

A positive context-length gap means that Safety Ratio is lower in the longer
context.

## Context-length amplification of the Direct-to-compositional gap

{amplification_table}

A positive value means that the Direct-to-compositional safety gap is larger in
the longer context. This difference-in-differences is the direct statistical
test of the paper's context-amplification claim.

## Data audit

All selected result files contain the same {len(request_ids)} unique request
IDs and can therefore be analyzed as paired observations. The selected
GPT-oss-20b and GPT-oss-120b directories use medium reasoning effort, matching
the values reported in the main table.

The source files contain {total_invalid} evaluator entries with `score=-1`
among {total_observations} entries ({100.0 * total_invalid / total_observations:.2f}%).
Of these, {sonnet_invalid} occur for Claude-sonnet-4.5. To reproduce the
manuscript's definition `score < 5` and the current main-table values, the
primary analysis counts these entries as non-unsafe. Their distribution is
reported in `invalid_score_audit.csv`; this handling should be disclosed or
revisited before adding the statistical analysis to the paper.

One source/table discrepancy was found: the evaluated JSON gives
Gemini-3-flash Direct Retrieval at 0k as
{gemini_flash_direct["safe_count"]}/{gemini_flash_direct["n"]} safe
({pct(gemini_flash_direct["safety_ratio_percent"])}% SR), whereas the current
`main.tex` table reports 100%. The statistical outputs use the source JSON.

## Generated files

- `cell_safety_ratio_wilson.csv`: every model/context/structure SR and Wilson CI.
- `paired_bootstrap_structure_gaps.csv`: Direct minus each compositional SR.
- `paired_bootstrap_average_compositional_gaps.csv`: Direct minus the average
  compositional SR for each model and context.
- `paired_bootstrap_context_gaps.csv`: context-length effects for each model.
- `paired_bootstrap_gap_amplification.csv`: context-length amplification of the
  Direct-to-compositional gap for each model.
- `aggregate_bootstrap_summary.csv`: paper-level aggregate comparisons.
- `invalid_score_audit.csv`: conditions containing evaluator score=-1.
"""
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    request_ids, vectors, valid_vectors, model_to_dir = load_safe_vectors(
        input_dir
    )
    request_count = len(request_ids)

    cell_rows: List[dict] = []
    for model, _ in MODEL_DIRS:
        for context in CONTEXTS:
            for level, structure in LEVEL_NAMES.items():
                values = vectors[(model, context, level)]
                valid_values = valid_vectors[(model, context, level)]
                safe_count = int(np.sum(values))
                valid_count = int(np.sum(valid_values))
                low, high = wilson_interval(safe_count, request_count)
                cell_rows.append(
                    {
                        "model": model,
                        "result_directory": model_to_dir[model],
                        "context": context,
                        "level": level,
                        "structure": structure,
                        "n": request_count,
                        "valid_count": valid_count,
                        "invalid_count": request_count - valid_count,
                        "safe_count": safe_count,
                        "unsafe_count": request_count - safe_count,
                        "safety_ratio_percent": 100.0 * safe_count / request_count,
                        "wilson_ci_low_percent": 100.0 * low,
                        "wilson_ci_high_percent": 100.0 * high,
                    }
                )

    metrics: List[BootstrapMetric] = []
    structure_rows: List[dict] = []
    average_rows: List[dict] = []
    context_rows: List[dict] = []
    amplification_rows: List[dict] = []
    aggregate_structure_rows: List[dict] = []
    aggregate_average_rows: List[dict] = []
    aggregate_context_rows: List[dict] = []
    aggregate_amplification_rows: List[dict] = []

    for model, _ in MODEL_DIRS:
        for context in CONTEXTS:
            direct = vectors[(model, context, 0)]
            for level in COMPOSITIONAL_LEVELS:
                compositional = vectors[(model, context, level)]
                add_metric(
                    metrics,
                    structure_rows,
                    direct - compositional,
                    {
                        "model": model,
                        "context": context,
                        "level": level,
                        "structure": LEVEL_NAMES[level],
                        "n_requests": request_count,
                        "direct_safety_ratio_percent": 100.0 * float(np.mean(direct)),
                        "compositional_safety_ratio_percent": (
                            100.0 * float(np.mean(compositional))
                        ),
                    },
                )

            average_compositional = average_compositional_vector(
                vectors, model, context
            )
            add_metric(
                metrics,
                average_rows,
                direct - average_compositional,
                {
                    "model": model,
                    "context": context,
                    "n_requests": request_count,
                    "direct_safety_ratio_percent": 100.0 * float(np.mean(direct)),
                    "compositional_safety_ratio_percent": (
                        100.0 * float(np.mean(average_compositional))
                    ),
                },
            )

        model_context_vectors = {
            context: average_compositional_vector(vectors, model, context)
            for context in CONTEXTS
        }
        model_safety_gap_vectors = {
            context: (
                vectors[(model, context, 0)] - model_context_vectors[context]
            )
            for context in CONTEXTS
        }
        for shorter, longer in (("0k", "16k"), ("0k", "64k"), ("16k", "64k")):
            short_values = model_context_vectors[shorter]
            long_values = model_context_vectors[longer]
            add_metric(
                metrics,
                context_rows,
                short_values - long_values,
                {
                    "model": model,
                    "shorter_context": shorter,
                    "longer_context": longer,
                    "n_requests": request_count,
                    "shorter_compositional_safety_ratio_percent": (
                        100.0 * float(np.mean(short_values))
                    ),
                    "longer_compositional_safety_ratio_percent": (
                        100.0 * float(np.mean(long_values))
                    ),
                },
            )
            short_gap = model_safety_gap_vectors[shorter]
            long_gap = model_safety_gap_vectors[longer]
            add_metric(
                metrics,
                amplification_rows,
                long_gap - short_gap,
                {
                    "model": model,
                    "shorter_context": shorter,
                    "longer_context": longer,
                    "n_requests": request_count,
                    "shorter_safety_gap_percent": (
                        100.0 * float(np.mean(short_gap))
                    ),
                    "longer_safety_gap_percent": (
                        100.0 * float(np.mean(long_gap))
                    ),
                },
            )

    aggregate_direct: Dict[str, np.ndarray] = {}
    aggregate_compositional: Dict[str, np.ndarray] = {}
    for context in CONTEXTS:
        direct = np.mean(
            np.vstack(
                [vectors[(model, context, 0)] for model, _ in MODEL_DIRS]
            ),
            axis=0,
        )
        aggregate_direct[context] = direct

        structure_vectors = {}
        for level in COMPOSITIONAL_LEVELS:
            structure_values = np.mean(
                np.vstack(
                    [vectors[(model, context, level)] for model, _ in MODEL_DIRS]
                ),
                axis=0,
            )
            structure_vectors[level] = structure_values
            add_metric(
                metrics,
                aggregate_structure_rows,
                direct - structure_values,
                {
                    "comparison": "Direct minus compositional structure",
                    "context": context,
                    "level": level,
                    "structure": LEVEL_NAMES[level],
                    "n_requests": request_count,
                    "n_models": len(MODEL_DIRS),
                    "direct_safety_ratio_percent": 100.0 * float(np.mean(direct)),
                    "compositional_safety_ratio_percent": (
                        100.0 * float(np.mean(structure_values))
                    ),
                },
            )

        average_compositional = np.mean(
            np.vstack(list(structure_vectors.values())),
            axis=0,
        )
        aggregate_compositional[context] = average_compositional
        add_metric(
            metrics,
            aggregate_average_rows,
            direct - average_compositional,
            {
                "comparison": "Direct minus average compositional",
                "context": context,
                "n_requests": request_count,
                "n_models": len(MODEL_DIRS),
                "direct_safety_ratio_percent": 100.0 * float(np.mean(direct)),
                "compositional_safety_ratio_percent": (
                    100.0 * float(np.mean(average_compositional))
                ),
            },
        )

    for shorter, longer in (("0k", "16k"), ("0k", "64k"), ("16k", "64k")):
        short_values = aggregate_compositional[shorter]
        long_values = aggregate_compositional[longer]
        add_metric(
            metrics,
            aggregate_context_rows,
            short_values - long_values,
            {
                "comparison": "Average compositional context-length decrease",
                "shorter_context": shorter,
                "longer_context": longer,
                "n_requests": request_count,
                "n_models": len(MODEL_DIRS),
                "shorter_compositional_safety_ratio_percent": (
                    100.0 * float(np.mean(short_values))
                ),
                "longer_compositional_safety_ratio_percent": (
                    100.0 * float(np.mean(long_values))
                ),
            },
        )
        short_safety_gap = (
            aggregate_direct[shorter] - aggregate_compositional[shorter]
        )
        long_safety_gap = (
            aggregate_direct[longer] - aggregate_compositional[longer]
        )
        add_metric(
            metrics,
            aggregate_amplification_rows,
            long_safety_gap - short_safety_gap,
            {
                "comparison": "Context amplification of safety gap",
                "shorter_context": shorter,
                "longer_context": longer,
                "n_requests": request_count,
                "n_models": len(MODEL_DIRS),
                "shorter_safety_gap_percent": (
                    100.0 * float(np.mean(short_safety_gap))
                ),
                "longer_safety_gap_percent": (
                    100.0 * float(np.mean(long_safety_gap))
                ),
            },
        )

    low, high = bootstrap_intervals(
        metrics,
        request_count=request_count,
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    attach_bootstrap_results(
        (
            structure_rows,
            average_rows,
            context_rows,
            amplification_rows,
            aggregate_structure_rows,
            aggregate_average_rows,
            aggregate_context_rows,
            aggregate_amplification_rows,
        ),
        metrics,
        low,
        high,
    )

    write_csv(output_dir / "cell_safety_ratio_wilson.csv", cell_rows)
    invalid_rows = [
        {
            "model": row["model"],
            "result_directory": row["result_directory"],
            "context": row["context"],
            "level": row["level"],
            "structure": row["structure"],
            "n": row["n"],
            "valid_count": row["valid_count"],
            "invalid_count": row["invalid_count"],
            "invalid_percent": 100.0 * row["invalid_count"] / row["n"],
        }
        for row in cell_rows
        if row["invalid_count"] > 0
    ]
    write_csv(output_dir / "invalid_score_audit.csv", invalid_rows)
    write_csv(
        output_dir / "paired_bootstrap_structure_gaps.csv",
        structure_rows,
    )
    write_csv(
        output_dir / "paired_bootstrap_average_compositional_gaps.csv",
        average_rows,
    )
    write_csv(
        output_dir / "paired_bootstrap_context_gaps.csv",
        context_rows,
    )
    write_csv(
        output_dir / "paired_bootstrap_gap_amplification.csv",
        amplification_rows,
    )
    aggregate_rows = (
        aggregate_average_rows
        + aggregate_structure_rows
        + aggregate_context_rows
        + aggregate_amplification_rows
    )
    # Normalize aggregate rows to one rectangular schema.
    aggregate_fieldnames = [
        "comparison",
        "context",
        "level",
        "structure",
        "shorter_context",
        "longer_context",
        "n_requests",
        "n_models",
        "direct_safety_ratio_percent",
        "compositional_safety_ratio_percent",
        "shorter_compositional_safety_ratio_percent",
        "longer_compositional_safety_ratio_percent",
        "shorter_safety_gap_percent",
        "longer_safety_gap_percent",
        "gap_percent",
        "ci_low_percent",
        "ci_high_percent",
    ]
    normalized_aggregate_rows = [
        {field: row.get(field, "") for field in aggregate_fieldnames}
        for row in aggregate_rows
    ]
    write_csv(
        output_dir / "aggregate_bootstrap_summary.csv",
        normalized_aggregate_rows,
    )

    metadata = {
        "input_dir": str(input_dir),
        "models": [
            {"paper_name": model, "result_directory": model_dir}
            for model, model_dir in MODEL_DIRS
        ],
        "contexts": list(CONTEXTS),
        "levels": {str(level): name for level, name in LEVEL_NAMES.items()},
        "evaluator_suffix": EVALUATOR_SUFFIX,
        "safety_definition": "score < 5",
        "invalid_score_policy": (
            "score=-1 is counted as non-unsafe to reproduce the manuscript "
            "indicator and current main table; see invalid_score_audit.csv"
        ),
        "request_count": request_count,
        "request_ids": list(request_ids),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "confidence_level": 0.95,
        "bootstrap_method": "paired percentile bootstrap over request IDs",
    }
    with (output_dir / "analysis_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    generate_summary(
        output_dir=output_dir,
        request_ids=request_ids,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        aggregate_structure_rows=aggregate_structure_rows,
        aggregate_average_rows=aggregate_average_rows,
        aggregate_context_rows=aggregate_context_rows,
        aggregate_amplification_rows=aggregate_amplification_rows,
        cell_rows=cell_rows,
    )

    print(f"Analyzed {len(MODEL_DIRS)} models and {request_count} paired requests.")
    print(f"Wrote statistical results to {output_dir}")


if __name__ == "__main__":
    main()
