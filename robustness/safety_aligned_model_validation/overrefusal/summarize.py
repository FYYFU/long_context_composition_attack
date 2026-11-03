#!/usr/bin/env python3
"""Summarize XSTest over-refusal results for the six checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", default=None)
    parser.add_argument("--output-prefix", default="xstest_overrefusal")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models: List[Dict[str, str]] = json.loads(
        (PARENT / "models.json").read_text(encoding="utf-8")
    )["models"]
    if args.family:
        models = [
            model
            for model in models
            if model["family"] == args.family
        ]
    rows: List[Dict[str, Any]] = []
    for model in models:
        path = ROOT / "results" / model["alias"] / "xstest_eval.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload["summary"]
        rows.append(
            {
                "alias": model["alias"],
                "family": model["family"],
                "alignment": model["alignment"],
                "compliance_rate": summary["compliance_rate"],
                "over_refusal_rate": summary["over_refusal_rate"],
                "valid": summary["valid"],
            }
        )

    base_rates = {
        row["family"]: row["over_refusal_rate"]
        for row in rows
        if row["alignment"] == "original"
    }
    for row in rows:
        row["delta_vs_base"] = (
            row["over_refusal_rate"] - base_rates[row["family"]]
        )

    summary_dir = ROOT / "results" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "alias",
        "family",
        "alignment",
        "compliance_rate",
        "over_refusal_rate",
        "delta_vs_base",
        "valid",
    ]
    with (summary_dir / f"{args.output_prefix}.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    markdown = [
        "| Model | Compliance | Over-refusal | Delta vs. base | Valid |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['alias']} | {row['compliance_rate']:.1f} | "
            f"{row['over_refusal_rate']:.1f} | "
            f"{row['delta_vs_base']:+.1f} | {row['valid']} |"
        )
    text = "\n".join(markdown) + "\n"
    (summary_dir / f"{args.output_prefix}.md").write_text(
        text,
        encoding="utf-8",
    )
    print(text)


if __name__ == "__main__":
    main()
