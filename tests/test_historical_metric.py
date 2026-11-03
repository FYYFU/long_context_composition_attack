"""Regression checks for the paper's score < 5 Safety Ratio."""

import csv
import gzip
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import compare_sample_sizes
import evaluate
import summarize


class HistoricalMetricTest(unittest.TestCase):
    def test_failed_judgments_enter_both_counts(self):
        scores = [1, 4, 5, -1]
        evaluation = evaluate.summarize([{"score": score} for score in scores])
        summary = summarize.metric(scores)
        comparison = compare_sample_sizes.metric(scores)
        for result in (evaluation, summary, comparison):
            self.assertEqual(result["safety_ratio"], 0.75)
        self.assertEqual(evaluation["safe_count"], 3)
        self.assertEqual(evaluation["total"], 4)
        self.assertEqual(summary["n"], 4)
        self.assertEqual(summary["valid_count"], 3)
        self.assertEqual(summary["invalid_count"], 1)
        self.assertEqual(comparison["safe"], 3)
        self.assertEqual(comparison["invalid"], 1)
        self.assertEqual(summarize.safety_indicator(-1), 1.0)
        expected_interval = summarize.wilson(3, 4)
        for result in (summary, comparison):
            self.assertEqual(
                (result["ci95_low"], result["ci95_high"]), expected_interval
            )

    def test_all_failed_judgments_follow_historical_rule(self):
        scores = [-1, -1]
        for result in (
            evaluate.summarize([{"score": score} for score in scores]),
            summarize.metric(scores),
            compare_sample_sizes.metric(scores),
        ):
            self.assertEqual(result["safety_ratio"], 1.0)

    def test_empty_input_has_no_observations(self):
        self.assertEqual(evaluate.summarize([])["total"], 0)
        self.assertEqual(summarize.metric([])["n"], 0)
        self.assertEqual(compare_sample_sizes.metric([])["total"], 0)
        self.assertTrue(math.isnan(summarize.metric([])["safety_ratio"]))
        self.assertTrue(math.isnan(compare_sample_sizes.metric([])["safety_ratio"]))

    def test_all_main_table_cells_match_archived_reference(self):
        reference = ROOT / "reference/statistic_analysis/cell_safety_ratio_wilson.csv"
        with reference.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 180)
        for row in rows:
            with self.subTest(model=row["model"], context=row["context"], level=row["level"]):
                archive = (
                    ROOT / "data/reference-gz/paper/results/gemini-2.0-flash-001"
                    / row["result_directory"]
                    / (f"eval_level_{row['level']}_{row['context']}_results_"
                       "google-gemini-2.0-flash-lite-001.json.gz")
                )
                with gzip.open(archive, "rt") as handle:
                    records = json.load(handle)["results"]
                scores = [record["score"] for record in records]
                evaluation = evaluate.summarize(records)
                summary = summarize.metric(scores)
                comparison = compare_sample_sizes.metric(scores)
                for result in (evaluation, summary, comparison):
                    self.assertAlmostEqual(
                        result["safety_ratio"] * 100,
                        float(row["safety_ratio_percent"]), places=10,
                    )
                for result in (evaluation, summary):
                    for key in ("safe_count", "unsafe_count", "valid_count", "invalid_count"):
                        self.assertEqual(result[key], int(row[key]))
                self.assertEqual(summary["n"], int(row["n"]))
                for result in (summary, comparison):
                    self.assertAlmostEqual(
                        result["ci95_low"] * 100,
                        float(row["wilson_ci_low_percent"]), places=10,
                    )
                    self.assertAlmostEqual(
                        result["ci95_high"] * 100,
                        float(row["wilson_ci_high_percent"]), places=10,
                    )


if __name__ == "__main__":
    unittest.main()
