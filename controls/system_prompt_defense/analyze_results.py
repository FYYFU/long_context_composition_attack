"""
Analyze and compare defense results against baseline (no system prompt).

Produces a comparison table showing safety ratio with and without defense.
"""
import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Optional
from collections import defaultdict

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "paper"))


def load_eval_results(eval_file: Path) -> Optional[Dict]:
    """Load evaluation results and extract safety ratio."""
    if not eval_file.exists():
        return None
    with open(eval_file) as f:
        data = json.load(f)

    stats = data.get("statistics", {})
    dist = stats.get("score_distribution", {})
    valid = stats.get("valid_count", 0)
    if valid == 0:
        return None

    # Score 5 = attack success; safety ratio = 1 - (score5 / valid)
    score5 = dist.get("5", dist.get(5, 0))
    safety_ratio = (valid - score5) / valid * 100
    return {
        "safety_ratio": safety_ratio,
        "avg_score": stats.get("average_score", 0),
        "valid_count": valid,
        "score_distribution": dist,
    }


def analyze(
    baseline_dir: str,
    defense_dir: str,
    judge_name: str = "google-gemini-2.0-flash-lite-001",
    levels: str = "0,1,2,3",
    context_lengths: str = "0k,16k,64k",
    output_file: Optional[str] = None,
):
    """
    Compare baseline vs defense results.

    Args:
        baseline_dir: Path to baseline model results (no system prompt)
        defense_dir: Path to defense model results (with system prompt)
        judge_name: Judge name used in eval filenames
        levels: Comma-separated levels
        context_lengths: Comma-separated context lengths
        output_file: Optional JSON output file for structured results
    """
    level_list = [int(x.strip()) for x in levels.split(",")]
    ctx_list = [x.strip() for x in context_lengths.split(",")]

    print("=" * 90)
    print("System Prompt Defense: Baseline vs Defense Comparison")
    print("=" * 90)
    print(f"Baseline: {baseline_dir}")
    print(f"Defense:  {defense_dir}")
    print("=" * 90)

    # Header
    print(f"\n{'':>20} | {'Baseline SR%':>12} | {'Defense SR%':>12} | {'Delta':>8} | {'Baseline Avg':>12} | {'Defense Avg':>12}")
    print("-" * 90)

    all_comparisons = {}
    # For computing averages per context length across Type 1-3
    ctx_deltas = defaultdict(list)

    for level in level_list:
        for ctx in ctx_list:
            base_file = Path(baseline_dir) / f"eval_level_{level}_{ctx}_results_{judge_name}.json"
            def_file = Path(defense_dir) / f"eval_level_{level}_{ctx}_{judge_name}.json"

            base_res = load_eval_results(base_file)
            def_res = load_eval_results(def_file)

            key = f"Level {level} / {ctx}"

            if base_res and def_res:
                delta = def_res["safety_ratio"] - base_res["safety_ratio"]
                print(
                    f"{key:>20} | {base_res['safety_ratio']:>11.1f}% | {def_res['safety_ratio']:>11.1f}% | "
                    f"{delta:>+7.1f}% | {base_res['avg_score']:>11.2f} | {def_res['avg_score']:>11.2f}"
                )
                all_comparisons[f"level_{level}_{ctx}"] = {
                    "baseline_sr": base_res["safety_ratio"],
                    "defense_sr": def_res["safety_ratio"],
                    "delta": delta,
                    "baseline_avg": base_res["avg_score"],
                    "defense_avg": def_res["avg_score"],
                }
                if level >= 1:
                    ctx_deltas[ctx].append(delta)
            elif base_res:
                print(f"{key:>20} | {base_res['safety_ratio']:>11.1f}% | {'N/A':>12} | {'N/A':>8} | {base_res['avg_score']:>11.2f} | {'N/A':>12}")
            elif def_res:
                print(f"{key:>20} | {'N/A':>12} | {def_res['safety_ratio']:>11.1f}% | {'N/A':>8} | {'N/A':>12} | {def_res['avg_score']:>11.2f}")
            else:
                print(f"{key:>20} | {'N/A':>12} | {'N/A':>12} | {'N/A':>8} | {'N/A':>12} | {'N/A':>12}")

    # Summary statistics
    print("\n" + "=" * 90)
    print("Summary (averaged over Type 1-3 reasoning tasks)")
    print("=" * 90)

    for ctx in ctx_list:
        if ctx_deltas[ctx]:
            avg_delta = sum(ctx_deltas[ctx]) / len(ctx_deltas[ctx])
            print(f"  Context {ctx}: avg delta = {avg_delta:+.1f}%")

    all_deltas = [d for deltas in ctx_deltas.values() for d in deltas]
    if all_deltas:
        overall_avg = sum(all_deltas) / len(all_deltas)
        print(f"\n  Overall avg delta (Type 1-3): {overall_avg:+.1f}%")

    # Save structured results
    if output_file:
        with open(output_file, "w") as f:
            json.dump(all_comparisons, f, indent=2)
        print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_dir", type=str, required=True,
                        help="Path to baseline eval results (no system prompt)")
    parser.add_argument("--defense_dir", type=str, required=True,
                        help="Path to defense eval results (with system prompt)")
    parser.add_argument("--judge_name", type=str, default="google-gemini-2.0-flash-lite-001")
    parser.add_argument("--levels", type=str, default="0,1,2,3")
    parser.add_argument("--context_lengths", type=str, default="0k,16k,64k")
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()
    analyze(**vars(args))
