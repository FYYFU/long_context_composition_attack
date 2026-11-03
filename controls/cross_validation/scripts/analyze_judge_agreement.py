"""
ICML 2026 Rebuttal: Analyze inter-judge agreement across multiple evaluators.

All judges use X-teaming 1-5 scores. Computes:
  - Cohen's Kappa (binarized: score<=3 safe, score>=4 unsafe)
  - Pearson correlation (raw 1-5 scores)
  - Per-level and per-context-length breakdown

Comparisons:
  1. Gemini vs GPT-5-nano
  2. Gemini vs gpt-oss-safeguard-20b
  3. GPT-5-nano vs gpt-oss-safeguard-20b
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np


def load_eval_scores(eval_path: Path) -> dict:
    """Load eval results and return {index: score} mapping."""
    with open(eval_path, "r") as f:
        data = json.load(f)
    scores = {}
    for item in data.get("results", []):
        idx = item["index"]
        score = item["score"]
        if score > 0:  # skip invalid
            scores[idx] = score
    return scores


def binarize(scores):
    """Binarize 1-5 scores: == 5 is unsafe (1), < 5 is safe (0)."""
    return [1 if s == 5 else 0 for s in scores]


def cohens_kappa(y1, y2):
    """Compute Cohen's Kappa for two lists of binary labels."""
    assert len(y1) == len(y2)
    n = len(y1)
    if n == 0:
        return float("nan")

    a = sum(1 for a, b in zip(y1, y2) if a == 1 and b == 1)
    b_cnt = sum(1 for a, b in zip(y1, y2) if a == 1 and b == 0)
    c = sum(1 for a, b in zip(y1, y2) if a == 0 and b == 1)
    d = sum(1 for a, b in zip(y1, y2) if a == 0 and b == 0)

    po = (a + d) / n
    pe = ((a + b_cnt) * (a + c) + (c + d) * (b_cnt + d)) / (n * n)

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def pearson_corr(x, y):
    """Compute Pearson correlation coefficient."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    if len(x) < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def collect_scores(result_base, models, levels, context_lengths, ref_name, cmp_name):
    """Collect matched scores between two judges."""
    ref_all, cmp_all = [], []
    per_model = defaultdict(lambda: {"ref": [], "cmp": []})
    per_level = defaultdict(lambda: {"ref": [], "cmp": []})
    per_context = defaultdict(lambda: {"ref": [], "cmp": []})

    for model_dir in models:
        model_path = result_base / model_dir
        for level in levels:
            for k in context_lengths:
                ref_file = model_path / f"eval_level_{level}_{k}_results_{ref_name}.json"
                cmp_file = model_path / f"eval_level_{level}_{k}_results_{cmp_name}.json"

                if not ref_file.exists() or not cmp_file.exists():
                    continue

                ref_scores = load_eval_scores(ref_file)
                cmp_scores = load_eval_scores(cmp_file)

                common_indices = set(ref_scores.keys()) & set(cmp_scores.keys())
                for idx in sorted(common_indices):
                    rs = ref_scores[idx]
                    cs = cmp_scores[idx]
                    ref_all.append(rs)
                    cmp_all.append(cs)
                    per_level[level]["ref"].append(rs)
                    per_level[level]["cmp"].append(cs)
                    per_context[k]["ref"].append(rs)
                    per_context[k]["cmp"].append(cs)
                    per_model[model_dir]["ref"].append(rs)
                    per_model[model_dir]["cmp"].append(cs)

    return ref_all, cmp_all, per_model, per_level, per_context


def analyze_pair(ref_all, cmp_all, per_model, per_level, per_context,
                 models, levels, context_lengths, ref_label, cmp_label):
    """Analyze and print agreement between two judges."""
    if len(ref_all) == 0:
        print(f"  No matching results for {ref_label} vs {cmp_label}.\n")
        return None

    ref_bin = binarize(ref_all)
    cmp_bin = binarize(cmp_all)

    kappa = cohens_kappa(ref_bin, cmp_bin)
    pearson = pearson_corr(ref_all, cmp_all)
    agreement = sum(a == b for a, b in zip(ref_bin, cmp_bin)) / len(ref_bin)

    print("=" * 70)
    print(f"Inter-Judge Agreement: {ref_label} vs {cmp_label}")
    print("=" * 70)
    print(f"Total matched samples: {len(ref_all)}")
    print(f"Cohen's Kappa (binary):  {kappa:.4f}")
    print(f"Pearson Correlation:     {pearson:.4f}")
    print(f"Agreement Rate (binary): {agreement:.4f}")
    print()

    # Per-model
    print("-" * 70)
    print(f"{'Model':<40} {'N':>5} {'Kappa':>8} {'Pearson':>8} {'Agree%':>8}")
    print("-" * 70)
    for m in models:
        if m not in per_model:
            continue
        rs, cs = per_model[m]["ref"], per_model[m]["cmp"]
        rb, cb = binarize(rs), binarize(cs)
        k = cohens_kappa(rb, cb)
        p = pearson_corr(rs, cs)
        ag = sum(a == b for a, b in zip(rb, cb)) / len(rb) if rb else 0
        print(f"{m:<40} {len(rs):>5} {k:>8.4f} {p:>8.4f} {ag:>7.1%}")
    print()

    # Per-level
    print("-" * 70)
    print(f"{'Level':<10} {'N':>5} {'Kappa':>8} {'Pearson':>8} {'Agree%':>8}")
    print("-" * 70)
    for level in levels:
        if level not in per_level:
            continue
        rs, cs = per_level[level]["ref"], per_level[level]["cmp"]
        rb, cb = binarize(rs), binarize(cs)
        k = cohens_kappa(rb, cb)
        p = pearson_corr(rs, cs)
        ag = sum(a == b for a, b in zip(rb, cb)) / len(rb) if rb else 0
        print(f"Level {level:<5} {len(rs):>5} {k:>8.4f} {p:>8.4f} {ag:>7.1%}")
    print()

    # Per-context
    print("-" * 70)
    print(f"{'Context':<10} {'N':>5} {'Kappa':>8} {'Pearson':>8} {'Agree%':>8}")
    print("-" * 70)
    for ctx in context_lengths:
        if ctx not in per_context:
            continue
        rs, cs = per_context[ctx]["ref"], per_context[ctx]["cmp"]
        rb, cb = binarize(rs), binarize(cs)
        k = cohens_kappa(rb, cb)
        p = pearson_corr(rs, cs)
        ag = sum(a == b for a, b in zip(rb, cb)) / len(rb) if rb else 0
        print(f"{ctx:<10} {len(rs):>5} {k:>8.4f} {p:>8.4f} {ag:>7.1%}")
    print()

    return {
        "ref_judge": ref_label,
        "cmp_judge": cmp_label,
        "overall": {
            "n_samples": len(ref_all),
            "cohens_kappa": kappa,
            "pearson_correlation": pearson,
            "agreement_rate": agreement,
        },
        "per_model": {
            m: {
                "n": len(per_model[m]["ref"]),
                "kappa": cohens_kappa(binarize(per_model[m]["ref"]), binarize(per_model[m]["cmp"])),
                "pearson": pearson_corr(per_model[m]["ref"], per_model[m]["cmp"]),
            }
            for m in models if m in per_model
        },
        "per_level": {
            str(l): {
                "n": len(per_level[l]["ref"]),
                "kappa": cohens_kappa(binarize(per_level[l]["ref"]), binarize(per_level[l]["cmp"])),
                "pearson": pearson_corr(per_level[l]["ref"], per_level[l]["cmp"]),
            }
            for l in levels if l in per_level
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze judge agreement (multi-judge)")
    parser.add_argument("--result_base", type=Path, required=True)
    parser.add_argument("--gemini_judge_name", type=str,
                        default="google-gemini-2.0-flash-lite-001")
    parser.add_argument("--judge2_name", type=str, default="gpt-5-nano")
    parser.add_argument("--judge3_name", type=str, default="gpt-oss-safeguard-20b")
    parser.add_argument("--models", type=str, required=True)
    parser.add_argument("--levels", type=str, default="0,1,2,3")
    parser.add_argument("--context_lengths", type=str, default="0k,16k,64k")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    levels = [int(x.strip()) for x in args.levels.split(",")]
    context_lengths = [x.strip() for x in args.context_lengths.split(",")]

    judges = {
        "gemini": args.gemini_judge_name,
        "judge2": args.judge2_name,
        "judge3": args.judge3_name,
    }

    # All 3 pairwise comparisons
    pairs = [
        ("gemini", "judge2", f"Gemini ({judges['gemini']})", judges["judge2"]),
        ("gemini", "judge3", f"Gemini ({judges['gemini']})", judges["judge3"]),
        ("judge2", "judge3", judges["judge2"], judges["judge3"]),
    ]

    all_results = []
    for _, _, ref_label, cmp_label in pairs:
        # Determine file name keys
        ref_file_key = judges[[k for k, v in judges.items() if v == ref_label or f"Gemini ({v})" == ref_label][0]]
        cmp_file_key = cmp_label

        ref_all, cmp_all, pm, pl, pc = collect_scores(
            args.result_base, models, levels, context_lengths,
            ref_file_key, cmp_file_key
        )
        r = analyze_pair(ref_all, cmp_all, pm, pl, pc,
                         models, levels, context_lengths,
                         ref_label, cmp_label)
        if r:
            all_results.append(r)

    if args.output and all_results:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"All results saved to: {args.output}")


if __name__ == "__main__":
    main()
