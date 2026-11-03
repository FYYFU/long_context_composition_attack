"""
Analyze fragment distribution experiment results.
Compare three conditions:
  - Adjacent: Run in this experiment.
  - Concentrated: Reuse the 25% windows from Needle_Position, averaging across 4 positions.
  - Dispersed: Reuse the main experiment baseline data.
"""
import json
import os
import glob
from collections import defaultdict
from fire import Fire


def load_results(model_dir):
    """Load safe/unsafe counts for all level x context_len combinations from one model's results directory."""
    stats = defaultdict(lambda: defaultdict(lambda: {"safe": 0, "unsafe": 0, "total": 0}))
    result_files = glob.glob(os.path.join(model_dir, "level_*_results.json"))
    for rf in result_files:
        basename = os.path.basename(rf)
        parts = basename.replace("_results.json", "").split("_")
        level = int(parts[1])
        ctx_len = parts[2]
        with open(rf, "r", encoding="utf-8") as f:
            data = json.load(f)
        safe = sum(1 for item in data if item["generated"]["safe"])
        total = len(data)
        stats[level][ctx_len] = {"safe": safe, "unsafe": total - safe, "total": total}
    return stats


def load_needle_position_avg(needle_pos_root, model_name):
    """
    Average results across 4 Needle_Position locations for the concentrated condition.
    """
    positions = ["position_start", "position_early_mid", "position_late_mid", "position_end"]
    # level -> ctx_len -> list of ASR values
    asr_lists = defaultdict(lambda: defaultdict(list))
    count_lists = defaultdict(lambda: defaultdict(lambda: {"safe": 0, "unsafe": 0, "total": 0}))

    for pos in positions:
        model_dir = os.path.join(needle_pos_root, pos, model_name)
        if not os.path.isdir(model_dir):
            print(f"[跳过] {model_dir}")
            continue
        pos_stats = load_results(model_dir)
        for level in pos_stats:
            for ctx in pos_stats[level]:
                s = pos_stats[level][ctx]
                count_lists[level][ctx]["safe"] += s["safe"]
                count_lists[level][ctx]["unsafe"] += s["unsafe"]
                count_lists[level][ctx]["total"] += s["total"]

    return count_lists


def print_table(label, stats, ctx_order):
    """Print ASR table rows for one condition."""
    row = f"{label:<35}"
    asr_values = []
    for c in ctx_order:
        s = stats.get(c, {"total": 0})
        if s["total"] > 0:
            asr = s["unsafe"] / s["total"]
            asr_values.append(asr)
            row += f"{asr*100:>7.1f}%"
        else:
            row += f"{'N/A':>8}"
    if asr_values:
        row += f"  {sum(asr_values)/len(asr_values)*100:>5.1f}%"
    print(row)


def analyze(
    adjacent_results: str = "./results/dist_adjacent",
    needle_pos_root: str = "../Needle_Position/results",
    dispersed_results: str = None,
    model_name: str = "gpt-oss-120b-medium",
    output_file: str = None,
):
    """
    Comparison across three conditions.

    Args:
        adjacent_results: Results directory for the adjacent condition.
        needle_pos_root: Root results directory for the Needle_Position experiment.
        dispersed_results: Main experiment baseline results directory (dispersed); skip if None.
        model_name: Model subdirectory name.
        output_file: Optional path for saving the summary JSON.
    """
    ctx_order = ["0k", "1k", "4k", "8k", "16k", "32k", "64k"]

    # 1. Adjacent
    adj_dir = os.path.join(adjacent_results, model_name)
    adj_stats = load_results(adj_dir) if os.path.isdir(adj_dir) else {}

    # 2. Concentrated: Average across 4 Needle_Position locations.
    conc_stats = load_needle_position_avg(needle_pos_root, model_name)

    # 3. Dispersed: Main experiment baseline.
    disp_stats = {}
    if dispersed_results:
        disp_dir = os.path.join(dispersed_results, model_name) if model_name else dispersed_results
        if os.path.isdir(disp_dir):
            disp_stats = load_results(disp_dir)

    all_levels = sorted(set(list(adj_stats.keys()) + list(conc_stats.keys()) + list(disp_stats.keys())))

    for level in all_levels:
        print(f"\n{'='*90}")
        print(f"Level {level} (n_facts={level+1})")
        print(f"{'='*90}")
        header = f"{'Distribution':<35}" + "".join(f"{'  '+c+' ':>8}" for c in ctx_order) + "    Avg"
        print(header)
        print("-" * len(header))

        if level in adj_stats:
            print_table("Adjacent (all together)", adj_stats[level], ctx_order)
        if level in conc_stats:
            print_table("Concentrated (25% window)", conc_stats[level], ctx_order)
        if level in disp_stats:
            print_table("Dispersed (100% random)", disp_stats[level], ctx_order)

    # Overall (averaged across levels)
    print(f"\n{'='*90}")
    print("Overall ASR (averaged across levels)")
    print(f"{'='*90}")
    header = f"{'Distribution':<35}" + "".join(f"{'  '+c+' ':>8}" for c in ctx_order) + "    Avg"
    print(header)
    print("-" * len(header))

    for label, stats_dict in [
        ("Adjacent (all together)", adj_stats),
        ("Concentrated (25% window)", conc_stats),
        ("Dispersed (100% random)", disp_stats),
    ]:
        if not stats_dict:
            continue
        row = f"{label:<35}"
        overall = []
        for c in ctx_order:
            vals = []
            for level in stats_dict:
                s = stats_dict[level].get(c, {"total": 0})
                if s["total"] > 0:
                    vals.append(s["unsafe"] / s["total"])
            if vals:
                avg = sum(vals) / len(vals)
                overall.append(avg)
                row += f"{avg*100:>7.1f}%"
            else:
                row += f"{'N/A':>8}"
        if overall:
            row += f"  {sum(overall)/len(overall)*100:>5.1f}%"
        print(row)

    # Save JSON.
    if output_file:
        summary = {"adjacent": {}, "concentrated": {}, "dispersed": {}}
        for label_key, stats_dict in [("adjacent", adj_stats), ("concentrated", conc_stats), ("dispersed", disp_stats)]:
            for level in stats_dict:
                summary[label_key][str(level)] = {}
                for ctx in stats_dict[level]:
                    s = stats_dict[level][ctx]
                    summary[label_key][str(level)][ctx] = {
                        "asr": s["unsafe"] / s["total"] if s["total"] > 0 else None,
                        **s,
                    }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n汇总已保存到: {output_file}")


if __name__ == "__main__":
    Fire(analyze)
