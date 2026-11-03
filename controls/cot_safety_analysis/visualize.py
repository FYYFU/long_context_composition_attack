"""
Visualize CoT Safety Analysis results.
Create figures for the rebuttal.

Usage:
    python visualize.py
"""
from pathlib import Path

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
from collections import defaultdict

matplotlib.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

OUTPUT_DIR = Path(str(Path(__file__).resolve().parents[2] / 'controls/cot_safety_analysis/results'))
FIG_DIR = Path(str(Path(__file__).resolve().parents[2] / 'controls/cot_safety_analysis/figures'))

OUTCOME_COLORS = {
    "SAFE_REFUSAL": "#4CAF50",
    "BENIGN_REFRAMING": "#FF9800",
    "WARNING_COMPLIANCE": "#F44336",
    "LOGICAL_FALLACY": "#9C27B0",
    "HARM_ID_FAILURE": "#607D8B",
    "PARSE_ERROR": "#BDBDBD",
}

OUTCOME_LABELS = {
    "SAFE_REFUSAL": "Safe Refusal",
    "BENIGN_REFRAMING": "Benign Reframing",
    "WARNING_COMPLIANCE": "Warning + Compliance",
    "LOGICAL_FALLACY": "Logical Fallacy",
    "HARM_ID_FAILURE": "Harm ID Failure",
    "PARSE_ERROR": "Parse Error",
}

EFFORT_LABELS = {"low": "Low", "medium": "Medium", "high": "High"}


def load_all_results():
    result_files = sorted(OUTPUT_DIR.glob("cot_decompose_*.json"))
    all_results = []
    for fp in result_files:
        with open(fp) as f:
            all_results.extend(json.load(f))
    return all_results


def fig1_outcome_by_effort(results):
    """Figure 1: Stacked bar chart - Safety outcome distribution by reasoning effort"""
    fig, ax = plt.subplots(figsize=(8, 5))

    efforts = ["low", "medium", "high"]
    outcomes = ["SAFE_REFUSAL", "HARM_ID_FAILURE", "BENIGN_REFRAMING", "WARNING_COMPLIANCE", "LOGICAL_FALLACY"]

    data = {}
    for effort in efforts:
        entries = [r for r in results if r["effort"] == effort]
        total = len(entries)
        if total == 0:
            continue
        data[effort] = {o: sum(1 for r in entries if r["safety_outcome"] == o) / total * 100
                        for o in outcomes}

    x = np.arange(len(efforts))
    width = 0.5
    bottom = np.zeros(len(efforts))

    for outcome in outcomes:
        values = [data.get(e, {}).get(outcome, 0) for e in efforts]
        bars = ax.bar(x, values, width, bottom=bottom,
                      color=OUTCOME_COLORS.get(outcome, "#999"),
                      label=OUTCOME_LABELS.get(outcome, outcome),
                      edgecolor="white", linewidth=0.5)
        # Add numeric labels to sufficiently large regions.
        for i, (v, b) in enumerate(zip(values, bottom)):
            if v > 3:
                ax.text(x[i], b + v / 2, f"{v:.1f}%", ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white")
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels([EFFORT_LABELS[e] for e in efforts])
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("Reasoning Effort")
    ax.set_title("Safety Outcome Distribution by Reasoning Effort\n(gpt-oss-20b)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    ax.set_ylim(0, 105)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_outcome_by_effort.pdf")
    fig.savefig(FIG_DIR / "fig1_outcome_by_effort.png")
    plt.close(fig)
    print("Saved fig1_outcome_by_effort")


def fig2_self_jailbreak_rate(results):
    """Figure 2: Self-jailbreak vs Harm-ID-Failure among unsafe cases"""
    fig, ax = plt.subplots(figsize=(6, 4))

    efforts = ["low", "medium", "high"]
    sj_rates = []
    hf_rates = []

    for effort in efforts:
        unsafe = [r for r in results if r["effort"] == effort and r["safety_outcome"] != "SAFE_REFUSAL"]
        if not unsafe:
            sj_rates.append(0)
            hf_rates.append(0)
            continue
        sj = sum(1 for r in unsafe if r.get("recognized_risk") is True)
        sj_rates.append(sj / len(unsafe) * 100)
        hf_rates.append((len(unsafe) - sj) / len(unsafe) * 100)

    x = np.arange(len(efforts))
    width = 0.35

    ax.bar(x - width / 2, sj_rates, width, color="#FF9800", label="Self-Jailbreak\n(recognized risk, still complied)")
    ax.bar(x + width / 2, hf_rates, width, color="#607D8B", label="Harm ID Failure\n(failed to recognize risk)")

    for i, (s, h) in enumerate(zip(sj_rates, hf_rates)):
        if s > 0:
            ax.text(x[i] - width / 2, s + 1, f"{s:.0f}%", ha="center", fontsize=9)
        if h > 0:
            ax.text(x[i] + width / 2, h + 1, f"{h:.0f}%", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([EFFORT_LABELS[e] for e in efforts])
    ax.set_ylabel("Percentage of Unsafe Cases (%)")
    ax.set_xlabel("Reasoning Effort")
    ax.set_title("Failure Mode Among Unsafe Outputs")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_self_jailbreak_rate.pdf")
    fig.savefig(FIG_DIR / "fig2_self_jailbreak_rate.png")
    plt.close(fig)
    print("Saved fig2_self_jailbreak_rate")


def fig3_trigger_position(results):
    """Figure 3: Safety trigger position distribution by effort"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    efforts = ["low", "medium", "high"]
    positions = ["early", "middle", "late", "none"]
    pos_colors = {"early": "#4CAF50", "middle": "#2196F3", "late": "#FF9800", "none": "#F44336"}

    for ax, effort in zip(axes, efforts):
        entries = [r for r in results if r["effort"] == effort]
        if not entries:
            continue
        total = len(entries)
        counts = [sum(1 for r in entries if r.get("safety_trigger_position") == p) for p in positions]
        pcts = [c / total * 100 for c in counts]

        bars = ax.bar(positions, pcts, color=[pos_colors[p] for p in positions], edgecolor="white")
        for bar, pct in zip(bars, pcts):
            if pct > 2:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{pct:.0f}%", ha="center", fontsize=9)

        ax.set_title(f"{EFFORT_LABELS[effort]} Effort")
        ax.set_ylabel("Percentage (%)" if effort == "low" else "")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Position of First Safety Trigger in Thinking Trace", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_trigger_position.pdf")
    fig.savefig(FIG_DIR / "fig3_trigger_position.png")
    plt.close(fig)
    print("Saved fig3_trigger_position")


def fig4_outcome_by_level(results):
    """Figure 4: Safe refusal rate by attack level, grouped by effort"""
    fig, ax = plt.subplots(figsize=(7, 5))

    efforts = ["low", "medium", "high"]
    levels = sorted({r["level"] for r in results})
    effort_colors = {"low": "#F44336", "medium": "#FF9800", "high": "#4CAF50"}

    for effort in efforts:
        safe_rates = []
        for level in levels:
            entries = [r for r in results if r["effort"] == effort and r["level"] == level]
            if entries:
                safe_rates.append(sum(1 for r in entries if r["safety_outcome"] == "SAFE_REFUSAL") / len(entries) * 100)
            else:
                safe_rates.append(0)
        ax.plot(levels, safe_rates, "o-", color=effort_colors[effort],
                label=f"{EFFORT_LABELS[effort]} Effort", linewidth=2, markersize=8)

    ax.set_xlabel("Attack Level (Reasoning Complexity)")
    ax.set_ylabel("Safe Refusal Rate (%)")
    ax.set_title("Safe Refusal Rate by Attack Level")
    ax.set_xticks(levels)
    ax.set_xticklabels([f"L{l}" for l in levels])
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 105)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_outcome_by_level.pdf")
    fig.savefig(FIG_DIR / "fig4_outcome_by_level.png")
    plt.close(fig)
    print("Saved fig4_outcome_by_level")


def fig5_thinking_length_vs_outcome(results):
    """Figure 5: Box plot of thinking length by safety outcome"""
    fig, ax = plt.subplots(figsize=(8, 5))

    outcomes = ["SAFE_REFUSAL", "BENIGN_REFRAMING", "WARNING_COMPLIANCE", "HARM_ID_FAILURE"]
    data = []
    labels = []
    colors = []

    for outcome in outcomes:
        entries = [r for r in results if r["safety_outcome"] == outcome]
        if entries:
            data.append([r["thinking_length"] for r in entries])
            labels.append(OUTCOME_LABELS.get(outcome, outcome))
            colors.append(OUTCOME_COLORS.get(outcome, "#999"))

    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Thinking Trace Length (chars)")
    ax.set_title("Thinking Trace Length by Safety Outcome")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=15)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_thinking_length_vs_outcome.pdf")
    fig.savefig(FIG_DIR / "fig5_thinking_length_vs_outcome.png")
    plt.close(fig)
    print("Saved fig5_thinking_length_vs_outcome")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    results = load_all_results()
    if not results:
        print("No results found. Run analyze_cot.py decompose first.")
        return

    print(f"Loaded {len(results)} entries")

    fig1_outcome_by_effort(results)
    fig2_self_jailbreak_rate(results)
    fig3_trigger_position(results)
    fig4_outcome_by_level(results)
    fig5_thinking_length_vs_outcome(results)

    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
