"""
MAD-Graph Results Analysis

Analyzes the CSV output from mad_graph_eval.py and produces:
- Overall accuracy
- Agent agreement rate (how often Phase 1 resolved without debate)
- Optional comparison plots vs. a baseline CSV
"""

import argparse
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["is_correct"] = df["is_correct"].astype(int)
    return df


def compute_selection_frequency(df: pd.DataFrame) -> pd.Series:
    return df["predicted_answer"].value_counts(normalize=True).sort_index()


def analyze_agent_agreement(df: pd.DataFrame, label: str = ""):
    if not all(c in df.columns for c in ["agent_1_ans", "agent_2_ans", "agent_3_ans"]):
        return

    total = len(df)

    unanimous = (
        (df["agent_1_ans"] == df["agent_2_ans"]) &
        (df["agent_2_ans"] == df["agent_3_ans"]) &
        df["agent_1_ans"].isin(["A", "B", "C", "D"])
    ).sum()
    disagreement = total - unanimous

    unanimous_mask = (
        (df["agent_1_ans"] == df["agent_2_ans"]) &
        (df["agent_2_ans"] == df["agent_3_ans"])
    )
    acc_unanimous = df[unanimous_mask]["is_correct"].mean() if unanimous > 0 else 0
    acc_debate = df[~unanimous_mask]["is_correct"].mean() if disagreement > 0 else 0

    # Build the output text
    lines = [
        "Agent Agreement Analysis:",
        f"  All 3 agreed (no debate needed) : {unanimous} ({unanimous/total:.1%})",
        f"  Debate triggered                : {disagreement} ({disagreement/total:.1%})",
        f"  Accuracy (unanimous)            : {acc_unanimous:.3f}",
        f"  Accuracy (after debate)         : {acc_debate:.3f}"
    ]

    # Include individual accuracies if the correct_answer column exists
    if "correct_answer" in df.columns:
        lines.append("\n  Individual Agent Accuracies:")
        lines.append(f"  Agent 1 Accuracy                : {(df['agent_1_ans'] == df['correct_answer']).mean():.3f}")
        lines.append(f"  Agent 2 Accuracy                : {(df['agent_2_ans'] == df['correct_answer']).mean():.3f}")
        lines.append(f"  Agent 3 Accuracy                : {(df['agent_3_ans'] == df['correct_answer']).mean():.3f}")

    output_text = "\n".join(lines)
    
    # Print to console
    print(f"\n{output_text}")

    # Save to text file
    summary_dir = Path("mad_graph/summary")
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{label}_agreement.txt" if label else "agreement_summary.txt"
    out_path = summary_dir / filename
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output_text + "\n")
        
    print(f"\nAgent agreement summary saved to: {out_path}")


def print_summary(df: pd.DataFrame, label: str = ""):
    print(f"\n{'='*50}")
    if label:
        print(f"  {label}")
    print(f"{'='*50}")

    total = len(df)
    overall_acc = df["is_correct"].mean()
    print(f"Total questions : {total}")
    print(f"Overall accuracy: {overall_acc:.3f} ({df['is_correct'].sum()}/{total})")

    sel_freq = compute_selection_frequency(df)
    print("\nSelection frequency by position:")
    for pos in ["A", "B", "C", "D"]:
        print(f"  {pos}: {sel_freq.get(pos, 0):.3f}")

    return overall_acc


def plot_comparison(baseline_df: pd.DataFrame, mad_df: pd.DataFrame,
                    output_dir: Path, label: str):
    positions = ["A", "B", "C", "D"]

    # Selection frequency (proxy for positional bias)
    baseline_freq = baseline_df["predicted_answer"].value_counts(normalize=True).reindex(positions, fill_value=0)
    mad_freq = mad_df["predicted_answer"].value_counts(normalize=True).reindex(positions, fill_value=0)

    x = np.arange(len(positions))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"MAD-Graph vs Baseline — {label}", fontsize=13)

    # Selection frequency chart
    ax = axes[0]
    bars1 = ax.bar(x - width/2, baseline_freq.values, width, label="Baseline", color="#5E81AC", alpha=0.85)
    bars2 = ax.bar(x + width/2, mad_freq.values, width, label="MAD-Graph", color="#A3BE8C", alpha=0.85)
    ax.axhline(0.25, color="gray", linestyle="--", linewidth=0.8, label="Uniform (0.25)")
    ax.set_xlabel("Selected Answer Position")
    ax.set_ylabel("Selection Frequency")
    ax.set_title("Answer Selection Frequency\n(uniform = unbiased)")
    ax.set_xticks(x)
    ax.set_xticklabels(positions)
    ax.set_ylim(0, 0.6)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.005, f"{h:.2f}", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.005, f"{h:.2f}", ha='center', va='bottom', fontsize=8)

    # Bias score (std of selection freq) and accuracy
    baseline_bias = float(baseline_freq.std())
    mad_bias = float(mad_freq.std())
    baseline_acc = baseline_df["is_correct"].mean()
    mad_acc = mad_df["is_correct"].mean()

    ax2 = axes[1]
    metrics = ["Overall Accuracy", "Selection Bias\n(std of freq)"]
    baseline_vals = [baseline_acc, baseline_bias]
    mad_vals = [mad_acc, mad_bias]

    x2 = np.arange(len(metrics))
    ax2.bar(x2 - width/2, baseline_vals, width, label="Baseline", color="#5E81AC", alpha=0.85)
    ax2.bar(x2 + width/2, mad_vals, width, label="MAD-Graph", color="#A3BE8C", alpha=0.85)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(metrics)
    ax2.set_title("Summary Metrics Comparison")
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    for i, (bv, mv) in enumerate(zip(baseline_vals, mad_vals)):
        ax2.text(i - width/2, bv + 0.005, f"{bv:.3f}", ha='center', va='bottom', fontsize=9)
        ax2.text(i + width/2, mv + 0.005, f"{mv:.3f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    out_path = output_dir / f"{label}_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze MAD-Graph evaluation results.")
    parser.add_argument("--mad-results", type=str, required=True,
                        help="Path to MAD-Graph results CSV")
    parser.add_argument("--baseline-results", type=str, default=None,
                        help="Path to baseline results CSV for comparison")
    parser.add_argument("--output-dir", type=str, default="mad_graph/analysis",
                        help="Directory to save plots")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mad_df = load_results(args.mad_results)
    label = Path(args.mad_results).stem

    print_summary(mad_df, label=f"MAD-Graph: {label}")
    analyze_agent_agreement(mad_df, label=label)

    if args.baseline_results:
        baseline_df = load_results(args.baseline_results)
        print_summary(baseline_df, label=f"Baseline: {Path(args.baseline_results).stem}")
        short_label = label.replace("_mad_graph", "")
        plot_comparison(baseline_df, mad_df, output_dir, short_label)


if __name__ == "__main__":
    main()
