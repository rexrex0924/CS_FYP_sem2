"""
MAD-Graph Results Analysis

Compares three stages for each run:
  - Phase 1  : majority vote of the three agents BEFORE any debate
  - Post-debate (Final) : resolved answer AFTER debate / graph resolution
  - Baseline (optional) : a separate vanilla CSV for further comparison

Produces console summary + comparison plots.
"""

import argparse
import io
import sys
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class _Tee:
    """Mirror every write to stdout AND an in-memory buffer."""
    def __init__(self, real_stdout):
        self._real = real_stdout   # captured before sys.stdout is replaced
        self._buf = io.StringIO()

    def write(self, text: str):
        self._real.write(text)     # always write to the original stdout
        self._buf.write(text)

    def flush(self):
        self._real.flush()

    def getvalue(self) -> str:
        return self._buf.getvalue()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["is_correct"] = df["is_correct"].astype(int)
    return df


def majority_vote(row) -> str:
    """Return the Phase-1 majority-vote answer from the three agent columns."""
    votes = [
        str(row.get("agent_1_ans", "")).strip().upper(),
        str(row.get("agent_2_ans", "")).strip().upper(),
        str(row.get("agent_3_ans", "")).strip().upper(),
    ]
    valid = [v for v in votes if v in ("A", "B", "C", "D")]
    if not valid:
        return ""
    return Counter(valid).most_common(1)[0][0]


def add_phase1_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'phase1_answer' column (majority vote before debate)."""
    df = df.copy()
    if all(c in df.columns for c in ("agent_1_ans", "agent_2_ans", "agent_3_ans")):
        df["phase1_answer"] = df.apply(majority_vote, axis=1)
        df["phase1_is_correct"] = (df["phase1_answer"] == df["correct_answer"]).astype(int)
    return df


def selection_freq(series: pd.Series) -> pd.Series:
    return series.value_counts(normalize=True).reindex(["A", "B", "C", "D"], fill_value=0)


def get_reference_dist(dataset_csv: str | None) -> pd.Series:
    """
    Return the reference distribution to use as the 'unbiased' baseline.
    If a dataset CSV is provided, use the empirical answer key distribution.
    Otherwise fall back to uniform (0.25 each).
    """
    POSITIONS = ["A", "B", "C", "D"]
    if dataset_csv and Path(dataset_csv).exists():
        df = pd.read_csv(dataset_csv)
        ans_col = next((c for c in df.columns
                        if c.lower() in ("answer", "correct_answer", "label")), None)
        if ans_col:
            dist = (df[ans_col].astype(str).str.strip().str.upper()
                    .value_counts(normalize=True)
                    .reindex(POSITIONS, fill_value=0))
            return dist
    return pd.Series([0.25] * 4, index=POSITIONS)


def bias_score(pred_freq: pd.Series, ref_dist: pd.Series | None = None) -> float:
    """
    Mean absolute deviation of predicted selection frequencies from the
    reference distribution (empirical answer key dist, or uniform 0.25).
    Lower = less biased.
    """
    if ref_dist is None:
        ref_dist = pd.Series([0.25] * 4, index=["A", "B", "C", "D"])
    return float((pred_freq - ref_dist).abs().mean())


# ---------------------------------------------------------------------------
# Console reporting
# ---------------------------------------------------------------------------

def print_stage(df: pd.DataFrame, answer_col: str, correct_col: str, label: str,
                ref_dist: pd.Series | None = None):
    valid = df[df[answer_col].isin(["A", "B", "C", "D"])]
    total = len(valid)
    acc = valid[correct_col].mean() if total > 0 else 0.0
    freq = selection_freq(valid[answer_col])

    print(f"\n{'-'*52}")
    print(f"  {label}")
    print(f"{'-'*52}")
    print(f"  Valid responses : {total}")
    print(f"  Accuracy        : {acc:.3f}  ({valid[correct_col].sum()}/{total})")
    print(f"  Selection freq  : A={freq['A']:.3f}  B={freq['B']:.3f}  "
          f"C={freq['C']:.3f}  D={freq['D']:.3f}")
    print(f"  Bias score (MAD): {bias_score(freq, ref_dist):.4f}")
    return acc, freq


def print_summary(df: pd.DataFrame, label: str = "", ref_dist: pd.Series | None = None,
                  output_dir: Path | None = None):
    df = add_phase1_column(df)

    print(f"\n{'='*52}")
    if label:
        print(f"  {label}")
    print(f"{'='*52}")
    print(f"  Total rows : {len(df)}")

    phase1_acc, phase1_freq = None, None
    if "phase1_answer" in df.columns:
        phase1_acc, phase1_freq = print_stage(
            df, "phase1_answer", "phase1_is_correct",
            "Phase 1  (before debate - majority vote)", ref_dist)

    final_acc, final_freq = print_stage(
        df, "predicted_answer", "is_correct",
        "Final  (after debate / graph resolution)", ref_dist)

    if phase1_acc is not None:
        delta_acc = final_acc - phase1_acc
        delta_bias = bias_score(final_freq, ref_dist) - bias_score(phase1_freq, ref_dist)
        print(f"\n  Delta Accuracy  (Final - Phase1) : {delta_acc:+.3f}")
        print(f"  Delta Bias (MAD)(Final - Phase1) : {delta_bias:+.4f}  "
              f"({'reduced [OK]' if delta_bias < 0 else 'increased [!!]'})")

    # Debate trigger analysis
    print_agent_agreement(df, label=label, output_dir=output_dir)

    return df


def print_agent_agreement(df: pd.DataFrame, label: str = "",
                          output_dir: Path | None = None):
    if not all(c in df.columns for c in ("agent_1_ans", "agent_2_ans", "agent_3_ans")):
        return
    total = len(df)
    unanimous = (
        (df["agent_1_ans"] == df["agent_2_ans"]) &
        (df["agent_2_ans"] == df["agent_3_ans"]) &
        df["agent_1_ans"].isin(["A", "B", "C", "D"])
    ).sum()
    debated = total - unanimous

    mask = (
        (df["agent_1_ans"] == df["agent_2_ans"]) &
        (df["agent_2_ans"] == df["agent_3_ans"])
    )
    acc_u = df[mask]["is_correct"].mean() if unanimous > 0 else 0
    acc_d = df[~mask]["is_correct"].mean() if debated > 0 else 0

    lines = [
        "  Agent agreement (Phase 1):",
        f"    Unanimous (no debate needed) : {unanimous:4d}  ({unanimous/total:.1%})",
        f"    Debate triggered             : {debated:4d}  ({debated/total:.1%})",
        f"    Accuracy when unanimous      : {acc_u:.3f}",
        f"    Accuracy after debate        : {acc_d:.3f}",
    ]

    if "correct_answer" in df.columns:
        lines.append("\n  Individual Agent Accuracies (Phase 1):")
        for i in range(1, 4):
            col = f"agent_{i}_ans"
            acc = (df[col] == df["correct_answer"]).mean()
            lines.append(f"    Agent {i} Accuracy : {acc:.3f}")

    output_text = "\n".join(lines)
    print(f"\n{output_text}")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{label}_agreement.txt" if label else "agreement_summary.txt"
        out_path = output_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(output_text + "\n")
        print(f"\n  Agreement summary saved -> {out_path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = {
    "baseline": "#5E81AC",
    "phase1":   "#EBCB8B",
    "final":    "#A3BE8C",
}


def plot_comparison(df: pd.DataFrame, baseline_df: pd.DataFrame | None,
                    output_dir: Path, label: str, ref_dist: pd.Series | None = None):
    positions = ["A", "B", "C", "D"]

    phase1_freq  = selection_freq(df["phase1_answer"]) if "phase1_answer" in df.columns else None
    final_freq   = selection_freq(df["predicted_answer"])
    base_freq    = selection_freq(baseline_df["predicted_answer"]) if baseline_df is not None else None

    phase1_acc  = df["phase1_is_correct"].mean() if "phase1_is_correct" in df.columns else None
    final_acc   = df["is_correct"].mean()
    base_acc    = baseline_df["is_correct"].mean() if baseline_df is not None else None

    # Decide how many series to plot
    series = []
    if base_freq is not None:
        series.append(("Baseline",      base_freq,   base_acc,   COLORS["baseline"]))
    if phase1_freq is not None:
        series.append(("Phase 1\n(before debate)", phase1_freq, phase1_acc, COLORS["phase1"]))
    series.append(("Final\n(after debate)",  final_freq,  final_acc,  COLORS["final"]))

    n = len(series)
    x = np.arange(len(positions))
    total_width = 0.7
    w = total_width / n

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"MAD-Graph Analysis — {label}", fontsize=13, fontweight="bold")

    # --- Left: selection frequency ---
    ax = axes[0]
    for i, (lbl, freq, _, color) in enumerate(series):
        offset = (i - (n - 1) / 2) * w
        bars = ax.bar(x + offset, freq.values, w, label=lbl, color=color, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7)
    # Draw a reference bar per position instead of a single flat line,
    # so non-uniform datasets show the correct per-position target.
    if ref_dist is None:
        ref_dist = pd.Series([0.25] * 4, index=["A", "B", "C", "D"])
    is_uniform = (ref_dist - 0.25).abs().max() < 0.01
    ref_label = "Uniform (0.25)" if is_uniform else "Answer key dist (unbiased ref)"
    ax.bar(x, ref_dist.values, total_width + 0.05, color="none",
           edgecolor="gray", linewidth=1.5, linestyle="--", label=ref_label, zorder=0)
    ax.set_xlabel("Selected Answer Position")
    ax.set_ylabel("Selection Frequency")
    title_note = "(reference = uniform 0.25)" if is_uniform else "(reference = answer key distribution)"
    ax.set_title(f"Answer Selection Frequency\n{title_note}")
    ax.set_xticks(x)
    ax.set_xticklabels(positions)
    ax.set_ylim(0, 0.65)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # --- Right: accuracy + bias score ---
    ax2 = axes[1]
    metrics = ["Overall Accuracy", "Selection Bias\n(std of freq)"]
    x2 = np.arange(len(metrics))
    for i, (lbl, freq, acc, color) in enumerate(series):
        if acc is None:
            continue
        offset = (i - (n - 1) / 2) * w
        vals = [acc, bias_score(freq, ref_dist)]
        bars2 = ax2.bar(x2 + offset, vals, w, label=lbl, color=color, alpha=0.85)
        for bar, v in zip(bars2, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(metrics)
    bias_note = "vs uniform" if is_uniform else "vs answer key dist"
    ax2.set_title(f"Summary Metrics Comparison\n(bias = MAD {bias_note})")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / f"{label}_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved -> {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze MAD-Graph evaluation results.")
    parser.add_argument("--mad-results", type=str, required=True,
                        help="Path to MAD-Graph results CSV")
    parser.add_argument("--baseline-results", type=str, default=None,
                        help="Path to baseline results CSV for comparison")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to the original dataset CSV (used to compute the empirical "
                             "answer key distribution as the unbiased reference). "
                             "If omitted, falls back to uniform 0.25.")
    parser.add_argument("--output-dir", type=str, default="results/mad_graph/analysis",
                        help="Directory to save plots")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_dist = get_reference_dist(args.dataset)
    is_uniform = (ref_dist - 0.25).abs().max() < 0.01
    if args.dataset:
        print(f"\n  Reference distribution loaded from: {args.dataset}")
        if is_uniform:
            print("  Dataset is near-uniform — 0.25 line is appropriate.")
        else:
            print("  Dataset is NON-UNIFORM — using empirical answer key distribution as reference.")
            for pos, v in ref_dist.items():
                print(f"    {pos}: {v:.4f} ({v*100:.1f}%)")
    else:
        print("\n  No --dataset provided. Using uniform 0.25 as reference.")

    mad_df = load_results(args.mad_results)
    mad_df = add_phase1_column(mad_df)
    label  = Path(args.mad_results).stem

    # Capture everything printed so we can also save it to a txt report
    _real_stdout = sys.stdout
    tee = _Tee(_real_stdout)   # capture real stdout before replacing
    sys.stdout = tee

    try:
        print_summary(mad_df, label=label, ref_dist=ref_dist, output_dir=output_dir)

        baseline_df = None
        if args.baseline_results:
            baseline_df = load_results(args.baseline_results)
            baseline_label = Path(args.baseline_results).stem
            print_summary(baseline_df, label=baseline_label, ref_dist=ref_dist,
                          output_dir=output_dir)

        plot_comparison(mad_df, baseline_df, output_dir, label, ref_dist=ref_dist)
    finally:
        sys.stdout = _real_stdout

    # Save the full console output to a txt file alongside the plots
    report_path = output_dir / f"{label}_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(tee.getvalue())
    print(f"\n  Full report saved -> {report_path}")


if __name__ == "__main__":
    main()
