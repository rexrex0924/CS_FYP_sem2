"""
PriDe Selective Analysis

Reads the unified *_pride.csv produced by mad_graph_selective_pride_eval.py,
splits by the `confident` flag, and runs PriDe debiasing independently on:
  - All questions combined
  - Confident subset  (agents unanimously agreed in Phase 1)
  - Uncertain subset  (agents disagreed in Phase 1)

Generates a side-by-side comparison plot and a full text report so you can
see in one place whether PriDe helps more on uncertain questions (the
hypothesis) than on confident ones.

Usage:
  # Recommended: auto-derive path from model + dataset names
  python pride/pride_selective_analysis.py --model gemma3:12b --dataset dataset/college_cs.csv

  # Or pass the CSV directly
  python pride/pride_selective_analysis.py results/mad_graph_selective/college_cs-gemma3_12b_pride.csv

Output (in pride/results/selective/):
  <label>_selective_pride_comparison.png   comparison plot
  <label>_selective_pride_report.txt       full text report
"""

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Import from pride_detail_eval — works whether invoked as
# `python pride/pride_selective_analysis.py`  or from the project root.
try:
    from pride_detail_eval import PriDeDebiasing, compute_bias_metrics
except ImportError:
    from pride.pride_detail_eval import PriDeDebiasing, compute_bias_metrics

SELECTIVE_DIR = Path("results/mad_graph_selective")
OUTPUT_DIR    = Path("pride/results/selective")
ALPHAS        = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
POSITIONS     = ["A", "B", "C", "D"]
MIN_QUESTIONS = 8   # below this PriDe calibration is too unreliable to report


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

class Tee:
    """Mirror stdout to an in-memory buffer for saving as a report file."""
    def __init__(self, real):
        self._real = real
        self._buf  = io.StringIO()

    def write(self, text: str):
        self._real.write(text)
        self._buf.write(text)

    def flush(self):
        self._real.flush()

    def getvalue(self) -> str:
        return self._buf.getvalue()


# ---------------------------------------------------------------------------
# Core: run PriDe on one subset
# ---------------------------------------------------------------------------

def _run_pride_group(df: pd.DataFrame, label: str,
                     calibration_ratio: float) -> dict | None:
    """
    Grid-search best alpha, run PriDe, and return a metrics dict.
    Returns None if the group has fewer than MIN_QUESTIONS questions.
    """
    n_q = df["question_id"].nunique()
    if n_q < MIN_QUESTIONS:
        print(f"  {label}: {n_q} questions — below minimum ({MIN_QUESTIONS}). Skipping PriDe.")
        return None

    best_alpha, best_deb_acc, orig_acc_global = 0.0, -1.0, None

    for alpha in ALPHAS:
        p = PriDeDebiasing(calibration_ratio=calibration_ratio,
                           alpha=alpha, random_seed=42)
        test_deb, _ = p.fit_and_predict(df)
        if orig_acc_global is None:
            _, test_orig = p.split_calibration_test(df)
            orig_acc_global = (
                test_orig["predicted_answer"] == test_orig["correct_position"]
            ).mean()
        deb_acc = test_deb["debiased_is_correct"].mean()
        if deb_acc > best_deb_acc:
            best_deb_acc  = deb_acc
            best_alpha    = alpha

    # Final run with best alpha
    pride = PriDeDebiasing(calibration_ratio=calibration_ratio,
                           alpha=best_alpha, random_seed=42)
    test_df, cal_info = pride.fit_and_predict(df)

    orig_metrics = compute_bias_metrics(test_df, "predicted_answer")
    deb_metrics  = compute_bias_metrics(test_df, "debiased_predicted_answer")

    return {
        "label":        label,
        "n_questions":  n_q,
        "n_test":       test_df["question_id"].nunique(),
        "best_alpha":   best_alpha,
        "prior":        pride.global_prior.tolist(),
        "orig":         orig_metrics,
        "deb":          deb_metrics,
    }


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

_SEP  = "=" * 62
_SEP2 = "-" * 62


def _print_group(result: dict | None, title: str):
    if result is None:
        print(f"\n  {title}: insufficient data — skipped.")
        return

    o = result["orig"]
    d = result["deb"]

    def delta(a, b):
        return f"{b - a:+.4f}"

    acc_d  = d["overall_accuracy"] - o["overall_accuracy"]
    bias_d = d["position_bias_score"] - o["position_bias_score"]
    rstd_d = d["recall_std"] - o["recall_std"]

    print(f"\n{_SEP2}")
    print(f"  {title}  (n={result['n_questions']} total, {result['n_test']} test)")
    print(f"{_SEP2}")
    print(f"  Best alpha        : {result['best_alpha']:.2f}")
    print(f"  Prior (A/B/C/D)   : "
          f"{result['prior'][0]:.3f} / {result['prior'][1]:.3f} / "
          f"{result['prior'][2]:.3f} / {result['prior'][3]:.3f}")
    print(f"  {'Metric':<24} {'Before':>8} {'After':>8} {'Delta':>8}")
    print(f"  {'-'*50}")
    print(f"  {'Accuracy':<24} {o['overall_accuracy']:>8.4f} "
          f"{d['overall_accuracy']:>8.4f} {delta(o['overall_accuracy'], d['overall_accuracy']):>8}")
    print(f"  {'Position Bias Score':<24} {o['position_bias_score']:>8.2f} "
          f"{d['position_bias_score']:>8.2f} {delta(o['position_bias_score'], d['position_bias_score']):>8}")
    print(f"  {'RStd (%)':<24} {o['recall_std']:>8.2f} "
          f"{d['recall_std']:>8.2f} {delta(o['recall_std'], d['recall_std']):>8}")
    print(f"  {'Chi2 p-value':<24} {o['chi2_pvalue']:>8.4f} "
          f"{d['chi2_pvalue']:>8.4f}")

    # Selection frequencies
    sel_o = o["choice_percentages"]
    sel_d = d["choice_percentages"]
    print(f"\n  Selection freq (%)  Before -> After")
    for pos in POSITIONS:
        bar = "#" * int(sel_d.get(pos, 0) / 5)
        print(f"    {pos}:  {sel_o.get(pos, 0):5.1f}%  ->  {sel_d.get(pos, 0):5.1f}%  {bar}")


def _print_verdict(conf: dict | None, uncert: dict | None):
    print(f"\n{_SEP}")
    print(f"  HYPOTHESIS VERDICT")
    print(f"{_SEP}")

    if conf is None or uncert is None:
        print("  Cannot evaluate — one or both groups were too small.")
        return

    acc_gain_conf  = conf["deb"]["overall_accuracy"]  - conf["orig"]["overall_accuracy"]
    acc_gain_uncert = uncert["deb"]["overall_accuracy"] - uncert["orig"]["overall_accuracy"]
    bias_conf  = conf["orig"]["position_bias_score"]
    bias_uncert = uncert["orig"]["position_bias_score"]
    alpha_conf  = conf["best_alpha"]
    alpha_uncert = uncert["best_alpha"]

    print(f"\n  Accuracy gain from PriDe:")
    print(f"    Confident  : {acc_gain_conf:+.4f}")
    print(f"    Uncertain  : {acc_gain_uncert:+.4f}")
    print(f"\n  Pre-PriDe positional bias (bias score):")
    print(f"    Confident  : {bias_conf:.2f}")
    print(f"    Uncertain  : {bias_uncert:.2f}")
    print(f"\n  Optimal alpha (higher = more debiasing needed):")
    print(f"    Confident  : {alpha_conf:.2f}")
    print(f"    Uncertain  : {alpha_uncert:.2f}")

    # Verdict lines
    verdicts = []

    if acc_gain_uncert > acc_gain_conf + 0.01:
        verdicts.append("(+) PriDe helps uncertain questions MORE than confident ones.")
    elif abs(acc_gain_uncert - acc_gain_conf) <= 0.01:
        verdicts.append("(~) PriDe accuracy gain is similar for both groups.")
    else:
        verdicts.append("(-) PriDe unexpectedly helped confident questions more.")

    if bias_uncert > bias_conf + 1.0:
        verdicts.append("(+) Uncertain questions have higher pre-PriDe bias as expected.")
    else:
        verdicts.append("(~) Bias difference between groups is small.")

    if alpha_uncert > alpha_conf:
        verdicts.append("(+) Uncertain group needed stronger debiasing (higher alpha).")
    else:
        verdicts.append("(~) Both groups required similar debiasing strength.")

    n_positive = sum(1 for v in verdicts if v.startswith("(+)"))
    print()
    for v in verdicts:
        print(f"  {v}")

    print()
    if n_positive == 3:
        print("  => Overall: STRONG support for the hypothesis.")
    elif n_positive == 2:
        print("  => Overall: MODERATE support for the hypothesis.")
    elif n_positive == 1:
        print("  => Overall: WEAK / mixed support for the hypothesis.")
    else:
        print("  => Overall: Hypothesis NOT supported by this data.")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _plot(all_r, conf_r, uncert_r, output_dir: Path, label: str):
    groups  = []
    colors  = []
    results = [("All",       all_r,   "#5E81AC"),
               ("Confident", conf_r,  "#A3BE8C"),
               ("Uncertain", uncert_r,"#BF616A")]

    valid = [(name, r, c) for name, r, c in results if r is not None]
    if len(valid) < 2:
        print("  Not enough groups for comparison plot.")
        return

    metrics = [
        ("Accuracy",             "overall_accuracy",    False),
        ("Position Bias Score",  "position_bias_score", True),
        ("RStd (%)",             "recall_std",          True),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    fig.suptitle(f"PriDe Selective Analysis — {label}",
                 fontsize=13, fontweight="bold")

    for ax, (metric_label, key, lower_is_better) in zip(axes, metrics):
        x       = np.arange(len(valid))
        w       = 0.32
        before  = [r["orig"][key] for _, r, _ in valid]
        after   = [r["deb"][key]  for _, r, _ in valid]
        clrs    = [c for _, _, c in valid]
        max_val = max(before + after) if before + after else 1.0

        ax.bar(x - w / 2, before, w, color=clrs, alpha=0.35,
               edgecolor=clrs, linewidth=1.2)
        ax.bar(x + w / 2, after,  w, color=clrs, alpha=0.9)

        # Value labels just above each bar
        for bar_x, val in zip(
            list(x - w / 2) + list(x + w / 2),
            before + after,
        ):
            ax.text(bar_x, val + 0.005 * max_val,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

        # Delta annotations with bracket line above the tallest bar
        for i, (bef, aft) in enumerate(zip(before, after)):
            d     = aft - bef
            color = ("#2E7D32"
                     if (d < 0 and lower_is_better) or (d > 0 and not lower_is_better)
                     else "#C62828")
            top = max(bef, aft)
            ax.plot([x[i], x[i]], [top + 0.02 * max_val, top + 0.14 * max_val],
                    color=color, lw=0.8, alpha=0.6)
            ax.text(x[i], top + 0.16 * max_val,
                    f"{d:+.3f}", ha="center", va="bottom",
                    fontsize=8, color=color, fontweight="bold")

        ax.set_ylim(0, max_val * 1.55)
        ax.set_xticks(x)
        # Colour the x-tick group labels to match their bar colour
        ax.set_xticklabels([name for name, _, _ in valid])
        for tick, (_, _, c) in zip(ax.get_xticklabels(), valid):
            tick.set_color(c)
            tick.set_fontweight("bold")
        ax.set_title(metric_label, pad=18, fontsize=11)
        ax.set_ylabel(metric_label)
        ax.grid(axis="y", alpha=0.3)

    # ---- Shared figure-level legend ----
    # Row 1: shading = before / after
    before_patch = Patch(facecolor="#888888", alpha=0.35,
                         edgecolor="#888888", linewidth=1.2, label="Before PriDe")
    after_patch  = Patch(facecolor="#888888", alpha=0.9, label="After PriDe")
    # Row 2: group colours
    group_patches = [
        Patch(facecolor=c, alpha=0.85, label=f"{name}  (α={r['best_alpha']:.1f})")
        for name, r, c in valid
    ]
    fig.legend(
        handles=[before_patch, after_patch] + group_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=2 + len(valid),
        fontsize=9,
        framealpha=0.9,
        columnspacing=1.5,
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{label}_selective_pride_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved -> {out}")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyse(csv_path: str, calibration_ratio: float):
    label = Path(csv_path).stem.replace("_pride", "")

    old_stdout = sys.stdout
    tee = Tee(old_stdout)
    sys.stdout = tee

    try:
        _run(csv_path, label, calibration_ratio)
    finally:
        sys.stdout = old_stdout

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / f"{label}_selective_pride_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(tee.getvalue())
    print(f"\n  Full report saved -> {report_path}")


def _run(csv_path: str, label: str, calibration_ratio: float):
    print(f"\n{_SEP}")
    print(f"  PriDe Selective Analysis")
    print(f"  {label}")
    print(f"  Calibration ratio: {calibration_ratio:.0%}")
    print(f"{_SEP}")

    df = pd.read_csv(csv_path)
    df["confident"] = (
        df["confident"].astype(str).str.strip()
        .map({"1": 1, "0": 0, "True": 1, "False": 0, "true": 1, "false": 0})
        .fillna(0).astype(int)
    )

    n_all    = df["question_id"].nunique()
    n_conf   = df[df["confident"] == 1]["question_id"].nunique()
    n_uncert = df[df["confident"] == 0]["question_id"].nunique()

    print(f"\n  Total questions  : {n_all}")
    print(f"  Confident (1)    : {n_conf}  ({n_conf/n_all:.1%})")
    print(f"  Uncertain (0)    : {n_uncert}  ({n_uncert/n_all:.1%})")

    # Drop the confident column — PriDeDebiasing doesn't need it
    drop_cols = [c for c in ["confident", "correct_answer"] if c in df.columns]
    df_clean = df.drop(columns=drop_cols)

    conf_clean   = df[df["confident"] == 1].drop(columns=drop_cols)
    uncert_clean = df[df["confident"] == 0].drop(columns=drop_cols)

    print(f"\n  Running PriDe on all 3 groups (calibration={calibration_ratio:.0%}) ...")

    all_r    = _run_pride_group(df_clean,     "All",       calibration_ratio)
    conf_r   = _run_pride_group(conf_clean,   "Confident", calibration_ratio)
    uncert_r = _run_pride_group(uncert_clean, "Uncertain", calibration_ratio)

    # Print results
    for title, r in [("ALL QUESTIONS",  all_r),
                     ("CONFIDENT SET",  conf_r),
                     ("UNCERTAIN SET",  uncert_r)]:
        _print_group(r, title)

    # Summary comparison table
    valid = [(name, r) for name, r in
             [("All", all_r), ("Confident", conf_r), ("Uncertain", uncert_r)]
             if r is not None]
    if valid:
        print(f"\n{_SEP}")
        print(f"  SUMMARY TABLE")
        print(f"{_SEP}")
        print(f"  {'Group':<12} {'Alpha':>6} {'Acc Before':>11} {'Acc After':>10} "
              f"{'Acc Gain':>9} {'Bias Before':>12} {'Bias After':>11} {'RStd Before':>12} {'RStd After':>11}")
        print(f"  {'-'*100}")
        for name, r in valid:
            o, d = r["orig"], r["deb"]
            print(f"  {name:<12} {r['best_alpha']:>6.2f} "
                  f"{o['overall_accuracy']:>11.4f} {d['overall_accuracy']:>10.4f} "
                  f"{d['overall_accuracy']-o['overall_accuracy']:>+9.4f} "
                  f"{o['position_bias_score']:>12.2f} {d['position_bias_score']:>11.2f} "
                  f"{o['recall_std']:>12.2f} {d['recall_std']:>11.2f}")

    _print_verdict(conf_r, uncert_r)

    # Plot
    _plot(all_r, conf_r, uncert_r, OUTPUT_DIR, label)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run PriDe on the confident and uncertain subsets of a selective-pride CSV\n"
            "and generate a comparison report.\n\n"
            "  python pride/pride_selective_analysis.py --model gemma3:12b "
            "--dataset dataset/college_cs.csv\n"
            "  python pride/pride_selective_analysis.py "
            "results/mad_graph_selective/college_cs-gemma3_12b_pride.csv"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("pride_csv", nargs="?", default=None,
                        help="Path to *_pride.csv (optional if --model + --dataset given)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name as passed to the eval script (e.g. gemma3:12b)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset CSV path — used to derive the file name")
    parser.add_argument("--calibration-ratio", type=float, default=0.15,
                        help="Fraction of questions used for PriDe calibration (default: 0.15)")

    args = parser.parse_args()

    if args.pride_csv:
        csv_path = args.pride_csv
    elif args.model and args.dataset:
        dataset_stem = Path(args.dataset).stem
        model_name   = args.model.replace(":", "_").replace("/", "_")
        label        = f"{dataset_stem}-{model_name}"
        csv_path     = str(SELECTIVE_DIR / f"{label}_pride.csv")
        if not Path(csv_path).exists():
            parser.error(
                f"Could not find: {csv_path}\n"
                f"Run mad_graph_selective_pride_eval.py first."
            )
    else:
        parser.error(
            "Provide either a positional pride_csv path, "
            "or both --model and --dataset."
        )

    analyse(csv_path, calibration_ratio=args.calibration_ratio)


if __name__ == "__main__":
    main()
