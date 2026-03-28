"""
Batch Analysis Runner

Auto-discovers all evaluation CSVs and runs the appropriate analysis scripts.
Skips files that have already been processed (checks for existing output).

What it processes:
  results/baseline/*_baseline.csv     -> pride/pride_detail_eval.py
  results/mad_graph_selective/*_pride.csv -> pride/pride_selective_analysis.py

After all scripts finish, prints a side-by-side comparison table:
  Baseline vs MAD-Graph Selective — accuracy and bias before/after PriDe.

Usage:
  python run_all_analysis.py                   # process everything new
  python run_all_analysis.py --force           # re-run even if output exists
  python run_all_analysis.py --summary-only    # skip analysis, show table only
  python run_all_analysis.py --dataset college_cs  # filter to one dataset
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# PriDe core (for the final comparison table)
try:
    from pride.pride_detail_eval import PriDeDebiasing, compute_bias_metrics
except ImportError:
    from pride_detail_eval import PriDeDebiasing, compute_bias_metrics

BASELINE_DIR   = Path("results/baseline")
SELECTIVE_DIR  = Path("results/mad_graph_selective")
DATASET_DIR    = Path("dataset")
PRIDE_CSV_DIR  = Path("pride/results/csv")
SELECTIVE_OUT  = Path("pride/results/selective")
CALIB          = 0.15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_dataset_csv(label: str) -> Path | None:
    """Heuristic: try all dataset CSVs and return one whose stem is a prefix."""
    for csv in sorted(DATASET_DIR.glob("*.csv")):
        if label.startswith(csv.stem):
            return csv
    return None


def _already_done_pride(csv_path: Path) -> bool:
    """Check if pride_detail_eval.py has already produced output for this CSV."""
    stem = csv_path.stem
    return (PRIDE_CSV_DIR / f"{stem}_pride_debiased.csv").exists()


def _already_done_selective(csv_path: Path) -> bool:
    """Check if pride_selective_analysis.py has already produced output."""
    label = csv_path.stem.replace("_pride", "")
    return (SELECTIVE_OUT / f"{label}_selective_pride_report.txt").exists()


def _run(cmd: list[str], label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  [!] Script exited with code {result.returncode}")


# ---------------------------------------------------------------------------
# Metric computation (for comparison table)
# ---------------------------------------------------------------------------

def _compute_metrics(csv_path: Path, permutation_idx: int | None = 0) -> dict | None:
    """
    Run PriDe grid-search (alpha 0..1) and return key before/after metrics.

    permutation_idx: if set, pre-filter to that permutation (for bias check on
    the un-permuted view).  Pass None to use all permutations (full PriDe).
    """
    try:
        from pride.pride_detail_eval import (
            PriDeDebiasing, compute_bias_metrics, load_and_prepare_data,
        )
        df = load_and_prepare_data(str(csv_path))

        # For the final-table we want PriDe on all permutations
        if permutation_idx is not None and "permutation_idx" in df.columns:
            df_pride = df[df["permutation_idx"] != -999].copy()  # keep all
        else:
            df_pride = df.copy()

        # Grid-search best alpha (same random split every time via fixed seed)
        best_alpha, best_acc = 0.0, -1.0
        for alpha in np.arange(0.0, 1.05, 0.1):
            alpha = round(float(alpha), 1)
            p = PriDeDebiasing(calibration_ratio=CALIB, alpha=alpha, random_seed=42)
            test_deb, _ = p.fit_and_predict(df_pride)
            deb_acc = test_deb["debiased_is_correct"].mean()
            if deb_acc > best_acc:
                best_acc, best_alpha = deb_acc, alpha

        # Final run with best alpha
        p = PriDeDebiasing(calibration_ratio=CALIB, alpha=best_alpha, random_seed=42)
        test_deb, _ = p.fit_and_predict(df_pride)

        orig_m = compute_bias_metrics(test_deb, "predicted_answer")
        deb_m  = compute_bias_metrics(test_deb, "debiased_predicted_answer")

        return {
            "n":          df_pride["question_id"].nunique(),
            "best_alpha": best_alpha,
            "orig_acc":   orig_m["overall_accuracy"],
            "deb_acc":    deb_m["overall_accuracy"],
            "orig_bias":  orig_m["position_bias_score"],
            "deb_bias":   deb_m["position_bias_score"],
            "orig_rstd":  orig_m["recall_std"],
            "deb_rstd":   deb_m["recall_std"],
        }
    except Exception as e:
        print(f"  [!] Could not compute metrics for {csv_path.name}: {e}")
        return None


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def _print_comparison_table(dataset_filter: str | None):
    """Match baseline vs selective CSVs by label and print a comparison."""
    baseline_files = {
        f.stem.replace("_baseline", ""): f
        for f in sorted(BASELINE_DIR.glob("*_baseline.csv"))
    }
    selective_files = {
        f.stem.replace("_pride", ""): f
        for f in sorted(SELECTIVE_DIR.glob("*_pride.csv"))
        if "deprecated" not in str(f)
    }

    all_labels = sorted(set(baseline_files) | set(selective_files))
    if dataset_filter:
        all_labels = [l for l in all_labels if dataset_filter in l]

    if not all_labels:
        print("\n  No results found to compare.")
        return

    # Group by dataset
    from collections import defaultdict
    by_dataset: dict[str, list[str]] = defaultdict(list)
    for label in all_labels:
        ds = _find_dataset_csv(label)
        ds_name = ds.stem if ds else label.split("-")[0]
        by_dataset[ds_name].append(label)

    HDR = f"  {'Model':<32} {'α':>4} | {'Acc B':>6} {'Acc A':>6} {'Δ':>6} | {'Bias B':>7} {'Bias A':>7} {'Δ':>6} |"
    SEP = "  " + "-" * (len(HDR) - 2)

    for ds_name, labels in sorted(by_dataset.items()):
        print(f"\n{'='*70}")
        print(f"  Dataset: {ds_name}")
        print(f"{'='*70}")

        for label in labels:
            model_name = label.replace(f"{ds_name}-", "").replace("_", ":")
            print(f"\n  Model: {model_name}")

            for kind, files in [("Baseline ", baseline_files),
                                 ("MAD-Selec", selective_files)]:
                if label not in files:
                    print(f"    {kind}: (no CSV found)")
                    continue
                m = _compute_metrics(files[label])
                if m is None:
                    continue
                dacc  = m["deb_acc"]  - m["orig_acc"]
                dbias = m["deb_bias"] - m["orig_bias"]
                print(
                    f"    {kind} (n={m['n']:3d}, a={m['best_alpha']:.1f}) | "
                    f"Acc  {m['orig_acc']:.3f} -> {m['deb_acc']:.3f} ({dacc:+.3f}) | "
                    f"Bias {m['orig_bias']:5.2f} -> {m['deb_bias']:5.2f} ({dbias:+.2f})"
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run all pending analyses and print a comparison table."
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if output already exists")
    parser.add_argument("--summary-only", action="store_true",
                        help="Skip running scripts, only print the comparison table")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Filter to a specific dataset name (e.g. college_cs)")
    parser.add_argument("--calibration-ratio", type=float, default=CALIB,
                        help=f"PriDe calibration ratio (default: {CALIB})")
    args = parser.parse_args()

    # Allow overriding the module-level calibration ratio
    import run_all_analysis as _self
    _self.CALIB = args.calibration_ratio

    if not args.summary_only:
        # --- Baseline CSVs -> pride_detail_eval.py ---
        baseline_csvs = sorted(BASELINE_DIR.glob("*_baseline.csv"))
        if args.dataset:
            baseline_csvs = [f for f in baseline_csvs if args.dataset in f.name]

        print(f"\nFound {len(baseline_csvs)} baseline CSV(s).")
        for csv_path in baseline_csvs:
            if not args.force and _already_done_pride(csv_path):
                print(f"  [skip] {csv_path.name}  (already processed)")
                continue
            _run(
                [sys.executable, "pride/pride_detail_eval.py", str(csv_path)],
                f"PriDe on baseline: {csv_path.name}",
            )

        # --- Selective CSVs -> pride_selective_analysis.py ---
        selective_csvs = sorted([
            f for f in SELECTIVE_DIR.glob("*_pride.csv")
            if "deprecated" not in str(f)
        ])
        if args.dataset:
            selective_csvs = [f for f in selective_csvs if args.dataset in f.name]

        print(f"\nFound {len(selective_csvs)} selective pride CSV(s).")
        for csv_path in selective_csvs:
            if not args.force and _already_done_selective(csv_path):
                print(f"  [skip] {csv_path.name}  (already processed)")
                continue
            _run(
                [sys.executable, "pride/pride_selective_analysis.py", str(csv_path)],
                f"Selective analysis: {csv_path.name}",
            )

    # --- Final comparison table ---
    print(f"\n\n{'#'*70}")
    print(f"  BASELINE  vs  MAD-GRAPH SELECTIVE  —  Comparison Table")
    print(f"  (PriDe calibration ratio: {CALIB:.0%}, 'a' = best alpha)")
    print(f"{'#'*70}")
    _print_comparison_table(args.dataset)
    print()


if __name__ == "__main__":
    main()
