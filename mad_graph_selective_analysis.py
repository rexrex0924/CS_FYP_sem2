"""
MAD-Graph Selective PriDe Analysis

Reads the two CSVs produced by mad_graph_selective_pride_eval.py and tests
the research hypothesis:

  "When agents unanimously agree in Phase 1 the model knows the answer.
   When they disagree, positional bias fills the uncertainty gap."

If the hypothesis holds we expect:
  - Confident accuracy  >>  Uncertain accuracy
  - Uncertain answer distribution  more biased  than confident distribution

Outputs (in results/mad_graph_selective/analysis/):
  <label>_selective_analysis.txt   full console report saved to disk
  <label>_selective_comparison.png comparison plot
"""

import argparse
import io
import sys
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers (self-contained — no dependency on other project scripts)
# ---------------------------------------------------------------------------

POSITIONS = ["A", "B", "C", "D"]


def selection_freq(series: pd.Series) -> pd.Series:
    return series.value_counts(normalize=True).reindex(POSITIONS, fill_value=0)


def bias_score(freq: pd.Series, ref: pd.Series) -> float:
    """Mean Absolute Deviation from the reference distribution."""
    return float((freq - ref).abs().mean())


def get_reference_dist(dataset_csv: str | None) -> pd.Series:
    if dataset_csv and Path(dataset_csv).exists():
        df = pd.read_csv(dataset_csv)
        col = next((c for c in df.columns
                    if c.lower() in ("answer", "correct_answer", "label")), None)
        if col:
            return (df[col].astype(str).str.strip().str.upper()
                    .value_counts(normalize=True)
                    .reindex(POSITIONS, fill_value=0))
    return pd.Series([0.25] * 4, index=POSITIONS)


class Tee:
    """Write simultaneously to stdout and an in-memory buffer."""
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
# Analysis logic
# ---------------------------------------------------------------------------

def analyse(confident_csv: str, uncertain_csv: str,
            ref: pd.Series, output_dir: Path, label: str):

    old_stdout = sys.stdout
    tee = Tee(old_stdout)   # capture real stdout before replacing
    sys.stdout = tee

    try:
        _run_analysis(confident_csv, uncertain_csv, ref, output_dir, label)
    finally:
        sys.stdout = old_stdout

    report_path = output_dir / f"{label}_selective_analysis.txt"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(tee.getvalue())
    print(f"\n  Full report saved -> {report_path}")


def _run_analysis(confident_csv: str, uncertain_csv: str,
                  ref: pd.Series, output_dir: Path, label: str):

    conf_df  = pd.read_csv(confident_csv)  if confident_csv  else pd.DataFrame()
    uncert_df = pd.read_csv(uncertain_csv) if uncertain_csv  else pd.DataFrame()

    is_uniform = (ref - 0.25).abs().max() < 0.01
    ref_label  = "uniform 0.25" if is_uniform else "empirical answer key dist"

    n_conf   = len(conf_df)
    # Each uncertain question has 4 rows (one per permutation); count unique IDs
    n_uncert = uncert_df["question_id"].nunique() if not uncert_df.empty else 0
    total    = n_conf + n_uncert

    sep  = "=" * 58
    sep2 = "-" * 58

    print(f"\n{sep}")
    print(f"  MAD-Graph Selective PriDe Analysis")
    print(f"  {label}")
    print(f"{sep}")
    print(f"  Reference dist   : {ref_label}")
    print(f"  Total questions  : {total}")
    print(f"  Confident (3/3)  : {n_conf:4d}  ({n_conf/total:.1%})")
    print(f"  Uncertain (<3/3) : {n_uncert:4d}  ({n_uncert/total:.1%})")

    # ---- Confident set ----
    if not conf_df.empty:
        valid_c = conf_df[conf_df["predicted_answer"].isin(POSITIONS)]
        acc_c   = valid_c["is_correct"].mean() if len(valid_c) else 0.0
        freq_c  = selection_freq(valid_c["predicted_answer"])
        bias_c  = bias_score(freq_c, ref)

        print(f"\n{sep2}")
        print(f"  CONFIDENT SET  (agents unanimously agreed — model 'knows')")
        print(f"{sep2}")
        print(f"  N              : {len(valid_c)}")
        print(f"  Accuracy       : {acc_c:.3f}  ({valid_c['is_correct'].sum()}/{len(valid_c)})")
        print(f"  Selection freq : A={freq_c['A']:.3f}  B={freq_c['B']:.3f}  "
              f"C={freq_c['C']:.3f}  D={freq_c['D']:.3f}")
        print(f"  Bias (MAD)     : {bias_c:.4f}  vs {ref_label}")
    else:
        acc_c = freq_c = bias_c = None
        print("\n  No confident CSV provided / file is empty.")

    # ---- Uncertain set (permutation 0 = original layout) ----
    acc_u = freq_u = bias_u = None
    if not uncert_df.empty:
        # Use permutation_idx == 0 for pre-PriDe stats
        perm0 = uncert_df[uncert_df["permutation_idx"] == 0].copy()
        perm0["is_correct"] = (perm0["predicted_answer"] == perm0["correct_position"]).astype(int)
        valid_u = perm0[perm0["predicted_answer"].isin(POSITIONS)]
        acc_u   = valid_u["is_correct"].mean() if len(valid_u) else 0.0
        freq_u  = selection_freq(valid_u["predicted_answer"])
        bias_u  = bias_score(freq_u, ref)

        print(f"\n{sep2}")
        print(f"  UNCERTAIN SET  (agents disagreed — positional bias suspected)")
        print(f"{sep2}")
        print(f"  N              : {len(valid_u)}")
        print(f"  Pre-PriDe acc  : {acc_u:.3f}  ({valid_u['is_correct'].sum()}/{len(valid_u)})")
        print(f"  Selection freq : A={freq_u['A']:.3f}  B={freq_u['B']:.3f}  "
              f"C={freq_u['C']:.3f}  D={freq_u['D']:.3f}")
        print(f"  Bias (MAD)     : {bias_u:.4f}  vs {ref_label}")

        # Show per-position vote probability averages across all permutations
        avg_probs = uncert_df[["prob_A","prob_B","prob_C","prob_D"]].mean()
        print(f"\n  Avg vote probs across all permutations (PriDe input signal):")
        print(f"    prob_A={avg_probs['prob_A']:.3f}  prob_B={avg_probs['prob_B']:.3f}  "
              f"prob_C={avg_probs['prob_C']:.3f}  prob_D={avg_probs['prob_D']:.3f}")
    else:
        print("\n  No uncertain CSV provided / file is empty.")

    # ---- Combined overall ----
    if acc_c is not None and acc_u is not None:
        n_c_correct = int(round(acc_c * n_conf))
        n_u_correct = int(round(acc_u * n_uncert))
        acc_overall = (n_c_correct + n_u_correct) / total
        print(f"\n{sep2}")
        print(f"  OVERALL (confident + uncertain combined)")
        print(f"{sep2}")
        print(f"  Accuracy       : {acc_overall:.3f}  ({n_c_correct+n_u_correct}/{total})")
        print(f"  Accuracy gap   : confident={acc_c:.3f}  uncertain={acc_u:.3f}  "
              f"gap={acc_c - acc_u:+.3f}")
        if acc_c - acc_u > 0.05:
            print("  -> Hypothesis SUPPORTED: confident set is notably more accurate.")
        else:
            print("  -> Hypothesis WEAK: accuracy gap is small.")

        if bias_c is not None and bias_u is not None:
            if bias_u > bias_c + 0.01:
                print(f"  -> Bias gap SUPPORTS hypothesis: uncertain bias ({bias_u:.4f}) "
                      f"> confident bias ({bias_c:.4f})")
            else:
                print(f"  -> Bias gap INCONCLUSIVE: uncertain ({bias_u:.4f}) "
                      f"vs confident ({bias_c:.4f})")

    # ---- Next steps ----
    print(f"\n{sep2}")
    print(f"  NEXT STEPS")
    print(f"{sep2}")
    print(f"  To debias the uncertain set with PriDe, run:")
    print(f"    python pride/pride_detail_eval.py {uncertain_csv}")

    # ---- Plot ----
    if acc_c is not None and acc_u is not None:
        _plot(freq_c, freq_u, acc_c, acc_u, bias_c, bias_u,
              ref, ref_label, output_dir, label)


def _plot(freq_c, freq_u, acc_c, acc_u, bias_c, bias_u,
          ref, ref_label, output_dir: Path, label: str):
    x = np.arange(len(POSITIONS))
    w = 0.3
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"MAD-Graph Selective Analysis — {label}", fontsize=13, fontweight="bold")

    # Left: answer selection frequency
    ax = axes[0]
    bars_c = ax.bar(x - w/2, freq_c.values, w, label="Confident (unanimous)",
                    color="#A3BE8C", alpha=0.85)
    bars_u = ax.bar(x + w/2, freq_u.values, w, label="Uncertain (disagreed)",
                    color="#BF616A", alpha=0.85)
    for bar in list(bars_c) + list(bars_u):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f"{h:.2f}", ha="center", va="bottom", fontsize=7)
    ax.bar(x, ref.values, 0.75, color="none", edgecolor="gray",
           linewidth=1.5, linestyle="--", label=f"Reference ({ref_label})", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(POSITIONS)
    ax.set_xlabel("Selected Answer Position")
    ax.set_ylabel("Selection Frequency")
    ax.set_title("Answer Selection Frequency\n(confident vs uncertain set)")
    ax.set_ylim(0, 0.7)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Right: accuracy + bias comparison
    ax2 = axes[1]
    metrics = ["Accuracy", "Bias (MAD)"]
    x2 = np.arange(len(metrics))
    vals_c = [acc_c, bias_c]
    vals_u = [acc_u, bias_u]
    bars_c2 = ax2.bar(x2 - w/2, vals_c, w, label="Confident", color="#A3BE8C", alpha=0.85)
    bars_u2 = ax2.bar(x2 + w/2, vals_u, w, label="Uncertain (pre-PriDe)", color="#BF616A", alpha=0.85)
    for bar, v in zip(list(bars_c2) + list(bars_u2), vals_c + vals_u):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(metrics)
    ax2.set_title("Accuracy & Bias Comparison\n(if hypothesis holds: high acc+low bias vs low acc+high bias)")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{label}_selective_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved -> {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

SELECTIVE_DIR = Path("results/mad_graph_selective")


def _find_pairs(dataset_stem: str) -> list[tuple[Path, Path, str]]:
    """
    Scan SELECTIVE_DIR for all (*_confident.csv, *_uncertain_pride.csv) pairs
    whose name starts with `dataset_stem`.  Returns list of (conf, uncert, label).
    """
    pairs = []
    for conf in sorted(SELECTIVE_DIR.glob(f"{dataset_stem}-*_confident.csv")):
        label = conf.stem.replace("_confident", "")
        uncert = SELECTIVE_DIR / f"{label}_uncertain_pride.csv"
        if uncert.exists():
            pairs.append((conf, uncert, label))
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Analyse MAD-Graph Selective PriDe results — confident vs uncertain split.\n\n"
                    "Short form (recommended):\n"
                    "  python mad_graph_selective_analysis.py --model gemma3:4b "
                    "--dataset dataset/2012-2020_ICT_DSE.csv\n\n"
                    "Run all models for a dataset at once:\n"
                    "  python mad_graph_selective_analysis.py --all "
                    "--dataset dataset/2012-2020_ICT_DSE.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- short-form args (auto-derive paths) ---
    parser.add_argument("--model", type=str, default=None,
                        help="Model name as passed to the eval script (e.g. gemma3:4b). "
                             "Auto-derives --confident-csv and --uncertain-csv.")
    parser.add_argument("--all", action="store_true",
                        help="Analyse every model found in results/mad_graph_selective/ "
                             "for the given --dataset.")

    # --- explicit paths (original long form, still supported) ---
    parser.add_argument("--confident-csv", type=str, default=None,
                        help="Path to *_confident.csv  (optional if --model is given)")
    parser.add_argument("--uncertain-csv", type=str, default=None,
                        help="Path to *_uncertain_pride.csv  (optional if --model is given)")

    parser.add_argument("--dataset", type=str, default=None,
                        help="Original dataset CSV for empirical reference distribution.")
    parser.add_argument("--output-dir", type=str,
                        default="results/mad_graph_selective/analysis",
                        help="Directory for plots and report txt (default: results/mad_graph_selective/analysis)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ref = get_reference_dist(args.dataset)
    is_uniform = (ref - 0.25).abs().max() < 0.01

    if args.dataset:
        print(f"\n  Reference distribution : {args.dataset}")
        if not is_uniform:
            print("  Dataset is NON-UNIFORM — using empirical distribution as reference.")
            for pos, v in ref.items():
                print(f"    {pos}: {v:.4f}  ({v*100:.1f}%)")
    else:
        print("\n  No --dataset provided. Using uniform 0.25 as reference.")

    # ---- build list of (confident_csv, uncertain_csv, label) to process ----
    jobs: list[tuple[str, str, str]] = []

    if args.all:
        # Auto-find every model pair for this dataset
        if not args.dataset:
            parser.error("--all requires --dataset")
        dataset_stem = Path(args.dataset).stem
        pairs = _find_pairs(dataset_stem)
        if not pairs:
            print(f"\n  No results found in {SELECTIVE_DIR} for dataset '{dataset_stem}'.")
            return
        for conf, uncert, label in pairs:
            jobs.append((str(conf), str(uncert), label))

    elif args.model:
        # Short form: derive paths from model + dataset names
        if not args.dataset:
            parser.error("--model requires --dataset")
        dataset_stem = Path(args.dataset).stem
        model_name   = args.model.replace(":", "_").replace("/", "_")
        label        = f"{dataset_stem}-{model_name}"
        conf         = SELECTIVE_DIR / f"{label}_confident.csv"
        uncert       = SELECTIVE_DIR / f"{label}_uncertain_pride.csv"
        if not conf.exists():
            parser.error(f"Could not find: {conf}\n"
                         f"Run mad_graph_selective_pride_eval.py first.")
        if not uncert.exists():
            parser.error(f"Could not find: {uncert}\n"
                         f"Run mad_graph_selective_pride_eval.py first.")
        jobs.append((str(conf), str(uncert), label))

    elif args.confident_csv and args.uncertain_csv:
        # Original explicit long form
        stem  = Path(args.confident_csv).stem
        label = stem.replace("_confident", "")
        jobs.append((args.confident_csv, args.uncertain_csv, label))

    else:
        parser.error("Provide either --model, --all, or both --confident-csv and --uncertain-csv.")

    # ---- run ----
    for conf_path, uncert_path, label in jobs:
        analyse(conf_path, uncert_path, ref, output_dir, label)

    if len(jobs) > 1:
        print(f"\n  Finished analysing {len(jobs)} models.")


if __name__ == "__main__":
    main()
