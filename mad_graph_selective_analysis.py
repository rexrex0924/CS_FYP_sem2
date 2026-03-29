"""
MAD-Graph Selective PriDe Analysis

Reads the unified CSV produced by mad_graph_selective_pride_eval.py and tests
the research hypothesis:

  "When agents unanimously agree in Phase 1 the model knows the answer.
   When they disagree, positional bias fills the uncertainty gap."

If the hypothesis holds we expect:
  - Confident accuracy  >>  Uncertain accuracy
  - Uncertain answer distribution  more biased  than confident distribution

The CSV has 4 rows per question (one per cyclic permutation) and a `confident`
column (1 = unanimous agreement, 0 = disagreement).

Usage (recommended short form):
  python mad_graph_selective_analysis.py --model gemma3:4b --dataset dataset/foo.csv

Run all models for a dataset at once:
  python mad_graph_selective_analysis.py --all --dataset dataset/foo.csv

Explicit paths:
  python mad_graph_selective_analysis.py --pride-csv results/.../foo_pride.csv

Outputs (in results/mad_graph_selective/analysis/):
  <label>_selective_analysis.txt   full console report
  <label>_selective_comparison.png comparison plot
"""

import argparse
import io
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
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
        col = next(
            (c for c in df.columns if c.lower() in ("answer", "correct_answer", "label")),
            None,
        )
        if col:
            return (
                df[col].astype(str).str.strip().str.upper()
                .value_counts(normalize=True)
                .reindex(POSITIONS, fill_value=0)
            )
    return pd.Series([0.25] * 4, index=POSITIONS)


class Tee:
    """Write simultaneously to the real stdout and an in-memory buffer."""

    def __init__(self, real_stdout):
        self._real = real_stdout
        self._buf = io.StringIO()

    def write(self, text: str):
        self._real.write(text)
        self._buf.write(text)

    def flush(self):
        self._real.flush()

    def getvalue(self) -> str:
        return self._buf.getvalue()


# ---------------------------------------------------------------------------
# Analysis logic
# ---------------------------------------------------------------------------

def analyse(pride_csv: str, ref: pd.Series, output_dir: Path, label: str):
    old_stdout = sys.stdout
    tee = Tee(old_stdout)
    sys.stdout = tee
    try:
        _run_analysis(pride_csv, ref, output_dir, label)
    finally:
        sys.stdout = old_stdout

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{label}_selective_analysis.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(tee.getvalue())
    print(f"\n  Full report saved -> {report_path}")


def _run_analysis(pride_csv: str, ref: pd.Series, output_dir: Path, label: str):
    df = pd.read_csv(pride_csv)

    # Normalise the confident column to int (handles bool strings from CSV)
    df["confident"] = df["confident"].astype(str).str.strip().map(
        {"1": 1, "0": 0, "True": 1, "False": 0, "true": 1, "false": 0}
    ).fillna(0).astype(int)

    # Use only permutation_idx == 0 for accuracy and bias measurements so that
    # cyclic shifts don't artificially redistribute the position signal.
    perm0 = df[df["permutation_idx"] == 0].copy()
    perm0["is_correct"] = (
        perm0["predicted_answer"].str.strip().str.upper()
        == perm0["correct_position"].str.strip().str.upper()
    ).astype(int)

    conf_p0  = perm0[perm0["confident"] == 1]
    uncert_p0 = perm0[perm0["confident"] == 0]

    # All-permutation rows split (for avg prob display)
    conf_all  = df[df["confident"] == 1]
    uncert_all = df[df["confident"] == 0]

    n_conf   = len(conf_p0)
    n_uncert = len(uncert_p0)
    total    = n_conf + n_uncert

    is_uniform = (ref - 0.25).abs().max() < 0.01
    ref_label  = "uniform 0.25" if is_uniform else "empirical answer key dist"

    sep  = "=" * 60
    sep2 = "-" * 60

    print(f"\n{sep}")
    print(f"  MAD-Graph Selective PriDe Analysis")
    print(f"  {label}")
    print(f"{sep}")
    print(f"  Reference dist   : {ref_label}")
    if not is_uniform:
        for pos, v in ref.items():
            print(f"    {pos}: {v:.4f}  ({v*100:.1f}%)")
    print(f"  Total questions  : {total}")
    if total > 0:
        print(f"  Confident (3/3)  : {n_conf:4d}  ({n_conf/total:.1%})")
        print(f"  Uncertain (<3/3) : {n_uncert:4d}  ({n_uncert/total:.1%})")

    # ---- Confident set ----
    acc_c = freq_c = bias_c = None
    if not conf_p0.empty:
        valid_c = conf_p0[conf_p0["predicted_answer"].isin(POSITIONS)]
        acc_c   = valid_c["is_correct"].mean() if len(valid_c) else 0.0
        freq_c  = selection_freq(valid_c["predicted_answer"])
        bias_c  = bias_score(freq_c, ref)

        print(f"\n{sep2}")
        print(f"  CONFIDENT SET  (agents unanimously agreed)")
        print(f"{sep2}")
        print(f"  N              : {len(valid_c)}")
        print(f"  Accuracy       : {acc_c:.3f}  ({valid_c['is_correct'].sum()}/{len(valid_c)})")
        print(f"  Selection freq : A={freq_c['A']:.3f}  B={freq_c['B']:.3f}  "
              f"C={freq_c['C']:.3f}  D={freq_c['D']:.3f}")
        print(f"  Bias (MAD)     : {bias_c:.4f}  vs {ref_label}")

        if not conf_all.empty:
            avg_probs = conf_all[["prob_A", "prob_B", "prob_C", "prob_D"]].mean()
            print(f"  Avg vote probs (all perms, confident):")
            print(f"    prob_A={avg_probs['prob_A']:.3f}  prob_B={avg_probs['prob_B']:.3f}  "
                  f"prob_C={avg_probs['prob_C']:.3f}  prob_D={avg_probs['prob_D']:.3f}")
    else:
        print("\n  No confident questions found.")

    # ---- Uncertain set ----
    acc_u = freq_u = bias_u = None
    if not uncert_p0.empty:
        valid_u = uncert_p0[uncert_p0["predicted_answer"].isin(POSITIONS)]
        acc_u   = valid_u["is_correct"].mean() if len(valid_u) else 0.0
        freq_u  = selection_freq(valid_u["predicted_answer"])
        bias_u  = bias_score(freq_u, ref)

        print(f"\n{sep2}")
        print(f"  UNCERTAIN SET  (agents disagreed — positional bias suspected)")
        print(f"{sep2}")
        print(f"  N              : {len(valid_u)}")
        print(f"  Accuracy       : {acc_u:.3f}  ({valid_u['is_correct'].sum()}/{len(valid_u)})")
        print(f"  Selection freq : A={freq_u['A']:.3f}  B={freq_u['B']:.3f}  "
              f"C={freq_u['C']:.3f}  D={freq_u['D']:.3f}")
        print(f"  Bias (MAD)     : {bias_u:.4f}  vs {ref_label}")

        if not uncert_all.empty:
            avg_probs = uncert_all[["prob_A", "prob_B", "prob_C", "prob_D"]].mean()
            print(f"  Avg vote probs (all perms, uncertain):")
            print(f"    prob_A={avg_probs['prob_A']:.3f}  prob_B={avg_probs['prob_B']:.3f}  "
                  f"prob_C={avg_probs['prob_C']:.3f}  prob_D={avg_probs['prob_D']:.3f}")
    else:
        print("\n  No uncertain questions found.")

    # ---- Combined + hypothesis verdict ----
    if acc_c is not None and acc_u is not None:
        n_c_correct = int(round(acc_c * n_conf))
        n_u_correct = int(round(acc_u * n_uncert))
        acc_overall = (n_c_correct + n_u_correct) / total if total else 0.0

        print(f"\n{sep2}")
        print(f"  OVERALL (confident + uncertain combined)")
        print(f"{sep2}")
        print(f"  Accuracy       : {acc_overall:.3f}  ({n_c_correct + n_u_correct}/{total})")
        print(f"  Accuracy gap   : confident={acc_c:.3f}  uncertain={acc_u:.3f}  "
              f"gap={acc_c - acc_u:+.3f}")

        if acc_c - acc_u > 0.05:
            print("  -> Hypothesis SUPPORTED: confident set is notably more accurate.")
        else:
            print("  -> Hypothesis WEAK: accuracy gap is small (<5 pp).")

        if bias_c is not None and bias_u is not None:
            if bias_u > bias_c + 0.01:
                print(f"  -> Bias gap SUPPORTS hypothesis: uncertain ({bias_u:.4f}) "
                      f"> confident ({bias_c:.4f})")
            else:
                print(f"  -> Bias gap INCONCLUSIVE: uncertain ({bias_u:.4f}) "
                      f"vs confident ({bias_c:.4f})")

    # ---- Permutation consistency (new: empirical test of genuine knowledge) ----
    print(f"\n{sep2}")
    print(f"  PERMUTATION CONSISTENCY  (do all 4 shifts pick the same answer?)")
    print(f"{sep2}")
    for group_name, group_all in [("Confident", conf_all), ("Uncertain", uncert_all)]:
        if group_all.empty:
            continue
        consistent = 0
        total_q = 0
        for _, grp in group_all.groupby("question_id"):
            if len(grp) == 4:
                answers = grp["predicted_answer"].str.strip().str.upper()
                if answers.nunique() == 1 and answers.iloc[0] in POSITIONS:
                    consistent += 1
                total_q += 1
        if total_q:
            pct = consistent / total_q
            print(f"  {group_name:10s} : {consistent}/{total_q} questions had "
                  f"identical answer across all 4 perms  ({pct:.1%})")

    # ---- Next steps ----
    print(f"\n{sep2}")
    print(f"  NEXT STEPS")
    print(f"{sep2}")
    print(f"  To run PriDe on the uncertain subset:")
    print(f"    python -c \"import pandas as pd; "
          f"df=pd.read_csv('{pride_csv}'); "
          f"df[df.confident==0].drop(columns='confident').to_csv('uncertain_only.csv', index=False)\"")
    print(f"    python pride/pride_detail_eval.py uncertain_only.csv")

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
    bars_c = ax.bar(x - w / 2, freq_c.values, w, label="Confident (unanimous)",
                    color="#A3BE8C", alpha=0.85)
    bars_u = ax.bar(x + w / 2, freq_u.values, w, label="Uncertain (disagreed)",
                    color="#BF616A", alpha=0.85)
    for bar in list(bars_c) + list(bars_u):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                f"{h:.2f}", ha="center", va="bottom", fontsize=7)
    ax.bar(x, ref.values, 0.75, color="none", edgecolor="gray",
           linewidth=1.5, linestyle="--",
           label=f"Reference ({ref_label})", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(POSITIONS)
    ax.set_xlabel("Selected Answer Position")
    ax.set_ylabel("Selection Frequency")
    ax.set_title("Answer Selection Frequency\n(permutation 0 only)")
    ax.set_ylim(0, 0.75)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Right: accuracy + bias
    ax2 = axes[1]
    metrics = ["Accuracy", "Bias (MAD)"]
    x2 = np.arange(len(metrics))
    vals_c = [acc_c, bias_c]
    vals_u = [acc_u, bias_u]
    bars_c2 = ax2.bar(x2 - w / 2, vals_c, w, label="Confident",
                      color="#A3BE8C", alpha=0.85)
    bars_u2 = ax2.bar(x2 + w / 2, vals_u, w, label="Uncertain",
                      color="#BF616A", alpha=0.85)
    for bar, v in zip(list(bars_c2) + list(bars_u2), vals_c + vals_u):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(metrics)
    ax2.set_title("Accuracy & Bias\n(hypothesis: high acc+low bias vs low acc+high bias)")
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


def _find_results(dataset_stem: str) -> list[tuple[Path, str]]:
    """Return all *_pride.csv files whose name starts with `dataset_stem`."""
    results = []
    for f in sorted(SELECTIVE_DIR.glob(f"{dataset_stem}-*_pride.csv")):
        label = f.stem.replace("_pride", "")
        results.append((f, label))
    return results


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyse MAD-Graph Selective PriDe results.\n\n"
            "Short form (recommended):\n"
            "  python mad_graph_selective_analysis.py "
            "--model gemma3:4b --dataset dataset/foo.csv\n\n"
            "Run all models for one dataset:\n"
            "  python mad_graph_selective_analysis.py "
            "--all --dataset dataset/foo.csv"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--model", type=str, default=None,
                        help="Model name as passed to the eval script (e.g. gemma3:4b).")
    parser.add_argument("--all", action="store_true",
                        help="Analyse every model found for the given --dataset.")
    parser.add_argument("--pride-csv", type=str, default=None,
                        help="Explicit path to *_pride.csv (overrides --model/--all).")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Original dataset CSV for empirical reference distribution.")
    parser.add_argument("--output-dir", type=str,
                        default="results/mad_graph_selective/analysis",
                        help="Directory for output files.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ref = get_reference_dist(args.dataset)
    is_uniform = (ref - 0.25).abs().max() < 0.01

    if args.dataset:
        print(f"\n  Reference distribution : {args.dataset}")
        if not is_uniform:
            print("  Dataset is NON-UNIFORM — using empirical distribution as reference.")
    else:
        print("\n  No --dataset provided. Using uniform 0.25 as reference.")

    # Build list of (pride_csv_path, label)
    jobs: list[tuple[str, str]] = []

    if args.pride_csv:
        label = Path(args.pride_csv).stem.replace("_pride", "")
        jobs.append((args.pride_csv, label))

    elif args.all:
        if not args.dataset:
            parser.error("--all requires --dataset")
        dataset_stem = Path(args.dataset).stem
        found = _find_results(dataset_stem)
        if not found:
            print(f"\n  No results found in {SELECTIVE_DIR} for dataset '{dataset_stem}'.")
            return
        for csv_path, label in found:
            jobs.append((str(csv_path), label))

    elif args.model:
        if not args.dataset:
            parser.error("--model requires --dataset")
        dataset_stem = Path(args.dataset).stem
        model_name   = args.model.replace(":", "_").replace("/", "_")
        label        = f"{dataset_stem}-{model_name}"
        csv_path     = SELECTIVE_DIR / f"{label}_pride.csv"
        if not csv_path.exists():
            parser.error(
                f"Could not find: {csv_path}\n"
                f"Run mad_graph_selective_pride_eval.py first."
            )
        jobs.append((str(csv_path), label))

    else:
        parser.error("Provide --model, --all, or --pride-csv.")

    for csv_path, label in jobs:
        analyse(csv_path, ref, output_dir, label)

    if len(jobs) > 1:
        print(f"\n  Finished analysing {len(jobs)} models.")


if __name__ == "__main__":
    main()
