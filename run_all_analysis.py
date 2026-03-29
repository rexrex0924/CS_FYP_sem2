"""
Batch Analysis Runner for MAD-Graph Selective PriDe
====================================================
Reads all *_pride.csv from results/mad_graph_selective/output/
(also falls back to results/mad_graph_selective/ for legacy files)

For every CSV it:
  1. Runs PriDe debiasing (fixed alpha or grid-search)
  2. Computes the full metric suite from pride_batch_summary:
       overall_accuracy, accuracy_by_position, choice distribution,
       chi-square (distribution + accuracy-vs-position),
       position_bias_score, recall_std, consistency_score
  3. Computes per-agent (agent_1/2/3) accuracy, bias & recall_std,
       broken down by confident / uncertain group

Output (pride/results/selective/output/):
  by_dataset/<dataset>/          -- cross-model comparison plots
  by_model_dataset/<ds>-<mdl>/   -- individual model deep-dives
  SUMMARY_REPORT.txt             -- text summary table

Usage:
  python run_all_analysis.py                        # process everything new
  python run_all_analysis.py --force                # re-run even if done
  python run_all_analysis.py --dataset college_cs   # filter to one dataset
  python run_all_analysis.py --fixed-alpha 0.3      # skip grid-search
  python run_all_analysis.py --summary-only         # report only, no plots
"""

import argparse
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import chisquare, chi2_contingency

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SELECTIVE_INPUT_DIR = Path("results/mad_graph_selective/output")
LEGACY_INPUT_DIR    = Path("results/mad_graph_selective")   # fallback
OUTPUT_DIR          = Path("pride/results/selective/output")
CALIB_RATIO         = 0.15
DEFAULT_FIXED_ALPHA = 0.3    # set to None to grid-search per CSV

MODEL_ORDER = [
    "gemma3_1b", "gemma3_4b", "gemma3_12b",
    "llama3.2_latest", "llama3_8b-instruct-q6_K", "llama3_8b-instruct-q8_0",
    "mistral_latest", "mistral-nemo_latest", "mistral-small3.2_24b",
]

POSITIONS = ["A", "B", "C", "D"]
C_BASE  = "#FF6B6B"   # red  – baseline
C_DEB   = "#4ECDC4"   # teal – after PriDe
C_AG1   = "#5B8FF9"   # blue  – agent 1
C_AG2   = "#F6BD16"   # gold  – agent 2
C_AG3   = "#5AD8A6"   # green – agent 3
C_CONF  = "#7B68EE"   # purple – confident
C_UNCONF = "#FF8C69"  # salmon – uncertain


# ---------------------------------------------------------------------------
# PriDe core  (self-contained so we don't depend on pride_batch_summary globals)
# ---------------------------------------------------------------------------

class _PriDe:
    def __init__(self, calib: float = CALIB_RATIO, alpha: float = 1.0, seed: int = 42):
        self.calib, self.alpha, self.seed = calib, alpha, seed
        self._prior = None
        self._calibrated = False

    def _split(self, df):
        ids = df["question_id"].unique()
        rng = np.random.RandomState(self.seed)
        rng.shuffle(ids)
        n = max(1, int(len(ids) * self.calib))
        cal_ids, test_ids = ids[:n], ids[n:]
        return df[df["question_id"].isin(cal_ids)].copy(), df[df["question_id"].isin(test_ids)].copy()

    def _estimate_prior(self, cal):
        priors = []
        for _, grp in cal.groupby("question_id"):
            grp = grp.sort_values("permutation_idx")
            P = grp[["prob_A","prob_B","prob_C","prob_D"]].values
            P = P / (P.sum(axis=1, keepdims=True) + 1e-10)
            prior = np.exp(np.log(P + 1e-10).mean(axis=0))
            prior /= prior.sum()
            priors.append(prior)
        self._prior = np.mean(priors, axis=0)
        self._calibrated = True

    def _debias(self, test):
        out = test.copy()
        answers, correct = [], []
        for _, row in out.iterrows():
            obs = np.array([row[f"prob_{p}"] for p in POSITIONS])
            logits = np.log(obs + 1e-10) - self.alpha * np.log(self._prior + 1e-10)
            pred = POSITIONS[int(np.argmax(logits))]
            answers.append(pred)
            correct.append(int(pred == str(row["correct_position"]).upper().strip()))
        out["debiased_predicted_answer"] = answers
        out["debiased_is_correct"] = correct
        return out

    def fit_predict(self, df):
        cal, test = self._split(df)
        self._estimate_prior(cal)
        return self._debias(test), test


def _run_pride(df: pd.DataFrame, fixed_alpha):
    """Return (test_debiased, test_original, best_alpha)."""
    alphas = [fixed_alpha] if fixed_alpha is not None else np.round(np.arange(0.0, 1.05, 0.1), 1).tolist()
    best_alpha, best_acc, best_deb, best_orig = alphas[0], -1.0, None, None
    for a in alphas:
        p = _PriDe(calib=CALIB_RATIO, alpha=float(a))
        deb, orig = p.fit_predict(df)
        acc = deb["debiased_is_correct"].mean()
        if acc > best_acc:
            best_acc, best_alpha, best_deb, best_orig = acc, a, deb, orig
    return best_deb, best_orig, best_alpha


# ---------------------------------------------------------------------------
# Metric computation  (matches pride_batch_summary.compute_bias_metrics)
# ---------------------------------------------------------------------------

def _compute_metrics(df: pd.DataFrame, pred_col: str = "predicted_answer") -> dict:
    d = df.copy()
    d[pred_col] = d[pred_col].astype(str).str.upper().str.strip()
    d["correct_position"] = d["correct_position"].astype(str).str.upper().str.strip()

    valid = d[d[pred_col].isin(POSITIONS)].copy()
    valid["is_correct_eval"] = (valid[pred_col] == valid["correct_position"]).astype(int)

    counts = valid[pred_col].value_counts().reindex(POSITIONS, fill_value=0)
    total  = len(valid)
    pcts   = (counts / total * 100) if total else counts * 0.0

    chi2_stat, chi2_p = chisquare(counts.values, f_exp=[total / 4] * 4)
    bias_score = float(np.std(pcts.values))

    recalls = []
    for pos in POSITIONS:
        mask = valid["correct_position"] == pos
        recalls.append((valid.loc[mask, pred_col] == pos).mean() if mask.sum() > 0 else 0.0)
    recall_std = float(np.std(recalls) * 100)

    acc_by_pos = valid.groupby("correct_position")["is_correct_eval"].mean().to_dict()
    overall_acc = valid["is_correct_eval"].mean() if total else 0.0

    try:
        ct = pd.crosstab(valid["correct_position"], valid["is_correct_eval"])
        chi2_acc, p_acc, _, _ = chi2_contingency(ct)
    except Exception:
        chi2_acc, p_acc = 0.0, 1.0

    # Consistency: fraction of questions where model picks same *content* across all permutations
    id_col = "question_id" if "question_id" in valid.columns else "id"
    consistency_score = 0.0
    if id_col in valid.columns and "permutation_idx" in valid.columns:
        def _orig_choice(row):
            try:
                pred = str(row[pred_col]).upper().strip()
                if pred not in POSITIONS:
                    return None
                shift = int(row["permutation_idx"]) % 4
                return POSITIONS[(shift + POSITIONS.index(pred)) % 4]
            except Exception:
                return None
        valid["_orig"] = valid.apply(_orig_choice, axis=1)
        vv = valid.dropna(subset=["_orig"])
        if not vv.empty:
            uniq = vv.groupby(id_col)["_orig"].nunique()
            consistency_score = float((uniq == 1).mean() * 100)

    return {
        "overall_accuracy":    float(overall_acc),
        "accuracy_by_position": acc_by_pos,
        "choice_counts":       counts.to_dict(),
        "choice_percentages":  pcts.to_dict(),
        "chi2_stat":           float(chi2_stat),
        "chi2_pvalue":         float(chi2_p),
        "position_bias_score": bias_score,
        "recall_std":          recall_std,
        "recalls":             {p: recalls[i] for i, p in enumerate(POSITIONS)},
        "chi2_acc_stat":       float(chi2_acc),
        "chi2_acc_pvalue":     float(p_acc),
        "consistency_score":   consistency_score,
        "n_samples":           total,
    }


def _compute_agent_metrics(df_perm0: pd.DataFrame) -> dict:
    """
    Per-agent accuracy & bias from permutation_idx=0 rows.
    Returns {'agent_1': {...}, 'agent_2': {...}, 'agent_3': {...}}
    """
    out = {}
    for i in [1, 2, 3]:
        col = f"agent_{i}_ans"
        if col not in df_perm0.columns:
            continue
        sub = df_perm0[[col, "correct_position", "confident"]].copy()
        sub[col] = sub[col].astype(str).str.upper().str.strip()
        sub["correct_position"] = sub["correct_position"].astype(str).str.upper().str.strip()
        valid = sub[sub[col].isin(POSITIONS)].copy()
        if valid.empty:
            continue
        valid["is_correct"] = (valid[col] == valid["correct_position"]).astype(int)

        counts = valid[col].value_counts().reindex(POSITIONS, fill_value=0)
        total  = len(valid)
        pcts   = (counts / total * 100).to_dict() if total else {p: 0.0 for p in POSITIONS}
        bias   = float(np.std(list(pcts.values())))
        recalls = []
        for pos in POSITIONS:
            m = valid["correct_position"] == pos
            recalls.append((valid.loc[m, col] == pos).mean() if m.sum() > 0 else 0.0)
        recall_std = float(np.std(recalls) * 100)

        conf_mask   = valid["confident"] == 1
        unconf_mask = ~conf_mask
        conf_acc   = float(valid.loc[conf_mask,   "is_correct"].mean()) if conf_mask.sum()   > 0 else float("nan")
        unconf_acc = float(valid.loc[unconf_mask, "is_correct"].mean()) if unconf_mask.sum() > 0 else float("nan")

        out[f"agent_{i}"] = {
            "accuracy":    float(valid["is_correct"].mean()),
            "conf_acc":    conf_acc,
            "unconf_acc":  unconf_acc,
            "bias":        bias,
            "recall_std":  recall_std,
            "choice_pcts": pcts,
            "n":           total,
        }
    return out


# ---------------------------------------------------------------------------
# Label parsing & data loading
# ---------------------------------------------------------------------------

def _parse_label(filepath: Path):
    """Return (dataset, model) or (None, None)."""
    name = filepath.stem.replace("_pride", "").replace("_baseline", "")
    for prefix in ["ministral", "mistral", "gemma", "llama", "phi", "qwen"]:
        if prefix in name:
            parts = name.split(prefix, 1)
            return parts[0].strip("-"), prefix + parts[1]
    return None, None


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for p in POSITIONS:
        df[f"prob_{p}"] = pd.to_numeric(df[f"prob_{p}"], errors="coerce").fillna(0.0)
    return df


def _sort_models(models):
    def key(m):
        try:
            return MODEL_ORDER.index(m)
        except ValueError:
            return len(MODEL_ORDER)
    return sorted(models, key=key)


# ---------------------------------------------------------------------------
# Data processing
# ---------------------------------------------------------------------------

def process_all_csvs(csv_dirs: list[Path], fixed_alpha, dataset_filter: str | None) -> dict:
    results = defaultdict(lambda: defaultdict(dict))

    all_files = []
    seen = set()
    for d in csv_dirs:
        for f in sorted(d.glob("*_pride.csv")):
            if "deprecated" not in str(f) and f.name not in seen:
                all_files.append(f)
                seen.add(f.name)

    if dataset_filter:
        all_files = [f for f in all_files if dataset_filter in f.name]

    print(f"\nFound {len(all_files)} CSV(s) to process.\n")

    for f in all_files:
        dataset, model = _parse_label(f)
        if not dataset:
            print(f"  [skip] Cannot parse name: {f.name}")
            continue

        print(f"  {dataset} / {model}")
        try:
            df = _load(f)

            # --- full-set PriDe ---
            test_deb, test_orig, alpha = _run_pride(df, fixed_alpha)
            base_m = _compute_metrics(test_orig, "predicted_answer")
            deb_m  = _compute_metrics(test_deb,  "debiased_predicted_answer")

            # --- confident / uncertain sub-groups ---
            def _group_pride(mask_val):
                sub = df[df["confident"] == mask_val] if "confident" in df.columns else pd.DataFrame()
                if sub["question_id"].nunique() < 4:
                    return None
                try:
                    td, to, a = _run_pride(sub, fixed_alpha)
                    return {
                        "baseline": _compute_metrics(to, "predicted_answer"),
                        "debiased": _compute_metrics(td, "debiased_predicted_answer"),
                        "alpha": a,
                    }
                except Exception:
                    return None

            conf_m   = _group_pride(1)
            unconf_m = _group_pride(0)

            # --- per-agent metrics ---
            df_p0 = df[df["permutation_idx"] == 0].copy()
            agent_m = _compute_agent_metrics(df_p0)

            results[dataset][model] = {
                "baseline":    base_m,
                "debiased":    deb_m,
                "best_alpha":  alpha,
                "confident":   conf_m,
                "uncertain":   unconf_m,
                "agents":      agent_m,
                "n_questions": int(df["question_id"].nunique()),
                "n_confident": int(df_p0["confident"].sum()) if "confident" in df_p0.columns else 0,
                "csv_path":    str(f),
            }

            print(f"    a={alpha:.1f}  "
                  f"Acc {base_m['overall_accuracy']*100:.1f}%→{deb_m['overall_accuracy']*100:.1f}%  "
                  f"Bias {base_m['position_bias_score']:.2f}→{deb_m['position_bias_score']:.2f}")

        except Exception as e:
            print(f"    [ERROR] {e}")

    return dict(results)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _bar_labels(ax, bars, fmt="{:.2f}", fontsize=7, va_up="bottom", va_dn="top"):
    for bar in bars:
        h = bar.get_height()
        va = va_up if h >= 0 else va_dn
        ax.text(bar.get_x() + bar.get_width() / 2, h,
                fmt.format(h), ha="center", va=va, fontsize=fontsize, fontweight="bold")


# ---------------------------------------------------------------------------
# BY-DATASET plots  (ported faithfully from pride_batch_summary)
# ---------------------------------------------------------------------------

def plot_dataset_accuracy_comparison(dataset, models_data, output_path):
    models = _sort_models(models_data.keys())
    b_acc = [models_data[m]["baseline"]["overall_accuracy"] for m in models]
    d_acc = [models_data[m]["debiased"]["overall_accuracy"]  for m in models]
    x = np.arange(len(models)); w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    bars1 = ax1.bar(x - w/2, b_acc, w, label="Baseline",    color=C_BASE, alpha=0.8)
    bars2 = ax1.bar(x + w/2, d_acc, w, label="After PriDe", color=C_DEB,  alpha=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Accuracy"); ax1.set_title(f"{dataset}: Accuracy Comparison", fontweight="bold")
    ax1.legend(); ax1.grid(axis="y", alpha=0.3)
    _bar_labels(ax1, bars1, "{:.1%}"); _bar_labels(ax1, bars2, "{:.1%}")

    imps = [d - b for b, d in zip(b_acc, d_acc)]
    colors = ["green" if i > 0 else "red" for i in imps]
    bars3 = ax2.bar(range(len(models)), [i * 100 for i in imps], color=colors, alpha=0.7)
    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_xticks(range(len(models))); ax2.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Accuracy Improvement (pp)"); ax2.set_title(f"{dataset}: Accuracy Improvement", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    _bar_labels(ax2, bars3, "{:+.2f}pp")

    plt.tight_layout(); plt.savefig(output_path, dpi=300, bbox_inches="tight"); plt.close()


def plot_dataset_bias_comparison(dataset, models_data, output_path):
    models = _sort_models(models_data.keys())
    x = np.arange(len(models)); w = 0.35
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    panels = [
        (axes[0, 0], "position_bias_score", "Position Bias Score (Lower=Better)"),
        (axes[0, 1], "recall_std",           "Recall Std % (Lower=Better)"),
        (axes[1, 0], "chi2_stat",            "Chi-square Stat (Lower=More Uniform)"),
        (axes[1, 1], "chi2_pvalue",          "Chi-square p-value (Higher=More Uniform)"),
    ]
    for ax, key, title in panels:
        bv = [models_data[m]["baseline"][key] for m in models]
        dv = [models_data[m]["debiased"][key]  for m in models]
        b1 = ax.bar(x - w/2, bv, w, label="Baseline",    color=C_BASE, alpha=0.8)
        b2 = ax.bar(x + w/2, dv, w, label="After PriDe", color=C_DEB,  alpha=0.8)
        if key == "chi2_pvalue":
            ax.axhline(0.05, color="red", linestyle="--", linewidth=2, label="p=0.05")
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{dataset}: {title}", fontweight="bold"); ax.legend(); ax.grid(axis="y", alpha=0.3)
        fmt = "{:.3f}" if key == "chi2_pvalue" else "{:.2f}"
        _bar_labels(ax, b1, fmt); _bar_labels(ax, b2, fmt)

    plt.tight_layout(); plt.savefig(output_path, dpi=300, bbox_inches="tight"); plt.close()


def plot_dataset_distribution_comparison(dataset, models_data, output_path):
    models = _sort_models(models_data.keys())
    fig, axes = plt.subplots(len(models), 2, figsize=(14, 4 * len(models)))
    if len(models) == 1:
        axes = axes.reshape(1, -1)
    bar_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"]
    for idx, m in enumerate(models):
        for col, kind in [(0, "baseline"), (1, "debiased")]:
            ax = axes[idx, col]
            pcts = [models_data[m][kind]["choice_percentages"][p] for p in POSITIONS]
            bars = ax.bar(POSITIONS, pcts, color=bar_colors, alpha=0.8)
            ax.axhline(25, color="red", linestyle="--", linewidth=2, label="Uniform 25%")
            bias = models_data[m][kind]["position_bias_score"]
            alpha_tag = f" (a={models_data[m]['best_alpha']:.1f})" if kind == "debiased" else ""
            ax.set_title(f"{m} – {'Baseline' if kind=='baseline' else 'After PriDe'+alpha_tag}\nBias={bias:.2f}",
                         fontweight="bold", fontsize=9)
            ax.set_ylim(0, max(pcts + [26]) * 1.25)
            ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
            _bar_labels(ax, bars, "{:.1f}%", fontsize=8)
    fig.suptitle(f"{dataset}: Choice Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(output_path, dpi=300, bbox_inches="tight"); plt.close()


def plot_dataset_accuracy_by_position(dataset, models_data, output_path):
    models = _sort_models(models_data.keys())
    fig, axes = plt.subplots(len(models), 1, figsize=(14, 5 * len(models)))
    if len(models) == 1:
        axes = [axes]
    x = np.arange(4); w = 0.35
    for idx, m in enumerate(models):
        ax = axes[idx]
        b_acc = [models_data[m]["baseline"]["accuracy_by_position"].get(p, 0) for p in POSITIONS]
        d_acc = [models_data[m]["debiased"]["accuracy_by_position"].get(p, 0)  for p in POSITIONS]
        b1 = ax.bar(x - w/2, b_acc, w, label="Baseline",    color=C_BASE, alpha=0.8)
        b2 = ax.bar(x + w/2, d_acc, w, label="After PriDe", color=C_DEB,  alpha=0.8)
        ax.axhline(models_data[m]["baseline"]["overall_accuracy"], color=C_BASE, linestyle="--", linewidth=1.5)
        ax.axhline(models_data[m]["debiased"]["overall_accuracy"],  color=C_DEB,  linestyle="--", linewidth=1.5)
        ax.set_xticks(x); ax.set_xticklabels(POSITIONS)
        ax.set_title(f"{m}: Accuracy by Position", fontweight="bold"); ax.legend(); ax.grid(axis="y", alpha=0.3)
        _bar_labels(ax, b1, "{:.1%}", fontsize=8); _bar_labels(ax, b2, "{:.1%}", fontsize=8)
    fig.suptitle(f"{dataset}: Accuracy by Position", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(output_path, dpi=300, bbox_inches="tight"); plt.close()


def plot_dataset_consistency_comparison(dataset, models_data, output_path):
    models = _sort_models(models_data.keys())
    b_cons = [models_data[m]["baseline"].get("consistency_score", 0) for m in models]
    d_cons = [models_data[m]["debiased"].get("consistency_score", 0)  for m in models]
    x = np.arange(len(models)); w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    b1 = ax1.bar(x - w/2, b_cons, w, label="Baseline",    color=C_BASE, alpha=0.8)
    b2 = ax1.bar(x + w/2, d_cons, w, label="After PriDe", color=C_DEB,  alpha=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Consistency Score (%)"); ax1.set_title(f"{dataset}: Consistency", fontweight="bold")
    ax1.legend(); ax1.grid(axis="y", alpha=0.3)
    _bar_labels(ax1, b1, "{:.1f}%"); _bar_labels(ax1, b2, "{:.1f}%")

    imps = [d - b for b, d in zip(b_cons, d_cons)]
    bars3 = ax2.bar(range(len(models)), imps, color=["green" if i > 0 else "red" for i in imps], alpha=0.7)
    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_xticks(range(len(models))); ax2.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Consistency Improvement (pp)"); ax2.set_title(f"{dataset}: Consistency Improvement", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3); _bar_labels(ax2, bars3, "{:+.2f}pp")

    plt.tight_layout(); plt.savefig(output_path, dpi=300, bbox_inches="tight"); plt.close()


# ---------------------------------------------------------------------------
# NEW: Agent accuracy & bias comparison (by dataset)
# ---------------------------------------------------------------------------

def plot_dataset_agent_comparison(dataset, models_data, output_path):
    """
    For each model, show Agent 1 / 2 / 3 accuracy, bias, and recall_std
    side by side.  Also shows confident vs uncertain accuracy per agent.
    """
    models = _sort_models(models_data.keys())
    ag_keys = ["agent_1", "agent_2", "agent_3"]
    ag_labels = ["Agent 1\n(Step-by-step)", "Agent 2\n(Trap-avoid)", "Agent 3\n(Intuitive)"]
    ag_colors = [C_AG1, C_AG2, C_AG3]

    fig, axes = plt.subplots(3, len(models), figsize=(6 * len(models), 14), squeeze=False)
    fig.suptitle(f"{dataset}: Per-Agent Analysis", fontsize=15, fontweight="bold")

    for col, m in enumerate(models):
        agents = models_data[m].get("agents", {})

        # Row 0: overall accuracy
        ax = axes[0, col]
        accs = [agents.get(a, {}).get("accuracy", float("nan")) for a in ag_keys]
        bars = ax.bar(ag_labels, accs, color=ag_colors, alpha=0.85)
        ax.set_ylim(0, 1.0); ax.set_title(f"{m}\nOverall Accuracy", fontsize=9, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        _bar_labels(ax, bars, "{:.1%}", fontsize=8)

        # Row 1: confident vs uncertain accuracy
        ax = axes[1, col]
        conf_accs   = [agents.get(a, {}).get("conf_acc",   float("nan")) for a in ag_keys]
        unconf_accs = [agents.get(a, {}).get("unconf_acc", float("nan")) for a in ag_keys]
        x = np.arange(len(ag_keys)); w = 0.35
        b1 = ax.bar(x - w/2, conf_accs,   w, label="Confident",   color=C_CONF,   alpha=0.85)
        b2 = ax.bar(x + w/2, unconf_accs, w, label="Uncertain",    color=C_UNCONF, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(ag_labels, fontsize=7)
        ax.set_ylim(0, 1.1); ax.set_title("Confident vs Uncertain Accuracy", fontsize=9, fontweight="bold")
        ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
        _bar_labels(ax, b1, "{:.1%}", fontsize=7); _bar_labels(ax, b2, "{:.1%}", fontsize=7)

        # Row 2: bias score & recall_std
        ax = axes[2, col]
        biases = [agents.get(a, {}).get("bias",       float("nan")) for a in ag_keys]
        rstds  = [agents.get(a, {}).get("recall_std", float("nan")) for a in ag_keys]
        x = np.arange(len(ag_keys)); w = 0.35
        b1 = ax.bar(x - w/2, biases, w, label="Bias Score", color="#E76F51", alpha=0.85)
        b2 = ax.bar(x + w/2, rstds,  w, label="Recall Std", color="#457B9D", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(ag_labels, fontsize=7)
        ax.set_title("Bias Score & Recall Std (Lower=Better)", fontsize=9, fontweight="bold")
        ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
        _bar_labels(ax, b1, "{:.2f}", fontsize=7); _bar_labels(ax, b2, "{:.2f}", fontsize=7)

    plt.tight_layout(); plt.savefig(output_path, dpi=300, bbox_inches="tight"); plt.close()


# ---------------------------------------------------------------------------
# BY-MODEL-DATASET individual plots  (ported from pride_batch_summary)
# ---------------------------------------------------------------------------

def plot_individual_accuracy_comparison(dataset, model, data, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    base, deb = data["baseline"], data["debiased"]

    bars = ax1.bar(["Baseline", "After PriDe"],
                   [base["overall_accuracy"] * 100, deb["overall_accuracy"] * 100],
                   color=[C_BASE, C_DEB], alpha=0.8, width=0.6)
    ax1.set_ylabel("Accuracy (%)"); ax1.set_title(f"{dataset} – {model}\nOverall Accuracy", fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, [base["overall_accuracy"], deb["overall_accuracy"]]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{v*100:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    delta = deb["overall_accuracy"] - base["overall_accuracy"]
    ax1.text(0.5, max(base["overall_accuracy"], deb["overall_accuracy"]) * 50,
             f"Delta = {delta*100:+.2f}pp", ha="center", fontsize=13, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7))

    x = np.arange(4); w = 0.35
    b_pos = [base["accuracy_by_position"].get(p, 0) * 100 for p in POSITIONS]
    d_pos = [deb["accuracy_by_position"].get(p, 0)  * 100 for p in POSITIONS]
    b1 = ax2.bar(x - w/2, b_pos, w, label="Baseline",    color=C_BASE, alpha=0.8)
    b2 = ax2.bar(x + w/2, d_pos, w, label="After PriDe", color=C_DEB,  alpha=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(POSITIONS)
    ax2.set_xlabel("Correct Answer Position"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy by Position", fontweight="bold"); ax2.legend(); ax2.grid(axis="y", alpha=0.3)
    _bar_labels(ax2, b1, "{:.1f}%", fontsize=9); _bar_labels(ax2, b2, "{:.1f}%", fontsize=9)

    plt.tight_layout(); plt.savefig(output_path, dpi=300, bbox_inches="tight"); plt.close()


def plot_individual_bias_metrics(dataset, model, data, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    base, deb = data["baseline"], data["debiased"]
    cats = ["Baseline", "After PriDe"]

    panels = [
        (axes[0, 0], [base["position_bias_score"], deb["position_bias_score"]], "Position Bias Score (Lower=Better)", "{:.2f}"),
        (axes[0, 1], [base["recall_std"],           deb["recall_std"]],           "Recall Std % (Lower=Better)",         "{:.2f}"),
        (axes[1, 0], [base["chi2_stat"],             deb["chi2_stat"]],             "Chi-square Stat (Lower=Uniform)",      "{:.2f}"),
        (axes[1, 1], [base["chi2_pvalue"],           deb["chi2_pvalue"]],           "Chi-square p-value (Higher=Uniform)",  "{:.4f}"),
    ]
    for ax, vals, title, fmt in panels:
        bars = ax.bar(cats, vals, color=[C_BASE, C_DEB], alpha=0.8, width=0.6)
        if "p-value" in title:
            ax.axhline(0.05, color="red", linestyle="--", linewidth=2, label="p=0.05")
            ax.legend()
        ax.set_title(title, fontweight="bold"); ax.grid(axis="y", alpha=0.3)
        _bar_labels(ax, bars, fmt, fontsize=10)

    fig.suptitle(f"{dataset} – {model}: Bias Metrics", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(output_path, dpi=300, bbox_inches="tight"); plt.close()


def plot_individual_distribution(dataset, model, data, output_path):
    base, deb = data["baseline"], data["debiased"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    bar_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"]

    for ax, kind, label in [(ax1, base, "Baseline"), (ax2, deb, f"After PriDe (a={data['best_alpha']:.1f})")]:
        pcts = [kind["choice_percentages"][p] for p in POSITIONS]
        bars = ax.bar(POSITIONS, pcts, color=bar_colors, alpha=0.85)
        ax.axhline(25, color="red", linestyle="--", linewidth=2, label="Uniform 25%")
        ax.set_title(f"{label}\nBias Score={kind['position_bias_score']:.2f}", fontweight="bold")
        ax.set_ylim(0, max(pcts + [26]) * 1.25); ax.legend(); ax.grid(axis="y", alpha=0.3)
        _bar_labels(ax, bars, "{:.1f}%")

    fig.suptitle(f"{dataset} – {model}: Choice Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(output_path, dpi=300, bbox_inches="tight"); plt.close()


def plot_individual_summary(dataset, model, data, output_path):
    """Comprehensive 3×3 dashboard (mirrors pride_batch_summary.plot_individual_summary)."""
    base, deb = data["baseline"], data["debiased"]
    alpha = data["best_alpha"]
    agents = data.get("agents", {})

    fig = plt.figure(figsize=(20, 14))
    gs  = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.3)
    fig.suptitle(f"{dataset} – {model}: Comprehensive Summary", fontsize=16, fontweight="bold")

    # 1. Overall accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(["Baseline", "After PriDe"],
                   [base["overall_accuracy"] * 100, deb["overall_accuracy"] * 100],
                   color=[C_BASE, C_DEB], alpha=0.8)
    ax1.set_title("Overall Accuracy", fontweight="bold"); ax1.grid(axis="y", alpha=0.3)
    _bar_labels(ax1, bars, "{:.1f}%", fontsize=9)

    # 2. Position bias score
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(["Baseline", "After PriDe"],
                   [base["position_bias_score"], deb["position_bias_score"]],
                   color=[C_BASE, C_DEB], alpha=0.8)
    ax2.set_title("Position Bias Score", fontweight="bold"); ax2.grid(axis="y", alpha=0.3)
    _bar_labels(ax2, bars, "{:.2f}", fontsize=9)

    # 3. Recall std
    ax3 = fig.add_subplot(gs[0, 2])
    bars = ax3.bar(["Baseline", "After PriDe"],
                   [base["recall_std"], deb["recall_std"]],
                   color=[C_BASE, C_DEB], alpha=0.8)
    ax3.set_title("Recall Std (%)", fontweight="bold"); ax3.grid(axis="y", alpha=0.3)
    _bar_labels(ax3, bars, "{:.2f}", fontsize=9)

    # 4. Choice distribution
    ax4 = fig.add_subplot(gs[1, :2])
    x = np.arange(4); w = 0.35
    b1 = ax4.bar(x - w/2, [base["choice_percentages"][p] for p in POSITIONS], w,
                  label="Baseline", color=C_BASE, alpha=0.8)
    b2 = ax4.bar(x + w/2, [deb["choice_percentages"][p]  for p in POSITIONS], w,
                  label="After PriDe", color=C_DEB, alpha=0.8)
    ax4.axhline(25, color="red", linestyle="--", linewidth=2)
    ax4.set_xticks(x); ax4.set_xticklabels(POSITIONS)
    ax4.set_title("Choice Distribution", fontweight="bold"); ax4.legend(); ax4.grid(axis="y", alpha=0.3)

    # 5. Accuracy by position
    ax5 = fig.add_subplot(gs[2, :2])
    b_pos = [base["accuracy_by_position"].get(p, 0) * 100 for p in POSITIONS]
    d_pos = [deb["accuracy_by_position"].get(p, 0)  * 100 for p in POSITIONS]
    b1 = ax5.bar(x - w/2, b_pos, w, label="Baseline",    color=C_BASE, alpha=0.8)
    b2 = ax5.bar(x + w/2, d_pos, w, label="After PriDe", color=C_DEB,  alpha=0.8)
    ax5.set_xticks(x); ax5.set_xticklabels(POSITIONS)
    ax5.set_title("Accuracy by Position", fontweight="bold"); ax5.legend(); ax5.grid(axis="y", alpha=0.3)
    _bar_labels(ax5, b1, "{:.1f}%", fontsize=8); _bar_labels(ax5, b2, "{:.1f}%", fontsize=8)

    # 6. Agent accuracy mini-bar (if available)
    ax6 = fig.add_subplot(gs[1, 2])
    if agents:
        ag_names = [k for k in ["agent_1", "agent_2", "agent_3"] if k in agents]
        ag_accs  = [agents[k]["accuracy"] * 100 for k in ag_names]
        ag_cols  = [C_AG1, C_AG2, C_AG3][: len(ag_names)]
        short_labels = ["Ag1\nStep", "Ag2\nTrap", "Ag3\nIntu"][: len(ag_names)]
        bars = ax6.bar(short_labels, ag_accs, color=ag_cols, alpha=0.85)
        ax6.set_title("Per-Agent Accuracy", fontweight="bold"); ax6.grid(axis="y", alpha=0.3)
        ax6.set_ylim(0, 100)
        _bar_labels(ax6, bars, "{:.1f}%", fontsize=9)
    else:
        ax6.axis("off")

    # 7. Summary text
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis("off")
    dacc  = deb["overall_accuracy"] - base["overall_accuracy"]
    dbias = deb["position_bias_score"] - base["position_bias_score"]
    drstd = deb["recall_std"] - base["recall_std"]
    dcons = deb.get("consistency_score", 0) - base.get("consistency_score", 0)
    n_q   = data["n_questions"]
    n_c   = data["n_confident"]

    best_ag = max(agents, key=lambda k: agents[k]["accuracy"]) if agents else "N/A"
    best_ag_acc = agents[best_ag]["accuracy"] if agents else float("nan")

    txt = (
        f"SUMMARY  (alpha={alpha:.2f})\n"
        f"N questions : {n_q}  (conf={n_c})\n\n"
        f"Accuracy\n"
        f"  Before : {base['overall_accuracy']*100:.1f}%\n"
        f"  After  : {deb['overall_accuracy']*100:.1f}%\n"
        f"  Delta  : {dacc*100:+.2f}pp\n\n"
        f"Consistency\n"
        f"  Before : {base.get('consistency_score',0):.1f}%\n"
        f"  After  : {deb.get('consistency_score',0):.1f}%\n"
        f"  Delta  : {dcons:+.2f}pp\n\n"
        f"Bias Score\n"
        f"  Before : {base['position_bias_score']:.2f}\n"
        f"  After  : {deb['position_bias_score']:.2f}\n"
        f"  Delta  : {dbias:+.2f}\n\n"
        f"Recall Std\n"
        f"  Before : {base['recall_std']:.2f}%\n"
        f"  After  : {deb['recall_std']:.2f}%\n"
        f"  Delta  : {drstd:+.2f}%\n\n"
        f"Chi-sq (dist)\n"
        f"  stat: {base['chi2_stat']:.2f}->{deb['chi2_stat']:.2f}\n"
        f"  p   : {base['chi2_pvalue']:.4f}->{deb['chi2_pvalue']:.4f}\n\n"
        f"Best agent : {best_ag} ({best_ag_acc*100:.1f}%)"
    )
    ax7.text(0.05, 0.97, txt, fontsize=8, verticalalignment="top", family="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5), transform=ax7.transAxes)

    plt.savefig(output_path, dpi=300, bbox_inches="tight"); plt.close()


def plot_individual_agent_detail(dataset, model, data, output_path):
    """Per-agent breakdown: accuracy (overall/conf/unconf), choice dist, bias."""
    agents = data.get("agents", {})
    if not agents:
        return

    ag_keys   = [k for k in ["agent_1", "agent_2", "agent_3"] if k in agents]
    ag_labels = {"agent_1": "Agent 1\n(Step-by-step)",
                 "agent_2": "Agent 2\n(Trap-avoid)",
                 "agent_3": "Agent 3\n(Intuitive)"}
    ag_colors = {"agent_1": C_AG1, "agent_2": C_AG2, "agent_3": C_AG3}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{dataset} – {model}: Agent Detail", fontsize=14, fontweight="bold")

    # 1. Overall accuracy
    ax = axes[0, 0]
    bars = ax.bar([ag_labels[k] for k in ag_keys],
                  [agents[k]["accuracy"] * 100 for k in ag_keys],
                  color=[ag_colors[k] for k in ag_keys], alpha=0.85)
    ax.set_ylim(0, 100); ax.set_title("Overall Accuracy", fontweight="bold"); ax.grid(axis="y", alpha=0.3)
    _bar_labels(ax, bars, "{:.1f}%", fontsize=9)

    # 2. Confident vs uncertain
    ax = axes[0, 1]
    x = np.arange(len(ag_keys)); w = 0.35
    c_acc = [agents[k].get("conf_acc",   float("nan")) * 100 for k in ag_keys]
    u_acc = [agents[k].get("unconf_acc", float("nan")) * 100 for k in ag_keys]
    b1 = ax.bar(x - w/2, c_acc, w, label="Confident",  color=C_CONF,   alpha=0.85)
    b2 = ax.bar(x + w/2, u_acc, w, label="Uncertain",  color=C_UNCONF, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([ag_labels[k] for k in ag_keys], fontsize=8)
    ax.set_ylim(0, 110); ax.set_title("Confident vs Uncertain Accuracy", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    _bar_labels(ax, b1, "{:.1f}%", fontsize=8); _bar_labels(ax, b2, "{:.1f}%", fontsize=8)

    # 3. Bias score & Recall Std
    ax = axes[1, 0]
    biases = [agents[k]["bias"]       for k in ag_keys]
    rstds  = [agents[k]["recall_std"] for k in ag_keys]
    b1 = ax.bar(x - w/2, biases, w, label="Bias Score", color="#E76F51", alpha=0.85)
    b2 = ax.bar(x + w/2, rstds,  w, label="Recall Std", color="#457B9D", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([ag_labels[k] for k in ag_keys], fontsize=8)
    ax.set_title("Bias Score & Recall Std (Lower=Better)", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    _bar_labels(ax, b1, "{:.2f}", fontsize=8); _bar_labels(ax, b2, "{:.2f}", fontsize=8)

    # 4. Choice distribution per agent
    ax = axes[1, 1]
    bar_w = 0.25
    xi = np.arange(4)
    for i, k in enumerate(ag_keys):
        pcts = [agents[k]["choice_pcts"].get(p, 0) for p in POSITIONS]
        bars = ax.bar(xi + (i - 1) * bar_w, pcts, bar_w,
                      label=ag_labels[k].replace("\n", " "), color=ag_colors[k], alpha=0.85)
    ax.axhline(25, color="red", linestyle="--", linewidth=2, label="Uniform 25%")
    ax.set_xticks(xi); ax.set_xticklabels(POSITIONS)
    ax.set_title("Choice Distribution per Agent", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(); plt.savefig(output_path, dpi=300, bbox_inches="tight"); plt.close()


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def generate_summary_report(results: dict, output_dir: Path):
    path = output_dir / "SUMMARY_REPORT.txt"
    lines = [
        "=" * 100,
        "MAD-GRAPH SELECTIVE PRIDE  –  BATCH SUMMARY REPORT",
        "=" * 100,
    ]

    for dataset in sorted(results):
        lines += ["", "=" * 100, f"DATASET: {dataset}", "=" * 100]
        header = (f"{'Model':<40} {'a':>4} | "
                  f"{'Acc B':>6} {'Acc A':>6} {'Acc D':>7} | "
                  f"{'Bias B':>6} {'Bias A':>6} | "
                  f"{'RStd B':>6} {'RStd A':>6} | "
                  f"{'Cons B':>6} {'Cons A':>6} | "
                  f"{'Ag1':>6} {'Ag2':>6} {'Ag3':>6}")
        lines += [header, "-" * len(header)]

        for model in _sort_models(results[dataset].keys()):
            d = results[dataset][model]
            b, de = d["baseline"], d["debiased"]
            ags = d.get("agents", {})
            a1 = ags.get("agent_1", {}).get("accuracy", float("nan"))
            a2 = ags.get("agent_2", {}).get("accuracy", float("nan"))
            a3 = ags.get("agent_3", {}).get("accuracy", float("nan"))
            dacc = de["overall_accuracy"] - b["overall_accuracy"]

            lines.append(
                f"{model:<40} {d['best_alpha']:>4.1f} | "
                f"{b['overall_accuracy']*100:>5.1f}% {de['overall_accuracy']*100:>5.1f}% {dacc*100:>+6.2f}pp | "
                f"{b['position_bias_score']:>6.2f} {de['position_bias_score']:>6.2f} | "
                f"{b['recall_std']:>6.2f} {de['recall_std']:>6.2f} | "
                f"{b.get('consistency_score',0):>5.1f}% {de.get('consistency_score',0):>5.1f}% | "
                f"{a1*100:>5.1f}% {a2*100:>5.1f}% {a3*100:>5.1f}%"
            )

        # Agent ranking summary
        lines += ["", "  Agent Accuracy Ranking:"]
        for model in _sort_models(results[dataset].keys()):
            ags = results[dataset][model].get("agents", {})
            if not ags:
                continue
            ranked = sorted(ags.items(), key=lambda kv: kv[1]["accuracy"], reverse=True)
            rank_str = "  >  ".join(
                f"{k.replace('agent_','Ag')} {v['accuracy']*100:.1f}% (bias={v['bias']:.2f})"
                for k, v in ranked
            )
            lines.append(f"    {model}: {rank_str}")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch analysis for MAD-Graph Selective PriDe CSVs.")
    parser.add_argument("--force",        action="store_true",  help="Re-run even if output exists")
    parser.add_argument("--summary-only", action="store_true",  help="Skip plots, only write report")
    parser.add_argument("--dataset",      type=str, default=None, help="Filter to one dataset")
    parser.add_argument("--fixed-alpha",  type=float, default=DEFAULT_FIXED_ALPHA,
                        help=f"Fixed PriDe alpha (default {DEFAULT_FIXED_ALPHA}). Set 0 to grid-search.")
    parser.add_argument("--input-dir",    type=str, default=None,
                        help="Override input directory (default: results/mad_graph_selective/output)")
    parser.add_argument("--output-dir",   type=str, default=None,
                        help="Override output directory (default: pride/results/selective/output)")
    args = parser.parse_args()

    fixed_alpha = args.fixed_alpha if args.fixed_alpha != 0 else None

    sel_dir = Path(args.input_dir) if args.input_dir else SELECTIVE_INPUT_DIR
    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR

    # Also scan legacy directory for backward compat
    csv_dirs = [sel_dir]
    if LEGACY_INPUT_DIR.exists() and sel_dir != LEGACY_INPUT_DIR:
        csv_dirs.append(LEGACY_INPUT_DIR)

    print("\n" + "=" * 70)
    print("  MAD-GRAPH SELECTIVE PriDe  –  BATCH ANALYSIS")
    print("=" * 70)
    print(f"  Input  : {[str(d) for d in csv_dirs]}")
    print(f"  Output : {out_dir}")
    print(f"  Alpha  : {'grid-search' if fixed_alpha is None else fixed_alpha}")

    results = process_all_csvs(csv_dirs, fixed_alpha, args.dataset)

    if not results:
        print("\nNo results found. Run mad_graph_selective_pride_eval.py first.")
        return

    by_dataset_dir    = out_dir / "by_dataset"
    by_model_dir      = out_dir / "by_model_dataset"
    by_dataset_dir.mkdir(parents=True, exist_ok=True)
    by_model_dir.mkdir(parents=True, exist_ok=True)

    if not args.summary_only:
        print("\n  Generating plots...")
        for dataset, models_data in results.items():
            dd = by_dataset_dir / dataset
            dd.mkdir(exist_ok=True)
            print(f"\n  [Dataset] {dataset} ({len(models_data)} model(s))")

            plot_dataset_accuracy_comparison(dataset, models_data, dd / f"{dataset}_accuracy.png")
            plot_dataset_bias_comparison(dataset, models_data,     dd / f"{dataset}_bias.png")
            plot_dataset_distribution_comparison(dataset, models_data, dd / f"{dataset}_distribution.png")
            plot_dataset_accuracy_by_position(dataset, models_data,   dd / f"{dataset}_acc_by_pos.png")
            plot_dataset_consistency_comparison(dataset, models_data,  dd / f"{dataset}_consistency.png")
            plot_dataset_agent_comparison(dataset, models_data,        dd / f"{dataset}_agents.png")

            for model, data in models_data.items():
                md = by_model_dir / f"{dataset}-{model}"
                md.mkdir(exist_ok=True)
                print(f"    [Model] {model}")
                plot_individual_accuracy_comparison(dataset, model, data, md / "accuracy.png")
                plot_individual_bias_metrics(dataset, model, data,        md / "bias_metrics.png")
                plot_individual_distribution(dataset, model, data,        md / "distribution.png")
                plot_individual_summary(dataset, model, data,             md / "summary.png")
                plot_individual_agent_detail(dataset, model, data,        md / "agents.png")

    generate_summary_report(results, out_dir)

    print("\n" + "=" * 70)
    print("  DONE")
    print(f"  by_dataset/     -> {by_dataset_dir}")
    print(f"  by_model_dataset/ -> {by_model_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
