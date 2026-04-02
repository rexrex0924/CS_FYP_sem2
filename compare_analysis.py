"""
Comparative Analysis: Baseline vs Agent vs PriDe
=================================================
Produces three pairwise comparisons per matched (dataset, model) pair:

  Comp 1: Baseline (raw)  vs  Agent (raw)
          Does multi-agent debate improve raw answer quality?

  Comp 2: Baseline (raw)  vs  Agent + PriDe
          Does the full MAD-Graph + PriDe pipeline beat a naive single-call baseline?

  Comp 3: Agent (raw)     vs  Agent + PriDe
          Does PriDe debiasing add value on top of the agent structure?

Note: Baseline vs Baseline+PriDe was already analysed in semester 1.

Every comparison reports ALL statistical tests from pride_batch_summary.py:
  - Overall accuracy (natural: perm=0; cross-perm: all 4 shifts)
  - Accuracy by position (A/B/C/D breakdown)
  - Choice distribution (choice_counts, choice_percentages)
  - Chi-square test on distribution uniformity (chi2_stat, chi2_pvalue)
  - Position bias score (std of choice percentages)
  - Recall Standard Deviation / RStd (from PriDe paper)
  - Chi-square test on accuracy-vs-position independence
  - Consistency score (same content chosen across all 4 permutations)
  - McNemar's test on paired accuracy (significance of accuracy difference)
  - Chi-square between two choice distributions (distributional shift significance)

Input:
  results/baseline/*_baseline.csv
  results/mad_graph_selective/output/*_pride.csv
  (matched by <dataset>-<model> label)

Output:
  pride/results/comparison/
    by_dataset/<dataset>/
      comp1_baseline_vs_agent.png
      comp2_baseline_vs_agent_pride.png
      comp3_agent_vs_agent_pride.png
      overview_all_comparisons.png
    by_model/<dataset>-<model>/
      comprehensive_dashboard.png
      distributions_all_methods.png
      accuracy_by_position.png
      statistical_tests.png
    COMPARISON_REPORT.txt

Usage:
  python compare_analysis.py
  python compare_analysis.py --dataset 2012-2020_ICT_DSE
  python compare_analysis.py --fixed-alpha 0.3
  python compare_analysis.py --summary-only
"""

import argparse
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import chisquare, chi2_contingency, chi2 as chi2_dist

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASELINE_DIR  = Path("results/baseline")
SELECTIVE_DIR = Path("results/mad_graph_selective/output")
OUTPUT_DIR    = Path("pride/results/comparison")
CALIB_RATIO   = 0.15
DEFAULT_ALPHA = 0.3   # set to None to grid-search

POSITIONS = ["A", "B", "C", "D"]

# Colours — consistent with pride_batch_summary palette
C_BASE      = "#FF6B6B"   # red       baseline raw
C_AGENT     = "#5B8FF9"   # blue      agent raw
C_BPRIDE    = "#FFA07A"   # salmon    baseline + PriDe (reference)
C_APRIDE    = "#4ECDC4"   # teal      agent + PriDe
C_DELTA_POS = "#2ECC71"   # green     positive delta
C_DELTA_NEG = "#E74C3C"   # red       negative delta
C_SIG       = "#F39C12"   # amber     significant

METHOD_LABELS = {
    "baseline_raw":   "Baseline (raw)",
    "agent_raw":      "Agent (raw)",
    "agent_pride":    "Agent + PriDe",
}
METHOD_COLORS = {
    "baseline_raw": C_BASE,
    "agent_raw":    C_AGENT,
    "agent_pride":  C_APRIDE,
}

COMP_TITLES = {
    "comp1": "Comp 1: Baseline vs Agent",
    "comp2": "Comp 2: Baseline vs Agent+PriDe",
    "comp3": "Comp 3: Agent vs Agent+PriDe",
}
COMP_A = {"comp1": "baseline_raw", "comp2": "baseline_raw", "comp3": "agent_raw"}
COMP_B = {"comp1": "agent_raw",    "comp2": "agent_pride",  "comp3": "agent_pride"}

MODEL_ORDER = [
    "gemma3_1b", "gemma3_4b", "gemma3_12b",
    "llama3.2_latest", "llama3_8b-instruct-q6_K", "llama3_8b-instruct-q8_0",
    "mistral_latest", "mistral-nemo_latest", "mistral-small3.2_24b", 
    "Qwen_Qwen2.5-1.5B-Instruct_transformers", "Qwen_Qwen2.5-3B-Instruct_transformers", "Qwen_Qwen2.5-7B-Instruct_transformers",
    "microsoft_Phi-3-mini-4k-instruct_transformers", "microsoft_Phi-3.5-mini-instruct_transformers", "microsoft_Phi-4-mini-instruct_transformers"
]


# ---------------------------------------------------------------------------
# PriDe  (self-contained)
# ---------------------------------------------------------------------------

class _PriDe:
    def __init__(self, calib=CALIB_RATIO, alpha=1.0, seed=42):
        self.calib, self.alpha, self.seed = calib, alpha, seed
        self._prior = None

    def _split(self, df):
        ids = df["question_id"].unique()
        rng = np.random.RandomState(self.seed)
        rng.shuffle(ids)
        n = max(1, int(len(ids) * self.calib))
        cal_ids, test_ids = ids[:n], ids[n:]
        return (df[df["question_id"].isin(cal_ids)].copy(),
                df[df["question_id"].isin(test_ids)].copy(),
                test_ids)

    def _fit(self, cal):
        priors = []
        for _, grp in cal.groupby("question_id"):
            grp = grp.sort_values("permutation_idx")
            P = grp[["prob_A", "prob_B", "prob_C", "prob_D"]].values
            P = P / (P.sum(axis=1, keepdims=True) + 1e-10)
            log_mean = np.log(P + 1e-10).mean(axis=0)
            prior = np.exp(log_mean - log_mean.max())
            prior /= prior.sum()
            priors.append(prior)
        self._prior = np.mean(priors, axis=0)

    def _debias(self, test):
        out = test.copy()
        preds, correct = [], []
        for _, row in out.iterrows():
            obs = np.array([row[f"prob_{p}"] for p in POSITIONS])
            logits = np.log(obs + 1e-10) - self.alpha * np.log(self._prior + 1e-10)
            pred = POSITIONS[int(np.argmax(logits))]
            preds.append(pred)
            correct.append(int(pred == str(row["correct_position"]).upper().strip()))
        out["debiased_predicted_answer"] = preds
        out["debiased_is_correct"] = correct
        return out

    def fit_predict(self, df):
        """Returns (test_debiased, test_original, test_question_ids)."""
        cal, test, test_ids = self._split(df)
        self._fit(cal)
        return self._debias(test), test, test_ids


def _run_pride(df, fixed_alpha):
    """
    Grid-search or fixed alpha.
    Returns (test_debiased, test_original, best_alpha, test_qids).
    """
    alphas = [fixed_alpha] if fixed_alpha is not None \
             else np.round(np.arange(0.0, 1.05, 0.1), 1).tolist()

    best_alpha, best_acc = alphas[0], -1.0
    best_deb = best_orig = best_ids = None

    for a in alphas:
        p = _PriDe(calib=CALIB_RATIO, alpha=float(a))
        deb, orig, ids = p.fit_predict(df)
        acc = deb["debiased_is_correct"].mean()
        if acc > best_acc:
            best_acc, best_alpha = acc, a
            best_deb, best_orig, best_ids = deb, orig, ids

    return best_deb, best_orig, best_alpha, best_ids


# ---------------------------------------------------------------------------
# Metric computation  (mirrors pride_batch_summary.compute_bias_metrics exactly)
# ---------------------------------------------------------------------------

def _compute_metrics(df, pred_col="predicted_answer"):
    """Full metric suite: accuracy, bias, recall_std, chi-square, consistency."""
    d = df.copy()
    d[pred_col] = d[pred_col].astype(str).str.upper().str.strip()
    d["correct_position"] = d["correct_position"].astype(str).str.upper().str.strip()

    valid = d[d[pred_col].isin(POSITIONS)].copy()
    valid["is_correct_eval"] = (valid[pred_col] == valid["correct_position"]).astype(int)

    counts = valid[pred_col].value_counts().reindex(POSITIONS, fill_value=0)
    total = len(valid)
    pcts = (counts / total * 100) if total > 0 else counts * 0.0

    # Chi-square: distribution uniformity
    try:
        chi2_stat, chi2_p = chisquare(counts.values, f_exp=[total / 4] * 4)
    except Exception:
        chi2_stat, chi2_p = 0.0, 1.0

    # Position bias score
    bias_score = float(np.std(pcts.values))

    # Recall Std (RStd — from PriDe paper)
    recalls = []
    for pos in POSITIONS:
        m = valid["correct_position"] == pos
        recalls.append(float((valid.loc[m, pred_col] == pos).mean()) if m.sum() > 0 else 0.0)
    recall_std = float(np.std(recalls) * 100)

    # Accuracy by position
    acc_by_pos = valid.groupby("correct_position")["is_correct_eval"].mean().to_dict()
    overall_acc = float(valid["is_correct_eval"].mean()) if total > 0 else 0.0

    # Chi-square: accuracy vs position independence
    try:
        ct = pd.crosstab(valid["correct_position"], valid["is_correct_eval"])
        chi2_acc, p_acc, _, _ = chi2_contingency(ct)
    except Exception:
        chi2_acc, p_acc = 0.0, 1.0

    # Consistency score (same content chosen across all permutations)
    consistency_score = 0.0
    id_col = "question_id" if "question_id" in valid.columns else "id"
    if id_col in valid.columns and "permutation_idx" in valid.columns:
        def _orig(row):
            try:
                pred = str(row[pred_col]).upper().strip()
                if pred not in POSITIONS:
                    return None
                shift = int(row["permutation_idx"]) % 4
                return POSITIONS[(shift + POSITIONS.index(pred)) % 4]
            except Exception:
                return None
        valid["_orig_choice"] = valid.apply(_orig, axis=1)
        vv = valid.dropna(subset=["_orig_choice"])
        if not vv.empty:
            uniq = vv.groupby(id_col)["_orig_choice"].nunique()
            consistency_score = float((uniq == 1).mean() * 100)

    return {
        "overall_accuracy":     overall_acc,
        "accuracy_by_position": acc_by_pos,
        "choice_counts":        counts.to_dict(),
        "choice_percentages":   pcts.to_dict(),
        "chi2_stat":            float(chi2_stat),
        "chi2_pvalue":          float(chi2_p),
        "position_bias_score":  bias_score,
        "recall_std":           recall_std,
        "recalls":              {p: recalls[i] for i, p in enumerate(POSITIONS)},
        "chi2_acc_stat":        float(chi2_acc),
        "chi2_acc_pvalue":      float(p_acc),
        "consistency_score":    consistency_score,
        "n_samples":            total,
    }


def _natural_accuracy(df_perm0, pred_col="predicted_answer"):
    """Accuracy and per-position accuracy on permutation_idx=0 only."""
    d = df_perm0.copy()
    d[pred_col] = d[pred_col].astype(str).str.upper().str.strip()
    d["correct_position"] = d["correct_position"].astype(str).str.upper().str.strip()
    valid = d[d[pred_col].isin(POSITIONS)].copy()
    valid["is_correct"] = (valid[pred_col] == valid["correct_position"]).astype(int)
    overall = float(valid["is_correct"].mean()) if len(valid) > 0 else 0.0
    by_pos = valid.groupby("correct_position")["is_correct"].mean().to_dict()
    correct_vec = valid.set_index("question_id")["is_correct"] if "question_id" in valid.columns else None
    return overall, by_pos, correct_vec


# ---------------------------------------------------------------------------
# Statistical comparison helpers
# ---------------------------------------------------------------------------

def _mcnemar(correct_A: pd.Series, correct_B: pd.Series):
    """
    McNemar's test for paired binary outcomes (with continuity correction).
    Returns (statistic, p_value, contingency_dict).
    """
    df = pd.DataFrame({"A": correct_A, "B": correct_B}).dropna()
    if df.empty:
        return float("nan"), float("nan"), {}

    both  = int(((df.A == 1) & (df.B == 1)).sum())
    A_only = int(((df.A == 1) & (df.B == 0)).sum())
    B_only = int(((df.A == 0) & (df.B == 1)).sum())
    neither = int(((df.A == 0) & (df.B == 0)).sum())

    b, c = B_only, A_only
    ct = {"both_correct": both, "A_only_correct": A_only,
          "B_only_correct": B_only, "neither_correct": neither}

    if b + c == 0:
        return 0.0, 1.0, ct

    stat = (abs(b - c) - 1) ** 2 / (b + c)   # with continuity correction
    p = float(1 - chi2_dist.cdf(stat, df=1))
    return float(stat), p, ct


def _chi2_between_distributions(counts_A: dict, counts_B: dict):
    """
    Chi-square test asking: are the two choice distributions significantly different?
    Rescales B to the same total as A, then runs chisquare(A, f_exp=B_scaled).
    """
    a = np.array([counts_A.get(p, 0) for p in POSITIONS], dtype=float)
    b = np.array([counts_B.get(p, 0) for p in POSITIONS], dtype=float)
    total_a, total_b = a.sum(), b.sum()
    if total_a == 0 or total_b == 0:
        return float("nan"), float("nan")
    b_scaled = b / total_b * total_a
    b_scaled = np.where(b_scaled == 0, 0.001, b_scaled)  # avoid div-by-zero
    try:
        stat, p = chisquare(a, f_exp=b_scaled)
        return float(stat), float(p)
    except Exception:
        return float("nan"), float("nan")


def _compare(m_A, m_B, nat_A, nat_B, corr_A, corr_B):
    """
    Build a comparison dict from two metric dicts.
    corr_A / corr_B: pd.Series indexed by question_id (perm=0 binary correctness).
    """
    # Align on common questions for McNemar
    if corr_A is not None and corr_B is not None:
        common = corr_A.index.intersection(corr_B.index)
        mcn_stat, mcn_p, mcn_ct = _mcnemar(corr_A.loc[common], corr_B.loc[common])
    else:
        mcn_stat, mcn_p, mcn_ct = float("nan"), float("nan"), {}

    dist_stat, dist_p = _chi2_between_distributions(
        m_A["choice_counts"], m_B["choice_counts"]
    )

    return {
        # Natural accuracy deltas
        "nat_acc_A":       nat_A,
        "nat_acc_B":       nat_B,
        "delta_nat_acc":   nat_B - nat_A,
        # Cross-perm accuracy deltas
        "acc_A":           m_A["overall_accuracy"],
        "acc_B":           m_B["overall_accuracy"],
        "delta_acc":       m_B["overall_accuracy"] - m_A["overall_accuracy"],
        # Bias
        "bias_A":          m_A["position_bias_score"],
        "bias_B":          m_B["position_bias_score"],
        "delta_bias":      m_B["position_bias_score"] - m_A["position_bias_score"],
        # Recall Std
        "rstd_A":          m_A["recall_std"],
        "rstd_B":          m_B["recall_std"],
        "delta_rstd":      m_B["recall_std"] - m_A["recall_std"],
        # Consistency
        "cons_A":          m_A["consistency_score"],
        "cons_B":          m_B["consistency_score"],
        "delta_cons":      m_B["consistency_score"] - m_A["consistency_score"],
        # Chi-square: distribution uniformity
        "chi2_stat_A":     m_A["chi2_stat"],
        "chi2_p_A":        m_A["chi2_pvalue"],
        "chi2_stat_B":     m_B["chi2_stat"],
        "chi2_p_B":        m_B["chi2_pvalue"],
        "delta_chi2_stat": m_B["chi2_stat"] - m_A["chi2_stat"],
        "delta_chi2_p":    m_B["chi2_pvalue"] - m_A["chi2_pvalue"],
        # Chi-square: accuracy vs position
        "chi2_acc_stat_A": m_A["chi2_acc_stat"],
        "chi2_acc_p_A":    m_A["chi2_acc_pvalue"],
        "chi2_acc_stat_B": m_B["chi2_acc_stat"],
        "chi2_acc_p_B":    m_B["chi2_acc_pvalue"],
        # McNemar test (paired accuracy significance)
        "mcnemar_stat":    mcn_stat,
        "mcnemar_p":       mcn_p,
        "mcnemar_ct":      mcn_ct,
        "sig_accuracy":    (mcn_p < 0.05) if not np.isnan(mcn_p) else False,
        # Distribution shift significance
        "dist_chi2_stat":  dist_stat,
        "dist_chi2_p":     dist_p,
        "sig_distribution": (dist_p < 0.05) if not np.isnan(dist_p) else False,
    }


# ---------------------------------------------------------------------------
# Data loading & pairing
# ---------------------------------------------------------------------------

def _load(path):
    df = pd.read_csv(path)
    for p in POSITIONS:
        col = f"prob_{p}"
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def _parse_label(path: Path):
    name = path.stem.replace("_baseline", "").replace("_pride", "")
    for prefix in ["ministral", "mistral", "gemma", "llama", "microsoft_Phi", "Qwen"]:
        if prefix in name:
            parts = name.split(prefix, 1)
            return parts[0].strip("-"), prefix + parts[1]
    return None, None


def _sort_models(models):
    def key(m):
        try:
            return MODEL_ORDER.index(m)
        except ValueError:
            return len(MODEL_ORDER)
    return sorted(models, key=key)


def _find_pairs(dataset_filter):
    """Return list of (label, baseline_path, selective_path)."""
    baseline_map = {}
    for f in BASELINE_DIR.glob("*_baseline.csv"):
        ds, mdl = _parse_label(f)
        if ds and mdl:
            baseline_map[f"{ds}-{mdl}"] = f

    pairs = []
    for f in SELECTIVE_DIR.glob("*_pride.csv"):
        if "deprecated" in str(f):
            continue
        ds, mdl = _parse_label(f)
        if not ds or not mdl:
            continue
        label = f"{ds}-{mdl}"
        if dataset_filter and dataset_filter not in label:
            continue
        if label in baseline_map:
            pairs.append((ds, mdl, label, baseline_map[label], f))
        else:
            print(f"  [skip] No matching baseline for {f.name}")
    return pairs


# ---------------------------------------------------------------------------
# Per-pair processing
# ---------------------------------------------------------------------------

def process_pair(ds, mdl, baseline_path, selective_path, fixed_alpha):
    """
    Compute all 4 condition metrics + 3 comparisons for one (dataset, model) pair.
    """
    bl = _load(baseline_path)
    sl = _load(selective_path)

    # ---- PriDe runs (same seed -> same test/calibration split on same question_ids)
    bl_deb, bl_orig, bl_alpha, bl_test_ids = _run_pride(bl, fixed_alpha)
    sl_deb, sl_orig, sl_alpha, sl_test_ids = _run_pride(sl, fixed_alpha)

    # ---- Full-dataset metrics (all 4 permutations)
    m_base_raw   = _compute_metrics(bl,     "predicted_answer")
    m_agent_raw  = _compute_metrics(sl,     "predicted_answer")
    m_agent_deb  = _compute_metrics(sl_deb, "debiased_predicted_answer")

    # ---- Natural accuracy (permutation_idx = 0 only)
    def _p0(df, col): return df[df["permutation_idx"] == 0] if "permutation_idx" in df.columns else df
    nat_base,  nat_base_bypos,  corr_base  = _natural_accuracy(_p0(bl,     "x"),     "predicted_answer")
    nat_agent, nat_agent_bypos, corr_agent = _natural_accuracy(_p0(sl,     "x"),     "predicted_answer")
    nat_adeb,  nat_adeb_bypos,  corr_adeb  = _natural_accuracy(_p0(sl_deb, "x"),     "debiased_predicted_answer")

    # Recompute properly (the lambda above breaks col – use direct filter)
    def _nat(df, col):
        sub = df[df["permutation_idx"] == 0].copy() if "permutation_idx" in df.columns else df.copy()
        return _natural_accuracy(sub, col)

    nat_base,  nat_base_bypos,  corr_base  = _nat(bl,     "predicted_answer")
    nat_agent, nat_agent_bypos, corr_agent = _nat(sl,     "predicted_answer")
    nat_adeb,  nat_adeb_bypos,  corr_adeb  = _nat(sl_deb, "debiased_predicted_answer")

    # ---- Build 3 comparisons
    comp1 = _compare(m_base_raw,  m_agent_raw, nat_base,  nat_agent, corr_base,  corr_agent)
    comp2 = _compare(m_base_raw,  m_agent_deb, nat_base,  nat_adeb,  corr_base,  corr_adeb)
    comp3 = _compare(m_agent_raw, m_agent_deb, nat_agent, nat_adeb,  corr_agent, corr_adeb)

    return {
        # Raw metric dicts for all methods
        "baseline_raw":  m_base_raw,
        "agent_raw":     m_agent_raw,
        "agent_pride":   m_agent_deb,
        # Natural accuracies + per-position breakdown
        "nat_acc": {
            "baseline_raw": nat_base,
            "agent_raw":    nat_agent,
            "agent_pride":  nat_adeb,
        },
        "nat_acc_bypos": {
            "baseline_raw": nat_base_bypos,
            "agent_raw":    nat_agent_bypos,
            "agent_pride":  nat_adeb_bypos,
        },
        # PriDe alphas
        "alpha_baseline": bl_alpha,
        "alpha_agent":    sl_alpha,
        # Comparisons
        "comp1": comp1,   # baseline_raw vs agent_raw
        "comp2": comp2,   # baseline_raw vs agent_pride
        "comp3": comp3,   # agent_raw    vs agent_pride
        # Metadata
        "n_questions": int(sl["question_id"].nunique()),
        "n_confident": int(sl[sl["permutation_idx"] == 0]["confident"].sum())
                       if "confident" in sl.columns else 0,
    }


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _bar_labels(ax, bars, fmt="{:.2f}", fontsize=7):
    for bar in bars:
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, h,
                fmt.format(h), ha="center", va=va, fontsize=fontsize, fontweight="bold")


def _sig_star(p):
    if np.isnan(p):  return "n/a"
    if p < 0.001:    return "***"
    if p < 0.01:     return "**"
    if p < 0.05:     return "*"
    return "ns"


# ---------------------------------------------------------------------------
# BY-DATASET: one plot per comparison
# ---------------------------------------------------------------------------

def _plot_comp_panel(ax_acc, ax_bias, ax_rstd, ax_cons,
                     dataset, models, models_data, comp_key, title):
    """Draw four sub-axes for one comparison: accuracy, bias, rstd, consistency."""
    x = np.arange(len(models)); w = 0.35

    key_A = COMP_A[comp_key]; key_B = COMP_B[comp_key]
    col_A = METHOD_COLORS[key_A]; col_B = METHOD_COLORS[key_B]
    lbl_A = METHOD_LABELS[key_A]; lbl_B = METHOD_LABELS[key_B]

    # --- Accuracy ---
    acc_A = [models_data[m]["nat_acc"][key_A]  for m in models]
    acc_B = [models_data[m]["nat_acc"][key_B]  for m in models]
    b1 = ax_acc.bar(x - w/2, acc_A, w, label=lbl_A, color=col_A, alpha=0.85)
    b2 = ax_acc.bar(x + w/2, acc_B, w, label=lbl_B, color=col_B, alpha=0.85)
    _bar_labels(ax_acc, b1, "{:.1%}"); _bar_labels(ax_acc, b2, "{:.1%}")
    ax_acc.set_xticks(x); ax_acc.set_xticklabels(models, rotation=45, ha="right", fontsize=7)
    ax_acc.set_title(f"{title}\nNatural Accuracy (perm=0)", fontweight="bold", fontsize=9)
    ax_acc.legend(fontsize=7); ax_acc.grid(axis="y", alpha=0.3)

    # significance stars above pairs
    for i, m in enumerate(models):
        p = models_data[m][comp_key]["mcnemar_p"]
        star = _sig_star(p)
        if star != "ns":
            ax_acc.text(x[i], max(acc_A[i], acc_B[i]) + 0.02, star,
                        ha="center", fontsize=8, color=C_SIG, fontweight="bold")

    # --- Bias ---
    bias_A = [models_data[m]["baseline_raw" if key_A == "baseline_raw" else key_A]["position_bias_score"] for m in models]
    bias_B = [models_data[m][key_B]["position_bias_score"] for m in models]
    b3 = ax_bias.bar(x - w/2, bias_A, w, label=lbl_A, color=col_A, alpha=0.85)
    b4 = ax_bias.bar(x + w/2, bias_B, w, label=lbl_B, color=col_B, alpha=0.85)
    _bar_labels(ax_bias, b3, "{:.2f}"); _bar_labels(ax_bias, b4, "{:.2f}")
    ax_bias.set_xticks(x); ax_bias.set_xticklabels(models, rotation=45, ha="right", fontsize=7)
    ax_bias.set_title("Position Bias Score (lower=better)", fontweight="bold", fontsize=9)
    ax_bias.legend(fontsize=7); ax_bias.grid(axis="y", alpha=0.3)

    # --- RStd ---
    rstd_A = [models_data[m]["baseline_raw" if key_A == "baseline_raw" else key_A]["recall_std"] for m in models]
    rstd_B = [models_data[m][key_B]["recall_std"] for m in models]
    b5 = ax_rstd.bar(x - w/2, rstd_A, w, label=lbl_A, color=col_A, alpha=0.85)
    b6 = ax_rstd.bar(x + w/2, rstd_B, w, label=lbl_B, color=col_B, alpha=0.85)
    _bar_labels(ax_rstd, b5, "{:.2f}"); _bar_labels(ax_rstd, b6, "{:.2f}")
    ax_rstd.set_xticks(x); ax_rstd.set_xticklabels(models, rotation=45, ha="right", fontsize=7)
    ax_rstd.set_title("Recall Std % (lower=better)", fontweight="bold", fontsize=9)
    ax_rstd.legend(fontsize=7); ax_rstd.grid(axis="y", alpha=0.3)

    # --- Consistency ---
    cons_A = [models_data[m]["baseline_raw" if key_A == "baseline_raw" else key_A]["consistency_score"] for m in models]
    cons_B = [models_data[m][key_B]["consistency_score"] for m in models]
    b7 = ax_cons.bar(x - w/2, cons_A, w, label=lbl_A, color=col_A, alpha=0.85)
    b8 = ax_cons.bar(x + w/2, cons_B, w, label=lbl_B, color=col_B, alpha=0.85)
    _bar_labels(ax_cons, b7, "{:.1f}%"); _bar_labels(ax_cons, b8, "{:.1f}%")
    ax_cons.set_xticks(x); ax_cons.set_xticklabels(models, rotation=45, ha="right", fontsize=7)
    ax_cons.set_title("Consistency Score % (higher=better)", fontweight="bold", fontsize=9)
    ax_cons.legend(fontsize=7); ax_cons.grid(axis="y", alpha=0.3)


def plot_dataset_single_comparison(dataset, models_data, comp_key, output_path):
    """
    6-panel figure for one comparison across all models:
    Row 1: accuracy + delta | Row 2: bias + chi2 | Row 3: rstd + consistency
    """
    models = _sort_models(models_data.keys())
    key_A = COMP_A[comp_key]; key_B = COMP_B[comp_key]
    col_A = METHOD_COLORS[key_A]; col_B = METHOD_COLORS[key_B]
    lbl_A = METHOD_LABELS[key_A]; lbl_B = METHOD_LABELS[key_B]
    title = COMP_TITLES[comp_key]
    x = np.arange(len(models)); w = 0.35

    fig, axes = plt.subplots(3, 2, figsize=(18, 16))
    fig.suptitle(f"{dataset}  |  {title}", fontsize=14, fontweight="bold")

    def _vals(metric_key, subkey, source="metrics"):
        if source == "metrics":
            return [models_data[m][metric_key][subkey] for m in models]
        return [models_data[m][comp_key][subkey] for m in models]

    # --- Row 0: Natural accuracy (absolute + delta) ---
    acc_A = [models_data[m]["nat_acc"][key_A] for m in models]
    acc_B = [models_data[m]["nat_acc"][key_B] for m in models]
    ax = axes[0, 0]
    b1 = ax.bar(x - w/2, acc_A, w, label=lbl_A, color=col_A, alpha=0.85)
    b2 = ax.bar(x + w/2, acc_B, w, label=lbl_B, color=col_B, alpha=0.85)
    _bar_labels(ax, b1, "{:.1%}"); _bar_labels(ax, b2, "{:.1%}")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.set_title("Natural Accuracy (perm=0)", fontweight="bold"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    for i, m in enumerate(models):
        p = models_data[m][comp_key]["mcnemar_p"]
        ax.text(x[i], max(acc_A[i], acc_B[i]) + 0.015, _sig_star(p),
                ha="center", fontsize=9, color=C_SIG, fontweight="bold")

    deltas_acc = [b - a for a, b in zip(acc_A, acc_B)]
    ax = axes[0, 1]
    bars = ax.bar(x, [d * 100 for d in deltas_acc],
                  color=[C_DELTA_POS if d >= 0 else C_DELTA_NEG for d in deltas_acc], alpha=0.85)
    ax.axhline(0, color="black", linewidth=1)
    _bar_labels(ax, bars, "{:+.2f}pp", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.set_title("Accuracy Delta (pp)  +/- = better/worse for B", fontweight="bold"); ax.grid(axis="y", alpha=0.3)

    # --- Row 1: Position bias score + Chi-square distribution test ---
    bias_A = _vals(key_A, "position_bias_score")
    bias_B = _vals(key_B, "position_bias_score")
    ax = axes[1, 0]
    b1 = ax.bar(x - w/2, bias_A, w, label=lbl_A, color=col_A, alpha=0.85)
    b2 = ax.bar(x + w/2, bias_B, w, label=lbl_B, color=col_B, alpha=0.85)
    _bar_labels(ax, b1, "{:.2f}"); _bar_labels(ax, b2, "{:.2f}")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.set_title("Position Bias Score (lower=better)", fontweight="bold"); ax.legend(); ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 1]
    chi2_A = _vals(key_A, "chi2_stat")
    chi2_B = _vals(key_B, "chi2_stat")
    b1 = ax.bar(x - w/2, chi2_A, w, label=f"Chi2: {lbl_A}", color=col_A, alpha=0.85)
    b2 = ax.bar(x + w/2, chi2_B, w, label=f"Chi2: {lbl_B}", color=col_B, alpha=0.85)
    _bar_labels(ax, b1, "{:.2f}"); _bar_labels(ax, b2, "{:.2f}")
    # p-value annotation
    p_A = _vals(key_A, "chi2_pvalue")
    p_B = _vals(key_B, "chi2_pvalue")
    for i in range(len(models)):
        ax.text(x[i] - w/2, chi2_A[i] + 0.1, f"p={p_A[i]:.3f}", ha="center", fontsize=6, color="grey")
        ax.text(x[i] + w/2, chi2_B[i] + 0.1, f"p={p_B[i]:.3f}", ha="center", fontsize=6, color="grey")
    ax.axhline(3.84, color="red", linestyle="--", linewidth=1.2, label="chi2 crit (p=0.05)")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.set_title("Distribution Chi-sq (lower=more uniform; red=5% threshold)", fontweight="bold")
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)

    # --- Row 2: Recall Std + Consistency ---
    rstd_A = _vals(key_A, "recall_std")
    rstd_B = _vals(key_B, "recall_std")
    ax = axes[2, 0]
    b1 = ax.bar(x - w/2, rstd_A, w, label=lbl_A, color=col_A, alpha=0.85)
    b2 = ax.bar(x + w/2, rstd_B, w, label=lbl_B, color=col_B, alpha=0.85)
    _bar_labels(ax, b1, "{:.2f}"); _bar_labels(ax, b2, "{:.2f}")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.set_title("Recall Std % (lower=better)", fontweight="bold"); ax.legend(); ax.grid(axis="y", alpha=0.3)

    cons_A = _vals(key_A, "consistency_score")
    cons_B = _vals(key_B, "consistency_score")
    ax = axes[2, 1]
    b1 = ax.bar(x - w/2, cons_A, w, label=lbl_A, color=col_A, alpha=0.85)
    b2 = ax.bar(x + w/2, cons_B, w, label=lbl_B, color=col_B, alpha=0.85)
    _bar_labels(ax, b1, "{:.1f}%"); _bar_labels(ax, b2, "{:.1f}%")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.set_title("Consistency Score % (higher=better)", fontweight="bold"); ax.legend(); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_dataset_distribution_comparison(dataset, models_data, comp_key, output_path):
    """Choice distribution for A vs B across all models for one comparison."""
    models = _sort_models(models_data.keys())
    key_A = COMP_A[comp_key]; key_B = COMP_B[comp_key]
    lbl_A = METHOD_LABELS[key_A]; lbl_B = METHOD_LABELS[key_B]
    col_A = METHOD_COLORS[key_A]; col_B = METHOD_COLORS[key_B]
    bar_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"]

    n = len(models)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle(f"{dataset}  |  {COMP_TITLES[comp_key]}  — Choice Distributions",
                 fontsize=13, fontweight="bold")

    for i, m in enumerate(models):
        for col, key, lbl in [(0, key_A, lbl_A), (1, key_B, lbl_B)]:
            ax = axes[i, col]
            pcts = [models_data[m][key]["choice_percentages"].get(p, 0) for p in POSITIONS]
            bars = ax.bar(POSITIONS, pcts, color=bar_colors, alpha=0.85)
            ax.axhline(25, color="red", linestyle="--", linewidth=1.5, label="Uniform 25%")
            bias = models_data[m][key]["position_bias_score"]
            chi2s = models_data[m][key]["chi2_stat"]
            chi2p = models_data[m][key]["chi2_pvalue"]
            ax.set_title(f"{m}\n{lbl}  bias={bias:.2f}  chi2={chi2s:.2f} (p={chi2p:.3f})",
                         fontsize=8, fontweight="bold")
            ax.set_ylim(0, max(pcts + [26]) * 1.3); ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
            _bar_labels(ax, bars, "{:.1f}%", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_dataset_accuracy_by_position(dataset, models_data, comp_key, output_path):
    """Accuracy by position for A vs B across all models."""
    models = _sort_models(models_data.keys())
    key_A = COMP_A[comp_key]; key_B = COMP_B[comp_key]
    lbl_A = METHOD_LABELS[key_A]; lbl_B = METHOD_LABELS[key_B]
    col_A = METHOD_COLORS[key_A]; col_B = METHOD_COLORS[key_B]

    n = len(models)
    fig, axes = plt.subplots(n, 1, figsize=(14, 5 * n))
    if n == 1:
        axes = [axes]
    fig.suptitle(f"{dataset}  |  {COMP_TITLES[comp_key]}  — Accuracy by Position",
                 fontsize=13, fontweight="bold")

    x = np.arange(4); w = 0.35
    for i, m in enumerate(models):
        ax = axes[i]
        acc_A = [models_data[m]["nat_acc_bypos"][key_A].get(p, 0) for p in POSITIONS]
        acc_B = [models_data[m]["nat_acc_bypos"][key_B].get(p, 0) for p in POSITIONS]
        b1 = ax.bar(x - w/2, acc_A, w, label=lbl_A, color=col_A, alpha=0.85)
        b2 = ax.bar(x + w/2, acc_B, w, label=lbl_B, color=col_B, alpha=0.85)
        ax.axhline(models_data[m]["nat_acc"][key_A], color=col_A, linestyle="--", linewidth=1.2,
                   label=f"{lbl_A} avg")
        ax.axhline(models_data[m]["nat_acc"][key_B], color=col_B, linestyle="--", linewidth=1.2,
                   label=f"{lbl_B} avg")
        _bar_labels(ax, b1, "{:.1%}", fontsize=8); _bar_labels(ax, b2, "{:.1%}", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(POSITIONS)
        ax.set_title(f"{m}", fontweight="bold"); ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_dataset_statistical_tests(dataset, models_data, output_path):
    """
    4-panel plot showing all statistical test results for all 3 comparisons:
    McNemar p-values, distribution shift p-values, chi2 stats, acc-vs-pos chi2.
    """
    models = _sort_models(models_data.keys())
    x = np.arange(len(models)); w = 0.25
    comps = ["comp1", "comp2", "comp3"]
    comp_colors = [C_BASE, C_AGENT, C_APRIDE]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f"{dataset}  —  Statistical Test Summary (all 3 comparisons)",
                 fontsize=14, fontweight="bold")

    # Panel 1: McNemar p-values (accuracy significance)
    ax = axes[0, 0]
    for i, (ck, cc) in enumerate(zip(comps, comp_colors)):
        pvals = [models_data[m][ck]["mcnemar_p"] for m in models]
        pvals_clipped = [min(v, 1.0) if not np.isnan(v) else 1.0 for v in pvals]
        ax.bar(x + (i-1)*w, pvals_clipped, w, label=COMP_TITLES[ck], color=cc, alpha=0.8)
    ax.axhline(0.05, color="red", linestyle="--", linewidth=2, label="p=0.05")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("McNemar p-value (accuracy significance)")
    ax.set_title("McNemar Test: Is Accuracy Difference Significant?\n(below red line = significant)", fontweight="bold")
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)

    # Panel 2: Distribution shift chi2 p-values
    ax = axes[0, 1]
    for i, (ck, cc) in enumerate(zip(comps, comp_colors)):
        pvals = [models_data[m][ck]["dist_chi2_p"] for m in models]
        pvals_clipped = [min(v, 1.0) if not np.isnan(v) else 1.0 for v in pvals]
        ax.bar(x + (i-1)*w, pvals_clipped, w, label=COMP_TITLES[ck], color=cc, alpha=0.8)
    ax.axhline(0.05, color="red", linestyle="--", linewidth=2, label="p=0.05")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Chi-square p-value")
    ax.set_title("Distribution Shift Test: Are Choice Distributions Significantly Different?\n(below red line = significant shift)", fontweight="bold")
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)

    # Panel 3: Chi-square uniformity stats for all methods (showing improvement)
    ax = axes[1, 0]
    w2 = 0.22
    all_methods = ["baseline_raw", "agent_raw", "agent_pride"]
    all_cols = [C_BASE, C_AGENT, C_APRIDE]
    all_lbls = [METHOD_LABELS[k] for k in all_methods]
    for i, (mk, mc, ml) in enumerate(zip(all_methods, all_cols, all_lbls)):
        chi2s = [models_data[m][mk]["chi2_stat"] for m in models]
        ax.bar(x + (i-1.5)*w2, chi2s, w2, label=ml, color=mc, alpha=0.8)
    ax.axhline(3.84, color="red", linestyle="--", linewidth=1.5, label="chi2 crit (p<0.05)")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Chi-square statistic")
    ax.set_title("Distribution Uniformity Chi-sq for All Methods\n(lower = more uniform, less biased)", fontweight="bold")
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)

    # Panel 4: Accuracy-vs-Position chi2 stats (are some positions harder?)
    ax = axes[1, 1]
    for i, (mk, mc, ml) in enumerate(zip(all_methods, all_cols, all_lbls)):
        chi2s = [models_data[m][mk]["chi2_acc_stat"] for m in models]
        p_acc = [models_data[m][mk]["chi2_acc_pvalue"] for m in models]
        bars = ax.bar(x + (i-1.5)*w2, chi2s, w2, label=ml, color=mc, alpha=0.8)
        for j, (bar, p) in enumerate(zip(bars, p_acc)):
            if p < 0.05:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        "*", ha="center", fontsize=9, color=C_SIG, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Chi-square statistic")
    ax.set_title("Accuracy-vs-Position Chi-sq for All Methods\n(* = p<0.05: accuracy differs significantly by position)", fontweight="bold")
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_dataset_overview(dataset, models_data, output_path):
    """
    Compact 3×3 overview: each row = one comparison, each col = accuracy/bias/rstd.
    Shows all 3 comparisons side by side for a high-level picture.
    """
    models = _sort_models(models_data.keys())
    x = np.arange(len(models)); w = 0.35
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle(f"{dataset}  —  All Comparisons Overview", fontsize=15, fontweight="bold")

    metrics = [
        ("nat_acc",            "Natural Accuracy (perm=0)", "{:.1%}"),
        ("position_bias_score","Position Bias Score",       "{:.2f}"),
        ("recall_std",         "Recall Std %",              "{:.2f}"),
    ]

    for row, comp_key in enumerate(["comp1", "comp2", "comp3"]):
        key_A = COMP_A[comp_key]; key_B = COMP_B[comp_key]
        col_A = METHOD_COLORS[key_A]; col_B = METHOD_COLORS[key_B]
        lbl_A = METHOD_LABELS[key_A]; lbl_B = METHOD_LABELS[key_B]

        for col, (metric, label, fmt) in enumerate(metrics):
            ax = axes[row, col]
            if metric == "nat_acc":
                vals_A = [models_data[m]["nat_acc"][key_A] for m in models]
                vals_B = [models_data[m]["nat_acc"][key_B] for m in models]
            else:
                vals_A = [models_data[m][key_A][metric] for m in models]
                vals_B = [models_data[m][key_B][metric] for m in models]

            b1 = ax.bar(x - w/2, vals_A, w, label=lbl_A, color=col_A, alpha=0.85)
            b2 = ax.bar(x + w/2, vals_B, w, label=lbl_B, color=col_B, alpha=0.85)
            _bar_labels(ax, b1, fmt, fontsize=6)
            _bar_labels(ax, b2, fmt, fontsize=6)
            ax.set_xticks(x); ax.set_xticklabels(models, rotation=45, ha="right", fontsize=6)
            comp_short = ["Baseline vs Agent", "Baseline vs Agent+PriDe", "Agent vs Agent+PriDe"][row]
            ax.set_title(f"{comp_short}\n{label}", fontsize=8, fontweight="bold")
            ax.legend(fontsize=6); ax.grid(axis="y", alpha=0.3)

            if metric == "nat_acc":
                for i, m in enumerate(models):
                    p = models_data[m][comp_key]["mcnemar_p"]
                    star = _sig_star(p)
                    if star not in ("ns", "n/a"):
                        ax.text(x[i], max(vals_A[i], vals_B[i]) + 0.01, star,
                                ha="center", fontsize=7, color=C_SIG, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# BY-MODEL: comprehensive individual dashboards
# ---------------------------------------------------------------------------

def plot_model_comprehensive(dataset, model, data, output_path):
    """
    Large dashboard for one (dataset, model) pair.
    Rows: comp1 / comp2 / comp3
    Cols: accuracy delta | bias delta | rstd delta | consistency delta | McNemar/chi2 stats
    """
    comps = ["comp1", "comp2", "comp3"]
    comp_labels = ["Baseline vs Agent", "Baseline vs Agent+PriDe", "Agent vs Agent+PriDe"]

    fig, axes = plt.subplots(3, 5, figsize=(24, 14))
    fig.suptitle(f"{dataset}  —  {model}  |  Comprehensive Comparison Dashboard",
                 fontsize=14, fontweight="bold")

    cats_2 = ["A", "B"]  # generic A vs B labels per comparison

    for row, (ck, clabel) in enumerate(zip(comps, comp_labels)):
        c = data[ck]
        key_A = COMP_A[ck]; key_B = COMP_B[ck]
        lbl_A = METHOD_LABELS[key_A]; lbl_B = METHOD_LABELS[key_B]
        col_A = METHOD_COLORS[key_A]; col_B = METHOD_COLORS[key_B]

        # Col 0: Natural accuracy
        ax = axes[row, 0]
        bars = ax.bar([lbl_A, lbl_B], [c["nat_acc_A"] * 100, c["nat_acc_B"] * 100],
                      color=[col_A, col_B], alpha=0.85)
        _bar_labels(ax, bars, "{:.1f}%", fontsize=9)
        delta = c["delta_nat_acc"] * 100
        ax.text(0.5, 0.05, f"Delta={delta:+.2f}pp", transform=ax.transAxes,
                ha="center", fontsize=8, fontweight="bold",
                color=C_DELTA_POS if delta >= 0 else C_DELTA_NEG,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        mcn_star = _sig_star(c["mcnemar_p"])
        ax.set_title(f"{clabel}\nNatural Acc  McNemar:{mcn_star}", fontsize=8, fontweight="bold")
        ax.set_ylabel("Accuracy (%)"); ax.grid(axis="y", alpha=0.3)

        # Col 1: Position bias score
        ax = axes[row, 1]
        bars = ax.bar([lbl_A, lbl_B], [c["bias_A"], c["bias_B"]],
                      color=[col_A, col_B], alpha=0.85)
        _bar_labels(ax, bars, "{:.2f}", fontsize=9)
        delta_bias = c["delta_bias"]
        ax.text(0.5, 0.05, f"Delta={delta_bias:+.2f}", transform=ax.transAxes,
                ha="center", fontsize=8, fontweight="bold",
                color=C_DELTA_POS if delta_bias <= 0 else C_DELTA_NEG,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_title("Position Bias Score\n(lower = less biased)", fontsize=8, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        # Col 2: Recall Std
        ax = axes[row, 2]
        bars = ax.bar([lbl_A, lbl_B], [c["rstd_A"], c["rstd_B"]],
                      color=[col_A, col_B], alpha=0.85)
        _bar_labels(ax, bars, "{:.2f}", fontsize=9)
        delta_rstd = c["delta_rstd"]
        ax.text(0.5, 0.05, f"Delta={delta_rstd:+.2f}%", transform=ax.transAxes,
                ha="center", fontsize=8, fontweight="bold",
                color=C_DELTA_POS if delta_rstd <= 0 else C_DELTA_NEG,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_title("Recall Std %\n(lower = more balanced)", fontsize=8, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        # Col 3: Consistency score
        ax = axes[row, 3]
        bars = ax.bar([lbl_A, lbl_B], [c["cons_A"], c["cons_B"]],
                      color=[col_A, col_B], alpha=0.85)
        _bar_labels(ax, bars, "{:.1f}%", fontsize=9)
        delta_cons = c["delta_cons"]
        ax.text(0.5, 0.05, f"Delta={delta_cons:+.2f}pp", transform=ax.transAxes,
                ha="center", fontsize=8, fontweight="bold",
                color=C_DELTA_POS if delta_cons >= 0 else C_DELTA_NEG,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_title("Consistency Score %\n(higher = more consistent)", fontsize=8, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        # Col 4: Statistical tests text panel
        ax = axes[row, 4]
        ax.axis("off")
        ct = c.get("mcnemar_ct", {})
        txt = (
            f"{clabel}\n"
            f"{'='*28}\n\n"
            f"ACCURACY (natural perm=0)\n"
            f"  {lbl_A}: {c['nat_acc_A']*100:.2f}%\n"
            f"  {lbl_B}: {c['nat_acc_B']*100:.2f}%\n"
            f"  Delta:  {c['delta_nat_acc']*100:+.2f}pp\n\n"
            f"McNemar Test (paired acc):\n"
            f"  stat = {c['mcnemar_stat']:.4f}\n"
            f"  p    = {c['mcnemar_p']:.4f}  {_sig_star(c['mcnemar_p'])}\n"
            f"  Both correct:  {ct.get('both_correct','?')}\n"
            f"  A only:        {ct.get('A_only_correct','?')}\n"
            f"  B only:        {ct.get('B_only_correct','?')}\n"
            f"  Neither:       {ct.get('neither_correct','?')}\n\n"
            f"BIAS / DISTRIBUTION\n"
            f"  Bias   {c['bias_A']:.3f} -> {c['bias_B']:.3f} ({c['delta_bias']:+.3f})\n"
            f"  RStd   {c['rstd_A']:.2f}% -> {c['rstd_B']:.2f}% ({c['delta_rstd']:+.2f}%)\n"
            f"  Cons   {c['cons_A']:.1f}% -> {c['cons_B']:.1f}% ({c['delta_cons']:+.2f}pp)\n\n"
            f"Chi-sq Uniformity:\n"
            f"  A: stat={c['chi2_stat_A']:.3f}  p={c['chi2_p_A']:.4f}\n"
            f"  B: stat={c['chi2_stat_B']:.3f}  p={c['chi2_p_B']:.4f}\n\n"
            f"Chi-sq Acc vs Position:\n"
            f"  A: stat={c['chi2_acc_stat_A']:.3f}  p={c['chi2_acc_p_A']:.4f}\n"
            f"  B: stat={c['chi2_acc_stat_B']:.3f}  p={c['chi2_acc_p_B']:.4f}\n\n"
            f"Distribution Shift:\n"
            f"  chi2={c['dist_chi2_stat']:.3f}  p={c['dist_chi2_p']:.4f}\n"
            f"  Significant: {c['sig_distribution']}"
        )
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=7.5,
                verticalalignment="top", family="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_model_distributions_all(dataset, model, data, output_path):
    """Choice distribution for all 3 methods side by side."""
    methods = ["baseline_raw", "agent_raw", "agent_pride"]
    labels  = [METHOD_LABELS[k] for k in methods]
    colors_  = [METHOD_COLORS[k] for k in methods]
    bar_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{dataset}  —  {model}  |  Choice Distributions: All Methods",
                 fontsize=13, fontweight="bold")

    for ax, mk, lbl in zip(axes, methods, labels):
        pcts = [data[mk]["choice_percentages"].get(p, 0) for p in POSITIONS]
        bars = ax.bar(POSITIONS, pcts, color=bar_colors, alpha=0.85)
        ax.axhline(25, color="red", linestyle="--", linewidth=2, label="Uniform 25%")
        bias = data[mk]["position_bias_score"]
        chi2s = data[mk]["chi2_stat"]
        chi2p = data[mk]["chi2_pvalue"]
        cons  = data[mk]["consistency_score"]
        ax.set_title(f"{lbl}\nBias={bias:.2f}  Chi2={chi2s:.2f} (p={chi2p:.3f})\nConsistency={cons:.1f}%",
                     fontweight="bold", fontsize=9)
        ax.set_ylim(0, max(pcts + [26]) * 1.3)
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        _bar_labels(ax, bars, "{:.1f}%")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_model_accuracy_by_position_all(dataset, model, data, output_path):
    """Accuracy by position for all 3 methods."""
    methods = ["baseline_raw", "agent_raw", "agent_pride"]
    labels  = [METHOD_LABELS[k] for k in methods]
    colors_  = [METHOD_COLORS[k] for k in methods]

    x = np.arange(4); w = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(f"{dataset}  —  {model}  |  Accuracy by Answer Position: All Methods",
                 fontsize=13, fontweight="bold")

    for i, (mk, lbl, cc) in enumerate(zip(methods, labels, colors_)):
        acc_by_pos = [data["nat_acc_bypos"][mk].get(p, 0) for p in POSITIONS]
        bars = ax.bar(x + (i-1)*w, acc_by_pos, w, label=lbl, color=cc, alpha=0.85)
        overall = data["nat_acc"][mk]
        ax.axhline(overall, color=cc, linestyle="--", linewidth=1.2, alpha=0.7)
        _bar_labels(ax, bars, "{:.1%}", fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels(POSITIONS)
    ax.set_xlabel("Correct Answer Position")
    ax.set_ylabel("Accuracy (natural, perm=0)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_model_bias_all_metrics(dataset, model, data, output_path):
    """All 4 bias-related metrics for all 3 methods."""
    methods = ["baseline_raw", "agent_raw", "agent_pride"]
    labels  = [METHOD_LABELS[k] for k in methods]
    colors_  = [METHOD_COLORS[k] for k in methods]

    panels = [
        ("position_bias_score", "Position Bias Score (lower=better)"),
        ("recall_std",          "Recall Std % (lower=better)"),
        ("chi2_stat",           "Chi-sq Uniformity Stat (lower=more uniform)"),
        ("consistency_score",   "Consistency Score % (higher=better)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{dataset}  —  {model}  |  All Bias Metrics: All Methods",
                 fontsize=13, fontweight="bold")

    for ax, (metric, title) in zip(axes.flat, panels):
        vals = [data[mk][metric] for mk in methods]
        bars = ax.bar(labels, vals, color=colors_, alpha=0.85)
        _bar_labels(ax, bars, "{:.2f}", fontsize=9)
        ax.set_title(title, fontweight="bold"); ax.grid(axis="y", alpha=0.3)
        if metric == "chi2_stat":
            ax.axhline(3.84, color="red", linestyle="--", linewidth=1.5, label="crit (p<0.05)")
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def generate_report(all_results, output_dir):
    path = output_dir / "COMPARISON_REPORT.txt"
    sep  = "=" * 110
    sep2 = "-" * 110
    lines = [sep, "COMPARATIVE ANALYSIS REPORT", "Baseline vs Agent  |  Baseline vs Agent+PriDe  |  Agent vs Agent+PriDe", sep, ""]

    COMP_DESCS = {
        "comp1": "Comp 1: Baseline (raw) vs Agent (raw)  — Does multi-agent debate improve raw accuracy?",
        "comp2": "Comp 2: Baseline (raw) vs Agent+PriDe  — Does agent structure + PriDe beat single-call baseline?",
        "comp3": "Comp 3: Agent (raw)    vs Agent+PriDe  — Does PriDe debiasing add value on top of agents?",
    }

    for dataset in sorted(all_results):
        models_data = all_results[dataset]
        lines += [sep, f"DATASET: {dataset}", sep, ""]

        # --- Summary table per comparison ---
        for ck, desc in COMP_DESCS.items():
            key_A = COMP_A[ck]; key_B = COMP_B[ck]
            lines += [f"  {desc}", sep2]

            hdr = (f"  {'Model':<38} | "
                   f"{'NatAcc_A':>9} {'NatAcc_B':>9} {'DeltaAcc':>9} | "
                   f"{'Bias_A':>7} {'Bias_B':>7} {'DBias':>7} | "
                   f"{'RStd_A':>7} {'RStd_B':>7} {'DRStd':>7} | "
                   f"{'Cons_A':>7} {'Cons_B':>7} {'DCons':>7} | "
                   f"{'McNemar':>8} {'p':>8} {'Sig':>4}")
            lines += [hdr, sep2]

            for m in _sort_models(models_data.keys()):
                c = models_data[m][ck]
                sig = "YES" if c["sig_accuracy"] else "no"
                lines.append(
                    f"  {m:<38} | "
                    f"{c['nat_acc_A']*100:>8.2f}% {c['nat_acc_B']*100:>8.2f}% {c['delta_nat_acc']*100:>+8.2f}pp | "
                    f"{c['bias_A']:>7.3f} {c['bias_B']:>7.3f} {c['delta_bias']:>+7.3f} | "
                    f"{c['rstd_A']:>7.2f} {c['rstd_B']:>7.2f} {c['delta_rstd']:>+7.2f} | "
                    f"{c['cons_A']:>6.1f}% {c['cons_B']:>6.1f}% {c['delta_cons']:>+6.1f}pp | "
                    f"{c['mcnemar_stat']:>8.4f} {c['mcnemar_p']:>8.4f} {sig:>4}"
                )
            lines += [""]

        # --- Detailed per-model breakdown ---
        lines += [f"  DETAILED STATISTICS PER MODEL", sep2]
        for m in _sort_models(models_data.keys()):
            d = models_data[m]
            lines += ["", f"  [{m}]"]
            lines += [f"    Method metrics (all permutations):"]
            for mk in ["baseline_raw", "agent_raw", "agent_pride"]:
                mm = d[mk]
                lines.append(
                    f"      {METHOD_LABELS[mk]:<22}: "
                    f"acc={mm['overall_accuracy']*100:.2f}%  "
                    f"bias={mm['position_bias_score']:.3f}  "
                    f"rstd={mm['recall_std']:.2f}%  "
                    f"cons={mm['consistency_score']:.1f}%  "
                    f"chi2={mm['chi2_stat']:.3f}(p={mm['chi2_pvalue']:.4f})  "
                    f"chi2_acc={mm['chi2_acc_stat']:.3f}(p={mm['chi2_acc_pvalue']:.4f})"
                )

            for ck in ["comp1", "comp2", "comp3"]:
                c = d[ck]
                ct = c.get("mcnemar_ct", {})
                lines += [
                    f"    {COMP_DESCS[ck].split('—')[0].strip()}:",
                    f"      McNemar: stat={c['mcnemar_stat']:.4f}  p={c['mcnemar_p']:.4f}  {_sig_star(c['mcnemar_p'])}",
                    f"        Contingency: both={ct.get('both_correct','?')}  "
                    f"A_only={ct.get('A_only_correct','?')}  "
                    f"B_only={ct.get('B_only_correct','?')}  "
                    f"neither={ct.get('neither_correct','?')}",
                    f"      Dist shift: chi2={c['dist_chi2_stat']:.4f}  p={c['dist_chi2_p']:.4f}  "
                    f"{_sig_star(c['dist_chi2_p'])}  (sig={c['sig_distribution']})",
                ]

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Comparative analysis: Baseline vs Agent vs PriDe.")
    parser.add_argument("--dataset",      type=str,   default=None,          help="Filter to one dataset")
    parser.add_argument("--fixed-alpha",  type=float, default=DEFAULT_ALPHA,
                        help=f"PriDe alpha (default {DEFAULT_ALPHA}; set 0 to grid-search)")
    parser.add_argument("--summary-only", action="store_true", help="Write report only, skip plots")
    parser.add_argument("--output-dir",   type=str,   default=None,          help="Override output directory")
    args = parser.parse_args()

    fixed_alpha = args.fixed_alpha if args.fixed_alpha != 0 else None
    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    by_ds_dir    = out_dir / "by_dataset"
    by_model_dir = out_dir / "by_model"

    print("\n" + "=" * 70)
    print("  COMPARATIVE ANALYSIS: Baseline vs Agent vs Agent+PriDe")
    print("=" * 70)
    print(f"  Baseline dir : {BASELINE_DIR}")
    print(f"  Selective dir: {SELECTIVE_DIR}")
    print(f"  Output       : {out_dir}")
    print(f"  Alpha        : {'grid-search' if fixed_alpha is None else fixed_alpha}")

    pairs = _find_pairs(args.dataset)
    if not pairs:
        print("\n  No matching pairs found.")
        print(f"  Need both: {BASELINE_DIR}/<label>_baseline.csv")
        print(f"         and: {SELECTIVE_DIR}/<label>_pride.csv")
        return

    print(f"\n  Found {len(pairs)} matched pair(s).\n")

    all_results = defaultdict(dict)

    for ds, mdl, label, bl_path, sl_path in pairs:
        print(f"  Processing: {label}")
        try:
            data = process_pair(ds, mdl, bl_path, sl_path, fixed_alpha)
            all_results[ds][mdl] = data
            c1, c2, c3 = data["comp1"], data["comp2"], data["comp3"]
            print(f"    Comp1 (Base vs Agent)      : acc delta={c1['delta_nat_acc']*100:+.2f}pp  "
                  f"bias delta={c1['delta_bias']:+.3f}  McNemar p={c1['mcnemar_p']:.4f} {_sig_star(c1['mcnemar_p'])}")
            print(f"    Comp2 (Base vs Ag+PriDe)   : acc delta={c2['delta_nat_acc']*100:+.2f}pp  "
                  f"bias delta={c2['delta_bias']:+.3f}  McNemar p={c2['mcnemar_p']:.4f} {_sig_star(c2['mcnemar_p'])}")
            print(f"    Comp3 (Agent vs Ag+PriDe)  : acc delta={c3['delta_nat_acc']*100:+.2f}pp  "
                  f"bias delta={c3['delta_bias']:+.3f}  McNemar p={c3['mcnemar_p']:.4f} {_sig_star(c3['mcnemar_p'])}")
        except Exception as e:
            print(f"    [ERROR] {e}")
            import traceback; traceback.print_exc()

    if not all_results:
        print("\n  No results produced.")
        return

    if not args.summary_only:
        print("\n  Generating plots...")
        for dataset, models_data in all_results.items():
            dd = by_ds_dir / dataset
            dd.mkdir(parents=True, exist_ok=True)
            print(f"\n  [Dataset] {dataset} ({len(models_data)} model(s))")

            for ck in ["comp1", "comp2", "comp3"]:
                short = ck.replace("comp", "comp")
                print(f"    {COMP_TITLES[ck]}")
                plot_dataset_single_comparison(
                    dataset, models_data, ck,
                    dd / f"{short}_summary.png")
                plot_dataset_distribution_comparison(
                    dataset, models_data, ck,
                    dd / f"{short}_distributions.png")
                plot_dataset_accuracy_by_position(
                    dataset, models_data, ck,
                    dd / f"{short}_accuracy_by_position.png")

            plot_dataset_statistical_tests(dataset, models_data, dd / "statistical_tests.png")
            plot_dataset_overview(dataset, models_data, dd / "overview_all_comparisons.png")

            for model, data in models_data.items():
                md = by_model_dir / f"{dataset}-{model}"
                md.mkdir(parents=True, exist_ok=True)
                print(f"    [Model] {model}")
                plot_model_comprehensive(dataset, model, data,
                                         md / "comprehensive_dashboard.png")
                plot_model_distributions_all(dataset, model, data,
                                              md / "distributions_all_methods.png")
                plot_model_accuracy_by_position_all(dataset, model, data,
                                                    md / "accuracy_by_position.png")
                plot_model_bias_all_metrics(dataset, model, data,
                                             md / "bias_all_metrics.png")

    generate_report(dict(all_results), out_dir)

    print("\n" + "=" * 70)
    print("  DONE")
    print(f"  by_dataset/  -> {by_ds_dir}")
    print(f"  by_model/    -> {by_model_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
