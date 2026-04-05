"""
Comparative Analysis: Baseline vs Individual Agents vs Agent Full vs PriDe
==========================================================================
10 configurations compared:

  Raw (no debiasing):           With PriDe:
  ───────────────────           ───────────
  1. Baseline                   2. Baseline + PriDe
  3. Agent 1 / Analyst (raw)    6. Agent 1 + PriDe  [needs patch script]
  4. Agent 2 / Critic   (raw)   7. Agent 2 + PriDe  [needs patch script]
  5. Agent 3 / Intuitive(raw)   8. Agent 3 + PriDe  [needs patch script]
  9. Agent Full / voted (raw)  10. Agent Full + PriDe

NOTE: Agent 1/2/3 raw uses Phase-1 answers (temperature > 0, diversity check).
      All other methods use temperature = 0. Intra-agent comparisons (1 vs 2
      vs 3) are fair; cross-method comparisons carry this caveat.
      Run patch_agent_perm_answers.py to unlock methods 6-8.

Statistical tests (from pride_batch_summary.py):
  - Overall + per-position accuracy
  - Choice distribution + chi-square uniformity test
  - Position bias score (std of choice %)
  - Recall Std / RStd
  - Chi-square: accuracy vs position independence
  - Consistency score (same content across 4 permutations)
  - McNemar's test: paired accuracy significance
  - Chi-square: distribution shift between methods

Input:
  results/baseline/*_baseline.csv
  results/mad_graph_selective/output/*_pride.csv

Output:
  pride/results/comparison/
    by_dataset/<dataset>/
      leaderboard.png
      pride_benefit.png
      bias_comparison.png
      accuracy_by_position.png
      statistical_tests.png
    by_model/<dataset>-<model>/
      dashboard.png
      distributions.png
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
DEFAULT_ALPHA = 0.3

POSITIONS = ["A", "B", "C", "D"]

# Display order for all plots and tables
METHODS = [
    "baseline_raw",
    "baseline_pride",
    "agent1_raw",
    "agent2_raw",
    "agent3_raw",
    "agent1_pride",
    "agent2_pride",
    "agent3_pride",
    "agent_full_raw",
    "agent_full_pride",
]

METHOD_LABELS = {
    "baseline_raw":    "Baseline",
    "baseline_pride":  "Baseline+PriDe",
    "agent1_raw":      "Agent1/Analyst",
    "agent2_raw":      "Agent2/Critic",
    "agent3_raw":      "Agent3/Intuitive",
    "agent1_pride":    "Agent1+PriDe",
    "agent2_pride":    "Agent2+PriDe",
    "agent3_pride":    "Agent3+PriDe",
    "agent_full_raw":  "AgentFull",
    "agent_full_pride":"AgentFull+PriDe",
}

METHOD_COLORS = {
    "baseline_raw":    "#E74C3C",
    "baseline_pride":  "#FF8C69",
    "agent1_raw":      "#2980B9",
    "agent2_raw":      "#27AE60",
    "agent3_raw":      "#8E44AD",
    "agent1_pride":    "#85C1E9",
    "agent2_pride":    "#82E0AA",
    "agent3_pride":    "#C39BD3",
    "agent_full_raw":  "#F39C12",
    "agent_full_pride":"#FAD7A0",
}

RAW_PRIDE_PAIRS = [
    ("baseline_raw",   "baseline_pride"),
    ("agent1_raw",     "agent1_pride"),
    ("agent2_raw",     "agent2_pride"),
    ("agent3_raw",     "agent3_pride"),
    ("agent_full_raw", "agent_full_pride"),
]

# Key pairwise comparisons shown in report + plots
KEY_COMPARISONS = [
    ("baseline_raw",   "agent_full_raw",    "Baseline vs AgentFull"),
    ("baseline_raw",   "agent_full_pride",  "Baseline vs AgentFull+PriDe"),
    ("agent_full_raw", "agent_full_pride",  "AgentFull vs +PriDe"),
    ("baseline_raw",   "baseline_pride",    "Baseline vs +PriDe"),
    ("baseline_raw",   "agent1_raw",        "Baseline vs Agent1"),
    ("baseline_raw",   "agent2_raw",        "Baseline vs Agent2"),
    ("baseline_raw",   "agent3_raw",        "Baseline vs Agent3"),
    ("agent1_raw",     "agent2_raw",        "Agent1 vs Agent2"),
    ("agent2_raw",     "agent3_raw",        "Agent2 vs Agent3"),
    ("agent1_raw",     "agent3_raw",        "Agent1 vs Agent3"),
    ("agent1_raw",     "agent1_pride",      "Agent1 vs +PriDe"),
    ("agent2_raw",     "agent2_pride",      "Agent2 vs +PriDe"),
    ("agent3_raw",     "agent3_pride",      "Agent3 vs +PriDe"),
]


# ---------------------------------------------------------------------------
# PriDe (self-contained)
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
        return (df[df["question_id"].isin(ids[:n])].copy(),
                df[df["question_id"].isin(ids[n:])].copy())

    def _fit(self, cal):
        priors = []
        for _, grp in cal.groupby("question_id"):
            P = grp[["prob_A","prob_B","prob_C","prob_D"]].values
            P = P / (P.sum(axis=1, keepdims=True) + 1e-10)
            lm = np.log(P + 1e-10).mean(axis=0)
            pr = np.exp(lm - lm.max()); pr /= pr.sum()
            priors.append(pr)
        self._prior = np.mean(priors, axis=0)

    def _debias(self, test):
        out = test.copy()
        preds = []
        for _, row in out.iterrows():
            obs = np.array([row[f"prob_{p}"] for p in POSITIONS])
            logits = np.log(obs + 1e-10) - self.alpha * np.log(self._prior + 1e-10)
            preds.append(POSITIONS[int(np.argmax(logits))])
        out["debiased_predicted_answer"] = preds
        out["debiased_is_correct"] = [
            int(p == str(r).upper().strip())
            for p, r in zip(preds, out["correct_position"])
        ]
        return out

    def fit_predict(self, df):
        cal, test = self._split(df)
        self._fit(cal)
        return self._debias(test)


def _run_pride(df, fixed_alpha):
    alphas = ([fixed_alpha] if fixed_alpha is not None
              else np.round(np.arange(0.0, 1.05, 0.1), 1).tolist())
    best_alpha, best_acc, best_deb = alphas[0], -1.0, None
    for a in alphas:
        p = _PriDe(calib=CALIB_RATIO, alpha=float(a))
        deb = p.fit_predict(df)
        acc = deb["debiased_is_correct"].mean()
        if acc > best_acc:
            best_acc, best_alpha, best_deb = acc, a, deb
    return best_deb, best_alpha


def _build_agent_pride_df(sl, perm_col):
    """One-hot prob vector from per-permutation individual agent answers."""
    if perm_col not in sl.columns or sl[perm_col].fillna("").astype(str).str.strip().eq("").all():
        return None
    df = sl[["question_id", "permutation_idx", "correct_position"]].copy()
    ans = sl[perm_col].astype(str).str.upper().str.strip()
    for p in POSITIONS:
        df[f"prob_{p}"] = (ans == p).astype(float)
    df["predicted_answer"] = ans.where(ans.isin(POSITIONS), "")
    return df


# ---------------------------------------------------------------------------
# Metric computation (faithful to pride_batch_summary.py)
# ---------------------------------------------------------------------------

def _compute_metrics(df, pred_col="predicted_answer"):
    d = df.copy()
    d[pred_col] = d[pred_col].astype(str).str.upper().str.strip()
    d["correct_position"] = d["correct_position"].astype(str).str.upper().str.strip()
    valid = d[d[pred_col].isin(POSITIONS)].copy()
    valid["ok"] = (valid[pred_col] == valid["correct_position"]).astype(int)

    counts = valid[pred_col].value_counts().reindex(POSITIONS, fill_value=0)
    total  = len(valid)
    pcts   = counts / total * 100 if total > 0 else counts * 0.0

    try:    chi2_s, chi2_p = chisquare(counts.values, f_exp=[total/4]*4)
    except: chi2_s, chi2_p = 0.0, 1.0

    recalls = []
    for pos in POSITIONS:
        m = valid["correct_position"] == pos
        recalls.append(float((valid.loc[m, pred_col] == pos).mean()) if m.sum() > 0 else 0.0)

    try:
        ct = pd.crosstab(valid["correct_position"], valid["ok"])
        chi2_acc, p_acc, _, _ = chi2_contingency(ct)
    except: chi2_acc, p_acc = 0.0, 1.0

    # Consistency score
    cons = 0.0
    id_col = "question_id" if "question_id" in valid.columns else "id"
    if id_col in valid.columns and "permutation_idx" in valid.columns:
        def _orig(row):
            try:
                pred = str(row[pred_col]).upper().strip()
                if pred not in POSITIONS: return None
                shift = int(row["permutation_idx"]) % 4
                return POSITIONS[(shift + POSITIONS.index(pred)) % 4]
            except: return None
        valid["_orig"] = valid.apply(_orig, axis=1)
        vv = valid.dropna(subset=["_orig"])
        if not vv.empty:
            cons = float((vv.groupby(id_col)["_orig"].nunique() == 1).mean() * 100)

    return {
        "overall_accuracy":     float(valid["ok"].mean()) if total > 0 else 0.0,
        "accuracy_by_position": valid.groupby("correct_position")["ok"].mean().to_dict(),
        "choice_counts":        counts.to_dict(),
        "choice_percentages":   pcts.to_dict(),
        "chi2_stat":            float(chi2_s),
        "chi2_pvalue":          float(chi2_p),
        "position_bias_score":  float(np.std(pcts.values)),
        "recall_std":           float(np.std(recalls) * 100),
        "recalls":              {p: recalls[i] for i, p in enumerate(POSITIONS)},
        "chi2_acc_stat":        float(chi2_acc),
        "chi2_acc_pvalue":      float(p_acc),
        "consistency_score":    cons,
        "n_samples":            total,
    }


def _nat_accuracy(df, pred_col="predicted_answer"):
    """Accuracy at perm=0 only. Returns (overall, by_pos_dict, correct_series)."""
    sub = (df[df["permutation_idx"] == 0].copy()
           if "permutation_idx" in df.columns else df.copy())
    sub[pred_col] = sub[pred_col].astype(str).str.upper().str.strip()
    sub["correct_position"] = sub["correct_position"].astype(str).str.upper().str.strip()
    valid = sub[sub[pred_col].isin(POSITIONS)].copy()
    valid["ok"] = (valid[pred_col] == valid["correct_position"]).astype(int)
    overall = float(valid["ok"].mean()) if len(valid) > 0 else 0.0
    by_pos  = valid.groupby("correct_position")["ok"].mean().to_dict()
    corr    = (valid.set_index("question_id")["ok"]
               if "question_id" in valid.columns else None)
    return overall, by_pos, corr


# ---------------------------------------------------------------------------
# Statistical comparison helpers
# ---------------------------------------------------------------------------

def _mcnemar(sA, sB):
    df = pd.DataFrame({"A": sA, "B": sB}).dropna()
    if df.empty: return float("nan"), float("nan"), {}
    both    = int(((df.A==1)&(df.B==1)).sum())
    A_only  = int(((df.A==1)&(df.B==0)).sum())
    B_only  = int(((df.A==0)&(df.B==1)).sum())
    neither = int(((df.A==0)&(df.B==0)).sum())
    b, c = B_only, A_only
    ct = {"both": both, "A_only": A_only, "B_only": B_only, "neither": neither}
    if b+c == 0: return 0.0, 1.0, ct
    stat = (abs(b-c)-1)**2 / (b+c)
    return float(stat), float(1 - chi2_dist.cdf(stat, df=1)), ct


def _dist_chi2(cA, cB):
    a = np.array([cA.get(p,0) for p in POSITIONS], dtype=float)
    b = np.array([cB.get(p,0) for p in POSITIONS], dtype=float)
    if a.sum()==0 or b.sum()==0: return float("nan"), float("nan")
    bs = np.where(b/b.sum()*a.sum()==0, 0.001, b/b.sum()*a.sum())
    try:    return tuple(float(v) for v in chisquare(a, f_exp=bs))
    except: return float("nan"), float("nan")


def _sig(p):
    if np.isnan(p): return "n/a"
    if p < 0.001:   return "***"
    if p < 0.01:    return "**"
    if p < 0.05:    return "*"
    return "ns"


def _compare(mA, mB, natA, natB, cA, cB):
    if cA is not None and cB is not None:
        common = cA.index.intersection(cB.index)
        ms, mp, mct = _mcnemar(cA.loc[common], cB.loc[common])
    else:
        ms, mp, mct = float("nan"), float("nan"), {}
    ds, dp = _dist_chi2(mA["choice_counts"], mB["choice_counts"])
    return {
        "nat_acc_A": natA, "nat_acc_B": natB,
        "delta_nat_acc":  natB - natA,
        "delta_bias":     mB["position_bias_score"] - mA["position_bias_score"],
        "delta_rstd":     mB["recall_std"] - mA["recall_std"],
        "delta_cons":     mB["consistency_score"] - mA["consistency_score"],
        "delta_chi2":     mB["chi2_stat"] - mA["chi2_stat"],
        "mcnemar_stat": ms, "mcnemar_p": mp, "mcnemar_ct": mct,
        "dist_chi2_stat": ds, "dist_chi2_p": dp,
        "sig_accuracy":     (mp < 0.05) if not np.isnan(mp) else False,
        "sig_distribution": (dp < 0.05) if not np.isnan(dp) else False,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load(path):
    df = pd.read_csv(path)
    for p in POSITIONS:
        c = f"prob_{p}"
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def _parse_label(path: Path):
    name = path.stem.replace("_baseline","").replace("_pride","")
    for prefix in ["ministral","mistral","gemma","llama","phi","qwen",
                   "Qwen","microsoft","Phi","gpt-oss"]:
        if prefix.lower() in name.lower():
            idx = name.lower().index(prefix.lower())
            return name[:idx].strip("-"), name[idx:]
    return None, None


def _find_pairs(dataset_filter):
    bmap = {}
    for f in BASELINE_DIR.glob("*_baseline.csv"):
        ds, mdl = _parse_label(f)
        if ds and mdl: bmap[f"{ds}-{mdl}"] = f
    pairs = []
    for f in sorted(SELECTIVE_DIR.glob("*_pride.csv")):
        if "deprecated" in str(f): continue
        ds, mdl = _parse_label(f)
        if not ds or not mdl: continue
        label = f"{ds}-{mdl}"
        if dataset_filter and dataset_filter not in label: continue
        if label in bmap:
            pairs.append((ds, mdl, label, bmap[label], f))
        else:
            print(f"  [skip] No baseline for {f.name}")
    return pairs


# ---------------------------------------------------------------------------
# Per-pair processing
# ---------------------------------------------------------------------------

def process_pair(ds, mdl, bl_path, sl_path, fixed_alpha):
    bl = _load(bl_path)
    sl = _load(sl_path)

    # PriDe on baseline and full-agent selective
    bl_deb, bl_alpha = _run_pride(bl, fixed_alpha)
    sl_deb, sl_alpha = _run_pride(sl, fixed_alpha)

    # perm=0 rows for natural accuracy
    p0 = (sl[sl["permutation_idx"]==0].copy()
          if "permutation_idx" in sl.columns else sl.copy())

    # ---- Build metric dicts for all available methods ----
    m, nat, bypos, corr = {}, {}, {}, {}

    def _add(key, df, pred_col="predicted_answer"):
        m[key] = _compute_metrics(df, pred_col)
        nat[key], bypos[key], corr[key] = _nat_accuracy(df, pred_col)

    _add("baseline_raw",    bl)
    _add("baseline_pride",  bl_deb,  "debiased_predicted_answer")
    _add("agent_full_raw",  sl)
    _add("agent_full_pride",sl_deb,  "debiased_predicted_answer")

    # Individual agent raw — uses Phase-1 answers (temp>0, single ordering)
    # Only cross-perm bias metrics are unavailable; natural accuracy is valid.
    for i, col in [(1,"agent_1_ans"),(2,"agent_2_ans"),(3,"agent_3_ans")]:
        key = f"agent{i}_raw"
        if col in p0.columns and p0[col].fillna("").astype(str).str.strip().ne("").any():
            sub = p0[["question_id","correct_position",col]].copy()
            sub["predicted_answer"] = sub[col]
            sub["permutation_idx"]  = 0
            m[key] = _compute_metrics(sub)
            nat[key], bypos[key], corr[key] = _nat_accuracy(sub)

    # Individual agent + PriDe — requires perm_agent_X_ans (patch script)
    for i, col in [(1,"perm_agent_1_ans"),(2,"perm_agent_2_ans"),(3,"perm_agent_3_ans")]:
        key = f"agent{i}_pride"
        adf = _build_agent_pride_df(sl, col)
        if adf is not None:
            deb, _ = _run_pride(adf, fixed_alpha)
            _add(key, deb, "debiased_predicted_answer")

    available = [mk for mk in METHODS if mk in m]
    unavailable = [mk for mk in METHODS if mk not in m]

    # ---- Key pairwise comparisons ----
    comparisons = {}
    for kA, kB, label in KEY_COMPARISONS:
        if kA in m and kB in m:
            comparisons[label] = _compare(
                m[kA], m[kB], nat.get(kA,0.0), nat.get(kB,0.0),
                corr.get(kA), corr.get(kB))

    return {
        "metrics": m, "nat": nat, "bypos": bypos, "corr": corr,
        "comparisons": comparisons,
        "available": available, "unavailable": unavailable,
        "alpha_bl": bl_alpha, "alpha_ag": sl_alpha,
        "n_questions": int(sl["question_id"].nunique()),
        "n_confident": int(p0["confident"].sum()) if "confident" in p0.columns else 0,
    }


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _blabels(ax, bars, fmt="{:.2f}", fs=7):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, h, fmt.format(h),
                ha="center", va="bottom" if h>=0 else "top", fontsize=fs, fontweight="bold")


def _avail(data, cross_all=False):
    """Return methods present in this data dict (or common across a models_data dict)."""
    if cross_all:
        # data is models_data dict; return methods present in ALL models
        sets = [set(d["available"]) for d in data.values()]
        return [mk for mk in METHODS if all(mk in s for s in sets)]
    return data["available"]


# ---------------------------------------------------------------------------
# BY-DATASET plots
# ---------------------------------------------------------------------------

def plot_leaderboard(dataset, models_data, out):
    models = sorted(models_data)
    fig, axes = plt.subplots(1, len(models), figsize=(8*len(models), 9))
    if len(models)==1: axes=[axes]
    fig.suptitle(f"{dataset} — Method Leaderboard (Natural Accuracy perm=0)",
                 fontsize=13, fontweight="bold")
    for ax, mdl in zip(axes, models):
        d = models_data[mdl]
        avail = d["available"]
        accs  = [d["nat"].get(mk,0)*100 for mk in avail]
        order = np.argsort(accs)[::-1]
        accs_s  = [accs[i]  for i in order]
        lbls_s  = [METHOD_LABELS[avail[i]] for i in order]
        cols_s  = [METHOD_COLORS[avail[i]] for i in order]
        bars = ax.barh(range(len(accs_s)), accs_s, color=cols_s, alpha=0.85)
        for j,bar in enumerate(bars):
            ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                    f"{accs_s[j]:.1f}%", va="center", fontsize=8, fontweight="bold")
        ax.set_yticks(range(len(lbls_s))); ax.set_yticklabels(lbls_s, fontsize=8)
        ax.set_title(mdl, fontweight="bold", fontsize=9)
        ax.set_xlim(0, max(accs_s)*1.18 if accs_s else 100)
        ax.grid(axis="x", alpha=0.3)
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()


def plot_pride_benefit(dataset, models_data, out):
    models = sorted(models_data)
    # only pairs where at least one side exists in every model
    avail_all = _avail(models_data, cross_all=True)
    pairs = [(r,pr) for r,pr in RAW_PRIDE_PAIRS if r in avail_all or pr in avail_all]
    if not pairs: return
    x = np.arange(len(models)); w = 0.12
    fig, axes = plt.subplots(2,1, figsize=(max(12, 3*len(models)),12))
    fig.suptitle(f"{dataset} — PriDe Benefit per Method Family", fontsize=13, fontweight="bold")
    for ai, (metric, ylabel) in enumerate([("nat_acc","Natural Accuracy (%)"),
                                           ("position_bias_score","Position Bias Score")]):
        ax = axes[ai]
        n = len(pairs)
        for i,(rk,pk) in enumerate(pairs):
            offset = (i - n/2 + 0.5)*w*2
            if metric=="nat_acc":
                vr = [models_data[m]["nat"].get(rk,0)*100 for m in models]
                vp = [models_data[m]["nat"].get(pk,0)*100 for m in models]
            else:
                vr = [models_data[m]["metrics"].get(rk,{}).get(metric,0) for m in models]
                vp = [models_data[m]["metrics"].get(pk,{}).get(metric,0) for m in models]
            ax.bar(x+offset-w/2, vr, w, color=METHOD_COLORS[rk], alpha=0.85,
                   label=METHOD_LABELS[rk])
            ax.bar(x+offset+w/2, vp, w, color=METHOD_COLORS[pk], alpha=0.85,
                   label=METHOD_LABELS[pk])
            for j,(r,p) in enumerate(zip(vr,vp)):
                d=p-r; better=(d>0)==(ai==0)
                ax.text(x[j]+offset, max(r,p)+0.4, f"{d:+.1f}",
                        ha="center", fontsize=6,
                        color="#27AE60" if better else "#E74C3C")
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylabel); ax.legend(fontsize=6, ncol=5); ax.grid(axis="y",alpha=0.3)
        ax.set_title(ylabel + (" (higher=better)" if ai==0 else " (lower=better)"),
                     fontweight="bold")
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()


def plot_bias_comparison(dataset, models_data, out):
    models  = sorted(models_data)
    avail_all = _avail(models_data, cross_all=True)
    n = len(avail_all); x = np.arange(len(models))
    w = max(0.05, 0.8/max(n,1))
    panels = [("position_bias_score","Position Bias Score (lower=better)"),
              ("recall_std","Recall Std % (lower=better)"),
              ("chi2_stat","Chi-sq Uniformity (lower=more uniform)")]
    fig,axes = plt.subplots(3,1, figsize=(max(14, 3*len(models)),15))
    fig.suptitle(f"{dataset} — Bias Metrics: All Methods", fontsize=13, fontweight="bold")
    for ax,(metric,title) in zip(axes,panels):
        for i,mk in enumerate(avail_all):
            offset = (i-n/2+0.5)*w
            vals = [models_data[m]["metrics"].get(mk,{}).get(metric,0) for m in models]
            ax.bar(x+offset, vals, w, color=METHOD_COLORS[mk],
                   label=METHOD_LABELS[mk], alpha=0.85)
        if metric=="chi2_stat":
            ax.axhline(3.84, color="red", linestyle="--", linewidth=1.5, label="p<0.05")
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=6, ncol=5); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()


def plot_accuracy_by_position(dataset, models_data, out):
    models = sorted(models_data)
    avail_all = _avail(models_data, cross_all=True)
    n = len(avail_all); x = np.arange(4); w = max(0.04, 0.7/max(n,1))
    fig,axes = plt.subplots(1,len(models), figsize=(9*len(models),7))
    if len(models)==1: axes=[axes]
    fig.suptitle(f"{dataset} — Accuracy by Position", fontsize=13, fontweight="bold")
    for ax,mdl in zip(axes,models):
        d = models_data[mdl]
        for i,mk in enumerate(avail_all):
            if mk not in d["bypos"]: continue
            offset = (i-n/2+0.5)*w
            vals = [d["bypos"][mk].get(p,0)*100 for p in POSITIONS]
            ax.bar(x+offset, vals, w, color=METHOD_COLORS[mk],
                   label=METHOD_LABELS[mk], alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(POSITIONS)
        ax.set_title(mdl, fontweight="bold"); ax.legend(fontsize=6, ncol=2)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()


def plot_statistical_tests(dataset, models_data, out):
    models = sorted(models_data)
    comps = [(lbl,kA,kB) for kA,kB,lbl in KEY_COMPARISONS
             if all(kA in models_data[m]["available"] and
                    kB in models_data[m]["available"] for m in models)]
    if not comps: return
    x = np.arange(len(models)); w = max(0.04, 0.7/max(len(comps),1))
    palette = plt.cm.tab20(np.linspace(0,1,len(comps)))
    fig,axes = plt.subplots(2,1, figsize=(max(14, 2*len(models)),12))
    fig.suptitle(f"{dataset} — Statistical Significance: Key Comparisons",
                 fontsize=13, fontweight="bold")
    for ai,(pkey,ptitle) in enumerate([
        ("mcnemar_p","McNemar p-value (accuracy significance)"),
        ("dist_chi2_p","Distribution shift chi-sq p-value"),
    ]):
        ax = axes[ai]
        for i,(lbl,kA,kB) in enumerate(comps):
            offset = (i-len(comps)/2+0.5)*w
            pvals = [min(models_data[m]["comparisons"].get(lbl,{}).get(pkey,1.0),1.0)
                     for m in models]
            bars = ax.bar(x+offset, pvals, w, label=lbl, color=palette[i], alpha=0.8)
            for j,pv in enumerate(pvals):
                if pv < 0.05:
                    ax.text(x[j]+offset, pv+0.01, "*", ha="center",
                            fontsize=9, fontweight="bold")
        ax.axhline(0.05, color="red", linestyle="--", linewidth=2, label="p=0.05")
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{ptitle}\n(below red = significant)", fontweight="bold")
        ax.legend(fontsize=6, ncol=3); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()


# ---------------------------------------------------------------------------
# BY-MODEL plots
# ---------------------------------------------------------------------------

def plot_model_dashboard(dataset, model, data, out):
    avail = data["available"]
    fig,axes = plt.subplots(2,2, figsize=(18,14))
    fig.suptitle(f"{dataset} — {model} | Dashboard", fontsize=13, fontweight="bold")

    # [0,0] Ranking
    ax = axes[0,0]
    accs  = [data["nat"].get(mk,0)*100 for mk in avail]
    order = np.argsort(accs)[::-1]
    bars  = ax.barh(range(len(order)),
                    [accs[i] for i in order],
                    color=[METHOD_COLORS[avail[i]] for i in order], alpha=0.85)
    for j,bar in enumerate(bars):
        ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                f"{[accs[i] for i in order][j]:.1f}%", va="center", fontsize=8, fontweight="bold")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([METHOD_LABELS[avail[i]] for i in order], fontsize=8)
    ax.set_title("Accuracy Ranking", fontweight="bold"); ax.grid(axis="x", alpha=0.3)

    # [0,1] PriDe benefit
    ax = axes[0,1]
    avail_pairs = [(r,pr) for r,pr in RAW_PRIDE_PAIRS if r in avail or pr in avail]
    x = np.arange(len(avail_pairs)); w = 0.35
    raw_a  = [data["nat"].get(r, 0)*100  for r,_ in avail_pairs]
    prd_a  = [data["nat"].get(pr,0)*100  for _,pr in avail_pairs]
    ax.bar(x-w/2, raw_a,  w, color="#E74C3C", alpha=0.8, label="raw")
    ax.bar(x+w/2, prd_a, w, color="#27AE60", alpha=0.8, label="+PriDe")
    for j,(r,p) in enumerate(zip(raw_a,prd_a)):
        d=p-r
        ax.text(x[j], max(r,p)+0.5, f"{d:+.1f}pp", ha="center", fontsize=8, fontweight="bold",
                color="#27AE60" if d>=0 else "#E74C3C")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[r] for r,_ in avail_pairs], rotation=25, ha="right", fontsize=8)
    ax.set_title("Accuracy: raw vs +PriDe", fontweight="bold"); ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # [1,0] Bias metrics
    ax = axes[1,0]
    x2 = np.arange(len(avail)); w2 = 0.25
    for k,(mkey,mlbl) in enumerate([("position_bias_score","Bias"),
                                     ("recall_std","RStd"),
                                     ("consistency_score","Cons%")]):
        vals = [data["metrics"].get(mk,{}).get(mkey,0) for mk in avail]
        ax.bar(x2+(k-1)*w2, vals, w2, label=mlbl, alpha=0.8)
    ax.set_xticks(x2)
    ax.set_xticklabels([METHOD_LABELS[mk] for mk in avail], rotation=35, ha="right", fontsize=7)
    ax.set_title("Bias / RStd / Consistency", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # [1,1] Key comparisons text
    ax = axes[1,1]; ax.axis("off")
    lines = ["Key Comparisons\n",
             f"{'Comparison':<34} {'DeltaAcc':>9} {'DeltaBias':>10} {'McNemar-p':>10} {'Sig':>4}",
             "-"*68]
    for comp_lbl,comp in data["comparisons"].items():
        lines.append(
            f"{comp_lbl:<34}"
            f"{comp.get('delta_nat_acc',0)*100:>+8.2f}pp"
            f"{comp.get('delta_bias',0):>+10.3f}"
            f"{comp.get('mcnemar_p',float('nan')):>10.4f}"
            f"  {_sig(comp.get('mcnemar_p',float('nan')))}")
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, fontsize=7.5,
            va="top", family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()


def plot_model_distributions(dataset, model, data, out):
    avail = data["available"]
    cols = 5; rows = (len(avail)+cols-1)//cols
    bar_colors = ["#FF6B6B","#4ECDC4","#45B7D1","#FFA07A"]
    fig,axes = plt.subplots(rows, cols, figsize=(18, 5*rows))
    axes_flat = list(axes.flat) if hasattr(axes,"flat") else [axes]
    fig.suptitle(f"{dataset} — {model} | Choice Distributions", fontsize=12, fontweight="bold")
    for ax,mk in zip(axes_flat, avail):
        mm = data["metrics"][mk]
        pcts = [mm["choice_percentages"].get(p,0) for p in POSITIONS]
        bars = ax.bar(POSITIONS, pcts, color=bar_colors, alpha=0.85)
        ax.axhline(25, color="red", linestyle="--", linewidth=1.5)
        ax.set_title(f"{METHOD_LABELS[mk]}\nbias={mm['position_bias_score']:.2f} "
                     f"chi2={mm['chi2_stat']:.2f}", fontsize=8, fontweight="bold")
        ax.set_ylim(0, max(pcts+[26])*1.3); ax.grid(axis="y", alpha=0.3)
        _blabels(ax, bars, "{:.1f}%", 7)
    for ax in axes_flat[len(avail):]: ax.set_visible(False)
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()


# ---------------------------------------------------------------------------
# Report (concise)
# ---------------------------------------------------------------------------

def generate_report(all_results, out_dir):
    path = out_dir / "COMPARISON_REPORT.txt"
    sep  = "=" * 88
    lines = [
        sep,
        "COMPARATIVE ANALYSIS REPORT  (up to 10 methods)",
        "NOTE: Agent1/2/3 raw = Phase-1 answers (temp>0). All others = temp=0.",
        "      Intra-agent comparisons (A1 vs A2 vs A3) are fair.",
        "      Run patch_agent_perm_answers.py to add Agent1/2/3+PriDe.",
        sep, "",
    ]

    for dataset in sorted(all_results):
        lines += [f"DATASET: {dataset}", ""]
        for model in sorted(all_results[dataset]):
            d = all_results[dataset][model]
            n_q, n_c = d["n_questions"], d["n_confident"]
            conf_pct = n_c/n_q*100 if n_q else 0
            avail, unavail = d["available"], d["unavailable"]

            lines += [
                f"  Model: {model}",
                f"  Q={n_q}  Confident={n_c} ({conf_pct:.0f}%)  "
                f"alpha_agent={d['alpha_ag']}  alpha_baseline={d['alpha_bl']}",
                f"  Available methods: {len(avail)}/10"
                + (f"  |  Missing: {', '.join(METHOD_LABELS[m] for m in unavail)}"
                   if unavail else ""),
                "",
            ]

            # -- Accuracy & bias table --
            hdr = (f"  {'Method':<18} {'Acc%':>6} {'Bias':>6} {'RStd':>6} "
                   f"{'Cons%':>6} {'Chi2':>6} {'Chi2p':>7} "
                   f"{'AccA':>6} {'AccB':>6} {'AccC':>6} {'AccD':>6}")
            lines += ["  METRICS:", hdr, "  " + "-"*86]
            for mk in avail:
                mm = d["metrics"][mk]
                na = d["nat"].get(mk, 0)
                bp = d["bypos"].get(mk, {})
                temp_note = "*" if mk in ("agent1_raw","agent2_raw","agent3_raw") else " "
                lines.append(
                    f"  {METHOD_LABELS[mk]+temp_note:<18}"
                    f"{na*100:>6.1f}%"
                    f"{mm['position_bias_score']:>6.2f}"
                    f"{mm['recall_std']:>6.2f}"
                    f"{mm['consistency_score']:>5.1f}%"
                    f"{mm['chi2_stat']:>6.2f}"
                    f"{mm['chi2_pvalue']:>7.4f}"
                    f"  {bp.get('A',0)*100:>4.1f}%"
                    f" {bp.get('B',0)*100:>4.1f}%"
                    f" {bp.get('C',0)*100:>4.1f}%"
                    f" {bp.get('D',0)*100:>4.1f}%"
                )
            lines += ["  (* = Phase-1 temp>0; interpret cross-method accuracy gaps cautiously)", ""]

            # -- Comparisons table --
            hdr2 = (f"  {'Comparison':<34} {'DeltaAcc':>9} {'DeltaBias':>10} "
                    f"{'DeltaRStd':>10} {'DeltaCons':>10} "
                    f"{'McNemar-p':>10} {'Sig':>4} {'DistShift-p':>12}")
            lines += ["  KEY COMPARISONS:", hdr2, "  " + "-"*100]
            for comp_lbl, comp in d["comparisons"].items():
                sig = _sig(comp.get("mcnemar_p", float("nan")))
                lines.append(
                    f"  {comp_lbl:<34}"
                    f"{comp.get('delta_nat_acc',0)*100:>+8.2f}pp"
                    f"{comp.get('delta_bias',0):>+10.3f}"
                    f"{comp.get('delta_rstd',0):>+10.2f}"
                    f"{comp.get('delta_cons',0):>+10.2f}pp"
                    f"{comp.get('mcnemar_p',float('nan')):>10.4f}"
                    f"  {sig:>4}"
                    f"{comp.get('dist_chi2_p',float('nan')):>12.4f}"
                )
            lines += ["", "  " + "-"*88, ""]

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="10-method comparative analysis.")
    parser.add_argument("--dataset",      type=str,   default=None)
    parser.add_argument("--fixed-alpha",  type=float, default=DEFAULT_ALPHA,
                        help=f"PriDe alpha (default {DEFAULT_ALPHA}; 0=grid-search)")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--output-dir",   type=str,   default=None)
    args = parser.parse_args()

    fixed_alpha = args.fixed_alpha if args.fixed_alpha != 0 else None
    out_dir      = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    by_ds_dir    = out_dir / "by_dataset"
    by_model_dir = out_dir / "by_model"

    print("\n" + "="*70)
    print("  10-METHOD COMPARATIVE ANALYSIS")
    print("="*70)
    print(f"  Baseline : {BASELINE_DIR}")
    print(f"  Selective: {SELECTIVE_DIR}")
    print(f"  Output   : {out_dir}")
    print(f"  Alpha    : {'grid-search' if fixed_alpha is None else fixed_alpha}")

    pairs = _find_pairs(args.dataset)
    if not pairs:
        print(f"\n  No matched pairs found. Need *_baseline.csv + *_pride.csv.")
        return

    print(f"\n  Found {len(pairs)} pair(s).\n")
    all_results = defaultdict(dict)

    for ds, mdl, label, bl_path, sl_path in pairs:
        print(f"  Processing: {label}")
        try:
            data = process_pair(ds, mdl, bl_path, sl_path, fixed_alpha)
            all_results[ds][mdl] = data
            avail, unavail = data["available"], data["unavailable"]
            print(f"    Methods: {len(avail)}/10 available"
                  + (f"  (missing: {', '.join(unavail)})" if unavail else ""))
            for lbl2, comp in data["comparisons"].items():
                sig = _sig(comp.get("mcnemar_p", float("nan")))
                if sig not in ("ns","n/a"):
                    print(f"    [SIG] {lbl2}: dAcc={comp.get('delta_nat_acc',0)*100:+.2f}pp  "
                          f"p={comp.get('mcnemar_p',float('nan')):.4f} {sig}")
        except Exception as e:
            print(f"    [ERROR] {e}")
            import traceback; traceback.print_exc()

    if not all_results:
        print("\n  No results produced."); return

    if not args.summary_only:
        print("\n  Generating plots...")
        for dataset, models_data in all_results.items():
            dd = by_ds_dir / dataset
            dd.mkdir(parents=True, exist_ok=True)
            print(f"\n  [Dataset] {dataset}")
            plot_leaderboard(dataset, models_data, dd/"leaderboard.png")
            plot_pride_benefit(dataset, models_data, dd/"pride_benefit.png")
            plot_bias_comparison(dataset, models_data, dd/"bias_comparison.png")
            plot_accuracy_by_position(dataset, models_data, dd/"accuracy_by_position.png")
            plot_statistical_tests(dataset, models_data, dd/"statistical_tests.png")
            for model, data in models_data.items():
                md = by_model_dir / f"{dataset}-{model}"
                md.mkdir(parents=True, exist_ok=True)
                print(f"    [Model] {model}")
                plot_model_dashboard(dataset, model, data, md/"dashboard.png")
                plot_model_distributions(dataset, model, data, md/"distributions.png")

    generate_report(dict(all_results), out_dir)

    print("\n" + "="*70 + "\n  DONE\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
