import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import chisquare
import re

RESULTS_DIR = r"mad_graph/results"
POSITIONS = ["A", "B", "C", "D"]

def compute_metrics(df, pred_col):
    """Computes Acc, Bias, RStd, and choice distributions."""
    d = df.copy()
    # Normalize
    d[pred_col] = d[pred_col].astype(str).str.upper().str.strip()
    d["correct_answer"] = d["correct_answer"].astype(str).str.upper().str.strip()
    
    # Filter valid
    valid = d[d[pred_col].isin(POSITIONS)].copy()
    valid["ok"] = (valid[pred_col] == valid["correct_answer"]).astype(int)
    
    total = len(valid)
    if total == 0:
        return None

    # Calculate empirical ground-truth distribution percentage
    emp_counts = valid["correct_answer"].value_counts().reindex(POSITIONS, fill_value=0)
    emp_pcts = emp_counts / total * 100

    # Calculate model's predicted distribution percentage
    counts = valid[pred_col].value_counts().reindex(POSITIONS, fill_value=0)
    pcts = counts / total * 100
    
    # Bias: standard deviation of (Predicted % - Empirical Ground-Truth %)
    # This perfectly aligns the metric scale with cyclic permutations (where empirical is uniformly 25%)
    bias = float(np.std(pcts.values - emp_pcts.values))
    
    # Chi2
    try:
        chi2_s, chi2_p = chisquare(counts.values, f_exp=[total/4]*4)
    except:
        chi2_s, chi2_p = 0.0, 1.0

    # Recall per position
    recalls = []
    for pos in POSITIONS:
        m = valid["correct_answer"] == pos
        if m.sum() > 0:
            recalls.append(float((valid.loc[m, pred_col] == pos).mean()) * 100.0)
        else:
            recalls.append(0.0)
            
    # RStd (std of recalls)
    rstd = float(np.std(recalls))
    
    # Overall Accuracy
    acc = float(valid["ok"].mean()) * 100.0
    
    return {
        "Acc": acc,
        "Bias": bias,
        "RStd": rstd,
        "Chi2": chi2_s,
        "Chi2p": chi2_p,
        "AccA": recalls[0],
        "AccB": recalls[1],
        "AccC": recalls[2],
        "AccD": recalls[3]
    }

def main():
    csv_files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
    
    output = []
    output.append("="*80)
    output.append("ORIGINAL MAD-GRAPH ANALYSIS REPORT")
    output.append("Metrics comparable to the modified semantic voting system")
    output.append("="*80 + "\n")
    
    # Group by dataset
    parsed_files = []
    for f in csv_files:
        basename = os.path.basename(f)
        
        # Datasets are known, let's extract them correctly
        ds = "Unknown"
        mdl = "Unknown"
        # Ordered by length descending so we match the longer prefix first
        known_datasets = [
             "2012-2020_ICT_DSE", "arc_challenge", "college_cs", "college_maths", 
             "college_physics", "formal_logic", "professional_law", "sociology"
        ]
        for kds in known_datasets:
             if basename.startswith(kds + "-"):
                 ds = kds
                 # Everything after dataset- and before _mad_graph
                 # e.g., dataset-gemma3_12b_mad_graph.csv -> gemma3_12b
                 suffix = basename[len(kds)+1:]
                 mdl = suffix.split("_mad_graph")[0]
                 break
                 
        if ds != "Unknown":
            parsed_files.append((ds, mdl, f))
    
    # Sort and group
    parsed_files.sort(key=lambda x: (x[0], x[1]))
    
    current_ds = None
    for ds, mdl, fpath in parsed_files:
        if ds != current_ds:
            output.append(f"\nDATASET: {ds}")
            output.append("-" * 80)
            current_ds = ds
            
        try:
            df = pd.read_csv(fpath)
        except pd.errors.EmptyDataError:
            print(f"Skipping empty or invalid file {fpath}")
            continue
        
        output.append(f"\n  Model: {mdl}")
        output.append(f"  Q={len(df)}")
        output.append("")
        output.append("  Method            Acc%   Bias    RStd   Chi2    Chi2p    AccA   AccB   AccC   AccD")
        output.append("  " + "-"*82)
        
        # We want to analyze 4 methods: agent_1_ans, agent_2_ans, agent_3_ans, predicted_answer
        methods = [
            ("Agent 1", "agent_1_ans"),
            ("Agent 2", "agent_2_ans"),
            ("Agent 3", "agent_3_ans"),
            ("Agent Full", "predicted_answer")
        ]
        
        for name, col in methods:
            if col not in df.columns:
                # Agent full?
                if name == "Agent Full" and "predicted_answer" not in df.columns:
                     if "final_answer" in df.columns:
                         col = "final_answer"
                     else:
                         continue
                else:
                     continue
            
            m = compute_metrics(df, col)
            if not m:
                continue
                
            line = f"  {name:15} {m['Acc']:5.1f}% {m['Bias']:6.2f} {m['RStd']:6.2f} "
            line += f"{m['Chi2']:6.2f} {m['Chi2p']:6.4f}   "
            line += f"{m['AccA']:5.1f}% {m['AccB']:5.1f}% {m['AccC']:5.1f}% {m['AccD']:5.1f}%"
            output.append(line)
            
    with open("original_mad_graph_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output))
        
    print("Saved original_mad_graph_summary.txt")

if __name__ == "__main__":
    main()