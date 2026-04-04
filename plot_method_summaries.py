import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Constants
POSITIONS = ["A", "B", "C", "D"]
METHODS = ["baseline", "pride", "agents", "agents_pride"]

def calculate_metrics(df, pred_col, ground_truth_col="correct_position"):
    df = df.copy()
    # Normalize strings
    df[pred_col] = df[pred_col].astype(str).str.upper().str.strip()
    if ground_truth_col in df.columns:
        df[ground_truth_col] = df[ground_truth_col].astype(str).str.upper().str.strip()
    
    valid = df[df[pred_col].isin(POSITIONS)].copy()
    if len(valid) == 0:
        return {"Accuracy": 0.0, "Bias": 0.0, "RStd": 0.0, "Consistency": 0.0}
    
    # 1. Accuracy
    valid["is_correct"] = (valid[pred_col] == valid[ground_truth_col]).astype(int)
    acc = valid["is_correct"].mean() * 100.0
    
    # 2. Position Bias Score
    counts = valid[pred_col].value_counts().reindex(POSITIONS, fill_value=0)
    pcts = counts / len(valid) * 100.0
    bias = float(np.std(pcts.values))
    
    # 3. Recall Standard Deviation (RStd)
    recalls = []
    for pos in POSITIONS:
        m = valid[ground_truth_col] == pos
        if m.sum() > 0:
            recalls.append(float((valid.loc[m, pred_col] == pos).mean()) * 100.0)
        else:
            recalls.append(0.0)
    rstd = float(np.std(recalls))
    
    # 4. Consistency Score
    cons = 0.0
    if "question_id" in valid.columns and "permutation_idx" in valid.columns:
        def _orig(row):
            pred = str(row[pred_col]).upper().strip()
            if pred not in POSITIONS: return None
            shift = int(row["permutation_idx"]) % 4
            # Consistent with compare_analysis logic
            return POSITIONS[(shift + POSITIONS.index(pred)) % 4]
            
        valid["_orig"] = valid.apply(_orig, axis=1)
        vv = valid.dropna(subset=["_orig"])
        if not vv.empty:
            cons = float((vv.groupby("question_id")["_orig"].nunique() == 1).mean() * 100.0)
            
    return {"Accuracy": acc, "Bias": bias, "RStd": rstd, "Consistency": cons}

def parse_model_dataset(filename):
    """Attempt to split filename into dataset and model"""
    # Ex: 2012-2020_ICT_DSE-gemma3_12b_pride.csv -> 2012-2020_ICT_DSE, gemma3_12b
    name = Path(filename).stem
    for suffix in ["_pride", "_baseline", ".csv"]:
        name = name.replace(suffix, "")
        
    for prefix in ["ministral", "mistral", "gemma", "llama", "phi", "qwen", "microsoft", "Qwen"]:
        if prefix.lower() in name.lower():
            idx = name.lower().index(prefix.lower())
            ds = name[:idx].strip("-")
            mdl = name[idx:]
            return ds, mdl
    return "unknown", name

def get_target_col(method_name, df):
    # Determines which prediction column to use
    if method_name in ["pride", "agents_pride"]:
        if "debiased_predicted_answer" in df.columns:
            return "debiased_predicted_answer"
    return "predicted_answer"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--organized-dir", default="organized")
    parser.add_argument("--out-dir", default="organized/summary_plots")
    args = parser.parse_args()

    base_dir = Path(args.organized_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Locate all available CSV files by dataset and model
    results = {} # ds -> model -> method -> metrics
    
    method_folders = {
        "baseline": base_dir / "baseline_csv",
        "pride": base_dir / "pride_csv",
        "agents": base_dir / "agents_csv",
        "agents_pride": base_dir / "agents_pride_csv",
    }
    
    print("Gathering data...")
    # Gather all file identifiers
    all_files = []
    for m, folder in method_folders.items():
        if folder.exists():
            for f in folder.glob("*.csv"):
                ds, mdl = parse_model_dataset(f.name)
                all_files.append((ds, mdl, m, f))
                if ds not in results: results[ds] = {}
                if mdl not in results[ds]: results[ds][mdl] = {}
    
    # Process files
    for ds, mdl, method, path in all_files:
        try:
            df = pd.read_csv(path)
            col = get_target_col(method, df)
            if col not in df.columns:
                print(f"Warning: {col} not found in {path.name}")
                continue
            metrics = calculate_metrics(df, col)
            results[ds][mdl][method] = metrics
        except Exception as e:
            print(f"Error processing {path.name}: {e}")

    # Plot graphs and generate report
    print("\nGenerating plots and summary report...")
    metrics_list = ["Accuracy", "Consistency", "Bias", "RStd"]
    bar_colors = ["#2ECC71", "#3498DB", "#E74C3C", "#9B59B6"] # Green, Blue, Red, Purple
    
    report_lines = ["METHOD SUMMARIES REPORT", "="*80, ""]
    
    for ds, models_dict in results.items():
        models = sorted(models_dict.keys())
        if not models: continue
        
        report_lines.append(f"DATASET: {ds}")
        report_lines.append("-" * 80)
        
        for mdl in models:
            report_lines.append(f"  Model: {mdl}")
            hdr = f"    {'Method':<15} {'Accuracy%':>10} {'Bias':>10} {'RStd':>10} {'Consistency%':>12}"
            report_lines.append(hdr)
            report_lines.append("    " + "-"*60)
            for method in METHODS:
                if method in models_dict[mdl]:
                    m_dict = models_dict[mdl][method]
                    report_lines.append(
                        f"    {method:<15} "
                        f"{m_dict.get('Accuracy',0):>10.2f} "
                        f"{m_dict.get('Bias',0):>10.2f} "
                        f"{m_dict.get('RStd',0):>10.2f} "
                        f"{m_dict.get('Consistency',0):>12.2f}"
                    )
            report_lines.append("")
        
        # We will make 1 plot per method
        for method in METHODS:
            # Check if this method has any data for this dataset
            valid_models = [m for m in models if method in models_dict[m]]
            if not valid_models:
                continue
                
            fig, ax = plt.subplots(figsize=(max(10, len(valid_models)*1.5), 6))
            x = np.arange(len(valid_models))
            width = 0.2
            
            for i, metric in enumerate(metrics_list):
                vals = [models_dict[m][method].get(metric, 0) for m in valid_models]
                offset = (i - len(metrics_list)/2 + 0.5) * width
                bars = ax.bar(x + offset, vals, width, label=metric, color=bar_colors[i], alpha=0.85)
                
                # Add text labels on bars
                for j, val in enumerate(vals):
                    ax.text(x[j] + offset, val + 1, f"{val:.1f}", ha='center', va='bottom', fontsize=8, rotation=90)

            ax.set_xticks(x)
            ax.set_xticklabels(valid_models, rotation=30, ha="right", fontsize=9)
            ax.set_ylabel("Score")
            ax.set_title(f"Dataset: {ds} | Method: {method.upper()}\nMetrics Comparison", fontweight="bold")
            ax.legend(title="Metrics", bbox_to_anchor=(1.01, 1), loc='upper left')
            ax.grid(axis='y', alpha=0.3)
            # Expand y limit slightly to fit text
            ymax = max([models_dict[m][method].get(met, 0) for m in valid_models for met in metrics_list])
            ax.set_ylim(0, ymax * 1.25 if ymax > 0 else 100)
            
            plt.tight_layout()
            out_path = out_dir / f"{ds}_{method}_metrics.png"
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"Saved plot -> {out_path}")
            
    report_path = out_dir / "SUMMARY_REPORT.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nSaved report -> {report_path}")

    print("Done!")

if __name__ == "__main__":
    main()

