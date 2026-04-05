import os
import re
import pandas as pd

COMPARISON_REPORT = "pride/results/comparison/COMPARISON_REPORT.txt"
ORIGINAL_MAD_REPORT = "original_mad_graph_summary.txt"

def normalize_model(model_name):
    name = model_name.lower().replace("_transformers", "").replace("-transformers", "")
    name = re.sub(r'^microsoft_', '', name)
    name = re.sub(r'^qwen_qwen', 'qwen', name)
    name = re.sub(r'^tinyllama_tinyllama', 'tinyllama', name)
    return name

data = []

with open(COMPARISON_REPORT, "r", encoding="utf-8") as f:
    lines = f.readlines()

current_dataset = None
current_model = None

for line in lines:
    line = line.strip()
    if line.startswith("DATASET:"):
        raw_ds = line.split("DATASET:")[1].strip()
        for pfx in ["-microsoft_", "-Qwen_", "-mistral_", "-gemma3_"]:
            if raw_ds.endswith(pfx):
                raw_ds = raw_ds[:-len(pfx)]
        current_dataset = raw_ds
    elif line.startswith("Model:"):
        model_raw = line.split("Model:")[1].strip()
        current_model = normalize_model(model_raw)
    elif (line.startswith("Baseline ") or line.startswith("Baseline+PriDe") or line.startswith("AgentFull ") or line.startswith("AgentFull+PriDe")) and "vs" not in line:
        parts = line.split()
        method = parts[0]
        acc_str = parts[1].replace('%', '')
        bias_str = parts[2]
        rstd_str = parts[3]
        
        if method == "Baseline":
            method_name = "Baseline"
        elif method == "Baseline+PriDe":
            method_name = "Baseline+PriDe"
        elif method == "AgentFull":
            method_name = "AgentFull (Semantic)"
        elif method == "AgentFull+PriDe":
            method_name = "AgentFull+PriDe (Semantic)"
        else:
            continue
            
        data.append({"Dataset": current_dataset, "Model": current_model, "Method": method_name, "Accuracy": float(acc_str), "Bias": float(bias_str), "RStd": float(rstd_str)})

with open(ORIGINAL_MAD_REPORT, "r", encoding="utf-8") as f:
    lines = f.readlines()

current_dataset = None
current_model = None

for line in lines:
    line = line.strip()
    if line.startswith("DATASET:"):
        current_dataset = line.split("DATASET:")[1].strip()
    elif line.startswith("Model:"):
        model_raw = line.split("Model:")[1].strip()
        current_model = normalize_model(model_raw)
    elif line.startswith("Agent Full") and "vs" not in line:
        parts = line.split()
        acc_str = parts[2].replace('%', '')
        bias_str = parts[3]
        rstd_str = parts[4]
        
        data.append({"Dataset": current_dataset, "Model": current_model, "Method": "Pure MAD-Graph", "Accuracy": float(acc_str), "Bias": float(bias_str), "RStd": float(rstd_str)})

df = pd.DataFrame(data)

with open("academic_paper_summary.txt", "w", encoding="utf-8") as out:
    out.write("="*95 + "\n")
    out.write("ACADEMIC PAPER FULL STATISTICS (MODEL-SPECIFIC)\n")
    out.write("Results broken down by dataset and individual models.\n")
    out.write("Use these values directly for the Results/Methodology section.\n")
    out.write("="*95 + "\n\n")

    datasets = sorted(df["Dataset"].unique())
    ordered_methods = ["Baseline", "Baseline+PriDe", "Pure MAD-Graph", "AgentFull (Semantic)", "AgentFull+PriDe (Semantic)"]
    
    for ds in datasets:
        out.write(f"=== DATASET: {ds} ===\n")
        out.write(f"{'Model':<30} | {'Method':<28} | {'Acc%':<10} | {'Bias':<10} | {'RStd':<10}\n")
        out.write("-" * 95 + "\n")
        ds_data = df[df["Dataset"] == ds]
        models = sorted(ds_data["Model"].dropna().unique())
        
        for model in models:
            model_data = ds_data[ds_data["Model"] == model]
            for method in ordered_methods:
                row = model_data[model_data["Method"] == method]
                if len(row) > 0:
                    acc = row.iloc[0]["Accuracy"]
                    bias = row.iloc[0]["Bias"]
                    rstd = row.iloc[0]["RStd"]
                    out.write(f"{model[:30]:<30} | {method:<28} | {acc:>9.2f}% | {bias:>10.2f} | {rstd:>10.2f}\n")
            out.write("-" * 95 + "\n")
        out.write("\n")

    out.write("="*95 + "\n")
    out.write("=== GLOBAL AVERAGES (OVER ALL DATASETS AND ALL MODELS) ===\n")
    out.write(f"{'Method':<30} | {'Global Avg Acc%':<20} | {'Global Avg Bias':<20} | {'Global Avg RStd':<20}\n")
    out.write("-" * 95 + "\n")
    global_mean_df = df.groupby("Method")[["Accuracy", "Bias", "RStd"]].mean().reset_index()
    for method in ordered_methods:
        row = global_mean_df[global_mean_df["Method"] == method]
        if len(row) > 0:
            acc = row.iloc[0]["Accuracy"]
            bias = row.iloc[0]["Bias"]
            rstd = row.iloc[0]["RStd"]
            out.write(f"{method:<30} | {acc:>19.2f}% | {bias:>20.2f} | {rstd:>20.2f}\n")

print('Done.')
