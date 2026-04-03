import re
import os

summary_path = r"sem1/results/pride_summary/SUMMARY_REPORT.txt"
comparison_path = r"pride/results/comparison/COMPARISON_REPORT.txt"

def normalize_name(name):
    # Remove common suffixes that might differ between semesters
    name = name.lower().replace("_transformers", "").replace("-transformers", "")
    # Remove prefix like "microsoft_" or "Qwen_" if it's duplicated
    name = re.sub(r'^(microsoft|qwen)_', '', name)
    return name

# Data structure: {(dataset, model): {'baseline_pride': {...}, 'agent_pride': {...}}}
data = {}

# 1. Parse SUMMARY_REPORT.txt (Last Sem)
if os.path.exists(summary_path):
    with open(summary_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    current_dataset = None
    for line in lines:
        ds_match = re.search(r"DATASET:\s*(\S+)", line)
        if ds_match:
            current_dataset = ds_match.group(1)
            continue
            
        row_match = re.match(r"^(\S+)\s+([0-9.]+)\s+([0-9.]+%?)→([0-9.]+%?)\s+([0-9.]+%?)→([0-9.]+%?)\s+([0-9.]+)→([0-9.]+)\s+([0-9.]+)→([0-9.]+)", line)
        if row_match and current_dataset:
            model = row_match.group(1)
            acc_d = row_match.group(4)
            cons_d = row_match.group(6)
            bias_d = row_match.group(8)
            rstd_d = row_match.group(10)
            
            norm_model = normalize_name(model)
            key = (current_dataset, norm_model)
            if key not in data:
                data[key] = {'original_name': model}
            data[key]['baseline_pride'] = {
                'Acc': acc_d, 'Cons': cons_d, 'Bias': bias_d, 'RStd': rstd_d
            }

# 2. Parse COMPARISON_REPORT.txt (This Sem)
if os.path.exists(comparison_path):
    with open(comparison_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    current_dataset = None
    current_model = None
    
    for line in lines:
        ds_match = re.search(r"^DATASET:\s*(\S+)", line)
        if ds_match:
            current_dataset = ds_match.group(1)
            continue
            
        mod_match = re.search(r"^\s+Model:\s*(\S+)", line)
        if mod_match:
            current_model = mod_match.group(1)
            continue
            
        row_match = re.match(r"^\s*AgentFull\+PriDe\s+([0-9.]+%?)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+%?)", line)
        if row_match and current_dataset and current_model:
            acc = row_match.group(1)
            bias = row_match.group(2)
            rstd = row_match.group(3)
            cons = row_match.group(4)
            
            norm_model = normalize_name(current_model)
            
            # Find closest match if not exact
            matched_key = None
            for (d, m) in data.keys():
                if d == current_dataset and (m in norm_model or norm_model in m):
                    matched_key = (d, m)
                    break
            
            if not matched_key:
                matched_key = (current_dataset, norm_model)
                if matched_key not in data:
                    data[matched_key] = {'original_name': current_model}
                    
            data[matched_key]['agent_pride'] = {
                'Acc': acc, 'Cons': cons, 'Bias': bias, 'RStd': rstd
            }

# 3. Build Comparison Text File
output_lines = []
output_lines.append("="*80)
output_lines.append("CROSS-SEMESTER COMPARISON: Baseline+PriDe (Last Sem) vs AgentFull+PriDe (This Sem)")
output_lines.append("="*80 + "\n")

datasets = set(k[0] for k in data.keys())

for ds in sorted(datasets):
    output_lines.append(f"DATASET: {ds}")
    output_lines.append("-" * 100)
    output_lines.append(f"{'Model':<30} | {'Acc (BP->AP)':<18} | {'Cons (BP->AP)':<18} | {'Bias (BP->AP)':<18} | {'RStd (BP->AP)':<18}")
    output_lines.append("-" * 100)
    
    # Filter and sort models for this dataset
    ds_models = {k: v for k, v in data.items() if k[0] == ds}
    
    for (d, m), metrics in sorted(ds_models.items(), key=lambda x: x[0][1]):
        bp = metrics.get('baseline_pride')
        ap = metrics.get('agent_pride')
        orig_name = metrics.get('original_name', m)
        
        if bp and ap:
            acc_str = f"{bp['Acc']} -> {ap['Acc']}"
            cons_str = f"{bp['Cons']} -> {ap['Cons']}"
            bias_str = f"{bp['Bias']} -> {ap['Bias']}"
            rstd_str = f"{bp['RStd']} -> {ap['RStd']}"
            
            output_lines.append(f"{orig_name[:28]:<30} | {acc_str:<18} | {cons_str:<18} | {bias_str:<18} | {rstd_str:<18}")
        elif bp:
            output_lines.append(f"{orig_name[:28]:<30} | Missing Agent+PriDe (This Sem)")
        elif ap:
            output_lines.append(f"{orig_name[:28]:<30} | Missing Baseline+PriDe (Last Sem)")

    output_lines.append("\n")

with open("semester_comparison_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))
    
print("Summary successfully written to semester_comparison_summary.txt")
