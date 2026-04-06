import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil

def parse_summary(filepath="academic_paper_summary.txt"):
    data = []
    current_dataset = None
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found!")
        return pd.DataFrame()

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("=== DATASET:"):
                current_dataset = line.split("=== DATASET:")[1].replace("===", "").strip()
            elif "|" in line and "Acc%" not in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) == 5:
                    try:
                        acc = float(parts[2].replace('%', ''))
                        bias = float(parts[3])
                        data.append({
                            "Dataset": current_dataset,
                            "Model": parts[0],
                            "Method": parts[1],
                            "Accuracy (%)": acc,
                            "Bias Score": bias
                        })
                    except ValueError:
                        continue
    return pd.DataFrame(data)

def generate_visuals(df, output_dir="visualizations"):
    # Delete the folder if it exists to keep everything clean
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    sns.set_theme(style="whitegrid", font_scale=1.1)
    method_order = ["Baseline", "Baseline+PriDe", "Pure MAD-Graph", "AgentFull (Semantic)", "AgentFull+PriDe (Semantic)"]
    clean_df = df[df["Accuracy (%)"] > 0].copy()

    print("Generating individual model charts per dataset...")
    for ds in clean_df["Dataset"].unique():
        ds_data = clean_df[clean_df["Dataset"] == ds]
        
        # --- 1. Grouped Bar Chart: Accuracy by Model & Method ---
        plt.figure(figsize=(16, 8))
        sns.barplot(data=ds_data, x="Model", y="Accuracy (%)", hue="Method", hue_order=method_order, palette="Blues")
        plt.title(f"Accuracy by Model - {ds} (Higher is Better)", pad=15, fontweight="bold")
        plt.ylim(0, 105)
        plt.xticks(rotation=30, ha='right')
        plt.legend(title="Methodology", bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"Bar_Accuracy_{ds}.png"), dpi=300)
        plt.close()

        # --- 2. Grouped Bar Chart: Bias by Model & Method ---
        plt.figure(figsize=(16, 8))
        sns.barplot(data=ds_data, x="Model", y="Bias Score", hue="Method", hue_order=method_order, palette="Reds")
        plt.title(f"Position Bias by Model - {ds} (Lower is Better)", pad=15, fontweight="bold")
        plt.xticks(rotation=30, ha='right')
        plt.legend(title="Methodology", bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"Bar_Bias_{ds}.png"), dpi=300)
        plt.close()

        # --- 3. Accuracy Heatmap ---
        acc_pivot = ds_data.pivot(index="Model", columns="Method", values="Accuracy (%)")
        avail_cols = [c for c in method_order if c in acc_pivot.columns]
        acc_pivot = acc_pivot[avail_cols]
        
        plt.figure(figsize=(10, len(acc_pivot) * 0.6 + 2))
        sns.heatmap(acc_pivot, annot=True, fmt=".1f", cmap="Blues", cbar_kws={'label': 'Accuracy %'}, linewidths=1)
        plt.title(f"Accuracy Matrix (%) - {ds}", pad=15, fontweight="bold")
        plt.xlabel("")
        plt.ylabel("")
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"Heatmap_Accuracy_{ds}.png"), dpi=300)
        plt.close()

        # --- 4. Bias Heatmap ---
        bias_pivot = ds_data.pivot(index="Model", columns="Method", values="Bias Score")
        bias_pivot = bias_pivot[avail_cols]
        
        plt.figure(figsize=(10, len(bias_pivot) * 0.6 + 2))
        sns.heatmap(bias_pivot, annot=True, fmt=".2f", cmap="Reds", cbar_kws={'label': 'Bias Score'}, linewidths=1)
        plt.title(f"Position Bias Matrix - {ds}", pad=15, fontweight="bold")
        plt.xlabel("")
        plt.ylabel("")
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"Heatmap_Bias_{ds}.png"), dpi=300)
        plt.close()

    print(f"✅ Generated {len(clean_df['Dataset'].unique()) * 4} individual model charts in '{output_dir}'.")

if __name__ == "__main__":
    df = parse_summary()
    generate_visuals(df)