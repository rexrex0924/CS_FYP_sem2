import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

CALIB_RATIO = 0.15
POSITIONS = ["A", "B", "C", "D"]

class PriDe:
    def __init__(self, calib=CALIB_RATIO, alpha=1.0, seed=42):
        self.calib, self.alpha, self.seed = calib, alpha, seed
        self._prior = None

    def _split(self, df):
        ids = df["question_id"].unique()
        rng = np.random.RandomState(self.seed)
        rng.shuffle(ids)
        n = max(1, int(len(ids) * self.calib))
        # cal, test
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

    def _debias(self, df_to_debias):
        out = df_to_debias.copy()
        preds = []
        for _, row in out.iterrows():
            obs = np.array([row.get(f"prob_{p}", 0.0) for p in POSITIONS])
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
        # Return debiased test set only to match analysis behavior
        return self._debias(test)

def run_pride(df, fixed_alpha=None):
    alphas = ([fixed_alpha] if fixed_alpha is not None
              else np.round(np.arange(0.0, 1.05, 0.1), 1).tolist())
    best_alpha, best_acc, best_deb = alphas[0], -1.0, None
    
    for a in alphas:
        p = PriDe(calib=CALIB_RATIO, alpha=float(a))
        deb = p.fit_predict(df)
        acc = deb["debiased_is_correct"].mean()
        if acc > best_acc:
            best_acc, best_alpha, best_deb = acc, a, deb
            
    return best_deb, best_alpha

def main():
    parser = argparse.ArgumentParser(description="Apply PriDe to AgentFull results and specify exact columns")
    parser.add_argument("--input-dir", type=str, default="results/mad_graph_selective/output", help="Directory with original PriDe CSVs")
    parser.add_argument("--output-dir", type=str, default="organized/agents_pride_csv", help="Output directory")
    parser.add_argument("--fixed-alpha", type=float, default=0.3, help="Alpha to use. If 0, does grid search.")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        print(f"Error: Input directory {in_dir} does not exist.")
        return

    fixed_alpha = args.fixed_alpha if args.fixed_alpha != 0.0 else None

    # Required column order
    cols_order = [
        "question_id", "permutation_idx", "prob_A", "prob_B", "prob_C", "prob_D",
        "predicted_answer", "correct_position", "correct_answer", "model",
        "temperature", "is_correct_fixed", "debiased_predicted_answer", "debiased_is_correct"
    ]

    csv_files = list(in_dir.glob("*_pride.csv"))
    if not csv_files:
        csv_files = list(in_dir.glob("*.csv"))
        
    print(f"Found {len(csv_files)} files to process in {in_dir}")

    for fp in csv_files:
        print(f"Processing {fp.name}...")
        df = pd.read_csv(fp)
        
        # Format string matching
        if "predicted_answer" in df.columns:
            df["predicted_answer"] = df["predicted_answer"].astype(str).str.upper().str.strip()
        else:
            df["predicted_answer"] = ""
            
        if "correct_position" in df.columns:
            df["correct_position"] = df["correct_position"].astype(str).str.upper().str.strip()
        else:
            df["correct_position"] = ""
            
        # Create is_correct_fixed base on original prediction
        df["is_correct_fixed"] = (df["predicted_answer"] == df["correct_position"]).astype(int)

        # Make sure probability columns exist in case they're missing
        for p in POSITIONS:
            col = f"prob_{p}"
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
                
        # Fill in missing categorical columns if any
        for col in ["correct_answer", "model", "temperature"]:
            if col not in df.columns:
                df[col] = ""
                
        # Run PriDe
        deb_df, best_alpha = run_pride(df, fixed_alpha)
        before_acc = deb_df["is_correct_fixed"].mean() * 100
        after_acc = deb_df["debiased_is_correct"].mean() * 100
        
        def calculate_bias(d, col):
            valid = d[d[col].isin(POSITIONS)]
            if len(valid) == 0: return 0.0
            counts = valid[col].value_counts().reindex(POSITIONS, fill_value=0)
            return float(np.std((counts / len(valid) * 100).values))
            
        before_bias = calculate_bias(deb_df, "predicted_answer")
        after_bias = calculate_bias(deb_df, "debiased_predicted_answer")

        print(f"  -> Best Alpha Used: {best_alpha:.1f}")
        print(f"  -> Accuracy: {before_acc:.2f}% -> {after_acc:.2f}% ({(after_acc - before_acc):+.2f}%)")
        print(f"  -> Bias Score: {before_bias:.2f} -> {after_bias:.2f} (lower is better)")
        
        # Ensure we only export the requested columns (in order)
        final_df = deb_df[cols_order]
        
        out_fp = out_dir / fp.name
        final_df.to_csv(out_fp, index=False)
        print(f"  -> Saved {len(final_df)} structured rows to {out_fp}\n")

if __name__ == "__main__":
    main()
