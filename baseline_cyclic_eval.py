"""
Baseline Cyclic Permutation Evaluation

Runs the simplest possible PriDe-compatible evaluation: one model call per
cyclic permutation, no agent debate, no confidence split.

Purpose:
  Direct comparison baseline for mad_graph_selective_pride_eval.py.
  Both scripts use num_predict=3 (single-token generation), so any difference
  in PriDe debiasing effectiveness is attributable to the MAD-Graph structure
  (3 agents + confident/uncertain split) rather than generation quality.

Output (in results/baseline/):
  <dataset>-<model>_baseline.csv    4 rows per question (one per cyclic shift)

CSV format is identical to the MAD-Graph selective output EXCEPT:
  - No `confident` column  (all questions treated equally)
  - prob_A/B/C/D is a hard 1.0/0.0 signal from a single deterministic call

Feed directly into PriDe:
  python pride/pride_detail_eval.py results/baseline/<dataset>-<model>_baseline.csv

Or use the selective analysis for a side-by-side comparison:
  python pride/pride_selective_analysis.py results/baseline/<dataset>-<model>_baseline.csv
"""

import argparse
import csv
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from mad_graph_eval import (
    MCQ,
    load_mcq_csv,
    call_ollama,
    permute_options,
    correct_pos_in_permutation,
    _POSITIONS,
)

BASELINE_PROMPT = """\
Question: {question}

A. {A}
B. {B}
C. {C}
D. {D}

Answer with a single letter only (A, B, C, or D):"""

OUTPUT_DIR = Path("results/baseline")

FIELDNAMES = [
    "question_id", "permutation_idx",
    "prob_A", "prob_B", "prob_C", "prob_D",
    "predicted_answer", "correct_position", "correct_answer", "model", "temperature",
]


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_question_baseline(
    mcq: MCQ, model: str, host: str, seed: int
) -> List[Dict]:
    """
    Run one deterministic call per cyclic permutation and return 4 rows.

    Hard probabilities (1.0 / 0.0) are used since a single call cannot
    produce soft vote distributions.  PriDe still works — it estimates the
    prior from how often each position wins across the calibration set.
    """
    rows: List[Dict] = []
    for shift in range(4):
        opts = permute_options(mcq.options, shift) if shift > 0 else mcq.options

        prompt = BASELINE_PROMPT.format(
            question=mcq.question,
            A=opts["A"],
            B=opts["B"],
            C=opts["C"],
            D=opts["D"],
        )
        try:
            raw = call_ollama(model, prompt, host, temperature=0.0,
                              seed=seed, num_predict=3)
            predicted = next(
                (c for c in raw.strip().upper() if c in _POSITIONS), ""
            )
        except Exception:
            predicted = ""

        # Hard probability: chosen position = 1.0, all others = 0.0
        probs = {f"prob_{p}": (1.0 if p == predicted else 0.0) for p in _POSITIONS}
        correct_pos = correct_pos_in_permutation(mcq.answer, shift)

        rows.append({
            "question_id":      mcq.uid,
            "permutation_idx":  shift,
            **probs,
            "predicted_answer": predicted,
            "correct_position": correct_pos,
            "correct_answer":   mcq.answer,
            "model":            model,
            "temperature":      0.0,
        })

    return rows


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_baseline_evaluation(
    model: str,
    host: str,
    csv_path: str,
    seed: int,
    max_questions: Optional[int],
    num_workers: int,
):
    print("\n=== Baseline Cyclic Permutation Evaluation ===")
    print(f"Model         : {model}")
    print(f"Dataset       : {csv_path}")
    print(f"Mode          : Single call per permutation | num_predict=3 | temp=0")
    print(f"Workers       : {num_workers}")

    mcqs = load_mcq_csv(csv_path, max_questions=max_questions)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_name = Path(csv_path).stem
    model_name   = model.replace(":", "_").replace("/", "_")
    out_file     = OUTPUT_DIR / f"{dataset_name}-{model_name}_baseline.csv"

    # Checkpointing
    processed_ids: set = set()
    if out_file.exists():
        try:
            with open(out_file, "r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("question_id"):
                        processed_ids.add(row["question_id"])
        except Exception:
            pass
    if processed_ids:
        print(f"Resuming: {len(processed_ids)} questions already completed.")

    pending = [mcq for mcq in mcqs if mcq.uid not in processed_ids]

    with open(out_file, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if not processed_ids:
            writer.writeheader()

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_mcq = {
                executor.submit(
                    process_question_baseline, mcq, model, host, seed
                ): mcq
                for mcq in pending
            }

            with tqdm(total=len(mcqs), initial=len(processed_ids),
                      desc=f"Baseline [{model}]") as pbar:
                for future in concurrent.futures.as_completed(future_to_mcq):
                    try:
                        rows = future.result()
                        for row in rows:
                            writer.writerow(row)
                        fh.flush()
                    except Exception as e:
                        qid = future_to_mcq[future].uid
                        print(f"\nError on question {qid}: {e}")
                    pbar.update(1)

    print(f"\nDone.")
    print(f"  Output CSV -> {out_file}")
    print("\nNext steps:")
    print(f"  python pride/pride_detail_eval.py {out_file}")
    print(f"  python pride/pride_selective_analysis.py {out_file}  # side-by-side groups")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Baseline cyclic permutation evaluation (no MAD-Graph debate).\n"
            "Use as a comparison target against mad_graph_selective_pride_eval.py.\n"
            "Both scripts use num_predict=3, making results directly comparable."
        )
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Ollama model name (e.g. gemma3:4b)")
    parser.add_argument("--host", type=str, default="http://localhost:11434",
                        help="Ollama server URL")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to MCQ dataset CSV")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Cap number of questions evaluated")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Parallel workers")

    args = parser.parse_args()

    run_baseline_evaluation(
        model=args.model,
        host=args.host,
        csv_path=args.input,
        seed=args.seed,
        max_questions=args.max_questions,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
