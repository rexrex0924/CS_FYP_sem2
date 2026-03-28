"""
MAD-Graph Selective PriDe Evaluation

Research hypothesis: LLMs know the answer when agents unanimously agree;
when they disagree, positional bias fills the uncertainty gap.

Pipeline per question:
  1. Diversity check  — run 3 agents (full generation, temperature > 0, distinct seeds
                        + personas) on the original question to get a "confident" /
                        "uncertain" label.  Unanimous agreement -> confident=1, else 0.
  2. Permutation pass — run 3 agents (full generation, temperature = 0, deterministic)
                        on all 4 cyclic permutations of EVERY question regardless of
                        its label. Records vote-based soft probabilities in PriDe format.

Single-token generation (num_predict=3) is used throughout for speed (~15-30x
faster per call). Accuracy is compared against baseline_cyclic_eval.py which
uses the same constraint, making the comparison fair and apples-to-apples.

Output (in results/mad_graph_selective/):
  <dataset>-<model>_pride.csv    4 rows per question (one per cyclic shift).
                                 Extra columns: correct_answer, confident.

Downstream usage:
  - mad_graph_selective_analysis.py  splits by `confident` to test the hypothesis.
  - pride/pride_detail_eval.py       can consume the CSV directly (ignores extra cols).
    To run PriDe on the uncertain subset only, pre-filter: confident == 0.
"""

import argparse
import csv
import concurrent.futures
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from tqdm import tqdm

from mad_graph_eval import (
    MCQ,
    load_mcq_csv,
    call_ollama,
    AGENT_PROMPTS,
    permute_options,
    correct_pos_in_permutation,
    votes_to_probs,
    _POSITIONS,
)

# Lightweight prompt — each agent uses its own persona but only outputs a single
# letter.  Keeping generation to num_predict=3 makes each call ~15-30x faster
# than full chain-of-thought, which matters given 15 calls per question.
# Accuracy is compared against the baseline_cyclic_eval.py script which uses
# the *same* generation constraint, making the comparison fair.
ANSWER_ONLY_PROMPT = """\
{agent_prompt}

Question: {question}

A. {A}
B. {B}
C. {C}
D. {D}

Answer with a single letter only (A, B, C, or D):"""

OUTPUT_DIR = Path("results/mad_graph_selective")

FIELDNAMES = [
    "question_id", "permutation_idx",
    "prob_A", "prob_B", "prob_C", "prob_D",
    "predicted_answer", "correct_position", "correct_answer", "model", "confident",
]


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def _run_phase1(
    mcq: MCQ,
    model: str,
    host: str,
    temperature: float,
    seed: int,
    options_override: Dict[str, str] | None = None,
    deterministic: bool = False,
) -> Dict[int, str]:
    """
    Call 3 agents once each and return their answer letters.

      deterministic=False  (diversity check):
        Uses caller-supplied temperature and distinct seeds so agents can genuinely
        disagree based on different reasoning paths.

      deterministic=True  (permutation pass):
        Forces temperature=0 for a stable positional-preference signal.
        All 3 agents are effectively identical here; we still run 3 to keep
        the vote-probability format consistent across both modes.

    num_predict=3 gives enough room for any leading whitespace before the letter
    while keeping each call ~15-30x faster than full chain-of-thought.
    Accuracy is compared against baseline_cyclic_eval.py which uses the same
    generation constraint, so the comparison is apples-to-apples.
    """
    opts = options_override if options_override else mcq.options
    votes: Dict[int, str] = {}
    for agent_id, agent_persona in AGENT_PROMPTS.items():
        prompt = ANSWER_ONLY_PROMPT.format(
            agent_prompt=agent_persona,
            question=mcq.question,
            A=opts["A"],
            B=opts["B"],
            C=opts["C"],
            D=opts["D"],
        )
        try:
            t = 0.0 if deterministic else temperature
            raw = call_ollama(model, prompt, host, temperature=t,
                              seed=seed + agent_id, num_predict=3)
            letter = next((c for c in raw.strip().upper() if c in _POSITIONS), "")
            votes[agent_id] = letter
        except Exception:
            votes[agent_id] = ""
    return votes


def process_question_selective(
    mcq: MCQ, model: str, host: str, temperature: float, seed: int
) -> Tuple[bool, List[Dict]]:
    """
    Returns (is_confident, [4 rows]).

    Always produces 4 rows — one per cyclic permutation — so every question
    contributes to the PriDe bias signal regardless of its label.

    Step 1 uses temperature > 0 for genuine inter-agent diversity.
    Step 2 uses temperature = 0 for reproducible positional-preference measurement.
    """
    # --- Step 1: diversity check ---
    orig_votes = _run_phase1(mcq, model, host, temperature, seed, deterministic=False)
    valid_orig = {aid: ans for aid, ans in orig_votes.items() if ans in _POSITIONS}
    vote_values = list(valid_orig.values())
    is_confident = (len(set(vote_values)) == 1 and len(vote_values) >= 2)

    # --- Step 2: deterministic permutation passes (all 4 cyclic shifts) ---
    rows: List[Dict] = []
    for shift in range(4):
        perm_opts = permute_options(mcq.options, shift) if shift > 0 else None
        raw = _run_phase1(
            mcq, model, host, temperature,
            seed=seed + shift * 1000,
            options_override=perm_opts,
            deterministic=True,
        )
        perm_valid = {aid: ans for aid, ans in raw.items() if ans in _POSITIONS}

        probs = votes_to_probs(perm_valid)
        correct_pos = correct_pos_in_permutation(mcq.answer, shift)
        predicted = (
            Counter(perm_valid.values()).most_common(1)[0][0] if perm_valid else ""
        )

        rows.append({
            "question_id":      mcq.uid,
            "permutation_idx":  shift,
            "prob_A":           probs["prob_A"],
            "prob_B":           probs["prob_B"],
            "prob_C":           probs["prob_C"],
            "prob_D":           probs["prob_D"],
            "predicted_answer": predicted,
            "correct_position": correct_pos,
            "correct_answer":   mcq.answer,
            "model":            model,
            "confident":        int(is_confident),
        })

    return is_confident, rows


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_selective_evaluation(
    model: str,
    host: str,
    csv_path: str,
    seed: int,
    max_questions: Optional[int],
    temperature: float,
    num_workers: int,
):
    print("\n=== MAD-Graph Selective PriDe Evaluation ===")
    print(f"Model         : {model}")
    print(f"Dataset       : {csv_path}")
    print(f"Temperature   : {temperature}  (diversity check; permutation passes use temp=0)")
    print(f"Workers       : {num_workers}")
    print("Mode          : All questions -> 4 cyclic permutations + confident flag")

    mcqs = load_mcq_csv(csv_path, max_questions=max_questions)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_name = Path(csv_path).stem
    model_name = model.replace(":", "_").replace("/", "_")
    out_file = OUTPUT_DIR / f"{dataset_name}-{model_name}_pride.csv"

    # Checkpointing: question_id is repeated 4 times (one per perm),
    # so the set naturally deduplicates to the number of completed questions.
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

        n_confident = 0
        n_uncertain = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_mcq = {
                executor.submit(
                    process_question_selective,
                    mcq, model, host, temperature, seed
                ): mcq
                for mcq in pending
            }

            with tqdm(total=len(mcqs), initial=len(processed_ids),
                      desc=f"Selective [{model}]") as pbar:
                for future in concurrent.futures.as_completed(future_to_mcq):
                    try:
                        is_conf, rows = future.result()
                        for row in rows:
                            writer.writerow(row)
                        fh.flush()
                        if is_conf:
                            n_confident += 1
                        else:
                            n_uncertain += 1
                    except Exception as e:
                        qid = future_to_mcq[future].uid
                        print(f"\nError on question {qid}: {e}")
                    pbar.update(1)

    total = n_confident + n_uncertain
    if total > 0:
        print(f"\nDone.")
        print(f"  Confident (unanimous) : {n_confident}  ({n_confident/total:.1%})")
        print(f"  Uncertain             : {n_uncertain}  ({n_uncertain/total:.1%})")
    print(f"\n  Output CSV -> {out_file}")
    print("\nNext steps:")
    print("  1. Run mad_graph_selective_analysis.py to compare confident vs uncertain.")
    print(f"  2. Run pride/pride_detail_eval.py {out_file}")
    print("     (Add --filter-confident=0 to run PriDe on uncertain subset only)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "MAD-Graph Selective PriDe: label questions by agent agreement, "
            "then run cyclic permutations on ALL questions for PriDe-compatible output."
        )
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Ollama model name (e.g. gemma3:4b)")
    parser.add_argument("--host", type=str, default="http://localhost:11434",
                        help="Ollama server URL")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to MCQ dataset CSV")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Sampling temperature for diversity check (must be > 0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Cap number of questions evaluated")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Parallel workers")

    args = parser.parse_args()

    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0 (needed for diversity check)")

    run_selective_evaluation(
        model=args.model,
        host=args.host,
        csv_path=args.input,
        seed=args.seed,
        max_questions=args.max_questions,
        temperature=args.temperature,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
