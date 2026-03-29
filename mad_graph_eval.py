"""
MAD-Graph: Graph-Based Multi-Agent Debate for Positional Bias Mitigation

Three LLM agent personas independently answer each MCQ. If they disagree,
they debate each other's reasoning. Final answer is resolved by vote counting
(In-Degree Centrality on a support graph).

References:
  - Du et al. (2023) https://arxiv.org/abs/2305.14325
  - Chan et al. (2023) https://arxiv.org/abs/2308.07201
  - Besta et al. (2023) https://arxiv.org/abs/2308.09687
"""

import argparse
import csv
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import concurrent.futures
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import requests


PROMPT_TEMPLATE = """Question: {question}

A. {A}
B. {B}
C. {C}
D. {D}

{agent_prompt}

End your response by clearly stating your final answer as exactly one letter (A, B, C, or D) inside <answer> tags, like this: <answer>A</answer>"""

DEBATE_PROMPT_TEMPLATE = """Question: {question}

A. {A}
B. {B}
C. {C}
D. {D}

You previously chose {my_ans}.

Other agents' current answers and reasoning:
{other_responses}
Carefully review the reasoning above. You may keep your answer or change it to any option (A, B, C, or D) if you find another argument more convincing.

End your response by clearly stating your final answer as exactly one letter (A, B, C, or D) inside <answer> tags, like this: <answer>B</answer>"""

# The three agent personas (Phase 1)
AGENT_PROMPTS = {
    1: "Solve this step-by-step using logical deduction.",
    2: "Identify the most common trap or misconception in this question and avoid it.",
    3: "Give your immediate, most confident answer based on core principles."
}


@dataclass
class MCQ:
    uid: str
    question: str
    options: Dict[str, str]
    answer: str


def load_mcq_csv(path: str, max_questions: Optional[int] = None) -> List[MCQ]:
    df = pd.read_csv(path, keep_default_na=False, na_values=[''])
    required_cols = {"id", "question", "option_a", "option_b", "option_c", "option_d", "answer"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing columns in CSV: {missing}")

    mcqs = []
    for _, row in df.iterrows():
        ans = str(row["answer"]).strip().upper()
        if ans not in {"A", "B", "C", "D"}:
            continue

        question = str(row["question"]).strip()
        options_text = [
            str(row["option_a"]).strip(),
            str(row["option_b"]).strip(),
            str(row["option_c"]).strip(),
            str(row["option_d"]).strip()
        ]

        if question in ["", "nan"] or any(opt in ["", "nan"] for opt in options_text):
            continue

        mcqs.append(MCQ(
            uid=str(row["id"]),
            question=question,
            options={"A": options_text[0], "B": options_text[1],
                     "C": options_text[2], "D": options_text[3]},
            answer=ans,
        ))

    if max_questions:
        mcqs = mcqs[:max_questions]

    print(f"Loaded {len(mcqs)} questions from {path}")
    return mcqs


def call_ollama(model: str, prompt: str, host: str = "http://localhost:11434",
                temperature: float = 0.7, seed: int = 42,
                retries: int = 3, timeout: int = 180,
                num_predict: int | None = None) -> str:
    url = f"{host.rstrip('/')}/api/generate"
    options: dict = {"temperature": temperature, "seed": seed}
    if num_predict is not None:
        options["num_predict"] = num_predict
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }
    for attempt in range(retries):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            text = response.json().get("response", "").strip()
            if not text:
                raise RuntimeError("Empty response from model")
            return text
        except requests.exceptions.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(1.0 + attempt * 0.5)
    return ""


def extract_answer_and_reasoning(response_text: str) -> Tuple[str, str]:
    """Extract letter from <answer> tags; fall back to last standalone letter."""
    match = re.search(r'<answer>\s*([A-D])\s*</answer>', response_text, re.IGNORECASE)
    if match:
        return match.group(1).upper(), response_text[:match.start()].strip()

    matches = re.findall(r'\b([A-D])\b', response_text.upper())
    if matches:
        return matches[-1], response_text.strip()

    return "", response_text.strip()


def run_mad_graph(mcq: MCQ, model: str, host: str, temperature: float, seed: int,
                  debate_rounds: int = 2) -> Tuple[str, Dict, Dict]:
    """
    Runs the full MAD-Graph pipeline for a single question.

    Phase 1 — Divergent Generation:
        Each agent independently answers with a distinct reasoning persona.
    Phase 2 — Multi-Round Debate:
        Each agent sees ALL other agents' current answers and reasoning, then
        updates its own answer. Repeated for `debate_rounds` rounds.
        (Du et al. 2023: "debate over multiple rounds to arrive at a common answer")
    Phase 3 — Graph Resolution:
        Majority vote over final answers (in-degree centrality approximation).

    Returns (final_answer, initial_responses).
    """
    # --- Phase 1: Divergent Generation ---
    initial_responses: Dict[int, Dict] = {}
    for agent_id, agent_prompt in AGENT_PROMPTS.items():
        prompt = PROMPT_TEMPLATE.format(
            question=mcq.question,
            A=mcq.options["A"],
            B=mcq.options["B"],
            C=mcq.options["C"],
            D=mcq.options["D"],
            agent_prompt=agent_prompt
        )
        try:
            response = call_ollama(model, prompt, host, temperature, seed=seed + agent_id)
            ans, reasoning = extract_answer_and_reasoning(response)
            initial_responses[agent_id] = {"ans": ans, "reasoning": reasoning}
        except Exception:
            initial_responses[agent_id] = {"ans": "", "reasoning": ""}

    valid_votes = {aid: r["ans"] for aid, r in initial_responses.items()
                  if r["ans"] in ["A", "B", "C", "D"]}

    if not valid_votes:
        return "", initial_responses, {}

    # All agents agree — no debate needed
    if len(set(valid_votes.values())) == 1:
        return list(valid_votes.values())[0], initial_responses, valid_votes

    # --- Phase 2: Multi-Round Debate ---
    # current_responses tracks the live answers+reasoning across rounds
    current_responses = {aid: dict(r) for aid, r in initial_responses.items()}

    for round_idx in range(debate_rounds):
        # Snapshot answers at the start of this round so all agents debate
        # against the same set of responses (simultaneous update)
        round_snapshot = {aid: dict(r) for aid, r in current_responses.items()}
        next_responses  = {aid: dict(r) for aid, r in current_responses.items()}

        for agent_id, r in round_snapshot.items():
            my_ans = r["ans"]
            if my_ans not in ["A", "B", "C", "D"]:
                continue

            # Build a summary of ALL other agents' current answers + reasoning
            other_lines = []
            for other_id, other_r in round_snapshot.items():
                if other_id == agent_id:
                    continue
                if other_r["ans"] not in ["A", "B", "C", "D"]:
                    continue
                snippet = other_r["reasoning"][:600].strip()
                other_lines.append(
                    f"Agent {other_id} chose {other_r['ans']}:\n{snippet}"
                )

            if not other_lines:
                continue  # no valid opponents — nothing to debate

            other_responses_text = "\n\n".join(other_lines)

            debate_prompt = DEBATE_PROMPT_TEMPLATE.format(
                question=mcq.question,
                A=mcq.options["A"],
                B=mcq.options["B"],
                C=mcq.options["C"],
                D=mcq.options["D"],
                my_ans=my_ans,
                other_responses=other_responses_text
            )

            try:
                response = call_ollama(
                    model, debate_prompt, host, temperature,
                    seed=seed + 100 * (round_idx + 1) + agent_id
                )
                new_ans, new_reasoning = extract_answer_and_reasoning(response)
                if new_ans in ["A", "B", "C", "D"]:
                    next_responses[agent_id]["ans"] = new_ans
                    next_responses[agent_id]["reasoning"] = new_reasoning
            except Exception:
                pass  # keep previous answer on failure

        current_responses = next_responses

        # Early exit if all valid agents now agree
        final_valid = {aid: r["ans"] for aid, r in current_responses.items()
                       if r["ans"] in ["A", "B", "C", "D"]}
        if len(set(final_valid.values())) == 1:
            break

    # --- Phase 3: Graph Resolution (In-Degree Centrality / Majority Vote) ---
    final_valid = {aid: r["ans"] for aid, r in current_responses.items()
                   if r["ans"] in ["A", "B", "C", "D"]}
    if not final_valid:
        return "", initial_responses, {}

    final_answer = Counter(final_valid.values()).most_common(1)[0][0]
    return final_answer, initial_responses, final_valid


def process_question(mcq: MCQ, model: str, host: str, temperature: float, seed: int,
                     debate_rounds: int = 2) -> Dict:
    predicted_answer, initial_responses, _ = run_mad_graph(
        mcq, model, host, temperature, seed, debate_rounds=debate_rounds)
    return {
        "question_id": mcq.uid,
        "model": model,
        "predicted_answer": predicted_answer,
        "correct_answer": mcq.answer,
        "is_correct": int(predicted_answer == mcq.answer),
        "agent_1_ans": initial_responses.get(1, {}).get("ans", ""),
        "agent_2_ans": initial_responses.get(2, {}).get("ans", ""),
        "agent_3_ans": initial_responses.get(3, {}).get("ans", ""),
        "question": mcq.question,
    }


# ---------------------------------------------------------------------------
# PriDe-compatible evaluation helpers
# ---------------------------------------------------------------------------

_POSITIONS = ["A", "B", "C", "D"]


def permute_options(options: Dict[str, str], shift: int) -> Dict[str, str]:
    """
    Cyclically shift answer options by `shift` positions.
    Position i in the output shows the original option at index (i + shift) % 4.
      shift=0: [A,B,C,D] -> [A,B,C,D]
      shift=1: [A,B,C,D] -> [B,C,D,A]   (original B now labelled A, etc.)
      shift=2: [A,B,C,D] -> [C,D,A,B]
      shift=3: [A,B,C,D] -> [D,A,B,C]
    """
    return {_POSITIONS[i]: options[_POSITIONS[(i + shift) % 4]] for i in range(4)}


def correct_pos_in_permutation(correct_answer: str, shift: int) -> str:
    """
    Given the original correct answer letter and the cyclic shift, return
    the label that letter appears under in the permuted layout.
    orig_idx=0 (A), shift=1  ->  new label = (0-1)%4 = 3 = D
    """
    orig_idx = _POSITIONS.index(correct_answer)
    return _POSITIONS[(orig_idx - shift) % 4]


def votes_to_probs(final_valid: Dict) -> Dict[str, float]:
    """
    Convert agent final-vote dict {agent_id: answer_letter} to a probability
    distribution over positions A/B/C/D (fraction of agents who voted each).
    With 3 agents, possible values are 0.0, 0.333, 0.667, 1.0.
    """
    n = len(final_valid) if final_valid else 1
    counts = Counter(final_valid.values())
    return {f"prob_{p}": counts.get(p, 0) / n for p in _POSITIONS}


def process_question_for_pride(mcq: MCQ, model: str, host: str, temperature: float,
                                seed: int, debate_rounds: int = 2) -> List[Dict]:
    """
    Run MAD-Graph on all 4 cyclic permutations of the question's options and
    return a list of 4 rows in PriDe-compatible CSV format:
      question_id, permutation_idx, prob_A, prob_B, prob_C, prob_D,
      predicted_answer, correct_position, model
    The soft probabilities (prob_*) are derived from the final agent vote
    distribution after debate, giving PriDe the signal it needs to estimate
    and subtract the model's positional prior.
    """
    rows = []
    for shift in range(4):
        perm_options = permute_options(mcq.options, shift)
        correct_pos = correct_pos_in_permutation(mcq.answer, shift)

        # Build a temporary MCQ with permuted options so run_mad_graph uses them
        perm_mcq = MCQ(
            uid=mcq.uid,
            question=mcq.question,
            options=perm_options,
            answer=correct_pos,  # correct answer in permuted space
        )
        _, _, final_valid = run_mad_graph(
            perm_mcq, model, host, temperature,
            seed=seed + shift * 1000, debate_rounds=debate_rounds
        )

        probs = votes_to_probs(final_valid)
        predicted = Counter(final_valid.values()).most_common(1)[0][0] if final_valid else ""

        rows.append({
            "question_id":      mcq.uid,
            "permutation_idx":  shift,
            "prob_A":           probs["prob_A"],
            "prob_B":           probs["prob_B"],
            "prob_C":           probs["prob_C"],
            "prob_D":           probs["prob_D"],
            "predicted_answer": predicted,
            "correct_position": correct_pos,
            "model":            model,
        })
    return rows


def run_evaluation(model: str, host: str, csv_path: str, seed: int,
                   max_questions: Optional[int], temperature: float, num_workers: int,
                   debate_rounds: int = 2, for_pride: bool = False):
    mode_label = "MAD-Graph (PriDe-compatible)" if for_pride else "MAD-Graph"
    print(f"\n=== Starting {mode_label} Evaluation ===")
    print(f"Model         : {model}")
    print(f"Dataset       : {csv_path}")
    print(f"Temperature   : {temperature}")
    print(f"Workers       : {num_workers}")
    print(f"Debate rounds : {debate_rounds}")
    if for_pride:
        print("Mode          : --for-pride  (4 permutations per question, vote probabilities)")

    mcqs = load_mcq_csv(csv_path, max_questions=max_questions)

    dataset_name = Path(csv_path).stem
    model_name = model.replace(':', '_').replace('/', '_')

    if for_pride:
        csv_dir = Path("results/mad_graph_pride")
        csv_output_file = csv_dir / f"{dataset_name}-{model_name}_mad_graph_pride.csv"
        fieldnames = [
            "question_id", "permutation_idx",
            "prob_A", "prob_B", "prob_C", "prob_D",
            "predicted_answer", "correct_position", "model",
        ]
    else:
        csv_dir = Path("results/mad_graph")
        csv_output_file = csv_dir / f"{dataset_name}-{model_name}_mad_graph.csv"
        fieldnames = [
            "question_id", "model", "predicted_answer", "correct_answer", "is_correct",
            "agent_1_ans", "agent_2_ans", "agent_3_ans", "question",
        ]

    csv_dir.mkdir(parents=True, exist_ok=True)

    # Checkpointing — track by question_id (all permutations re-run together)
    processed_ids: set = set()
    if csv_output_file.exists():
        try:
            with open(csv_output_file, 'r', newline='', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    if row.get('question_id'):
                        processed_ids.add(row['question_id'])
            print(f"Resuming: {len(processed_ids)} questions already completed.")
        except Exception:
            processed_ids = set()

    pending = [mcq for mcq in mcqs if mcq.uid not in processed_ids]

    with open(csv_output_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not processed_ids and f.tell() == 0:
            writer.writeheader()

        if for_pride:
            # Each task returns a list of 4 rows (one per permutation)
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_id = {
                    executor.submit(process_question_for_pride, mcq, model, host,
                                    temperature, seed, debate_rounds): mcq.uid
                    for mcq in pending
                }
                with tqdm(total=len(mcqs), initial=len(processed_ids),
                          desc=f"MAD-Graph PriDe [{model}]") as pbar:
                    for future in concurrent.futures.as_completed(future_to_id):
                        try:
                            rows = future.result()
                            for row in rows:
                                writer.writerow(row)
                            f.flush()
                        except Exception as e:
                            qid = future_to_id[future]
                            print(f"\nError on question {qid}: {e}")
                        pbar.update(1)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_id = {
                    executor.submit(process_question, mcq, model, host, temperature, seed,
                                    debate_rounds): mcq.uid
                    for mcq in pending
                }
                with tqdm(total=len(mcqs), initial=len(processed_ids),
                          desc=f"MAD-Graph [{model}]") as pbar:
                    for future in concurrent.futures.as_completed(future_to_id):
                        try:
                            result = future.result()
                            writer.writerow(result)
                            f.flush()
                        except Exception as e:
                            qid = future_to_id[future]
                            print(f"\nError on question {qid}: {e}")
                        pbar.update(1)

    print(f"\nDone. Results saved to: {csv_output_file}")
    if for_pride:
        print("Feed this CSV directly into pride/pride_detail_eval.py or pride/pride_batch_summary.py.")
    else:
        print("Run mad_graph_analysis.py to analyse the results.")


def main():
    parser = argparse.ArgumentParser(
        description="MAD-Graph: Graph-Based Multi-Agent Debate for MCQ positional bias mitigation."
    )
    parser.add_argument("--model", type=str, required=True, help="Ollama model name (e.g. gemma3:4b)")
    parser.add_argument("--host", type=str, default="http://localhost:11434", help="Ollama server URL")
    parser.add_argument("--input", type=str, required=True, help="Path to MCQ CSV file")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (> 0)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--max-questions", type=int, default=None, help="Cap number of questions evaluated")
    parser.add_argument("--num-workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--debate-rounds", type=int, default=2,
                        help="Number of debate rounds (default: 2, per Du et al. 2023)")
    parser.add_argument("--for-pride", action="store_true",
                        help="Run all 4 cyclic permutations per question and output "
                             "vote-based soft probabilities in PriDe-compatible CSV format. "
                             "Output goes to results/mad_graph_pride/. "
                             "Note: 4x more LLM calls than standard mode.")

    args = parser.parse_args()

    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0")

    run_evaluation(
        model=args.model,
        host=args.host,
        csv_path=args.input,
        seed=args.seed,
        max_questions=args.max_questions,
        temperature=args.temperature,
        num_workers=args.num_workers,
        debate_rounds=args.debate_rounds,
        for_pride=args.for_pride,
    )


if __name__ == "__main__":
    main()
