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

You previously chose {my_ans}. Another agent chose {other_ans} with this reasoning:
"{other_reasoning}"

Critique their reasoning. Do you concede to {other_ans}, or hold your ground on {my_ans}?
End your response by clearly stating your final choice as exactly one letter ({my_ans} or {other_ans}) inside <answer> tags, like this: <answer>{my_ans}</answer>"""

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
                retries: int = 3, timeout: int = 180) -> str:
    url = f"{host.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "seed": seed},
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


def run_mad_graph(mcq: MCQ, model: str, host: str, temperature: float, seed: int) -> Tuple[str, Dict]:
    """
    Runs the full MAD-Graph pipeline for a single question.
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
        return "", initial_responses

    # All agents agree — no debate needed
    if len(set(valid_votes.values())) == 1:
        return list(valid_votes.values())[0], initial_responses

    # --- Phase 2: Debate (Cross-Critique) ---
    final_votes = dict(valid_votes)

    for agent_id, my_ans in valid_votes.items():
        disagreeing = [(aid, a) for aid, a in valid_votes.items() if a != my_ans]
        if not disagreeing:
            continue

        other_agent_id, other_ans = disagreeing[0]
        other_reasoning = initial_responses[other_agent_id]["reasoning"]

        debate_prompt = DEBATE_PROMPT_TEMPLATE.format(
            question=mcq.question,
            A=mcq.options["A"],
            B=mcq.options["B"],
            C=mcq.options["C"],
            D=mcq.options["D"],
            my_ans=my_ans,
            other_ans=other_ans,
            other_reasoning=other_reasoning[:800]
        )

        try:
            response = call_ollama(model, debate_prompt, host, temperature, seed=seed + 10 + agent_id)
            new_ans, _ = extract_answer_and_reasoning(response)
            if new_ans in [my_ans, other_ans]:
                final_votes[agent_id] = new_ans
        except Exception:
            pass

    # --- Phase 3: Graph Resolution (In-Degree Centrality) ---
    final_answer = Counter(final_votes.values()).most_common(1)[0][0]
    return final_answer, initial_responses


def process_question(mcq: MCQ, model: str, host: str, temperature: float, seed: int) -> Dict:
    predicted_answer, initial_responses = run_mad_graph(mcq, model, host, temperature, seed)
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


def run_evaluation(model: str, host: str, csv_path: str, seed: int,
                   max_questions: Optional[int], temperature: float, num_workers: int):
    print("\n=== Starting MAD-Graph Evaluation ===")
    print(f"Model       : {model}")
    print(f"Dataset     : {csv_path}")
    print(f"Temperature : {temperature}")
    print(f"Workers     : {num_workers}")

    mcqs = load_mcq_csv(csv_path, max_questions=max_questions)

    csv_dir = Path("mad_graph/results")
    csv_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = Path(csv_path).stem
    model_name = model.replace(':', '_').replace('/', '_')
    csv_output_file = csv_dir / f"{dataset_name}-{model_name}_mad_graph.csv"

    fieldnames = [
        "question_id", "model", "predicted_answer", "correct_answer", "is_correct",
        "agent_1_ans", "agent_2_ans", "agent_3_ans", "question"
    ]

    # Checkpointing
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

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_id = {
                executor.submit(process_question, mcq, model, host, temperature, seed): mcq.uid
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
    )


if __name__ == "__main__":
    main()
