"""
MAD-Graph Selective PriDe Evaluation - Transformers version

Research hypothesis: LLMs know the answer when agents unanimously agree;
when they disagree, positional bias fills the uncertainty gap.

Pipeline per question:
  1. Diversity check  — run 3 agents (temperature > 0, distinct seeds + personas)
  2. Permutation pass — run 3 agents (deterministic) on all 4 cyclic permutations.

This script uses local Hugging Face generation. 
Outputs to: results/mad_graph_selective/<dataset>-<model>_pride.csv
"""

import argparse
import csv
import re
import torch
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Minimal Prompts ---
ANSWER_ONLY_PROMPT = """\
{agent_prompt}

Question: {question}

A. {A}
B. {B}
C. {C}
D. {D}

Answer with a single letter only (A, B, C, or D):"""

AGENT_PROMPTS = {
    1: "Solve this step-by-step using logical deduction.",
    2: "Identify the most common trap or misconception in this question and avoid it.",
    3: "Give your immediate, most confident answer based on core principles."
}

_POSITIONS = ["A", "B", "C", "D"]
OUTPUT_DIR = Path("../results/mad_graph_selective")

FIELDNAMES = [
    "question_id", "permutation_idx",
    "prob_A", "prob_B", "prob_C", "prob_D",
    "predicted_answer", "correct_position", "correct_answer", "model", "confident",
    "temperature", "agent_1_ans", "agent_2_ans", "agent_3_ans",
]


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
        raise ValueError(f"Missing columns in CSV: {required_cols - set(df.columns)}")

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
    return mcqs


class TransformersLLM:
    """Wrapper for a Hugging Face causal language model."""
    def __init__(self, model_name_or_path: str, device: str = "auto"):
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if device == "auto" else None
        )
        if self.device != "auto":
            self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str, temperature: float, do_sample: bool = True, seed: Optional[int] = None) -> str:
        if seed is not None:
            torch.manual_seed(seed)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        if self.device != "auto":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        kwargs = {
            "max_new_tokens": 10, # Keep it small for single-letter output
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        
        if do_sample and temperature > 0:
            kwargs.update({
                "do_sample": True,
                "temperature": temperature,
                "top_p": 0.9,
            })
        else:
            kwargs.update({"do_sample": False})

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **kwargs)

        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return response


# --- PriDe Helpers ---
def permute_options(options: Dict[str, str], shift: int) -> Dict[str, str]:
    return {_POSITIONS[i]: options[_POSITIONS[(i + shift) % 4]] for i in range(4)}

def correct_pos_in_permutation(correct_answer: str, shift: int) -> str:
    orig_idx = _POSITIONS.index(correct_answer)
    return _POSITIONS[(orig_idx - shift) % 4]

def votes_to_probs(final_valid: Dict) -> Dict[str, float]:
    n = len(final_valid) if final_valid else 1
    counts = Counter(final_valid.values())
    return {f"prob_{p}": counts.get(p, 0) / n for p in _POSITIONS}


def _run_phase1(
    mcq: MCQ,
    llm: TransformersLLM,
    temperature: float,
    seed: int,
    options_override: Dict[str, str] | None = None,
    deterministic: bool = False,
) -> Dict[int, str]:
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
            do_sample = not deterministic
            raw = llm.generate(prompt, temperature=temperature, do_sample=do_sample, seed=seed + agent_id)
            letter = next((c for c in raw.strip().upper() if c in _POSITIONS), "")
            votes[agent_id] = letter
        except Exception:
            votes[agent_id] = ""
    return votes


def process_question_selective(
    mcq: MCQ, llm: TransformersLLM, temperature: float, seed: int
) -> Tuple[bool, List[Dict]]:
    # 1. Diversity check
    orig_votes = _run_phase1(mcq, llm, temperature, seed, deterministic=False)
    valid_orig = {aid: ans for aid, ans in orig_votes.items() if ans in _POSITIONS}
    vote_values = list(valid_orig.values())
    is_confident = (len(set(vote_values)) == 1 and len(vote_values) >= 2)

    # Store Phase 1 individual agent answers (on original question ordering).
    agent_1_ans = orig_votes.get(1, "")
    agent_2_ans = orig_votes.get(2, "")
    agent_3_ans = orig_votes.get(3, "")

    # 2. Deterministic permutation passes
    rows: List[Dict] = []
    
    # We use model_name correctly from the config
    model_name = llm.model.config._name_or_path
    
    for shift in range(4):
        perm_opts = permute_options(mcq.options, shift) if shift > 0 else None
        raw = _run_phase1(
            mcq, llm, temperature,
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
            "model":            model_name,
            "confident":        int(is_confident),
            "temperature":      temperature,
            "agent_1_ans":      agent_1_ans,
            "agent_2_ans":      agent_2_ans,
            "agent_3_ans":      agent_3_ans,
        })

    return is_confident, rows


def run_selective_evaluation(
    model_name_or_path: str,
    csv_path: str,
    seed: int,
    max_questions: Optional[int],
    temperature: float,
    device: str,
):
    print("\n=== MAD-Graph Selective PriDe Evaluation (Transformers) ===")
    print(f"Model         : {model_name_or_path}")
    print(f"Dataset       : {csv_path}")
    print(f"Temperature   : {temperature} (diversity check; permutation passes use do_sample=False)")

    # Load model
    llm = TransformersLLM(model_name_or_path, device=device)
    mcqs = load_mcq_csv(csv_path, max_questions=max_questions)

    # Set up paths
    # Resolve relative to the script location (assuming run from transformers folder)
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir.parent / "results" / "mad_graph_selective"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_name = Path(csv_path).stem
    model_name_clean = model_name_or_path.replace(":", "_").replace("/", "_")
    out_file = output_dir / f"{dataset_name}-{model_name_clean}_transformers_pride.csv"

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

    n_confident = 0
    n_uncertain = 0

    with open(out_file, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if not processed_ids and fh.tell() == 0:
            writer.writeheader()

        for mcq in tqdm(pending, desc=f"Selective [{model_name_clean}]", total=len(pending)):
            try:
                is_conf, rows = process_question_selective(mcq, llm, temperature, seed)
                for row in rows:
                    writer.writerow(row)
                fh.flush()
                
                if is_conf:
                    n_confident += 1
                else:
                    n_uncertain += 1
            except Exception as e:
                print(f"\nError on question {mcq.uid}: {e}")

    total = n_confident + n_uncertain
    if total > 0:
        print(f"\nDone.")
        print(f"  Confident (unanimous) : {n_confident}  ({n_confident/total:.1%})")
        print(f"  Uncertain             : {n_uncertain}  ({n_uncertain/total:.1%})")
    print(f"\n  Output CSV -> {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description="MAD-Graph Selective PriDe (Transformers version)."
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Hugging Face model name or path")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to MCQ dataset CSV")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Sampling temperature for diversity check (must be > 0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Cap number of questions evaluated")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Device to run the model on")

    args = parser.parse_args()

    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0 (needed for diversity check)")

    run_selective_evaluation(
        model_name_or_path=args.model,
        csv_path=args.input,
        seed=args.seed,
        max_questions=args.max_questions,
        temperature=args.temperature,
        device=args.device,
    )

if __name__ == "__main__":
    main()


