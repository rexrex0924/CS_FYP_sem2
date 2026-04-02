import argparse
import csv
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm
import torch

# Add parent directory to path to import mad_graph_eval components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mad_graph_eval import (
    MCQ,
    load_mcq_csv,
    permute_options,
    correct_pos_in_permutation,
    _POSITIONS,
)

# Import Hugging Face components
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, QuantoConfig

BASELINE_PROMPT = """Question: {question}

A. {A}
B. {B}
C. {C}
D. {D}

Answer with a single letter only (A, B, C, or D):"""

OUTPUT_DIR = Path("../results/baseline")

FIELDNAMES = [
    "question_id", "permutation_idx",
    "prob_A", "prob_B", "prob_C", "prob_D",
    "predicted_answer", "correct_position", "correct_answer", "model", "temperature",
]

QUANT_CHOICES = ("fp16", "int8", "nf4", "fp4", "fp8")

def _build_quant_config(quantization: str):
    if quantization == "fp16":
        return None
    if quantization == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    if quantization in ("nf4", "fp4"):
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quantization,
            bnb_4bit_use_double_quant=(quantization == "nf4"),
            bnb_4bit_compute_dtype=torch.float16,
        )
    if quantization == "fp8":
        return QuantoConfig(weights="float8")
    raise ValueError(f"Unknown quantization: {quantization!r}. Choose from {QUANT_CHOICES}")

class TransformersLLM:
    """Wrapper for a Hugging Face causal language model."""
    def __init__(self, model_name_or_path: str, device: str = "auto",
                 temperature: float = 0.5, quantization: str = "fp16"):
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        quant_config = _build_quant_config(quantization)
        load_kwargs: dict = {"device_map": "auto" if device == "auto" else None}

        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.float16 if self.device == "cuda" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)

        if quant_config is None and device != "auto":
            self.model.to(self.device)
        self.model.eval()
        self.temperature = temperature

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str, temperature: float = None, max_new_tokens: int = 3, seed: int = None) -> str:
        """Generate text from the model given a prompt."""
        if seed is not None:
            torch.manual_seed(seed)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        if self.device != "auto":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        temp = temperature if temperature is not None else self.temperature
        do_sample = temp > 0.0

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temp if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return response


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_question_baseline(
    mcq: MCQ, llm: TransformersLLM, seed: int
) -> List[Dict]:
    """
    Run one deterministic call per cyclic permutation and return 4 rows.
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
            # Temperature 0.0 for deterministic baseline
            raw = llm.generate(prompt, temperature=0.0, seed=seed, max_new_tokens=3)
            predicted = next(
                (c for c in raw.strip().upper() if c in _POSITIONS), ""
            )
        except Exception as e:
            print(f"Error during generation: {e}")
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
            "model":            llm.model.config._name_or_path,
            "temperature":      llm.temperature,
        })

    return rows


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_baseline_evaluation(
    model_name_or_path: str,
    csv_path: str,
    seed: int,
    max_questions: Optional[int],
    device: str,
    quantization: str = "fp16"
):
    print("\n=== Baseline Cyclic Permutation Evaluation (Transformers) ===")
    print(f"Model         : {model_name_or_path}")
    print(f"Dataset       : {csv_path}")
    print(f"Mode          : Single call per permutation | max_new_tokens=3 | temp=0.0")
    print(f"Device        : {device}")
    print(f"Quantization  : {quantization}")

    # Load model
    llm = TransformersLLM(model_name_or_path, device=device,
                          temperature=0.5, quantization=quantization)

    mcqs = load_mcq_csv(csv_path, max_questions=max_questions)

    # Use relative path from the script execution context
    output_dir = Path("results/baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_name = Path(csv_path).stem
    model_name   = model_name_or_path.replace("/", "_")
    out_file     = output_dir / f"{dataset_name}-{model_name}_baseline_transformers.csv"

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
        if not processed_ids and fh.tell() == 0:
            writer.writeheader()

        # Sequential processing for local model inference
        for mcq in tqdm(pending, desc=f"Baseline [{model_name}]", total=len(pending), initial=0):
            try:
                rows = process_question_baseline(mcq, llm, seed)
                for row in rows:
                    writer.writerow(row)
                fh.flush()
            except Exception as e:
                print(f"\nError on question {mcq.uid}: {e}")

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
        description="Baseline cyclic permutation evaluation (Transformers version)."
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Hugging Face model name or path")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to MCQ dataset CSV")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Cap number of questions evaluated")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Device to run the model on")
    parser.add_argument(
        "--quantization", type=str, default="fp16", choices=QUANT_CHOICES,
        help="Weight quantization: fp16, int8, nf4, fp4, fp8",
    )

    args = parser.parse_args()

    run_baseline_evaluation(
        model_name_or_path=args.model,
        csv_path=args.input,
        seed=args.seed,
        max_questions=args.max_questions,
        device=args.device,
        quantization=args.quantization,
    )


if __name__ == "__main__":
    main()