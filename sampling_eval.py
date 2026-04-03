"""
Local LLM Positional Bias Data Collection (Sampling-Based)

This script supports both Ollama (REST API) and Transformers (Hugging Face) backends
to estimate probabilities by taking multiple temperature-sampled inferences.

Example Usages:

1. Run with Ollama API:
   python sampling_eval.py --backend ollama --model "qwen2.5:3b" \
       --input "dataset/arc_challenge.csv" --sampling-n 15 \
       --sampling-temp 0.7 --num-workers 8

2. Run with Transformers (Local GPU via HuggingFace):
   python sampling_eval.py --backend transformers --model "Qwen/Qwen2.5-3B-Instruct" \
       --input "dataset/arc_challenge.csv" --sampling-n 15 \
       --sampling-temp 0.7 --num-workers 4 --quantization fp16
"""

import argparse
import csv
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import concurrent.futures
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import requests
import torch

import sys
import os

# Try to import from mad_graph_eval for MCQ definition and utilities
try:
    from mad_graph_eval import MCQ, load_mcq_csv, permute_options, _POSITIONS, call_ollama, correct_pos_in_permutation
except ImportError:
    # Fallback definitions if not found
    _POSITIONS = ["A", "B", "C", "D"]
    
    @dataclass
    class MCQ:
        uid: str
        question: str
        options: Dict[str, str]  # keys A-D, values are option text
        answer: str  # correct letter A-D

    def load_mcq_csv(path: str, max_questions: int = None) -> List[MCQ]:
        """Load multiple choice questions from CSV file"""
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
            
            if (question in ["", "nan"] or any(opt in ["", "nan"] for opt in options_text)):
                continue
                
            mcq = MCQ(
                uid=str(row["id"]),
                question=question,
                options={"A": options_text[0], "B": options_text[1], "C": options_text[2], "D": options_text[3]},
                answer=ans,
            )
            mcqs.append(mcq)
        
        if max_questions:
            mcqs = mcqs[:max_questions]
        return mcqs

    def permute_options(options: Dict[str, str], perm_idx: int):
        from collections import OrderedDict
        out = OrderedDict()
        keys = list(options.keys())
        for i, k in enumerate(_POSITIONS):
            out[k] = options[keys[(i + perm_idx) % 4]]
        return out
        
    def correct_pos_in_permutation(original_ans, shift_idx):
        idx = _POSITIONS.index(original_ans)
        new_idx = (idx - shift_idx) % 4
        return _POSITIONS[new_idx]


# Prompt template
PROMPT_TEMPLATE = """Question: {question}

A. {A}
B. {B}
C. {C}
D. {D}

You must respond with exactly one letter (A, B, C, or D) and nothing else.
Answer:"""

def build_prompt(mcq: MCQ, permuted_options: Dict[str, str]) -> str:
    """Build the full prompt for the LLM"""
    return PROMPT_TEMPLATE.format(
        question=mcq.question,
        A=permuted_options["A"],
        B=permuted_options["B"],
        C=permuted_options["C"],
        D=permuted_options["D"],
    )

# Transformers lazy loaded
TransformersLLM_Class = None

def get_transformers_class():
    global TransformersLLM_Class
    if TransformersLLM_Class is not None:
        return TransformersLLM_Class
        
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    
    class TransformersLLM:
        def __init__(self, model_name_or_path: str, device: str = "auto",
                     temperature: float = 0.5, quantization: str = "fp16"):
            self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side='left')      

            load_kwargs = {"device_map": "auto" if device == "auto" else None}

            if quantization == "int8":
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
                load_kwargs["quantization_config"] = quant_config
                load_kwargs["device_map"] = "auto"
            else:
                load_kwargs["torch_dtype"] = torch.float16 if self.device == "cuda" else torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)

            if quantization != "int8" and device != "auto":
                self.model.to(self.device)
            self.model.eval()
            self.temperature = temperature

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        def generate_batch(self, prompts: List[str], temperature: float = None, seed: int = None, max_new_tokens: int = 50) -> List[str]:
            if seed is not None:
                torch.manual_seed(seed)

            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
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

            responses = []
            for i in range(len(prompts)):
                input_length = inputs["input_ids"][i].shape[0] if inputs["input_ids"].ndim == 2 else 0
                generated_ids = outputs[i][inputs["input_ids"].shape[1]:]
                resp = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                responses.append(resp)
            return responses

    TransformersLLM_Class = TransformersLLM
    return TransformersLLM_Class


def _ollama_worker(model: str, prompt: str, host: str, temperature: float, seed: int, timeout: int) -> str:
    url = f"{host.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "seed": seed,
        }
    }
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception:
        return ""

def parse_answer(response_text: str) -> str:
    response_text = response_text.strip().upper()
    
    if "<THINK>" in response_text:
        think_start = response_text.find("<THINK>")
        if "</THINK>" in response_text:
            think_end = response_text.find("</THINK>")
            think_content = response_text[think_start:think_end]
        else:
            think_content = response_text[think_start:]
        
        answer_patterns = [
            r'ANSWER\s+(?:SHOULD\s+BE|IS|MUST\s+BE)\s+([A-D])',
            r'SO\s+(?:THE\s+)?ANSWER\s+(?:SHOULD\s+BE|IS|MUST\s+BE)\s+([A-D])',
            r'(?:DEFINITELY|CLEARLY)\s+([A-D])',
            r'(?:SO|THEREFORE),?\s+(?:THE\s+ANSWER\s+IS\s+)?([A-D])',
            r'OPTION\s+([A-D])',
            r'CHOICE\s+([A-D])',
            r'([A-D]),?\s+(?:IS\s+THE\s+ANSWER|IS\s+CORRECT)'
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, think_content)
            if match:
                return match.group(1)
    
    reasoning_patterns = [
        r'(?:ANSWER|OPTION|CHOICE|SO|THEREFORE)\s+(?:IS\s+)?([A-D])',
        r'([A-D])\s+(?:IS\s+(?:THE\s+)?(?:ANSWER|CORRECT|RIGHT))',
        r'MUST\s+BE\s+([A-D])'
    ]
    for pattern in reasoning_patterns:
        matches = re.findall(pattern, response_text)
        if matches:
            return matches[-1]
            
    matches = re.findall(r'\b([A-D])\b', response_text)
    if matches:
        return matches[-1]
    return ""


def run_evaluation(model_type: str, model: str, host: str, csv_path: str, n_permutations: int,
                  seed: int, max_questions: int, sampling_n: int, sampling_temp: float, 
                  num_workers: int, device: str, quantization: str):
    
    print("\n=== Starting Positional Bias Evaluation (Sampling-Based) ===")
    print(f"Backend: {model_type}")
    print(f"Model: {model}")
    print(f"Dataset: {csv_path}")
    
    mcqs = load_mcq_csv(csv_path, max_questions=max_questions)
    
    csv_dir = Path("results/sampling")
    csv_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = Path(csv_path).stem
    model_name = model.replace(':', '_').replace('/', '_')
    output_filename_base = f"{dataset_name}-{model_name}_sampling_n{sampling_n}"
    csv_output_file = csv_dir / f"{output_filename_base}.csv"

    processed_tasks = set()
    fieldnames = [
        "question_id", "permutation_idx", 
        "prob_A", "prob_B", "prob_C", "prob_D", 
        "predicted_answer", "correct_position", "correct_answer", 
        "model", "temperature"
    ]

    if csv_output_file.exists():
        try:
            with open(csv_output_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('question_id') and row.get('permutation_idx'):
                        processed_tasks.add((row['question_id'], int(row['permutation_idx'])))
            print(f"Resuming: Found {len(processed_tasks)} completed tasks to skip.")
        except Exception as e:
            print(f"Warning: Could not parse existing results file, starting fresh. Error: {e}")
            processed_tasks = set()

    llm = None
    if model_type == "transformers":
        TransformersCls = get_transformers_class()
        llm = TransformersCls(model_name_or_path=model, device=device, temperature=sampling_temp, quantization=quantization)
        
    total_prompts = len(mcqs) * n_permutations
    
    with open(csv_output_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not processed_tasks and f.tell() == 0:
            writer.writeheader()

        with tqdm(total=total_prompts, desc=f"Evaluating {model}", initial=len(processed_tasks)) as pbar:
            for mcq in mcqs:
                for perm_idx in range(n_permutations):
                    if (mcq.uid, perm_idx) in processed_tasks:
                        continue

                    permuted_options = permute_options(mcq.options, perm_idx) if perm_idx > 0 else mcq.options
                    
                    correct_pos = correct_pos_in_permutation(mcq.answer, perm_idx)
                    prompt = build_prompt(mcq, permuted_options)
                    
                    choices = []
                    
                    if model_type == "ollama":
                        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                            futures = [
                                executor.submit(_ollama_worker, model, prompt, host, sampling_temp, seed + i, 180)
                                for i in range(sampling_n)
                            ]
                            for future in concurrent.futures.as_completed(futures):
                                ans = parse_answer(future.result())
                                if ans in _POSITIONS:
                                    choices.append(ans)
                    else:
                        # Transformers batched execution
                        prompts = [prompt] * sampling_n
                        # We use simple sequential or batch grouping here
                        batch_size = max(1, num_workers) # reuse param as batch size
                        for i in range(0, sampling_n, batch_size):
                            batch_prompts = prompts[i:i+batch_size]
                            raw_responses = llm.generate_batch(batch_prompts, temperature=sampling_temp, seed=seed+i)
                            for resp in raw_responses:
                                ans = parse_answer(resp)
                                if ans in _POSITIONS:
                                    choices.append(ans)
                                    
                    total = len(choices)
                    if total == 0:
                        probabilities = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
                        predicted_answer = ""
                    else:
                        counter = Counter(choices)
                        probabilities = {letter: counter.get(letter, 0) / total for letter in _POSITIONS}
                        predicted_answer = counter.most_common(1)[0][0]
                    
                    writer.writerow({
                        "question_id": mcq.uid,
                        "permutation_idx": perm_idx,
                        "prob_A": probabilities["A"],
                        "prob_B": probabilities["B"],
                        "prob_C": probabilities["C"],
                        "prob_D": probabilities["D"],
                        "predicted_answer": predicted_answer,
                        "correct_position": correct_pos,
                        "correct_answer": mcq.answer,
                        "model": model,
                        "temperature": sampling_temp,
                    })
                    f.flush()
                    pbar.update(1)

    print(f"\nEvaluation complete. Results saved to: {csv_output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate positional bias using sampling-based probability estimation.")
    parser.add_argument("--backend", type=str, choices=["ollama", "transformers"], required=True, help="Backend to use")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--host", type=str, default="http://localhost:11434", help="Ollama host URL (if using ollama)")
    parser.add_argument("--input", type=str, required=True, help="Path to MCQ CSV file")
    parser.add_argument("--n-permutations", type=int, default=4, help="Number of permutations per question")
    parser.add_argument("--sampling-n", type=int, default=15, help="Number of samples for probability estimation")
    parser.add_argument("--sampling-temp", type=float, default=0.5, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-questions", type=int, default=None, help="Maximum number of questions to evaluate")
    parser.add_argument("--num-workers", type=int, default=8, help="Parallel workers for Ollama / Batch size for Transformers")
    parser.add_argument("--device", type=str, default="auto", help="Device for transformers")
    parser.add_argument("--quantization", type=str, default="fp16", help="Quantization for transformers")
    
    args = parser.parse_args()
    
    if args.sampling_temp <= 0:
        raise ValueError("Sampling temperature must be greater than 0")
        
    run_evaluation(
        model_type=args.backend,
        model=args.model,
        host=args.host,
        csv_path=args.input,
        n_permutations=args.n_permutations,
        seed=args.seed,
        max_questions=args.max_questions,
        sampling_n=args.sampling_n,
        sampling_temp=args.sampling_temp,
        num_workers=args.num_workers,
        device=args.device,
        quantization=args.quantization
    )

if __name__ == "__main__":
    main()
