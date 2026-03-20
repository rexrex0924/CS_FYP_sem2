# MAD-Graph: Graph-Based Multi-Agent Debate for Positional Bias Mitigation

This folder implements the **MAD-Graph (Graph-Based Multi-Agent Debate)** method for mitigating positional bias in LLM-based MCQ evaluation.

## Method Overview

Based on the following papers:
- **Du et al. (2023)** — *Improving Factuality and Reasoning in Language Models through Multiagent Debate* — https://arxiv.org/abs/2305.14325
- **Chan et al. (2023)** — *ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate* — https://arxiv.org/abs/2308.07201
- **Besta et al. (2023)** — *Graph of Thoughts: Solving Elaborate Problems with Large Language Models* — https://arxiv.org/abs/2308.09687

### How It Works

**Phase 1 — Divergent Generation:**  
Three "agent" instances of the same LLM are queried in parallel using different system personas:
1. **The Analyst** – Solves step-by-step using logical deduction
2. **The Critic** – Identifies common traps/misconceptions and avoids them
3. **The Intuitive** – Gives an immediate confident answer based on core principles

If all three agree → final answer is returned immediately.

**Phase 2 — Debate (Cross-Critique):**  
When agents disagree, conflicting agents are paired and critique each other's reasoning. Each agent either concedes or holds ground.

**Phase 3 — Graph Resolution:**  
All final votes (post-debate) are aggregated using **In-Degree Centrality** — the answer with the most votes wins.

This approach reduces positional bias because different agent personas approach the problem from different angles, making it harder for any single positional artifact to dominate.

---

## Usage

### Evaluation

```bash
python mad_graph/mad_graph_eval.py \
  --model gemma3:4b \
  --input data/sample_mcq.csv \
  --n-permutations 4 \
  --temperature 0.7 \
  --num-workers 4
```

Results are saved to `mad_graph/results/`.

### Analysis

```bash
# Analyze MAD-Graph results only
python mad_graph/mad_graph_analysis.py \
  --mad-results mad_graph/results/college_cs-gemma3_4b_mad_graph.csv

# Compare with baseline sampling results
python mad_graph/mad_graph_analysis.py \
  --mad-results mad_graph/results/college_cs-gemma3_4b_mad_graph.csv \
  --baseline-results results/csv_results/college_cs-gemma3_4b_sampling_n15.csv \
  --output-dir mad_graph/analysis
```

---

## Arguments

### `mad_graph_eval.py`

| Argument | Default | Description |
|---|---|---|
| `--model` | *(required)* | Ollama model name |
| `--input` | *(required)* | Path to MCQ CSV file |
| `--host` | `http://localhost:11434` | Ollama server URL |
| `--n-permutations` | `4` | Number of option permutations per question |
| `--temperature` | `0.7` | Sampling temperature |
| `--seed` | `42` | Random seed |
| `--max-questions` | `None` | Limit number of questions |
| `--num-workers` | `4` | Parallel workers (one per question-permutation task) |

### `mad_graph_analysis.py`

| Argument | Default | Description |
|---|---|---|
| `--mad-results` | *(required)* | Path to MAD-Graph output CSV |
| `--baseline-results` | `None` | Path to baseline CSV for comparison |
| `--output-dir` | `mad_graph/analysis` | Where to save plots |

---

## CSV Input Format

Same as the rest of the project:

```
id,question,option_a,option_b,option_c,option_d,answer
```

## Output CSV Columns

| Column | Description |
|---|---|
| `question_id` | Question identifier |
| `permutation_idx` | Which cyclic permutation (0–3) |
| `predicted_answer` | MAD-Graph final answer |
| `correct_position` | Correct letter in this permutation |
| `is_correct` | 1/0 |
| `agent_1_ans` | Analyst's initial answer |
| `agent_2_ans` | Critic's initial answer |
| `agent_3_ans` | Intuitive agent's initial answer |
