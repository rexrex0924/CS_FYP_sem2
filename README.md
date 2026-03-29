# MAD-Graph + PriDe: Positional Bias Mitigation for LLM MCQ Evaluation

This project combines two complementary debiasing methods:

- **MAD-Graph** (pre-processing) — a multi-agent debate system where three LLM agents with different reasoning personas independently answer a question, debate when they disagree, and resolve via majority vote.
- **PriDe** (post-processing) — a mathematical debiasing method that estimates an LLM's inherent positional preference from cyclic permutations and subtracts it from predictions.

Three pipelines are provided, from simple to most research-relevant:

| Pipeline | What it does | When to use |
|---|---|---|
| [0] Cyclic baseline | Single model call per permutation (no debate) → PriDe | Comparison baseline for Pipeline C |
| [A] Standard MAD-Graph | Debate → analysis | Standalone debate effectiveness |
| [B] MAD-Graph → PriDe | Debate + cyclic permutations → PriDe debiasing | Full pipeline, no hypothesis split |
| [C] Selective PriDe *(recommended)* | Agent agreement split → cyclic permutations → PriDe | Research hypothesis testing |

Pipelines 0 and C both use `num_predict=3` (single-token generation), making them **directly comparable**. Any difference in PriDe effectiveness is attributable purely to the MAD-Graph agent structure.

**Recommended workflow:** Run Pipeline 0 and Pipeline C on the same models/datasets, then run `run_all_analysis.py` once to produce all plots, per-agent analysis, and the summary report automatically.

---

## How the Methods Work

### MAD-Graph — Phase 1–3

**Phase 1 — Divergent Generation:**  
Three agent instances of the same LLM answer the question independently using different personas:
1. **The Analyst** — solves step-by-step using logical deduction
2. **The Critic** — identifies common traps and misconceptions
3. **The Intuitive** — gives an immediate answer based on core principles

If all three agents unanimously agree → final answer returned immediately (no debate needed).

**Phase 2 — Multi-Round Debate:**  
When agents disagree, each agent is shown all other agents' current answers and reasoning, then updates its own. Repeats for `--debate-rounds` rounds (default: 2). Agents may change to any option, not just their opponent's.

**Phase 3 — Graph Resolution:**  
Final votes (post-debate) are aggregated using In-Degree Centrality — the answer with the most votes wins.

### PriDe — Post-Processing Debiasing

Runs the same question four times with cyclic permutations of the answer options (ABCD → BCDA → CDAB → DABC). The model's positional preference (prior) is estimated from how often each position is selected across permutations. The final prediction is the option that is consistently chosen *beyond* the model's positional habit.

### The Selective Hypothesis (Pipeline C)

> "When agents unanimously agree on an answer, the model knows it.  
> When they disagree, positional bias is filling the uncertainty gap."

Expected result if the hypothesis holds:
- Confident (unanimous) questions → higher accuracy, lower positional bias
- Uncertain (disagreed) questions → lower accuracy, higher positional bias, correctable with PriDe

---

---

## Pipeline 0 — Cyclic Permutation Baseline

The simplest possible PriDe input: one deterministic model call per cyclic shift, no debate, no agents. Use this as the comparison target for Pipeline C.

```bash
python baseline_cyclic_eval.py \
  --model gemma3:4b \
  --input dataset/college_cs.csv \
  --num-workers 4
```

Output: `results/baseline/<dataset>-<model>_baseline.csv`  
Then run PriDe: `python pride/pride_detail_eval.py results/baseline/<name>_baseline.csv`

| Argument | Default | Description |
|---|---|---|
| `--model` | *(required)* | Ollama model name |
| `--input` | *(required)* | Path to MCQ dataset CSV |
| `--host` | `http://localhost:11434` | Ollama server URL |
| `--seed` | `42` | Random seed |
| `--max-questions` | `None` | Cap number of questions |
| `--num-workers` | `4` | Parallel workers |

---

## Setup

### 1. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS
pip install -r requirements.txt
```

### 2. Start Ollama and pull a model

Install [Ollama](https://ollama.com), then:

```bash
ollama serve                  # starts the server (usually auto-starts)
ollama pull gemma3:4b         # or any model you want to test
```

---

## Dataset Preparation

Datasets must be in the project's standard CSV format:

```
id,question,option_a,option_b,option_c,option_d,answer
q1,"What is 2+2?","3","4","5","6","B"
```

### Using a built-in dataset

The repository includes `dataset/2012-2020_ICT_DSE.csv` (HKDSE ICT past papers). Use it directly.

### Extracting a dataset from HuggingFace

```bash
# General usage
python extract_mmlu_dataset.py <hf-dataset-name> --max-questions 200

# Examples
python extract_mmlu_dataset.py allenai/ai2_arc --config ARC-Challenge --split test --max-questions 200
python extract_mmlu_dataset.py cais/mmlu --config abstract_algebra --split test --max-questions 100
python extract_mmlu_dataset.py wzzzq/MMLU-PRO-Leveled-TinyBench --split combined --max-questions 200

# Dry run to preview detected columns without writing files
python extract_mmlu_dataset.py allenai/ai2_arc --config ARC-Challenge --dry-run
```

Output is saved to `dataset/`. If the script reports an unknown split, it will list valid split names.

#### `extract_mmlu_dataset.py` arguments

| Argument | Default | Description |
|---|---|---|
| `dataset` | *(required)* | HuggingFace dataset name or URL |
| `--config` | `None` | Dataset configuration / subset name |
| `--split` | `train` | Dataset split (e.g. `test`, `validation`, `combined`) |
| `--output` | `dataset/` | Output directory |
| `--name` | *(auto)* | Override output filename stem |
| `--max-questions` | `None` | Limit number of questions extracted |
| `--dry-run` | `False` | Preview column detection without writing |
| `--question-col` | *(auto)* | Override question column name |
| `--options-col` | *(auto)* | Override options column name |
| `--answer-col` | *(auto)* | Override answer column name |

---

## Pipeline A — Standard MAD-Graph

Best for: comparing debate vs. no-debate accuracy; understanding what multi-agent debate does.

### Step 1: Run evaluation

```bash
python mad_graph_eval.py \
  --model gemma3:4b \
  --input dataset/2012-2020_ICT_DSE.csv \
  --temperature 0.7 \
  --debate-rounds 2 \
  --num-workers 4
```

Output: `results/mad_graph/<dataset>-<model>_mad_graph.csv`

### Step 2: Analyse results

```bash
# MAD-Graph results only
python mad_graph_analysis.py \
  --mad-results results/mad_graph/2012-2020_ICT_DSE-gemma3_4b_mad_graph.csv \
  --dataset dataset/2012-2020_ICT_DSE.csv

# With a baseline comparison (e.g. standard sampling results)
python mad_graph_analysis.py \
  --mad-results results/mad_graph/2012-2020_ICT_DSE-gemma3_4b_mad_graph.csv \
  --baseline-results results/csv_results/2012-2020_ICT_DSE-gemma3_4b_sampling_n15.csv \
  --dataset dataset/2012-2020_ICT_DSE.csv
```

Output: `results/mad_graph/analysis/<label>_report.txt` and `<label>_*.png` plots.

#### `mad_graph_eval.py` arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | *(required)* | Ollama model name (e.g. `gemma3:4b`) |
| `--input` | *(required)* | Path to MCQ dataset CSV |
| `--host` | `http://localhost:11434` | Ollama server URL |
| `--temperature` | `0.7` | Sampling temperature (must be > 0) |
| `--debate-rounds` | `2` | Number of debate rounds (0 = Phase 1 only, no debate) |
| `--seed` | `42` | Base random seed |
| `--max-questions` | `None` | Cap number of questions evaluated |
| `--num-workers` | `4` | Parallel workers |
| `--for-pride` | `False` | Enable Pipeline B mode (see below) |

#### `mad_graph_analysis.py` arguments

| Argument | Default | Description |
|---|---|---|
| `--mad-results` | *(required)* | Path to MAD-Graph output CSV |
| `--baseline-results` | `None` | Optional baseline CSV for before/after comparison |
| `--dataset` | `None` | Original dataset CSV for empirical answer key distribution. Falls back to uniform 0.25 if omitted |
| `--output-dir` | `results/mad_graph/analysis` | Directory for plots and report |

---

## Pipeline B — MAD-Graph → PriDe (Full Permutations)

Best for: measuring positional bias across the full dataset and applying PriDe debiasing to everything.

### Step 1: Run evaluation with `--for-pride`

```bash
python mad_graph_eval.py \
  --model gemma3:4b \
  --input dataset/2012-2020_ICT_DSE.csv \
  --debate-rounds 0 \
  --num-workers 4 \
  --for-pride
```

> Use `--debate-rounds 0` here — debate adds no value for positional bias measurement since PriDe only needs raw positional preference, not reasoned answers.

Output: `results/mad_graph_pride/<dataset>-<model>_mad_graph_pride.csv`  
Format: 4 rows per question (one per cyclic permutation) with `prob_A/B/C/D` and `correct_position`.

### Step 2: Run PriDe debiasing

```bash
python pride/pride_detail_eval.py results/mad_graph_pride/2012-2020_ICT_DSE-gemma3_4b_mad_graph_pride.csv
```

Output:
- Debiased CSV: `pride/results/csv/<name>_pride_debiased.csv`
- Visualisation plots: `pride/results/visualizations/`

### Step 3 (optional): Batch summary across multiple models

```bash
python pride/pride_batch_summary.py
```

Output: `pride/results/summary/`

---

## Pipeline C — Selective PriDe *(Recommended)*

Best for: testing the research hypothesis. Each question gets a `confident` label based on whether agents unanimously agree in Phase 1. Cyclic permutations are run for ALL questions so bias can be compared empirically between the two groups.

### Step 1: Run selective evaluation

```bash
python mad_graph_selective_pride_eval.py \
  --model gemma3:4b \
  --input dataset/2012-2020_ICT_DSE.csv \
  --temperature 0.5 \
  --num-workers 4
```

What happens per question (15 LLM calls total, all single-token — very fast):
1. **Diversity check** (3 calls, temp > 0, different personas + seeds) → unanimous? → `confident=1`, else `confident=0`. Each agent's individual answer is recorded.
2. **Permutation passes** (12 calls = 4 shifts × 3 agents, temp=0) → records vote probabilities

Output: `results/mad_graph_selective/output/<dataset>-<model>_pride.csv`  
Format: 4 rows per question with `prob_A/B/C/D`, `correct_position`, `correct_answer`, `model`, `confident`, `temperature`, `agent_1_ans`, `agent_2_ans`, `agent_3_ans`.

#### `mad_graph_selective_pride_eval.py` arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | *(required)* | Ollama model name (e.g. `gemma3:4b`) |
| `--input` | *(required)* | Path to MCQ dataset CSV |
| `--host` | `http://localhost:11434` | Ollama server URL |
| `--temperature` | `0.5` | Sampling temperature for diversity check (must be > 0) |
| `--seed` | `42` | Base random seed |
| `--max-questions` | `None` | Cap number of questions evaluated |
| `--num-workers` | `4` | Parallel workers |

### Step 2: Run batch analysis

`run_all_analysis.py` auto-discovers all CSVs in `results/mad_graph_selective/output/`, runs PriDe, computes the full metric suite (accuracy, bias, recall std, chi-square, consistency score), analyses per-agent accuracy and bias, and generates all plots and a summary report.

```bash
# Process everything and generate all plots + report
python run_all_analysis.py

# Filter to one dataset
python run_all_analysis.py --dataset 2012-2020_ICT_DSE

# Report only (no plot regeneration)
python run_all_analysis.py --summary-only

# Use a specific fixed alpha instead of the default (0.3)
python run_all_analysis.py --fixed-alpha 0.5

# Force re-run even if output already exists
python run_all_analysis.py --force
```

Output in `pride/results/selective/output/`:
```
by_dataset/<dataset>/
    <dataset>_accuracy.png          # Baseline vs After PriDe across all models
    <dataset>_bias.png              # Bias score, Recall Std, Chi-square across models
    <dataset>_distribution.png      # Choice distribution per model
    <dataset>_acc_by_pos.png        # Accuracy by answer position per model
    <dataset>_consistency.png       # Consistency score across models
    <dataset>_agents.png            # Per-agent accuracy, bias, conf vs uncertain
by_model_dataset/<dataset>-<model>/
    accuracy.png                    # Overall + by-position accuracy
    bias_metrics.png                # All 4 bias metrics
    distribution.png                # Choice distribution before/after
    summary.png                     # Comprehensive 3×3 dashboard
    agents.png                      # Per-agent detail breakdown
SUMMARY_REPORT.txt                  # Full metrics table + agent ranking
```

#### `run_all_analysis.py` arguments

| Argument | Default | Description |
|---|---|---|
| `--fixed-alpha` | `0.3` | PriDe alpha (set to `0` to grid-search best alpha per CSV) |
| `--dataset` | `None` | Filter to one dataset name |
| `--force` | `False` | Re-run even if output files already exist |
| `--summary-only` | `False` | Skip plot generation, write report only |
| `--input-dir` | `results/mad_graph_selective/output` | Override CSV input directory |
| `--output-dir` | `pride/results/selective/output` | Override output directory |

### Step 3 (optional): Deeper per-model PriDe analysis

For a detailed per-model PriDe breakdown (alpha search, prior plots, per-permutation accuracy):

```bash
python pride/pride_detail_eval.py \
  results/mad_graph_selective/output/2012-2020_ICT_DSE-gemma3_4b_pride.csv
```

Output: `pride/results/csv/<name>_pride_debiased.csv` and `pride/results/visualizations/`

For a confident/uncertain split comparison with a single model:

```bash
python pride/pride_selective_analysis.py \
  results/mad_graph_selective/output/2012-2020_ICT_DSE-gemma3_4b_pride.csv
```

Output in `pride/results/selective/`: comparison plot + report with hypothesis verdict.

#### `pride/pride_selective_analysis.py` arguments

| Argument | Default | Description |
|---|---|---|
| `pride_csv` | *(positional, optional)* | Explicit path to `*_pride.csv` |
| `--model` | `None` | Model name — auto-derives CSV path (requires `--dataset`) |
| `--dataset` | `None` | Dataset CSV path — used to derive file name |
| `--calibration-ratio` | `0.15` | Fraction of questions for PriDe prior calibration |

---

## PriDe Arguments

#### `pride/pride_detail_eval.py`

```bash
python pride/pride_detail_eval.py <csv_path> [--calibration-ratio 0.15]
```

| Argument | Default | Description |
|---|---|---|
| `csv_path` | *(required, positional)* | Path to permuted CSV with `prob_A/B/C/D`, `permutation_idx`, `correct_position` |
| `--calibration-ratio` | `0.15` | Fraction of data used for prior estimation |

---

## CSV Formats

### Input dataset format

```
id,question,option_a,option_b,option_c,option_d,answer
q1,"What is 2+2?","3","4","5","6","B"
```

### Pipeline A output (`mad_graph/results/`)

| Column | Description |
|---|---|
| `question_id` | Question identifier |
| `model` | Ollama model name |
| `predicted_answer` | Final answer after debate |
| `correct_answer` | Ground truth |
| `is_correct` | 1 if correct, else 0 |
| `agent_1_ans` | Analyst's Phase 1 answer |
| `agent_2_ans` | Critic's Phase 1 answer |
| `agent_3_ans` | Intuitive agent's Phase 1 answer |
| `question` | Question text |

### Pipeline 0 output — Baseline (`results/baseline/`)

| Column | Description |
|---|---|
| `question_id` | Question identifier |
| `permutation_idx` | Cyclic shift index (0–3) |
| `prob_A` – `prob_D` | Hard probabilities (1.0 for chosen position, 0.0 for others) |
| `predicted_answer` | Chosen answer for this permutation |
| `correct_position` | Correct answer letter after applying the cyclic shift |
| `correct_answer` | Original (unshifted) correct answer |
| `model` | Model name |
| `temperature` | Always `0.0` (deterministic) |

### Pipeline B output — PriDe-compatible (`results/mad_graph_pride/`)

| Column | Description |
|---|---|
| `question_id` | Question identifier |
| `permutation_idx` | Cyclic shift index (0–3) |
| `prob_A` – `prob_D` | Vote fraction for each position (sums to 1.0) |
| `predicted_answer` | Majority vote answer for this permutation |
| `correct_position` | Correct answer letter after applying the cyclic shift |
| `model` | Model name |

### Pipeline C output — Selective PriDe (`results/mad_graph_selective/output/`)

| Column | Description |
|---|---|
| `question_id` | Question identifier |
| `permutation_idx` | Cyclic shift index (0–3) |
| `prob_A` – `prob_D` | Vote fraction for each position (sums to 1.0) |
| `predicted_answer` | Majority vote answer for this permutation |
| `correct_position` | Correct answer letter after applying the cyclic shift |
| `correct_answer` | Original (unshifted) correct answer |
| `model` | Model name |
| `confident` | `1` = all 3 agents unanimously agreed in Phase 1, `0` = disagreed |
| `temperature` | Phase 1 sampling temperature (logged for reproducibility) |
| `agent_1_ans` | Agent 1 (Analyst) Phase 1 answer on original question ordering |
| `agent_2_ans` | Agent 2 (Critic) Phase 1 answer on original question ordering |
| `agent_3_ans` | Agent 3 (Intuitive) Phase 1 answer on original question ordering |

---

## Folder Structure

```
sem2/
├── dataset/                            # MCQ datasets (CSV)
├── results/
│   ├── mad_graph/                      # Pipeline A: eval CSVs
│   │   └── analysis/                   # Pipeline A: analysis plots + reports
│   ├── mad_graph_pride/                # Pipeline B: eval CSVs
│   ├── mad_graph_selective/
│   │   ├── output/                     # Pipeline C: eval CSVs (new runs go here)
│   │   └── deprecated/                 # Old CSVs kept for reference
│   └── baseline/                       # Pipeline 0: baseline eval CSVs
├── pride/
│   ├── pride_detail_eval.py            # PriDe debiasing (single CSV)
│   ├── pride_batch_summary.py          # Batch summary for Pipeline B CSVs
│   ├── pride_selective_analysis.py     # Selective (confident/uncertain) comparison
│   └── results/
│       ├── csv/                        # Debiased CSVs (pride_detail_eval output)
│       ├── visualizations/             # Per-run plots (pride_detail_eval output)
│       ├── summary/                    # Batch summary plots (pride_batch_summary output)
│       └── selective/
│           ├── output/                 # run_all_analysis.py output (by_dataset/ + by_model_dataset/)
│           └── deprecated/             # Old selective analysis outputs
├── transformers/                       # HuggingFace Transformers variant
│   ├── mad_graph_analysis_transformers.py  # Transformers MAD-Graph eval
│   └── run_selective_transformers.cmd      # SLURM job script
├── run_all_analysis.py                 # Batch analysis: all pipelines → plots + report
├── baseline_cyclic_eval.py             # Pipeline 0: cyclic permutation baseline
├── mad_graph_eval.py                   # Pipeline A & B evaluation
├── mad_graph_analysis.py               # Pipeline A analysis
├── mad_graph_selective_pride_eval.py   # Pipeline C evaluation
├── mad_graph_selective_analysis.py     # Pipeline C single-model analysis
├── extract_mmlu_dataset.py             # HuggingFace dataset extractor
├── requirements.txt
└── README.md
```

---

## HuggingFace Transformers Variant

`transformers/mad_graph_analysis_transformers.py` runs the same MAD-Graph pipeline using HuggingFace models instead of Ollama. Supports quantization via `--quantization`:

| Flag | Backend | VRAM vs FP16 | Notes |
|---|---|---|---|
| `fp16` | PyTorch native | 1× (default) | No extra packages |
| `int8` | bitsandbytes | ~0.5× | `pip install bitsandbytes` |
| `nf4` | bitsandbytes | ~0.25× | Best 4-bit quality; closest to Ollama Q4_K_M |
| `fp4` | bitsandbytes | ~0.25× | Slightly faster kernel than nf4 |
| `fp8` | optimum-quanto | ~0.5× | Better quality than int8; `pip install optimum-quanto` |

```bash
python transformers/mad_graph_analysis_transformers.py \
  --model google/gemma-2-2b-it \
  --input dataset/2012-2020_ICT_DSE.csv \
  --quantization nf4 \
  --max-questions 100
```

> **Note:** Results from the Transformers variant (FP16) are **not directly comparable** to Ollama results (GGUF quantized). Keep the backend consistent across experiments.

---

## Tips

- **Checkpointing**: all evaluation scripts resume automatically if interrupted. Re-run the same command and already-processed questions are skipped.
- **Speed**: Pipeline C uses single-token generation (`num_predict=3`) — roughly 15–30× faster per LLM call than full-chain generation.
- **Temperature**: Pipeline C uses `--temperature 0.5` by default for the diversity check. Higher values (e.g. 0.7) generate more agent disagreement; lower values produce more confident labels. Keep this consistent across experiments — it is now recorded in the CSV `temperature` column.
- **Workers**: `--num-workers 4` works well for most local setups. Increase if your GPU/CPU can handle more parallel Ollama requests, but watch memory usage.
- **Datasets**: aim for 100–300 questions. Fewer than 50 makes bias statistics too noisy; more than 500 can take hours on a single GPU.
- **Model selection**: models in the 4B–12B range reveal clear positional bias without being so weak that accuracy collapses. Very strong models (>30B) may show ceiling effects.
- **Alpha**: `run_all_analysis.py` uses `--fixed-alpha 0.3` by default to avoid data leakage from tuning alpha on the test set. Set `--fixed-alpha 0` to grid-search per CSV instead.
