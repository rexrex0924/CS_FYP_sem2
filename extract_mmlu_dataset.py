"""
Generic HuggingFace MCQ Dataset Extractor

Give it any HuggingFace dataset name (or URL) and it auto-detects the
question / options / answer columns and saves to dataset/ in the project format:

  id, question, option_a, option_b, option_c, option_d, answer

Usage:
  python extract_mmlu_dataset.py wzzzq/MMLU-PRO-Leveled-TinyBench
  python extract_mmlu_dataset.py wzzzq/MMLU-PRO-Leveled-TinyBench --split hard_0.2_0.3
  python extract_mmlu_dataset.py cais/mmlu --config college_mathematics
  python extract_mmlu_dataset.py cais/mmlu --config college_mathematics --split test
  python extract_mmlu_dataset.py allenai/ai2_arc --config ARC-Challenge

  # Override auto-detected column names manually:
  python extract_mmlu_dataset.py some/dataset --question-col stem --options-col choices --answer-col answerKey

  # Preview without saving:
  python extract_mmlu_dataset.py wzzzq/MMLU-PRO-Leveled-TinyBench --dry-run

Install dependency:
  pip install datasets
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path("dataset")
POSITIONS  = ["A", "B", "C", "D"]

# ---------------------------------------------------------------------------
# Column auto-detection helpers
# ---------------------------------------------------------------------------

# Ranked name patterns: first match wins
_QUESTION_NAMES = ["question", "query", "prompt", "text", "sentence", "passage", "stem"]
_OPTIONS_NAMES  = ["options", "choices", "answers", "candidates"]   # list column
_ANSWER_NAMES   = ["answer", "label", "target", "correct", "answerkey", "correct_answer"]


def _find_col(columns: list[str], candidates: list[str]) -> str | None:
    """Return the first column whose lowercase name starts with any candidate."""
    col_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        for key, orig in col_lower.items():
            if key == cand or key.startswith(cand):
                return orig
    return None


def _detect_individual_option_cols(columns: list[str]) -> list[str] | None:
    """
    Detect datasets that store each option in its own column, e.g.
    A/B/C/D  or  option_a/option_b  or  choice_0/choice_1.
    Returns the 4 column names in order [A,B,C,D] if found, else None.
    """
    patterns = [
        # exact single-letter columns
        ["A", "B", "C", "D"],
        # option_a / option_b …
        ["option_a", "option_b", "option_c", "option_d"],
        # opa / opb …
        ["opa", "opb", "opc", "opd"],
        # choice_0 … choice_3
        ["choice_0", "choice_1", "choice_2", "choice_3"],
    ]
    col_set = {c.lower() for c in columns}
    col_map = {c.lower(): c for c in columns}
    for pat in patterns:
        if all(p in col_set for p in pat):
            return [col_map[p] for p in pat]
    return None


def _to_letter(value, options_len: int | None = None) -> str | None:
    """
    Normalise an answer value to a capital letter (A-D).
    Handles: int index, letter string, full answer text (matched against options).
    """
    if isinstance(value, int):
        return POSITIONS[value] if value < len(POSITIONS) else None
    if isinstance(value, str):
        v = value.strip().upper()
        if v in POSITIONS:
            return v
        # Some datasets use "1"/"2"/"3"/"4"
        if v.isdigit() and int(v) - 1 < len(POSITIONS):
            return POSITIONS[int(v) - 1]
    return None


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract(
    dataset_name: str,
    config:        str | None,
    split:         str,
    question_col:  str | None,
    options_col:   str | None,
    answer_col:    str | None,
    output_dir:    Path,
    max_questions: int | None,
    dry_run:       bool,
    output_name:   str | None,
):
    try:
        from datasets import load_dataset, get_dataset_config_names
    except ImportError:
        print("ERROR: 'datasets' library not found.\nRun:  pip install datasets")
        sys.exit(1)

    # ---- Load ----
    print(f"\nDataset : {dataset_name}")
    if config:
        print(f"Config  : {config}")
    print(f"Split   : {split}")
    print("Loading from HuggingFace...\n")

    try:
        kwargs = {}
        if config:
            kwargs["name"] = config
        ds = load_dataset(dataset_name, split=split, **kwargs)
    except ValueError as e:
        err = str(e)
        print(f"ERROR: {err}\n")
        # Show valid splits if the error mentions them
        if "Should be one of" in err or "Available splits" in err:
            print("Use --split <name> to pick one of the splits listed above.")
        else:
            # Might need a config name
            try:
                configs = get_dataset_config_names(dataset_name)
                print("This dataset requires a config name. Use --config:\n")
                for c in configs:
                    print(f"  {c}")
            except Exception:
                pass
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        sys.exit(1)

    df = ds.to_pandas()
    print(f"Loaded  : {len(df)} rows")
    print(f"Columns : {list(df.columns)}\n")

    if max_questions:
        df = df.head(max_questions)

    # ---- Detect columns ----
    cols = list(df.columns)

    q_col   = question_col or _find_col(cols, _QUESTION_NAMES)
    ans_col = answer_col   or _find_col(cols, _ANSWER_NAMES)

    # Options: either a single list column or individual A/B/C/D columns
    ind_cols = _detect_individual_option_cols(cols)    # [colA, colB, colC, colD] or None
    opt_col  = options_col or (None if ind_cols else _find_col(cols, _OPTIONS_NAMES))

    # Report detection
    print("Column mapping:")
    print(f"  Question : {q_col}")
    if ind_cols:
        print(f"  Options  : {ind_cols}  (individual columns)")
    else:
        print(f"  Options  : {opt_col}  (list column)")
    print(f"  Answer   : {ans_col}\n")

    if not q_col:
        print("ERROR: Could not detect question column.")
        print("Use --question-col to specify it manually.")
        sys.exit(1)
    if not ans_col:
        print("ERROR: Could not detect answer column.")
        print("Use --answer-col to specify it manually.")
        sys.exit(1)
    if not opt_col and not ind_cols:
        print("ERROR: Could not detect options column.")
        print("Use --options-col to specify it manually.")
        sys.exit(1)

    # ---- Build output rows ----
    rows   = []
    kept   = 0
    dropped_answer = 0
    dropped_opts   = 0

    for i, row in enumerate(df.itertuples(index=False), start=1):
        question = getattr(row, q_col)

        # Get options list
        if ind_cols:
            opts = [str(getattr(row, c)) for c in ind_cols]
        else:
            raw_opts = getattr(row, opt_col)
            # Handle nested dict format e.g. ARC: {"text": [...], "label": [...]}
            if isinstance(raw_opts, dict):
                raw_opts = raw_opts.get("text", raw_opts.get("texts", list(raw_opts.values())[0]))
            opts = list(raw_opts) if hasattr(raw_opts, "__iter__") and not isinstance(raw_opts, str) else [raw_opts]

        if len(opts) < 4:
            dropped_opts += 1
            continue

        # Normalise answer to letter
        raw_ans = getattr(row, ans_col)
        letter  = _to_letter(raw_ans)
        if letter is None:
            dropped_answer += 1
            continue

        rows.append({
            "id":       f"q{i}",
            "question": question,
            "option_a": opts[0],
            "option_b": opts[1],
            "option_c": opts[2],
            "option_d": opts[3],
            "answer":   letter,
        })
        kept += 1

    out_df = pd.DataFrame(rows)

    # ---- Report ----
    print(f"Kept    : {kept}")
    if dropped_answer:
        print(f"Dropped : {dropped_answer} (answer not A-D compatible)")
    if dropped_opts:
        print(f"Dropped : {dropped_opts} (fewer than 4 options)")

    if out_df.empty:
        print("\nERROR: No usable rows extracted. Check column mapping.")
        sys.exit(1)

    dist = out_df["answer"].value_counts().sort_index()
    print(f"\nAnswer distribution:")
    for ans, count in dist.items():
        print(f"  {ans}: {count:4d}  ({count/len(out_df)*100:.1f}%)")

    print(f"\nPreview:")
    print(out_df[["id", "question", "option_a", "answer"]].head(3).to_string(index=False))

    if dry_run:
        print("\n[dry-run] Nothing saved.")
        return out_df

    # ---- Save ----
    stem = output_name or (
        f"{dataset_name.split('/')[-1]}"
        + (f"_{config}" if config else "")
        + (f"_{split}" if split != "train" else "")
    )
    # Sanitise filename
    stem = stem.replace("/", "_").replace("\\", "_").replace(" ", "_")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{stem}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved   : {out_path}  ({len(out_df)} questions)")
    return out_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download any HuggingFace MCQ dataset and convert to project format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("dataset",
                        help="HuggingFace dataset name, e.g. wzzzq/MMLU-PRO-Leveled-TinyBench")
    parser.add_argument("--config",   type=str, default=None,
                        help="Dataset config/subset name (e.g. college_mathematics for cais/mmlu)")
    parser.add_argument("--split",    type=str, default="train",
                        help="Dataset split to use (default: train). "
                             "If the dataset has no 'train' split the error message "
                             "will list valid names — re-run with --split <name>.")
    parser.add_argument("--output",   type=str, default=str(OUTPUT_DIR),
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--name",     type=str, default=None,
                        help="Override output filename stem (without .csv)")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Cap number of questions extracted")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print detected columns and preview without saving")

    # Manual column overrides
    parser.add_argument("--question-col", type=str, default=None,
                        help="Override auto-detected question column name")
    parser.add_argument("--options-col",  type=str, default=None,
                        help="Override auto-detected options list column name")
    parser.add_argument("--answer-col",   type=str, default=None,
                        help="Override auto-detected answer column name")

    args = parser.parse_args()

    extract(
        dataset_name  = args.dataset,
        config        = args.config,
        split         = args.split,
        question_col  = args.question_col,
        options_col   = args.options_col,
        answer_col    = args.answer_col,
        output_dir    = Path(args.output),
        max_questions = args.max_questions,
        dry_run       = args.dry_run,
        output_name   = args.name,
    )


if __name__ == "__main__":
    main()
