#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-1.7B-Base}"
WORKDIR="${WORKDIR:-./tokenizer_workdir}"
OUTPUT_DIR="${OUTPUT_DIR:-./qwen3_extended}"

python step3_expand_tokenizer.py   --base_model "$BASE_MODEL"   --reviewed_csv "$WORKDIR/candidate_tokens.csv"   --output_dir "$OUTPUT_DIR"   --validation_text_file "$WORKDIR/merged_clean_corpus.txt"   --validation_samples 5000
