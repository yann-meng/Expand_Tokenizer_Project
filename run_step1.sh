#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-1.7B-Base}"
INPUT_PATH="${INPUT_PATH:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./tokenizer_workdir}"

python step1_mine_candidates.py   --base_model "$BASE_MODEL"   --input_path "$INPUT_PATH"   --output_dir "$OUTPUT_DIR"   --sp_vocab_size 16000   --max_candidates 5000   --min_freq 20
