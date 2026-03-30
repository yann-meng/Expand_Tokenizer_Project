#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Set, Tuple

import sentencepiece as spm
from transformers import AutoTokenizer

TEXT_FILE_EXTENSIONS = {
    ".txt", ".text", ".md", ".markdown", ".log", ".csv", ".tsv",
    ".jsonl", ".json", ".xml", ".html", ".htm"
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--recursive", action="store_true", default=True)
    parser.add_argument("--file_extensions", type=str, default=",".join(sorted(TEXT_FILE_EXTENSIONS)))
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--sample_rate", type=float, default=1.0)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--min_line_chars", type=int, default=2)
    parser.add_argument("--max_line_chars", type=int, default=20000)
    parser.add_argument("--normalize_whitespace", action="store_true", default=True)
    parser.add_argument("--deduplicate_lines", action="store_true", default=False)
    parser.add_argument("--sp_model_prefix", type=str, default="domain_sp")
    parser.add_argument("--sp_vocab_size", type=int, default=16000)
    parser.add_argument("--sp_model_type", type=str, default="unigram", choices=["unigram", "bpe", "char", "word"])
    parser.add_argument("--character_coverage", type=float, default=0.9995)
    parser.add_argument("--input_sentence_size", type=int, default=2000000)
    parser.add_argument("--shuffle_input_sentence", action="store_true", default=True)
    parser.add_argument("--max_sentencepiece_length", type=int, default=16)
    parser.add_argument("--max_candidates", type=int, default=5000)
    parser.add_argument("--min_freq", type=int, default=20)
    parser.add_argument("--min_piece_chars", type=int, default=2)
    parser.add_argument("--max_piece_chars", type=int, default=24)
    parser.add_argument("--min_old_token_len", type=int, default=2)
    parser.add_argument("--allow_whitespace_piece", action="store_true")
    parser.add_argument("--max_examples_per_token", type=int, default=3)
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    return parser.parse_args()

def set_seed(seed: int):
    random.seed(seed)

def ensure_exists(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

def is_binary_like(data: bytes) -> bool:
    if not data:
        return False
    if b"\x00" in data:
        return True
    sample = data[:4096]
    nontext = sum((b < 9 or (13 < b < 32)) for b in sample)
    return (nontext / max(1, len(sample))) > 0.10

def detect_readable_text_file(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return not is_binary_like(f.read(4096))
    except Exception:
        return False

def normalize_text(text: str, normalize_whitespace: bool = True) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u0000", "").replace("\ufeff", "")
    if normalize_whitespace:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def is_bad_text(text: str, min_chars: int, max_chars: int) -> bool:
    if not text or len(text) < min_chars or len(text) > max_chars:
        return True
    printable = sum(ch.isprintable() or ch in "\n\t" for ch in text)
    if printable / max(1, len(text)) < 0.85:
        return True
    if len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", text)) == 0:
        return True
    return False

def iter_text_from_json_obj(obj) -> Iterator[str]:
    if isinstance(obj, dict):
        for key in ["text", "content", "body", "document", "raw_text", "message"]:
            val = obj.get(key)
            if isinstance(val, str) and val.strip():
                yield val
        msgs = obj.get("messages")
        if isinstance(msgs, list):
            for m in msgs:
                if isinstance(m, dict):
                    for key in ["content", "text", "value"]:
                        val = m.get(key)
                        if isinstance(val, str) and val.strip():
                            yield val

def read_text_file(path: Path) -> Iterator[str]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".jsonl":
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        found = False
                        for text in iter_text_from_json_obj(obj):
                            found = True
                            yield text
                        if not found:
                            yield line
                    except Exception:
                        yield line
            return
        if suffix == ".json":
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                try:
                    obj = json.load(f)
                    found = False
                    if isinstance(obj, list):
                        for item in obj:
                            for text in iter_text_from_json_obj(item):
                                found = True
                                yield text
                    else:
                        for text in iter_text_from_json_obj(obj):
                            found = True
                            yield text
                    if found:
                        return
                except Exception:
                    pass
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                yield line.rstrip("\n")
    except Exception as e:
        print(f"[Warn] Failed to read file: {path} ({e})")

def collect_input_files(input_path: str, recursive: bool, allowed_exts: Set[str], max_files: int) -> List[Path]:
    p = Path(input_path)
    if p.is_file():
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(input_path)
    files = [x for x in (p.rglob("*") if recursive else p.glob("*")) if x.is_file()]
    result = [fp for fp in files if not allowed_exts or fp.suffix.lower() in allowed_exts]
    result.sort()
    return result[:max_files] if max_files > 0 else result

def prepare_clean_corpus(input_path: str, merged_corpus_path: str, recursive: bool, allowed_exts: Set[str], max_files: int,
                         min_line_chars: int, max_line_chars: int, normalize_whitespace: bool,
                         deduplicate_lines: bool, sample_rate: float, random_seed: int) -> Dict:
    print("[Step 1/4] Preparing clean corpus...")
    rng = random.Random(random_seed)
    files = collect_input_files(input_path, recursive, allowed_exts, max_files)
    if not files:
        raise RuntimeError("No valid input files found.")
    seen_lines = set() if deduplicate_lines else None
    stats = {"num_input_files": len(files), "num_binary_skipped_files": 0, "num_total_raw_lines": 0,
             "num_written_lines": 0, "num_empty_skipped": 0, "num_too_short_skipped": 0,
             "num_too_long_skipped": 0, "num_bad_text_skipped": 0, "num_sampled_out": 0,
             "num_dedup_skipped": 0}
    with open(merged_corpus_path, "w", encoding="utf-8") as out_f:
        for fp in files:
            if not detect_readable_text_file(fp):
                stats["num_binary_skipped_files"] += 1
                continue
            for raw in read_text_file(fp):
                stats["num_total_raw_lines"] += 1
                if sample_rate < 1.0 and rng.random() > sample_rate:
                    stats["num_sampled_out"] += 1
                    continue
                text = normalize_text(raw, normalize_whitespace=normalize_whitespace)
                if not text:
                    stats["num_empty_skipped"] += 1
                    continue
                if len(text) < min_line_chars:
                    stats["num_too_short_skipped"] += 1
                    continue
                if len(text) > max_line_chars:
                    stats["num_too_long_skipped"] += 1
                    continue
                if is_bad_text(text, min_line_chars, max_line_chars):
                    stats["num_bad_text_skipped"] += 1
                    continue
                if seen_lines is not None:
                    if text in seen_lines:
                        stats["num_dedup_skipped"] += 1
                        continue
                    seen_lines.add(text)
                out_f.write(text + "\n")
                stats["num_written_lines"] += 1
    if stats["num_written_lines"] == 0:
        raise RuntimeError("No cleaned lines were written.")
    return stats

def train_sentencepiece(corpus_path: str, model_prefix: str, vocab_size: int, model_type: str,
                        character_coverage: float, input_sentence_size: int,
                        shuffle_input_sentence: bool, max_sentencepiece_length: int) -> Tuple[str, str]:
    model_file = f"{model_prefix}.model"
    vocab_file = f"{model_prefix}.vocab"
    if os.path.exists(model_file) and os.path.exists(vocab_file):
        print(f"[Info] Reusing existing SP model: {model_file}")
        return model_file, vocab_file
    print("[Step 2/4] Training SentencePiece...")
    spm.SentencePieceTrainer.train(
        input=corpus_path, model_prefix=model_prefix, vocab_size=vocab_size, model_type=model_type,
        character_coverage=character_coverage, input_sentence_size=input_sentence_size,
        shuffle_input_sentence=shuffle_input_sentence, max_sentencepiece_length=max_sentencepiece_length,
        bos_id=-1, eos_id=-1, pad_id=-1, unk_id=0,
    )
    return model_file, vocab_file

def load_sp_processor(model_file: str):
    sp = spm.SentencePieceProcessor()
    sp.load(model_file)
    return sp

def normalize_sp_piece(piece: str, allow_whitespace_piece: bool) -> str:
    return piece if allow_whitespace_piece else piece.lstrip("▁")

def is_reasonable_token(piece: str, min_chars: int, max_chars: int) -> bool:
    if not piece or len(piece) < min_chars or len(piece) > max_chars:
        return False
    if re.fullmatch(r"[\W_]+", piece, flags=re.UNICODE):
        return False
    if any(ord(ch) < 32 for ch in piece):
        return False
    if piece.strip() != piece or any(ch.isspace() for ch in piece):
        return False
    return True

def tokenize_len(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])

def make_example(text: str, piece: str, max_len: int = 120) -> str:
    idx = text.find(piece)
    if idx < 0:
        return text[:max_len]
    left = max(0, idx - 40)
    right = min(len(text), idx + len(piece) + 40)
    snippet = text[left:right]
    if left > 0:
        snippet = "..." + snippet
    if right < len(text):
        snippet += "..."
    return snippet[:max_len]

def count_candidates_and_examples(corpus_path: str, sp, base_tokenizer, min_freq: int, min_piece_chars: int,
                                  max_piece_chars: int, min_old_token_len: int, max_candidates: int,
                                  allow_whitespace_piece: bool, max_examples_per_token: int) -> List[Dict]:
    print("[Step 3/4] Mining candidate tokens...")
    existing_vocab = set(base_tokenizer.get_vocab().keys())
    freq_counter, sample_cover_counter = Counter(), Counter()
    example_map = defaultdict(list)
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            text = line.strip()
            if not text:
                continue
            pieces = sp.encode(text, out_type=str)
            line_seen = set()
            for raw_piece in pieces:
                piece = normalize_sp_piece(raw_piece, allow_whitespace_piece=allow_whitespace_piece)
                if not is_reasonable_token(piece, min_piece_chars, max_piece_chars):
                    continue
                freq_counter[piece] += 1
                if piece not in line_seen:
                    sample_cover_counter[piece] += 1
                    line_seen.add(piece)
                if len(example_map[piece]) < max_examples_per_token and piece in text:
                    example_map[piece].append(make_example(text, piece))
            if (line_idx + 1) % 100000 == 0:
                print(f"  processed {line_idx + 1} lines")
    candidates = []
    for piece, freq in freq_counter.most_common():
        if freq < min_freq:
            break
        if piece in existing_vocab:
            continue
        old_len = tokenize_len(base_tokenizer, piece)
        if old_len < min_old_token_len:
            continue
        score = freq * (old_len - 1)
        candidates.append({"candidate": piece, "freq": int(freq), "sample_coverage": int(sample_cover_counter[piece]),
                           "old_token_len": int(old_len), "score": float(score),
                           "examples": example_map.get(piece, [])[:max_examples_per_token]})
    dedup = {}
    for item in candidates:
        prev = dedup.get(item["candidate"])
        if prev is None or item["score"] > prev["score"]:
            dedup[item["candidate"]] = item
    result = list(dedup.values())
    result.sort(key=lambda x: (-x["score"], -x["freq"], -x["sample_coverage"], -x["old_token_len"], x["candidate"]))
    return result[:max_candidates]

def write_candidate_csv(candidates: List[Dict], path: str):
    fieldnames = ["candidate", "freq", "sample_coverage", "old_token_len", "score", "examples",
                  "decision", "normalized_to", "comment"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in candidates:
            writer.writerow({"candidate": item["candidate"], "freq": item["freq"],
                             "sample_coverage": item["sample_coverage"], "old_token_len": item["old_token_len"],
                             "score": f"{item['score']:.4f}", "examples": " ||| ".join(item.get("examples", [])),
                             "decision": "", "normalized_to": "", "comment": ""})

def write_metadata(path: str, payload: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def main():
    args = parse_args()
    set_seed(args.random_seed)
    ensure_exists(args.input_path)
    os.makedirs(args.output_dir, exist_ok=True)
    allowed_exts = {x.strip().lower() for x in args.file_extensions.split(",") if x.strip()}
    merged_corpus_path = os.path.join(args.output_dir, "merged_clean_corpus.txt")
    sp_prefix = os.path.join(args.output_dir, args.sp_model_prefix)
    candidate_csv_path = os.path.join(args.output_dir, "candidate_tokens.csv")
    metadata_path = os.path.join(args.output_dir, "step1_metadata.json")
    corpus_stats = prepare_clean_corpus(args.input_path, merged_corpus_path, args.recursive, allowed_exts,
                                        args.max_files, args.min_line_chars, args.max_line_chars,
                                        args.normalize_whitespace, args.deduplicate_lines,
                                        args.sample_rate, args.random_seed)
    sp_model_file, sp_vocab_file = train_sentencepiece(merged_corpus_path, sp_prefix, args.sp_vocab_size,
                                                       args.sp_model_type, args.character_coverage,
                                                       args.input_sentence_size, args.shuffle_input_sentence,
                                                       args.max_sentencepiece_length)
    sp = load_sp_processor(sp_model_file)
    print("[Info] Loading base tokenizer...")
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code, use_fast=True)
    candidates = count_candidates_and_examples(merged_corpus_path, sp, base_tokenizer, args.min_freq,
                                               args.min_piece_chars, args.max_piece_chars,
                                               args.min_old_token_len, args.max_candidates,
                                               args.allow_whitespace_piece, args.max_examples_per_token)
    print("[Step 4/4] Writing candidate CSV...")
    write_candidate_csv(candidates, candidate_csv_path)
    write_metadata(metadata_path, {"base_model": args.base_model, "input_path": args.input_path,
                                   "merged_corpus_path": merged_corpus_path, "sp_model_file": sp_model_file,
                                   "sp_vocab_file": sp_vocab_file, "candidate_csv_path": candidate_csv_path,
                                   "num_candidates": len(candidates), "corpus_stats": corpus_stats,
                                   "args": vars(args)})
    print("Done.")
    print(f"Candidate CSV: {candidate_csv_path}")

if __name__ == "__main__":
    main()
