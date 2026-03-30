#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import random
from typing import Dict, List, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--reviewed_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--validation_text_file", type=str, default="")
    parser.add_argument("--validation_samples", type=int, default=5000)
    parser.add_argument("--validation_max_chars", type=int, default=2000)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--torch_dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--low_cpu_mem_usage", action="store_true", default=True)
    return parser.parse_args()

def ensure_exists(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)

def get_torch_dtype(name: str):
    return {"auto": "auto", "float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]

def load_reviewed_tokens(csv_path: str) -> Tuple[List[str], List[str], Dict]:
    normal_tokens, special_tokens = [], []
    stats = {"rows_total": 0, "accept_rows": 0, "special_rows": 0, "reject_rows": 0,
             "normalize_rows": 0, "empty_decision_rows": 0, "invalid_rows": 0}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats["rows_total"] += 1
            candidate = (row.get("candidate") or "").strip()
            decision = (row.get("decision") or "").strip().lower()
            normalized_to = (row.get("normalized_to") or "").strip()
            if not decision:
                stats["empty_decision_rows"] += 1
                continue
            if decision == "accept":
                if candidate:
                    normal_tokens.append(candidate); stats["accept_rows"] += 1
                else:
                    stats["invalid_rows"] += 1
            elif decision == "special":
                if candidate:
                    special_tokens.append(candidate); stats["special_rows"] += 1
                else:
                    stats["invalid_rows"] += 1
            elif decision == "reject":
                stats["reject_rows"] += 1
            elif decision == "normalize":
                if normalized_to:
                    normal_tokens.append(normalized_to); stats["normalize_rows"] += 1
                else:
                    stats["invalid_rows"] += 1
            else:
                stats["invalid_rows"] += 1
    def unique_keep_order(items: List[str]) -> List[str]:
        seen, result = set(), []
        for x in items:
            if x and x not in seen:
                seen.add(x); result.append(x)
        return result
    normal_tokens = unique_keep_order(normal_tokens)
    special_tokens = unique_keep_order(special_tokens)
    special_set = set(special_tokens)
    normal_tokens = [x for x in normal_tokens if x not in special_set]
    return normal_tokens, special_tokens, stats

def collect_existing_vocab_strings(tokenizer) -> set:
    return set(tokenizer.get_vocab().keys())

def init_new_embeddings_from_old_segments(model, tokenizer_before_add, tokenizer_after_add, new_tokens: Sequence[str]):
    input_emb = model.get_input_embeddings()
    if input_emb is None:
        raise RuntimeError("Model has no input embeddings.")
    input_weight = input_emb.weight.data
    output_emb = model.get_output_embeddings()
    output_weight = output_emb.weight.data if output_emb is not None and hasattr(output_emb, "weight") else None
    inited = skipped = 0
    for tok in new_tokens:
        new_id = tokenizer_after_add.convert_tokens_to_ids(tok)
        if new_id is None or new_id < 0:
            skipped += 1; continue
        old_ids = tokenizer_before_add(tok, add_special_tokens=False)["input_ids"]
        old_ids = [i for i in old_ids if i != new_id]
        if not old_ids:
            skipped += 1; continue
        input_weight[new_id] = input_weight[old_ids].mean(dim=0)
        if output_weight is not None and new_id < output_weight.shape[0]:
            output_weight[new_id] = output_weight[old_ids].mean(dim=0)
        inited += 1
    return {"initialized": inited, "skipped": skipped}

def load_validation_texts(path: str, n: int, max_chars: int, seed: int) -> List[str]:
    if not path or not os.path.exists(path):
        return []
    texts = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            text = line.strip()
            if text:
                texts.append(text[:max_chars])
    if len(texts) <= n:
        return texts
    return random.Random(seed).sample(texts, n)

def evaluate_compression(texts: List[str], tokenizer_before, tokenizer_after) -> Dict:
    if not texts:
        return {"num_eval_texts": 0, "avg_old_tokens": 0.0, "avg_new_tokens": 0.0,
                "token_reduction": 0.0, "relative_improvement_pct": 0.0}
    total_old = total_new = improved = unchanged = worsened = 0
    for text in texts:
        old_len = len(tokenizer_before(text, add_special_tokens=False)["input_ids"])
        new_len = len(tokenizer_after(text, add_special_tokens=False)["input_ids"])
        total_old += old_len; total_new += new_len
        if new_len < old_len: improved += 1
        elif new_len == old_len: unchanged += 1
        else: worsened += 1
    avg_old = total_old / len(texts)
    avg_new = total_new / len(texts)
    reduction = avg_old - avg_new
    improvement_pct = (reduction / avg_old * 100.0) if avg_old > 0 else 0.0
    return {"num_eval_texts": len(texts), "avg_old_tokens": avg_old, "avg_new_tokens": avg_new,
            "token_reduction": reduction, "relative_improvement_pct": improvement_pct,
            "num_improved_texts": improved, "num_unchanged_texts": unchanged, "num_worsened_texts": worsened}

def save_metadata(path: str, payload: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def main():
    args = parse_args()
    set_seed(args.random_seed)
    ensure_exists(args.reviewed_csv)
    os.makedirs(args.output_dir, exist_ok=True)
    normal_tokens, special_tokens, review_stats = load_reviewed_tokens(args.reviewed_csv)
    print(json.dumps(review_stats, ensure_ascii=False, indent=2))
    tokenizer_before_add = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code, use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=args.trust_remote_code,
        torch_dtype=get_torch_dtype(args.torch_dtype), low_cpu_mem_usage=args.low_cpu_mem_usage,
    )
    if args.device != "cpu":
        model = model.to(args.device)
    old_vocab_size = len(tokenizer)
    original_vocab = collect_existing_vocab_strings(tokenizer_before_add)
    added_normal = tokenizer.add_tokens(normal_tokens) if normal_tokens else 0
    added_special = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens}) if special_tokens else 0
    new_vocab_size = len(tokenizer)
    if new_vocab_size == old_vocab_size:
        print("[Warn] No token added."); return
    model.resize_token_embeddings(new_vocab_size)
    actually_added_tokens = []
    for tok in normal_tokens + special_tokens:
        if tok not in original_vocab:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid >= old_vocab_size:
                actually_added_tokens.append(tok)
    init_stats = init_new_embeddings_from_old_segments(model, tokenizer_before_add, tokenizer, actually_added_tokens)
    compression_metrics = {}
    eval_texts = load_validation_texts(args.validation_text_file, args.validation_samples,
                                       args.validation_max_chars, args.random_seed)
    if eval_texts:
        compression_metrics = evaluate_compression(eval_texts, tokenizer_before_add, tokenizer)
        print(json.dumps(compression_metrics, ensure_ascii=False, indent=2))
    tokenizer.save_pretrained(args.output_dir)
    model.save_pretrained(args.output_dir)
    save_metadata(os.path.join(args.output_dir, "step3_metadata.json"),
                  {"base_model": args.base_model, "reviewed_csv": args.reviewed_csv,
                   "old_vocab_size": old_vocab_size, "new_vocab_size": new_vocab_size,
                   "normal_tokens_count": len(normal_tokens), "special_tokens_count": len(special_tokens),
                   "normal_tokens_preview": normal_tokens[:200], "special_tokens_preview": special_tokens[:200],
                   "review_stats": review_stats, "init_stats": init_stats,
                   "compression_metrics": compression_metrics, "args": vars(args)})
    print(f"Expanded tokenizer/model saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
