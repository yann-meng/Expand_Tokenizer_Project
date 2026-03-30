
# 3) `step3_expand_tokenizer.py`

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 3:
读取人工审核后的 candidate_tokens.csv，正式扩充 tokenizer。

功能：
1. 读取审核后的 CSV
2. accept -> 普通 token
3. special -> special token
4. normalize -> 归并到 normalized_to 后再决定是否加入
5. resize embedding
6. 用旧 tokenizer 拆分结果初始化新 embedding
7. 保存 tokenizer / model
8. 验证压缩率提升

依赖：
    pip install -U transformers torch pandas

示例：
    python step3_expand_tokenizer.py \
      --base_model Qwen/Qwen3-1.7B-Base \
      --reviewed_csv ./tokenizer_workdir/candidate_tokens.csv \
      --output_dir ./qwen3_extended \
      --validation_text_file ./tokenizer_workdir/merged_clean_corpus.txt \
      --validation_samples 5000
"""

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
    parser.add_argument("--torch_dtype", type=str, default="auto",
                        choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--low_cpu_mem_usage", action="store_true", default=True)

    parser.add_argument("--keep_embed_dim", action="store_true", default=False)

    return parser.parse_args()


def ensure_exists(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def get_torch_dtype(name: str):
    if name == "auto":
        return "auto"
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def load_reviewed_tokens(csv_path: str) -> Tuple[List[str], List[str], Dict]:
    """
    decision 支持：
    - accept
    - special
    - reject
    - normalize

    规则：
    - accept: candidate 作为普通 token
    - special: candidate 作为 special token
    - reject: 丢弃
    - normalize: 使用 normalized_to 作为普通 token
    """
    normal_tokens = []
    special_tokens = []

    stats = {
        "rows_total": 0,
        "accept_rows": 0,
        "special_rows": 0,
        "reject_rows": 0,
        "normalize_rows": 0,
        "empty_decision_rows": 0,
        "invalid_rows": 0,
    }

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
                    normal_tokens.append(candidate)
                    stats["accept_rows"] += 1
                else:
                    stats["invalid_rows"] += 1

            elif decision == "special":
                if candidate:
                    special_tokens.append(candidate)
                    stats["special_rows"] += 1
                else:
                    stats["invalid_rows"] += 1

            elif decision == "reject":
                stats["reject_rows"] += 1

            elif decision == "normalize":
                if normalized_to:
                    normal_tokens.append(normalized_to)
                    stats["normalize_rows"] += 1
                else:
                    stats["invalid_rows"] += 1
            else:
                stats["invalid_rows"] += 1

    # 去重并保持顺序
    def unique_keep_order(items: List[str]) -> List[str]:
        seen = set()
        result = []
        for x in items:
            if x and x not in seen:
                seen.add(x)
                result.append(x)
        return result

    normal_tokens = unique_keep_order(normal_tokens)
    special_tokens = unique_keep_order(special_tokens)

    # 若同一个 token 同时在 normal / special，中间优先 special
    normal_tokens = [x for x in normal_tokens if x not in set(special_tokens)]

    return normal_tokens, special_tokens, stats


def collect_existing_vocab_strings(tokenizer) -> set:
    return set(tokenizer.get_vocab().keys())


def init_new_embeddings_from_old_segments(
    model,
    tokenizer_before_add,
    tokenizer_after_add,
    new_tokens: Sequence[str],
):
    input_emb = model.get_input_embeddings()
    if input_emb is None:
        raise RuntimeError("Model has no input embeddings.")
    input_weight = input_emb.weight.data

    output_emb = model.get_output_embeddings()
    output_weight = None
    if output_emb is not None and hasattr(output_emb, "weight"):
        output_weight = output_emb.weight.data

    inited = 0
    skipped = 0

    for tok in new_tokens:
        new_id = tokenizer_after_add.convert_tokens_to_ids(tok)
        if new_id is None or new_id < 0:
            skipped += 1
            continue

        old_ids = tokenizer_before_add(tok, add_special_tokens=False)["input_ids"]
        old_ids = [i for i in old_ids if i != new_id]

        if not old_ids:
            skipped += 1
            continue

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

    if not texts:
        return []

    rng = random.Random(seed)
    if len(texts) <= n:
        return texts
    return rng.sample(texts, n)


def evaluate_compression(texts: List[str], tokenizer_before, tokenizer_after) -> Dict:
    if not texts:
        return {
            "num_eval_texts": 0,
            "avg_old_tokens": 0.0,
            "avg_new_tokens": 0.0,
            "token_reduction": 0.0,
            "relative_improvement_pct": 0.0,
        }

    total_old = 0
    total_new = 0
    improved = 0
    unchanged = 0
    worsened = 0

    for text in texts:
        old_len = len(tokenizer_before(text, add_special_tokens=False)["input_ids"])
        new_len = len(tokenizer_after(text, add_special_tokens=False)["input_ids"])

        total_old += old_len
        total_new += new_len

        if new_len < old_len:
            improved += 1
        elif new_len == old_len:
            unchanged += 1
        else:
            worsened += 1

    avg_old = total_old / len(texts)
    avg_new = total_new / len(texts)
    reduction = avg_old - avg_new
    improvement_pct = (reduction / avg_old * 100.0) if avg_old > 0 else 0.0

    return {
        "num_eval_texts": len(texts),
        "avg_old_tokens": avg_old,
        "avg_new_tokens": avg_new,
        "token_reduction": reduction,
        "relative_improvement_pct": improvement_pct,
        "num_improved_texts": improved,
        "num_unchanged_texts": unchanged,
        "num_worsened_texts": worsened,
    }


def save_metadata(path: str, payload: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    set_seed(args.random_seed)
    ensure_exists(args.reviewed_csv)
    os.makedirs(args.output_dir, exist_ok=True)

    normal_tokens, special_tokens, review_stats = load_reviewed_tokens(args.reviewed_csv)

    print("[Info] Review stats:")
    print(json.dumps(review_stats, ensure_ascii=False, indent=2))
    print(f"[Info] normal tokens : {len(normal_tokens)}")
    print(f"[Info] special tokens: {len(special_tokens)}")

    print("[Info] Loading tokenizer and model...")
    tokenizer_before_add = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )

    torch_dtype = get_torch_dtype(args.torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
    )

    if args.device != "cpu":
        model = model.to(args.device)

    old_vocab_size = len(tokenizer)
    original_vocab = collect_existing_vocab_strings(tokenizer_before_add)

    added_normal = tokenizer.add_tokens(normal_tokens) if normal_tokens else 0
    added_special = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens}) if special_tokens else 0

    new_vocab_size = len(tokenizer)

    print(f"[Info] old vocab size: {old_vocab_size}")
    print(f"[Info] new vocab size: {new_vocab_size}")
    print(f"[Info] actually added normal : {added_normal}")
    print(f"[Info] actually added special: {added_special}")

    if new_vocab_size == old_vocab_size:
        print("[Warn] No token added.")
        return

    if not args.keep_embed_dim:
        model.resize_token_embeddings(new_vocab_size)

    actually_added_tokens = []
    for tok in normal_tokens + special_tokens:
        if tok not in original_vocab:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid >= old_vocab_size:
                actually_added_tokens.append(tok)

    init_stats = init_new_embeddings_from_old_segments(
        model=model,
        tokenizer_before_add=tokenizer_before_add,
        tokenizer_after_add=tokenizer,
        new_tokens=actually_added_tokens,
    )

    compression_metrics = {}
    eval_texts = load_validation_texts(
        path=args.validation_text_file,
        n=args.validation_samples,
        max_chars=args.validation_max_chars,
        seed=args.random_seed,
    )
    if eval_texts:
        compression_metrics = evaluate_compression(
            texts=eval_texts,
            tokenizer_before=tokenizer_before_add,
            tokenizer_after=tokenizer,
        )
        print("[Info] compression metrics:")
        print(json.dumps(compression_metrics, ensure_ascii=False, indent=2))

    tokenizer.save_pretrained(args.output_dir)
    model.save_pretrained(args.output_dir)

    metadata = {
        "base_model": args.base_model,
        "reviewed_csv": args.reviewed_csv,
        "old_vocab_size": old_vocab_size,
        "new_vocab_size": new_vocab_size,
        "normal_tokens_count": len(normal_tokens),
        "special_tokens_count": len(special_tokens),
        "normal_tokens_preview": normal_tokens[:200],
        "special_tokens_preview": special_tokens[:200],
        "review_stats": review_stats,
        "init_stats": init_stats,
        "compression_metrics": compression_metrics,
        "args": vars(args),
    }
    save_metadata(os.path.join(args.output_dir, "step3_metadata.json"), metadata)

    print("\nDone.")
    print(f"Expanded tokenizer/model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
