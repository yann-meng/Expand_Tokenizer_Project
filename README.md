# Tokenizer Expansion Workflow

A 3-step workflow for extending a Qwen3 tokenizer with domain-specific tokens:

1. **Mine candidates** from your corpus
2. **Review candidates manually**
3. **Expand tokenizer and model embeddings**

## Files

- `step1_mine_candidates.py`
- `step2_review_template.md`
- `step3_expand_tokenizer.py`
- `requirements.txt`
- `run_step1.sh`
- `run_step3.sh`
- `example_candidate_tokens.csv`

## Quick start

```bash
pip install -r requirements.txt

bash run_step1.sh
# edit candidate_tokens.csv manually

bash run_step3.sh
```

## Notes

- This workflow does **not** replace the original Qwen3 tokenizer.
- It appends reviewed tokens as normal tokens or special tokens.
- After expansion, you should still do continued pretraining or SFT.
