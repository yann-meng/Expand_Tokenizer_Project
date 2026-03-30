# Step 2: 人工审核 candidate_tokens.csv

请打开 `candidate_tokens.csv`，重点填写以下三列：

- `decision`
- `normalized_to`
- `comment`

## decision 参数选择

- `accept`: 作为普通 token 加入
- `special`: 作为 special token 加入
- `reject`: 不加入
- `normalize`: 规范到 `normalized_to` 指定形式后，作为普通 token 加入

## comment 写什么

写拒绝原因或规范原因，方便后续追踪。

例如：
- 随机 ID
- 只在单个文件重复
- 与已有 token 重复
- 规范到大写形式
- 应改为 special token


## 推荐保留

- 高频领域术语
- 原 tokenizer 会切很多段的词
- 语义稳定的缩写
- 高频代码/API 名称
- 结构化控制标记

## 推荐拒绝

- 随机串 / UUID / trace_id
- 哈希 / 时间戳 / 样本特有脏数据
- 高频但无独立语义的片段
- 泛化价值低的长串

## 审核完成后

```bash
python step3_expand_tokenizer.py   --base_model Qwen/Qwen3-1.7B-Base   --reviewed_csv ./tokenizer_workdir/candidate_tokens.csv   --output_dir ./qwen3_extended   --validation_text_file ./tokenizer_workdir/merged_clean_corpus.txt   --validation_samples 5000
```
