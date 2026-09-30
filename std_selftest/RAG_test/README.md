# RAG 评估说明

本目录存放第四章 RAG 评估相关脚本和测试。

目录结构：
```
  eval_ch04.py
    -> 检索策略比较
    -> 证据能否支持标准答案
  
  eval_ragas_ch04.py
    -> 最终生成回答的 RAGAS 质量评估
  ```

```text
RAG_test/
  scripts/
    eval_ch04.py
    eval_ragas_ch04.py
  tests/
    test_eval_ch04.py
    test_eval_ragas_ch04.py
```

## 四策略检索评估

运行：

```bash
uv run python RAG_test/scripts/eval_ch04.py \
  --strategies dense bm25 hybrid hybrid_rerank \
  --report-dir reports
```

该脚本会分别运行：

- `dense`
- `bm25`
- `hybrid`
- `hybrid_rerank`

计算指标：

- `Recall@3`
- `Recall@5`
- `Recall@10`
- `MRR`

报告输出位置：

```text
reports/ch04_eval_report.json
reports/ch04_eval_report.md
```

## 带自定义 Faithfulness 的检索评估

运行：

```bash
uv run python RAG_test/scripts/eval_ch04.py \
  --strategies dense bm25 hybrid hybrid_rerank \
  --report-dir reports \
  --with-faithfulness
```

## RAGAS 生成质量评估

运行：

```bash
uv run python RAG_test/scripts/eval_ragas_ch04.py \
  --strategies hybrid_rerank \
  --ragas-top-k 5
```

该脚本会调用 RAGAS，计算：

- `Faithfulness`
- `Answer Relevancy`
- `Context Precision`
- `Context Recall`

同时单独统计：

- `refusal_accuracy`

RAGAS 报告输出位置：

```text
reports/ch04_ragas_report.json
reports/ch04_ragas_report.md
reports/ch04_ragas_rows.json
```

## 快速试跑 RAGAS

只跑前 3 条样本：

```bash
uv run python RAG_test/scripts/eval_ragas_ch04.py \
  --strategies hybrid_rerank \
  --ragas-top-k 5 \
  --limit 3
```

## 运行测试

运行 RAG 评估相关测试：

```bash
uv run pytest RAG_test/tests -v
```

运行全量测试：

```bash
uv run pytest -v
```


# 完整运行顺序
cd D:\std_selftest

### 1. 启动 Milvus
docker compose -f D:\Tools\milvus\docker-compose.yml up -d

### 2. 可选：重建知识库
uv run python scripts/rebuild_hybrid_kb.py

### 3. 四策略检索评估
uv run --no-cache python RAG_test/scripts/eval_ch04.py `
  --strategies dense bm25 hybrid hybrid_rerank `
  --report-dir reports `
  --with-faithfulness

### 4. RAGAS 快速试跑
uv run --no-cache python RAG_test/scripts/eval_ragas_ch04.py `
  --strategies hybrid_rerank `
  --ragas-top-k 5 `
  --limit 3

### 5. RAGAS 全量评估
uv run --no-cache python RAG_test/scripts/eval_ragas_ch04.py `
  --strategies hybrid_rerank `
  --ragas-top-k 5
