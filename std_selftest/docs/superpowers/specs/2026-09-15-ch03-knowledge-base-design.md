# iHelp Ch03 Knowledge Base Design

## Goal

Upgrade `query_faq` from MySQL keyword search to Milvus dense vector retrieval while preserving its tool contract.

## Architecture

- Markdown documents are split into structure-aware chunks.
- Chunks are written to MySQL `knowledge_chunks` with `pending` status.
- Each chunk is embedded with Qwen3-Embedding-0.6B through SiliconFlow.
- Vectors are upserted into Milvus `knowledge`.
- `query_faq` embeds the question and searches Milvus Top-K.
- Historical conversations are mined into staged QA pairs and then inserted as knowledge chunks.

## Data Model

- `knowledge_chunks`: authoritative text, questions, category, metadata, vector status, Milvus ID, prev/next chunk.
- `extracted_qa_pairs`: staging table for LLM-mined QA pairs.

## Key Decisions

- Use PyMilvus `MilvusClient`.
- Use COSINE similarity; higher score is better in this Milvus version.
- Keep dense vector single-path retrieval only.
- Do not implement hybrid retrieval or reranking in this chapter.
