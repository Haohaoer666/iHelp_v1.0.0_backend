# iHelp Ch03 Knowledge Base Implementation Plan

**Goal:** Build Markdown knowledge base, double-write to MySQL and Milvus, and replace `query_faq` with semantic search.

**Architecture:** Markdown splitting, embedding API, Milvus vector store, MySQL authoritative source, LLM QA mining.

**Tech Stack:** FastAPI, SQLAlchemy, PyMilvus, SiliconFlow Qwen3-Embedding-0.6B, LangChain `@tool`.

## Tasks

- [x] Add embedding/Milvus configuration and dependencies.
- [x] Add `knowledge_chunks` and `extracted_qa_pairs` models.
- [x] Implement structure-aware Markdown splitting.
- [x] Implement embedding client and Milvus collection.
- [x] Implement MySQL-first double write and pending chunk sync.
- [x] Implement historical conversation QA mining.
- [x] Replace `query_faq` with `query_knowledge`.
- [x] Add document splitter tests and ch03 acceptance script.
