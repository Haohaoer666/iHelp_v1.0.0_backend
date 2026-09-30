# iHelp Ch04 Hybrid Retrieval and RAG Quality Design

## Goal

Upgrade the knowledge-based customer-service answer path from dense-only retrieval to hybrid dense + BM25 retrieval with reranking, citations, refusal controls, a low-confidence question pool, and a four-strategy evaluation harness.

## Scope

In scope:

- Milvus native BM25 on the existing `knowledge` collection, using the `chinese` analyzer.
- Hybrid retrieval: dense Top-10 + BM25 Top-10, fused with RRF.
- Metadata filtering on `category`.
- Query normalization and retrieval-only synonym expansion.
- Reranking with SiliconFlow `BAAI/bge-reranker-v2-m3`.
- Citation identifiers that map back to source chunk text and section path.
- Explicit refusal when evidence is missing or insufficient.
- `low_confidence_questions` persistence.
- A small difficulty-graded evaluation set and four-strategy comparison report.
- Frontend clickable citations and one-shot local satisfaction feedback.

Out of scope:

- Coreference resolution.
- Multi-turn query rewriting.
- Feedback persistence to the backend.
- Frontend category filter controls.

## Decisions

- Use the existing `query_faq` tool as the RAG orchestration entry point; do not create a separate RAG chat endpoint.
- Rebuild the existing `knowledge` collection in place and re-ingest current MySQL chunks.
- Rerank through SiliconFlow's rerank API with model `BAAI/bge-reranker-v2-m3`.
- Use LLM few-shot normalization plus a service-side synonym dictionary.
- Start with a 20-query seed evaluation set; support later replacement or expansion.
- Expose category filtering in the retrieval service and as an optional `query_faq` tool argument.
- Directly edit the existing frontend at `D:\std_selftest_frontend`.

## Current State

- Milvus server version: `3.0.0`.
- PyMilvus version: `3.0.1`.
- Existing collection: `Haohelper.knowledge`, 57 rows.
- Existing collection has dense `vector` but no BM25 function.
- Existing backend stores authoritative chunks in MySQL `knowledge_chunks`.
- Existing chat page is a Vite app at `D:\std_selftest_frontend`.

## Architecture

```text
user message
  -> customer-service model chooses query_faq
  -> query_faq(keyword, category?, conversation_id?)
  -> QueryUnderstanding
       normalized_query + synonym_terms + category
  -> HybridRetriever
       dense Top-10 + BM25 Top-10
       RRF fusion
       category filter
       bge-reranker-v2-m3 rerank
  -> EvidenceAssembler
       citation ids and first/last ordering
  -> SufficiencySelfCheck
       sufficient -> generate cited answer
       insufficient -> refusal + low-confidence insert
  -> SSE deltas + citations metadata
```

Responsibility boundaries:

- `VectorStore`: Milvus connection, schema, dense search, BM25/hybrid search.
- `HybridRetriever`: recall, filtering, fusion, rerank orchestration.
- `QueryUnderstanding`: normalization and synonym expansion.
- `RAGService`: end-to-end retrieval and quality control.
- `query_faq`: LangChain tool facade returning RAG results and citation metadata.
- `chat.py`: tool routing, conversation injection, SSE metadata emission.

## Data Model

Add:

```text
low_confidence_questions
  id
  question
  source_conversation_id
  entry_point
  reason
  created_at
```

The existing `knowledge_chunks` table remains the authoritative source for section paths and chunk text.

## Milvus and Hybrid Retrieval

Rebuild `knowledge` with:

- `text` VARCHAR with `enable_analyzer=True` and `analyzer_params={"type": "chinese"}`.
- `sparse` SPARSE_FLOAT_VECTOR.
- `Function("bm25", FunctionType.BM25, input_field_names=["text"], output_field_names=["sparse"])`.
- Dense index `AUTOINDEX` with `COSINE`.
- Sparse inverted index; exact index string verified against PyMilvus 3.0.1 and Milvus 3.0 before implementation.

`VectorStore` exposes:

- Dense search for `dense` strategy.
- BM25 search for `bm25` strategy.
- Hybrid search for `hybrid` and `hybrid_rerank` strategies.

Hybrid search uses `AnnSearchRequest` for both paths and `RRFRanker(k=60)`. Dense and BM25 requests share the same category filter expression.

Normalized hits contain:

- `chunk_id`
- `category`
- `text`
- `score` or `rank`

Section paths are enriched from MySQL by the RAG service.

## Query Understanding

`normalize_query()` uses the configured chat model with few-shot examples and `temperature=0`, returning:

```json
{
  "normalized_query": "...",
  "category": "optional category",
  "is_knowledge_question": true
}
```

`expand_synonyms()` reads a service-side dictionary. Expanded terms are only appended to the retrieval query; knowledge chunks are never duplicated or rewritten.

The retrieval query is:

- Dense path: embed the normalized query with expanded terms.
- BM25 path: submit the normalized query with expanded terms through the BM25 function.

## Reranking

Reranker configuration:

- `RERANK_BASE_URL`
- `RERANK_MODEL=BAAI/bge-reranker-v2-m3`
- `RERANK_API_KEY`
- `RERANK_TOP_K`
- `RERANK_SCORE_THRESHOLD`

The reranker receives the query and candidate chunk texts, then returns reranked top results. The exact SiliconFlow request and response schema is verified against current documentation or live API before implementation.

## Evidence Assembly and Generation

After reranking:

1. Sort by rerank score descending.
2. Place the strongest chunk first and the second-strongest chunk last; remaining chunks stay in the middle rather than being sorted monotonically.
3. Assign `[1]`, `[2]`, ... citation ids by assembled prompt position.
4. Preserve a mapping from citation id to `chunk_id`, `section_path`, and original text.

Generation prompt constraints:

- Cite evidence with `[n]` identifiers.
- Only cite chunks present in the current evidence block.
- Do not fabricate facts or sources.
- Refuse when evidence is insufficient.
- Never promise arrival time, logistics delivery time, refund arrival time, restock time, compensation amount, or other negative-knowledge commitments.

## Sufficiency Check and Low Confidence Pool

Before final generation, the model checks whether the evidence can answer the user question:

```json
{
  "sufficient": true,
  "reason": "..."
}
```

If `sufficient` is false:

- Return a refusal answer directly from the self-check step.
- Insert a `low_confidence_questions` row with:
  - the original user question,
  - `source_conversation_id`,
  - `entry_point=query_faq_self_check`,
  - the self-check reason,
  - current timestamp.

The `query_faq` tool accepts an optional `conversation_id`; `chat.py` injects the current session id when the model omits it.

## SSE Contract

Existing event types remain:

- `tool_status`
- `delta`

Add:

```json
{"type": "citations", "citations": [{"id": 1, "chunk_id": "...", "section_path": "...", "text": "..."}]}
```

`citations` is emitted before `[DONE]`. A refusal answer is streamed as normal `delta` with an empty citations list.

## Evaluation

Seed evaluation set:

`eval_data/ch04_eval_set.json`

Each item has:

- `id`
- `query`
- `type`
- `difficulty`
- `category`
- `ground_truth_chunk_ids`
- `ground_truth_answer`
- `expect_refusal`

Query types:

- `simple`
- `colloquial`
- `model_specific`
- `filtered`
- `out_of_kb`

Difficulty levels:

- `easy`
- `medium`
- `hard`

Strategies:

- `dense`
- `bm25`
- `hybrid`
- `hybrid_rerank`

Retrieval metrics:

- `Recall@3`
- `Recall@5`
- `Recall@10`
- `MRR`

Generation metric:

- `Faithfulness`, judged by the configured chat model against the answer and evidence.
- Refusals are reported separately using `expect_refusal`.

Reports:

- `reports/ch04_eval_report.json`
- `reports/ch04_eval_report.md`

Reports bucket results by query type and difficulty.

## Frontend

Files under `D:\std_selftest_frontend`:

- Render `[n]` references as clickable citation controls.
- Clicking a citation opens a source panel showing section path and original chunk text.
- Add one `👍 / 👎` group under each assistant answer.
- Feedback locks after one click and displays `已反馈`.
- Feedback is local-only and does not call the backend.

## Configuration

Add to `.env.example`:

```text
RERANK_BASE_URL=https://api.siliconflow.cn/v1
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RERANK_API_KEY=
RERANK_TOP_K=10
RERANK_SCORE_THRESHOLD=0.35
```

## Testing and Verification

- Unit tests cover retrieval strategy selection, category filter construction, rerank result normalization, evidence ordering, low-confidence insertion, citation mapping, and metric calculations.
- The prompt/eval-data tasks are verified by running `scripts/eval_ch04.py` against the labeled seed set.
- Backend code remains test-first where it is unit-testable.
- Frontend is implemented as Vibe Coding and verified in the browser.

## Risks

- Exact BM25 sparse search/index API can differ across Milvus versions; verify against installed PyMilvus 3.0.1 before writing the adapter.
- SiliconFlow rerank response fields may differ from OpenAI-style embeddings; verify before coding the client.
- Existing collection has data and must be rebuilt safely from MySQL chunks.
- Model self-check can be flaky; the refusal path is kept deterministic after self-check returns insufficient.
