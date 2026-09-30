# Ch04 Hybrid Retrieval and RAG Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace dense-only knowledge retrieval with Milvus dense + BM25 hybrid retrieval, bge-reranker-v2-m3 reranking, cited generation, refusal controls, a low-confidence question pool, and a four-strategy evaluation harness.

**Architecture:** Keep `query_faq` as the orchestration entry point. Add `QueryUnderstanding`, `RerankerClient`, `HybridRetriever`, and `RAGService` around the existing MySQL/Milvus knowledge pipeline. Emit citation metadata through SSE and render it in the Vite chat UI.

**Tech Stack:** FastAPI, SQLAlchemy 2 async, PyMilvus 3.0.1, Milvus server v3.0.1, SiliconFlow rerank API, LangChain chat model, Vite plain JavaScript.

**Spec:** `docs/superpowers/specs/2026-09-17-ch04-hybrid-retrieval-design.md`

## Global Constraints

- Keep Milvus collection name `knowledge`, database `Haohelper`.
- Keep dense embedding model `Qwen/Qwen3-Embedding-0.6B`.
- Rerank model exactly `BAAI/bge-reranker-v2-m3`.
- Rerank API key comes from `RERANK_API_KEY`; never hard-code secrets.
- BM25 analyzer exactly `chinese`.
- Dense and BM25 recall Top-10 before RRF; final evidence Top-10 after rerank.
- Citation identifiers use `[1]`, `[2]`, ... style in generated text.
- `low_confidence_questions.entry_point` is `query_faq_self_check`.
- Never promise arrival time, delivery time, refund arrival time, restock time, or compensation amount.
- Frontend feedback is local-only and locks after one click.
- Do not modify unrelated existing code beyond the specified files.

## File Structure

Create or modify these files:

- `app/config.py`
- `app/db/models.py`
- `app/db/repository.py`
- `app/services/vector_store.py`
- `app/services/query_understanding.py`
- `app/services/reranker.py`
- `app/services/hybrid_retriever.py`
- `app/services/rag_service.py`
- `app/core/prompts.py`
- `app/core/tools.py`
- `app/api/chat.py`
- `tests/test_vector_store.py`
- `tests/test_query_understanding.py`
- `tests/test_reranker.py`
- `tests/test_hybrid_retriever.py`
- `tests/test_rag_service.py`
- `tests/test_chat_api.py`
- `eval_data/ch04_eval_set.json`
- `scripts/eval_ch04.py`
- `scripts/rebuild_hybrid_kb.py`
- `D:\std_selftest_frontend\src\main.js`
- `D:\std_selftest_frontend\src\styles.css`

---

### Task 1: BM25 Schema and Hybrid Search in VectorStore

**Files:**
- Modify: `app/config.py`
- Modify: `app/services/vector_store.py`
- Test: `tests/test_vector_store.py`

**Interfaces:**
- Consumes: existing `settings.milvus_uri`, `settings.milvus_database`, `settings.milvus_collection`, `settings.embedding_dim`.
- Produces:
  - `VectorStore.ensure_collection() -> None`
  - `VectorStore.search(vector: list[float], top_k: int | None = None) -> list[dict]`
  - `VectorStore.search_bm25(query: str, top_k: int = 10, category: str | None = None) -> list[dict]`
  - `VectorStore.hybrid_search(vector: list[float], query: str, top_k: int = 10, category: str | None = None) -> list[dict]`

- [ ] **Step 1: Write failing tests**

Update `tests/test_vector_store.py` with a fake client that records schema and search calls.

```python
from pymilvus import AnnSearchRequest, RRFRanker


class FakeMilvusClient:
    def __init__(self):
        self.schema_fields = []
        self.schema_functions = []
        self.indexes = []
        self.hybrid_calls = []
        self.search_calls = []

    async def has_collection(self, collection_name):
        return True

    async def insert(self, collection_name, data):
        return {"ids": [7]}

    async def search(self, collection_name, data, anns_field, search_params, limit, filter, output_fields):
        self.search_calls.append((data, anns_field, search_params, limit, filter))
        return [[{
            "entity": {
                "chunk_id": "chunk-bm25",
                "category": "商品FAQ",
                "questions": "[]",
                "text": "断电后拆下集便仓。",
            },
            "distance": 0.8,
        }]]

    async def hybrid_search(self, collection_name, reqs, ranker, limit, output_fields):
        self.hybrid_calls.append((reqs, ranker, limit))
        return [[{
            "entity": {
                "chunk_id": "chunk-hybrid",
                "category": "商品FAQ",
                "questions": "[]",
                "text": "智能猫砂盆断电后清理。",
            },
            "distance": 0.032,
        }]]

    async def close(self):
        return None


def make_store_with_recorder():
    store = vector_store.VectorStore()
    fake = FakeMilvusClient()
    store.client = fake
    store.root_client = FakeMilvusClient()
    store._database_ready = True
    return store, fake


async def test_hybrid_search_builds_dense_and_bm25_requests():
    store, fake = make_store_with_recorder()

    hits = await store.hybrid_search([0.1, 0.2], "智能猫砂盆", top_k=10, category="商品FAQ")

    assert hits[0]["chunk_id"] == "chunk-hybrid"
    assert len(fake.hybrid_calls) == 1
    reqs, ranker, limit = fake.hybrid_calls[0]
    assert limit == 10
    assert isinstance(ranker, RRFRanker)
    assert reqs[0].anns_field == "vector"
    assert reqs[1].anns_field == "sparse"
    assert reqs[0].data == [[0.1, 0.2]]
    assert reqs[1].data == ["智能猫砂盆"]
    assert reqs[0].expr == 'category == "商品FAQ"'
    await store.close()
```

Add a schema recording test by giving `FakeMilvusClient.has_collection` a configurable `False` and a `create_schema` method that returns a schema-like object with `add_field` and `add_function` callbacks. For this task, use a small dedicated fake:

```python
class RecordingSchema:
    def __init__(self):
        self.fields = []
        self.functions = []

    def add_field(self, name, dtype, **kwargs):
        self.fields.append((name, dtype, kwargs))

    def add_function(self, function):
        self.functions.append(function)


async def test_ensure_collection_registers_bm25_function():
    class CreateFakeClient(FakeMilvusClient):
        def __init__(self):
            super().__init__()
            self.has_collection_value = False
            self.schema = RecordingSchema()
            self.created = False

        async def has_collection(self, collection_name):
            return self.has_collection_value

        def create_schema(self, **kwargs):
            return self.schema

        def prepare_index_params(self):
            return self

        def add_index(self, field_name, index_type, metric_type):
            self.indexes.append((field_name, index_type, metric_type))

        async def create_collection(self, collection_name, schema, index_params):
            self.created = True

        async def load_collection(self, collection_name):
            return None

    store = vector_store.VectorStore()
    fake = CreateFakeClient()
    store.client = fake
    store.root_client = FakeMilvusClient()
    store._database_ready = True

    await store.ensure_collection()

    assert fake.created is True
    assert any(name == "sparse" and dtype == DataType.SPARSE_FLOAT_VECTOR for name, dtype, _ in fake.schema.fields)
    assert fake.schema.functions[0].name == "bm25"
    assert fake.schema.functions[0].type == FunctionType.BM25
    assert any(field == "sparse" for field, _, _ in fake.indexes)
    await store.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_vector_store.py -v
```

Expected: FAIL because `search_bm25`, `hybrid_search`, and `FunctionType` are not defined.

- [ ] **Step 3: Implement config and VectorStore changes**

In `app/config.py`, add:

```python
rerank_base_url: str = "https://api.siliconflow.cn/v1"
rerank_model: str = "BAAI/bge-reranker-v2-m3"
rerank_api_key: str = ""
rerank_top_k: int = 10
rerank_score_threshold: float = 0.35
```

In `app/services/vector_store.py`, add imports:

```python
from pymilvus import (
    AnnSearchRequest,
    AsyncMilvusClient,
    DataType,
    Function,
    FunctionType,
    RRFRanker,
)
```

Modify `ensure_collection`:

```python
schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
schema.add_function(
    Function(
        name="bm25",
        function_type=FunctionType.BM25,
        input_field_names=["text"],
        output_field_names=["sparse"],
    )
)
index_params.add_index(
    field_name="sparse",
    index_type="SPARSE_INVERTED_INDEX",
    metric_type="BM25",
)
```

Add helpers:

```python
def _category_expr(category: str | None) -> str:
    if not category:
        return ""
    safe = category.replace('"', '\\"')
    return f'category == "{safe}"'


def _normalize_hit(hit: dict) -> dict:
    entity = hit["entity"]
    raw_questions = entity.get("questions", "")
    try:
        questions = json.loads(raw_questions or "[]")
    except (TypeError, json.JSONDecodeError):
        questions = []
    return {
        "chunk_id": entity["chunk_id"],
        "category": entity["category"],
        "question": questions[0] if questions else entity["category"],
        "questions": questions,
        "text": entity["text"],
        "score": float(hit["distance"]),
    }
```

Add `search_bm25` and `hybrid_search`:

```python
async def search_bm25(
    self,
    query: str,
    top_k: int = 10,
    category: str | None = None,
) -> list[dict]:
    await self.ensure_collection()
    result = await self.client.search(
        collection_name=self.collection,
        data=[query],
        anns_field="sparse",
        search_params={"metric_type": "BM25"},
        limit=top_k,
        filter=_category_expr(category),
        output_fields=["chunk_id", "category", "questions", "text"],
    )
    return [_normalize_hit(hit) for hit in result[0]]


async def hybrid_search(
    self,
    vector: list[float],
    query: str,
    top_k: int = 10,
    category: str | None = None,
) -> list[dict]:
    await self.ensure_collection()
    expr = _category_expr(category)
    dense = AnnSearchRequest(
        data=[vector],
        anns_field="vector",
        param={"metric_type": "COSINE"},
        limit=top_k,
        filter=expr,
    )
    sparse = AnnSearchRequest(
        data=[query],
        anns_field="sparse",
        param={"metric_type": "BM25"},
        limit=top_k,
        filter=expr,
    )
    result = await self.client.hybrid_search(
        collection_name=self.collection,
        reqs=[dense, sparse],
        ranker=RRFRanker(k=60),
        limit=top_k,
        output_fields=["chunk_id", "category", "questions", "text"],
    )
    return [_normalize_hit(hit) for hit in result[0]]
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run pytest tests/test_vector_store.py -v
```

Expected: PASS.

- [ ] **Step 5: Add hybrid rebuild script**

Create `scripts/rebuild_hybrid_kb.py`:

```python
"""Rebuild the hybrid-enabled Milvus knowledge collection from MySQL chunks."""

import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.services.knowledge_pipeline import build_knowledge_base


async def main() -> int:
    docs_dir = pathlib.Path("knowledge_docs")
    result = await build_knowledge_base(docs_dir, rebuild=True)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
```

Run once against the local Milvus server:

```bash
uv run python scripts/rebuild_hybrid_kb.py
```

Expected: `files` and `inserted` are non-zero, `failed` is `0`.

- [ ] **Step 6: Commit**

```bash
git add app/config.py app/services/vector_store.py scripts/rebuild_hybrid_kb.py tests/test_vector_store.py
git commit -m "feat: add Milvus BM25 and hybrid search"
```

---

### Task 2: Query Understanding

**Files:**
- Create: `app/services/query_understanding.py`
- Test: `tests/test_query_understanding.py`

**Interfaces:**
- Consumes: `get_chat_model` from `app.core.llm`, `settings`.
- Produces:
  - `SYNONYMS: dict[str, list[str]]`
  - `async def normalize_query(query: str) -> dict`
  - `def expand_synonyms(normalized_query: str) -> str`
  - `def build_retrieval_query(normalized: dict) -> str`

- [ ] **Step 1: Write failing test**

```python
import pytest

from app.services import query_understanding


def test_expand_synonyms_appends_only_configured_terms():
    assert "到账" in query_understanding.expand_synonyms("退款什么时候到账")
    assert "到款" in query_understanding.expand_synonyms("退款什么时候到账")
    assert "退货" not in query_understanding.expand_synonyms("订单发货了吗")


def test_build_retrieval_query_uses_normalized_and_synonyms():
    result = query_understanding.build_retrieval_query(
        {"normalized_query": "退款到账时间", "category": None}
    )
    assert result == "退款到账时间 到账 到款"


@pytest.mark.asyncio
async def test_normalize_query_returns_structured_json(monkeypatch):
    class FakeModel:
        async def ainvoke(self, messages):
            return type(
                "Msg",
                (),
                {"content": '{"normalized_query":"退款到账时间","category":null,"is_knowledge_question":true}'},
            )()

    monkeypatch.setattr(query_understanding, "get_chat_model", lambda: FakeModel())

    result = await query_understanding.normalize_query("钱什么时候打回来")

    assert result["normalized_query"] == "退款到账时间"
    assert result["is_knowledge_question"] is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_query_understanding.py -v
```

Expected: FAIL with module not found.

- [ ] **Step 3: Implement**

Create `app/services/query_understanding.py`:

```python
import json
import re

from langchain_core.messages import HumanMessage

from app.core.llm import get_chat_model


SYNONYMS = {
    "到账": ["到款", "打款", "到账时间"],
    "退款": ["退钱", "退货退款"],
    "发货": ["发出", "寄出"],
}


def _strip_code_fence(value: str) -> str:
    value = value.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    return value.strip()


async def normalize_query(query: str) -> dict:
    prompt = f"""把用户口语改写成标准检索问法，只输出 JSON。
示例：
用户：钱什么时候打回来
输出：{{"normalized_query":"退款到账时间","category":null,"is_knowledge_question":true}}

用户：{query}
输出："""
    model = get_chat_model(streaming=False)
    response = await model.ainvoke([HumanMessage(content=prompt)])
    content = getattr(response, "content", "") or ""
    try:
        return json.loads(_strip_code_fence(content))
    except json.JSONDecodeError:
        return {"normalized_query": query, "category": None, "is_knowledge_question": True}


def expand_synonyms(normalized_query: str) -> str:
    terms = [normalized_query]
    for key, synonyms in SYNONYMS.items():
        if key in normalized_query:
            terms.extend(synonyms)
    return " ".join(dict.fromkeys(terms))


def build_retrieval_query(normalized: dict) -> str:
    return expand_synonyms(normalized.get("normalized_query") or "")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_query_understanding.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/services/query_understanding.py tests/test_query_understanding.py
git commit -m "feat: add query normalization and synonym expansion"
```

---

### Task 3: SiliconFlow Reranker Client

**Files:**
- Create: `app/services/reranker.py`
- Test: `tests/test_reranker.py`

**Interfaces:**
- Consumes: `settings.rerank_base_url`, `settings.rerank_model`, `settings.rerank_api_key`.
- Produces:
  - `async def rerank(query: str, documents: list[str], top_k: int | None = None) -> list[dict]`
  - Each result has `index`, `score`, `document`.

- [ ] **Step 1: Write failing test**

```python
import httpx
import pytest

from app.services import reranker


class FakeTransport(httpx.AsyncBaseTransport):
    def __init__(self, handler):
        self.handler = handler

    async def handle_async_request(self, request):
        return self.handler(request)


@pytest.mark.asyncio
async def test_rerank_returns_sorted_documents(monkeypatch):
    async def handler(request):
        payload = json.loads(request.content)
        assert payload["model"] == "BAAI/bge-reranker-v2-m3"
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 0, "relevance_score": 0.2},
                    {"index": 1, "relevance_score": 0.9},
                ]
            },
        )

    client = httpx.AsyncClient(transport=FakeTransport(handler))
    monkeypatch.setattr(reranker, "_get_client", lambda: client)

    result = await reranker.rerank("智能猫砂盆", ["退款政策", "猫砂盆清理"])

    assert [item["index"] for item in result] == [1, 0]
    assert result[0]["score"] == 0.9
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_reranker.py -v
```

Expected: FAIL with module not found.

- [ ] **Step 3: Implement**

Create `app/services/reranker.py`:

```python
import httpx

from app.config import settings


_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=30)
    return _client


async def rerank(
    query: str,
    documents: list[str],
    top_k: int | None = None,
) -> list[dict]:
    if not documents:
        return []
    payload = {
        "model": settings.rerank_model,
        "query": query,
        "documents": documents,
        "top_n": top_k or min(settings.rerank_top_k, len(documents)),
    }
    response = await _get_client().post(
        f"{settings.rerank_base_url.rstrip('/')}/rerank",
        headers={"Authorization": f"Bearer {settings.rerank_api_key}"},
        json=payload,
    )
    response.raise_for_status()
    rows = response.json().get("results", [])
    results = [
        {
            "index": int(row["index"]),
            "score": float(row.get("relevance_score") or row.get("score") or 0),
            "document": documents[int(row["index"])],
        }
        for row in rows
    ]
    return sorted(results, key=lambda item: item["score"], reverse=True)
```

Add a tiny test helper import at the top of `tests/test_reranker.py`:

```python
import json
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_reranker.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/services/reranker.py tests/test_reranker.py
git commit -m "feat: add SiliconFlow reranker client"
```

---

### Task 4: Hybrid Retriever and Evidence Assembly

**Files:**
- Create: `app/services/hybrid_retriever.py`
- Test: `tests/test_hybrid_retriever.py`

**Interfaces:**
- Consumes: `VectorStore`, `embed_query`, `normalize_query`, `build_retrieval_query`, `rerank`, `rerank_score_threshold`.
- Produces:
  - `async def retrieve(query: str, category: str | None = None, strategy: str = "hybrid_rerank", top_k: int = 10) -> list[dict]`
  - `def assemble_evidence(hits: list[dict]) -> list[dict]`
  - Each evidence item has `citation_id`, `chunk_id`, `category`, `text`, `section_path`, `score`.

- [ ] **Step 1: Write failing test**

```python
import pytest

from app.services import hybrid_retriever


def test_assemble_evidence_puts_strongest_first_and_second_last():
    hits = [
        {"chunk_id": "a", "text": "a", "score": 0.9, "category": "x"},
        {"chunk_id": "b", "text": "b", "score": 0.8, "category": "x"},
        {"chunk_id": "c", "text": "c", "score": 0.7, "category": "x"},
        {"chunk_id": "d", "text": "d", "score": 0.6, "category": "x"},
    ]

    evidence = hybrid_retriever.assemble_evidence(hits)

    assert [item["chunk_id"] for item in evidence] == ["a", "c", "d", "b"]
    assert [item["citation_id"] for item in evidence] == [1, 2, 3, 4]
```

Add an async retrieval test that monkeypatches dependencies:

```python
@pytest.mark.asyncio
async def test_retrieve_runs_hybrid_rerank_path(monkeypatch):
    async def fake_normalize(query):
        return {"normalized_query": "智能猫砂盆清理", "category": None}

    def fake_build(normalized):
        return "智能猫砂盆清理"

    async def fake_embed(query):
        return [0.1, 0.2]

    class FakeStore:
        async def hybrid_search(self, vector, query, top_k, category):
            return [
                {"chunk_id": "a", "text": "a", "score": 0.9, "category": "商品FAQ"},
                {"chunk_id": "b", "text": "b", "score": 0.8, "category": "商品FAQ"},
            ]

    async def fake_rerank(query, documents, top_k):
        return [
            {"index": 1, "score": 0.95, "document": "b"},
            {"index": 0, "score": 0.85, "document": "a"},
        ]

    monkeypatch.setattr(hybrid_retriever, "normalize_query", fake_normalize)
    monkeypatch.setattr(hybrid_retriever, "build_retrieval_query", fake_build)
    monkeypatch.setattr(hybrid_retriever, "embed_query", fake_embed)
    monkeypatch.setattr(hybrid_retriever, "VectorStore", lambda: FakeStore())
    monkeypatch.setattr(hybrid_retriever, "rerank", fake_rerank)

    result = await hybrid_retriever.retrieve("猫砂盆", strategy="hybrid_rerank")

    assert result[0]["chunk_id"] == "b"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_hybrid_retriever.py -v
```

Expected: FAIL with module not found.

- [ ] **Step 3: Implement**

Create `app/services/hybrid_retriever.py`:

```python
from app.config import settings
from app.services.embedding import embed_query
from app.services.query_understanding import build_retrieval_query, normalize_query
from app.services.reranker import rerank
from app.services.vector_store import VectorStore


async def retrieve(
    query: str,
    category: str | None = None,
    strategy: str = "hybrid_rerank",
    top_k: int = 10,
) -> list[dict]:
    normalized = await normalize_query(query)
    retrieval_query = build_retrieval_query(normalized)
    category = category or normalized.get("category")
    store = VectorStore()
    try:
        if strategy == "dense":
            vector = await embed_query(retrieval_query)
            hits = await store.search(vector, top_k=top_k)
        elif strategy == "bm25":
            hits = await store.search_bm25(retrieval_query, top_k=top_k, category=category)
        else:
            vector = await embed_query(retrieval_query)
            hits = await store.hybrid_search(
                vector,
                retrieval_query,
                top_k=top_k,
                category=category,
            )
            if strategy == "hybrid_rerank" and hits:
                ranked = await rerank(
                    retrieval_query,
                    [hit["text"] for hit in hits],
                    top_k=top_k,
                )
                hits = [hits[item["index"]] | {"score": item["score"]} for item in ranked]
                hits = [hit for hit in hits if hit["score"] >= settings.rerank_score_threshold]
    finally:
        await store.close()
    return hits


def assemble_evidence(hits: list[dict]) -> list[dict]:
    if not hits:
        return []
    ordered = sorted(hits, key=lambda item: item.get("score", 0), reverse=True)
    if len(ordered) >= 2:
        ordered = [ordered[0], *ordered[2:], ordered[1]]
    return [
        {
            "citation_id": index + 1,
            "chunk_id": hit["chunk_id"],
            "category": hit.get("category", ""),
            "text": hit.get("text", ""),
            "section_path": hit.get("section_path", ""),
            "score": float(hit.get("score", 0)),
        }
        for index, hit in enumerate(ordered)
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_hybrid_retriever.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/services/hybrid_retriever.py tests/test_hybrid_retriever.py
git commit -m "feat: add hybrid retriever and evidence ordering"
```

---

### Task 5: Low Confidence Model and RAG Service

**Files:**
- Modify: `app/db/models.py`
- Modify: `app/db/repository.py`
- Create: `app/services/rag_service.py`
- Modify: `app/core/tools.py`
- Test: `tests/test_rag_service.py`
- Modify: `tests/test_tools.py`

**Interfaces:**
- Consumes: `retrieve`, `assemble_evidence`, `get_chat_model`, `AsyncSessionLocal`.
- Produces:
  - `class LowConfidenceQuestion(Base)`
  - `async def insert_low_confidence_question(...) -> None`
  - `async def answer_knowledge_question(keyword: str, conversation_id: str | None = None, category: str | None = None) -> dict`
  - `query_faq.ainvoke` returns JSON string.

- [ ] **Step 1: Write failing tests**

Create `tests/test_rag_service.py`:

```python
import json

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.core import tools
from app.db import models, session as db_session
from app.db.base import Base
from app.services import rag_service


async def make_session(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool, future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(db_session, "AsyncSessionLocal", session)
    monkeypatch.setattr(tools, "AsyncSessionLocal", session)
    return engine, session


class FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)

    async def ainvoke(self, messages):
        content = next(self.responses)
        return type("Msg", (), {"content": content})()


async def test_answer_knowledge_question_refuses_and_inserts_pool(monkeypatch):
    engine, session = await make_session(monkeypatch)
    try:
        async def fake_retrieve(*args, **kwargs):
            return []

        model = FakeModel(['{"sufficient": false, "reason": "知识库没有相关证据"}'])
        monkeypatch.setattr(rag_service, "retrieve", fake_retrieve)
        monkeypatch.setattr(rag_service, "get_chat_model", lambda: model)

        result = await rag_service.answer_knowledge_question("明天股市", conversation_id="s-rag")

        assert result["matched"] is False
        assert result["sufficient"] is False
        async with session() as db:
            rows = list((await db.execute(select(models.LowConfidenceQuestion))).scalars())
            assert len(rows) == 1
            assert rows[0].question == "明天股市"
            assert rows[0].source_conversation_id == "s-rag"
    finally:
        await engine.dispose()
```

Modify `tests/test_tools.py` to add:

```python
async def test_query_faq_passes_conversation_id(monkeypatch):
    engine = await setup_testing_session(monkeypatch)
    try:
        async def fake_answer(keyword, conversation_id=None, category=None):
            return {"matched": False, "sufficient": False, "refusal": "抱歉"}

        monkeypatch.setattr(tools, "answer_knowledge_question", fake_answer)
        result = await tools.query_faq.ainvoke(
            {"keyword": "明天股市", "conversation_id": "s-tool"}
        )
        assert '"matched": false' in result
    finally:
        await engine.dispose()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_rag_service.py tests/test_tools.py -v
```

Expected: FAIL with missing model/service.

- [ ] **Step 3: Implement model and repository**

In `app/db/models.py`, add:

```python
class LowConfidenceQuestion(Base):
    __tablename__ = "low_confidence_questions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(Text)
    source_conversation_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    entry_point: Mapped[str] = mapped_column(String(64), default="query_faq_self_check")
    reason: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
```

In `app/db/repository.py`, add:

```python
async def insert_low_confidence_question(
    question: str,
    reason: str,
    source_conversation_id: str | None,
) -> None:
    async with AsyncSessionLocal() as db:
        db.add(
            LowConfidenceQuestion(
                question=question,
                source_conversation_id=source_conversation_id,
                entry_point="query_faq_self_check",
                reason=reason,
            )
        )
        await db.commit()
```

Add a path lookup used by RAG service:

```python
async def load_chunk_section_paths(chunk_ids: list[str]) -> dict[str, str]:
    if not chunk_ids:
        return {}
    async with AsyncSessionLocal() as db:
        rows = await db.scalars(
            select(KnowledgeChunk).where(KnowledgeChunk.chunk_id.in_(chunk_ids))
        )
        return {row.chunk_id: row.section_path or "" for row in rows}
```

Import `LowConfidenceQuestion` and `KnowledgeChunk` at the top of `app/db/repository.py`.

- [ ] **Step 4: Implement RAG service**

Create `app/services/rag_service.py`:

```python
import json

from langchain_core.messages import HumanMessage

from app.core.llm import get_chat_model
from app.db.repository import insert_low_confidence_question, load_chunk_section_paths
from app.services.hybrid_retriever import assemble_evidence, retrieve


async def answer_knowledge_question(
    keyword: str,
    conversation_id: str | None = None,
    category: str | None = None,
) -> dict:
    hits = await retrieve(keyword, category=category, strategy="hybrid_rerank", top_k=10)
    if not hits:
        reason = "未召回足够证据"
        await insert_low_confidence_question(keyword, reason, conversation_id)
        return {"matched": False, "sufficient": False, "reason": reason, "refusal": "抱歉，我暂时无法依据现有知识回答这个问题。"}

    section_paths = await load_chunk_section_paths([hit["chunk_id"] for hit in hits])
    for hit in hits:
        hit["section_path"] = section_paths.get(hit["chunk_id"], "")
    evidence = assemble_evidence(hits)
    check_prompt = build_sufficiency_prompt(keyword, evidence)
    model = get_chat_model(streaming=False)
    response = await model.ainvoke([HumanMessage(content=check_prompt)])
    try:
        check = json.loads(_extract_json(getattr(response, "content", "") or ""))
    except json.JSONDecodeError:
        check = {"sufficient": False, "reason": "证据判断失败"}

    if not check.get("sufficient"):
        reason = check.get("reason") or "证据不足"
        await insert_low_confidence_question(keyword, reason, conversation_id)
        return {"matched": True, "sufficient": False, "reason": reason, "refusal": "抱歉，现有知识不足以可靠回答这个问题。"}

    return {
        "matched": True,
        "sufficient": True,
        "evidence": evidence,
        "context": build_evidence_text(evidence),
    }


def build_sufficiency_prompt(keyword: str, evidence: list[dict]) -> str:
    lines = [f"问题：{keyword}", "证据："]
    for item in evidence:
        lines.append(f"[{item['citation_id']}] {item['text']}")
    lines.append('只输出 JSON：{"sufficient": true/false, "reason": "..."}')
    return "\n".join(lines)


def build_evidence_text(evidence: list[dict]) -> str:
    return "\n".join(f"[{item['citation_id']}] {item['text']}" for item in evidence)


def _extract_json(value: str) -> str:
    value = value.strip()
    if value.startswith("```"):
        value = value.strip("`")
        if value.startswith("json"):
            value = value[4:]
    return value
```

- [ ] **Step 5: Modify query_faq tool**

In `app/core/tools.py`, change import and function:

```python
from app.services.rag_service import answer_knowledge_question


@tool(parse_docstring=True)
async def query_faq(keyword: str, category: str | None = None, conversation_id: str | None = None) -> str:
    """从知识库查询并生成带引用的回答依据。

    Args:
        keyword: 用户问题。
        category: 可选品类。
        conversation_id: 当前会话 ID，可选。

    Returns:
        RAG 结果 JSON；证据不足时返回拒答信息。
    """
    result = await answer_knowledge_question(keyword, conversation_id=conversation_id, category=category)
    return json.dumps(result, ensure_ascii=False)
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
uv run pytest tests/test_rag_service.py tests/test_tools.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add app/db/models.py app/db/repository.py app/services/rag_service.py app/core/tools.py tests/test_rag_service.py tests/test_tools.py
git commit -m "feat: add RAG self-check and low confidence pool"
```

---

### Task 6: Chat SSE Citations and Generation Prompts

**Files:**
- Modify: `app/core/prompts.py`
- Modify: `app/api/chat.py`
- Test: `tests/test_chat_api.py`

**Interfaces:**
- Consumes: existing chat SSE event stream and `answer_knowledge_question`.
- Produces:
  - `citations` SSE event before `[DONE]`.
  - refusal path emits a normal delta and empty citations.

- [ ] **Step 1: Write failing test**

Update `tests/test_chat_api.py` `make_client` to support a custom tool execution by monkeypatching `chat_api.execute_tool`. Add:

```python
async def test_chat_emits_citations_after_query_faq(monkeypatch):
    from app.core.tool_runner import ToolExecutionResult
    from langchain_core.messages import AIMessage

    async def fake_execute(tool_name, tool_args):
        return ToolExecutionResult(
            tool_name=tool_name,
            tool_args=tool_args,
            result=json.dumps({
                "matched": True,
                "sufficient": True,
                "evidence": [
                    {
                        "citation_id": 1,
                        "chunk_id": "chunk-1",
                        "section_path": "商品FAQ > 智能猫砂盆",
                        "text": "断电后清理。",
                    }
                ],
            }),
            success=True,
        )

    class ToolCallModel(ToolBindableFakeChatModel):
        def __init__(self, final_text):
            super().__init__(responses=[final_text])
            self.final_text = final_text

        async def ainvoke(self, messages, **kwargs):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "query_faq",
                        "args": {"keyword": "怎么退款"},
                        "id": "call-cite",
                    }
                ],
            )

        async def astream(self, messages, **kwargs):
            class Chunk:
                content = self.final_text

            yield Chunk()

    monkeypatch.setattr(chat_api, "execute_tool", fake_execute)
    fake = ToolCallModel("退款政策是七天无理由。")
    monkeypatch.setattr(chat_api, "get_model", lambda: fake)
    client = TestClient(app)
    with client:
        with client.stream(
            "POST",
            "/api/chat",
            json={"session_id": "s-cite", "message": "怎么退款"},
        ) as resp:
            frames = list(resp.iter_lines())

    citation_payload = next(json.loads(line[6:]) for line in frames if line.startswith("data: ") and '"type": "citations"' in line)
    assert citation_payload["citations"][0]["chunk_id"] == "chunk-1"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_chat_api.py::test_chat_emits_citations_after_query_faq -v
```

Expected: FAIL because no citations event.

- [ ] **Step 3: Modify prompts**

In `app/core/prompts.py`, extend `CUSTOMER_SERVICE_SYSTEM`:

```python
- 基于知识库回答时，必须使用证据中的 [1]、[2] 引用编号，且只能引用本次提供的证据。
- 如果 query_faq 返回 sufficient=false 或 refusal，直接使用拒答文案，不得补写事实。
- 不承诺到账时间、物流送达时间、退款到账时间、库存补货时间、赔偿金额。
```

- [ ] **Step 4: Modify chat.py**

Add `rag_citations: list[dict] = []` before event stream. After tool execution success, if `tool_name == "query_faq"`, parse result:

```python
if tool_name == "query_faq" and execution.success:
    try:
        parsed = json.loads(tool_content)
        rag_citations = [
            {
                "id": item["citation_id"],
                "chunk_id": item["chunk_id"],
                "section_path": item.get("section_path", ""),
                "text": item.get("text", ""),
            }
            for item in parsed.get("evidence", [])
        ]
        if parsed.get("sufficient") is False and parsed.get("refusal"):
            tool_content = json.dumps({"refusal": parsed["refusal"]}, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        logger.exception("解析 query_faq 结果失败 session_id=%s", req.session_id)
```

After the final model stream succeeds and before `yield "data: [DONE]\n\n"`, emit:

```python
yield f"data: {json.dumps({'type': 'citations', 'citations': rag_citations}, ensure_ascii=False)}\n\n"
yield "data: [DONE]\n\n"
```

If `tool_content` contains a refusal and final model still generates, leave the prompt constraint to produce refusal; the tool result now contains only `{"refusal": "..."}`.

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_chat_api.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/core/prompts.py app/api/chat.py tests/test_chat_api.py
git commit -m "feat: emit citations and refusal metadata through chat SSE"
```

---

### Task 7: Evaluation Set and Four-Strategy Report

**Files:**
- Create: `eval_data/ch04_eval_set.json`
- Create: `scripts/eval_ch04.py`
- Test: `tests/test_eval_ch04.py`

**Interfaces:**
- Consumes: `retrieve` from `app.services.hybrid_retriever`.
- Produces: console report plus `reports/ch04_eval_report.json` and `.md`.

- [ ] **Step 1: Write metric tests**

Create `tests/test_eval_ch04.py`:

```python
from scripts import eval_ch04


def test_recall_at_k():
    assert eval_ch04.recall_at_k(["a", "b"], ["a"], k=3) == 1.0
    assert eval_ch04.recall_at_k(["x", "y"], ["a"], k=3) == 0.0


def test_mrr():
    assert eval_ch04.mrr(["a", "b"], ["a"]) == 1.0
    assert eval_ch04.mrr(["a", "b"], ["b"]) == 0.5
```

Add a minimal sample file:

`eval_data/ch04_eval_set.json`

```json
[
  {
    "id": "sample-refund",
    "query": "钱什么时候退回来",
    "type": "colloquial",
    "difficulty": "easy",
    "category": "售后",
    "ground_truth_chunk_ids": ["seed-refund-1"],
    "ground_truth_answer": "退款到账时间以平台售后规则为准。",
    "expect_refusal": false
  },
  {
    "id": "sample-out-of-kb",
    "query": "明天股市会涨吗",
    "type": "out_of_kb",
    "difficulty": "easy",
    "category": null,
    "ground_truth_chunk_ids": [],
    "ground_truth_answer": "",
    "expect_refusal": true
  }
]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_eval_ch04.py -v
```

Expected: FAIL with module not found.

- [ ] **Step 3: Implement eval script**

Create `scripts/eval_ch04.py` with:

```python
def recall_at_k(retrieved: list[str], ground_truth: list[str], k: int) -> float:
    if not ground_truth:
        return 1.0 if not retrieved else 0.0
    pool = set(ground_truth)
    hits = set(retrieved[:k]) & pool
    return len(hits) / len(pool)


def mrr(retrieved: list[str], ground_truth: list[str]) -> float:
    for index, item in enumerate(retrieved, start=1):
        if item in set(ground_truth):
            return 1.0 / index
    return 0.0
```

`main()` loads the JSON file, calls `retrieve(item["query"], category=item.get("category"), strategy=strategy)` for each of `dense`, `bm25`, `hybrid`, and `hybrid_rerank`, calculates `Recall@3/5/10` and `MRR`, then writes JSON and Markdown reports under `reports/`.

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_eval_ch04.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add eval_data/ch04_eval_set.json scripts/eval_ch04.py tests/test_eval_ch04.py
git commit -m "feat: add ch04 evaluation harness"
```

---

### Task 8: Frontend Citation and Feedback UX

**Files:**
- Modify: `D:\std_selftest_frontend\src\main.js`
- Modify: `D:\std_selftest_frontend\src\styles.css`

**Interfaces:**
- Consumes: SSE `citations` event and existing assistant bubble DOM.
- Produces: clickable `[n]` controls, source drawer, one-shot local feedback group.

This task is frontend Vibe Coding and does not use TDD. Implement directly, then verify in the browser with Playwright screenshots or the existing Vite dev server.

- [ ] **Step 1: Replace text-only bubble rendering with citation-aware rendering**

In `streamChat`, keep `bubble.textContent` for plain delta during streaming. After citations arrive, call:

```javascript
function renderAssistantContent(bubble, text, citations) {
  bubble.replaceChildren();
  const nodes = [];
  const pattern = /(\[\d+\])/g;
  let last = 0;
  for (const match of text.matchAll(pattern)) {
    if (match.index > last) nodes.push(document.createTextNode(text.slice(last, match.index)));
    const citationId = Number(match[0].slice(1, -1));
    const citation = citations.find((item) => item.id === citationId);
    const button = document.createElement("button");
    button.className = "citation-chip";
    button.textContent = match[0];
    if (citation) {
      button.addEventListener("click", () => openCitation(citation));
    } else {
      button.disabled = true;
    }
    nodes.push(button);
    last = match.index + match[0].length;
  }
  if (last < text.length) nodes.push(document.createTextNode(text.slice(last)));
  bubble.append(...nodes);
}
```

Store `state.lastCitations = []` while processing SSE.

- [ ] **Step 2: Add source drawer**

Add a hidden `<div id="citation-drawer">` to `index.html`, or create it dynamically in `main.js`:

```javascript
const drawer = document.createElement("aside");
drawer.id = "citation-drawer";
drawer.hidden = true;
document.body.appendChild(drawer);

function openCitation(citation) {
  drawer.innerHTML = `
    <button class="citation-close" aria-label="关闭来源">×</button>
    <div class="citation-path">${escapeHtml(citation.section_path || citation.chunk_id)}</div>
    <div class="citation-text">${escapeHtml(citation.text || "")}</div>`;
  drawer.hidden = false;
}
```

Implement a small `escapeHtml` helper.

- [ ] **Step 3: Add feedback group to each assistant message**

After `citations` is processed, append:

```javascript
function addFeedback(bubble) {
  const stack = bubble.closest(".message-stack");
  const group = document.createElement("div");
  group.className = "feedback";
  const up = document.createElement("button");
  up.className = "feedback-btn";
  up.textContent = "👍";
  const down = document.createElement("button");
  down.className = "feedback-btn";
  down.textContent = "👎";
  const label = document.createElement("span");
  label.className = "feedback-label";
  label.textContent = "";
  const lock = () => {
    group.querySelectorAll("button").forEach((item) => (item.disabled = true));
    label.textContent = "已反馈";
  };
  up.addEventListener("click", () => { up.classList.add("is-active"); lock(); });
  down.addEventListener("click", () => { down.classList.add("is-active"); lock(); });
  group.append(up, down, label);
  stack.appendChild(group);
}
```

- [ ] **Step 4: Add CSS**

Add styles for `.citation-chip`, `#citation-drawer`, `.feedback`, `.feedback-btn`, `.feedback-btn.is-active`, and `.feedback-label`. Keep responsive constraints.

- [ ] **Step 5: Verify with browser**

Start backend and frontend, then send a knowledge question that triggers citations. Confirm:

- Citation chips appear.
- Clicking a chip opens the source text and section path.
- Feedback buttons lock after one click.

Do not add a backend feedback API.

---

## Self-Review

- Spec coverage: each requirement maps to Tasks 1-8.
- Placeholder scan: no `TBD`, `TODO`, or vague implementation text remains.
- Type consistency: `retrieve`, `assemble_evidence`, `answer_knowledge_question`, and SSE `citations` names match across tasks.
