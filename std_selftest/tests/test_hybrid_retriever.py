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

        async def close(self):
            return None

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


@pytest.mark.asyncio
async def test_retrieve_dense_passes_category_filter(monkeypatch):
    async def fake_normalize(query):
        return {"normalized_query": "显示器亮度", "category": None}

    async def fake_embed(query):
        return [0.1, 0.2]

    class FakeStore:
        def __init__(self):
            self.calls = []

        async def search(self, vector, top_k=None, category=None):
            self.calls.append((vector, top_k, category))
            return [{"chunk_id": "a", "text": "a", "distance": 0.9}]

        async def close(self):
            return None

    store = FakeStore()
    monkeypatch.setattr(hybrid_retriever, "normalize_query", fake_normalize)
    monkeypatch.setattr(
        hybrid_retriever,
        "build_retrieval_query",
        lambda normalized: "显示器亮度",
    )
    monkeypatch.setattr(hybrid_retriever, "embed_query", fake_embed)
    monkeypatch.setattr(hybrid_retriever, "VectorStore", lambda: store)

    await hybrid_retriever.retrieve(
        "显示器亮度",
        category="知识库：iHao V1 电竞显示器",
        strategy="dense",
    )

    assert store.calls[0][2] == "知识库：iHao V1 电竞显示器"
