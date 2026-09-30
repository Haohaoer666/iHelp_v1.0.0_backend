import pytest
from langchain_core.messages import AIMessage

from app.services import rag_service
from app.services.retrieval_queries import expand_retrieval_queries


class StaticModel:
    def __init__(self, content: str) -> None:
        self.content = content
        self.messages = []

    async def ainvoke(self, messages, **kwargs):
        self.messages = messages
        return AIMessage(content=self.content)


async def test_expansion_deduplicates_and_preserves_original_first():
    model = StaticModel(
        '{"queries":["七天无理由退货","签收后七天退货","七天无理由退货"]}'
    )

    result = await expand_retrieval_queries("这个能退吗", model)

    assert result.queries == ["这个能退吗", "七天无理由退货", "签收后七天退货"]
    assert result.source == "model"


async def test_expansion_rejects_extra_fields():
    model = StaticModel('{"queries":["退款"],"reason":"x"}')

    result = await expand_retrieval_queries("退款", model)

    assert result.queries == ["退款"]
    assert result.source == "fallback"


async def test_expansion_rejects_non_string_queries():
    model = StaticModel('{"queries":["退款", 123]}')

    result = await expand_retrieval_queries("退款", model)

    assert result.queries == ["退款"]
    assert result.source == "fallback"


async def test_expansion_limits_query_count():
    model = StaticModel('{"queries":["a","b","c","d","e"]}')

    result = await expand_retrieval_queries("退款", model, max_queries=3)

    assert result.queries == ["退款", "a", "b"]


async def test_multi_query_retrieval_merges_duplicate_chunks(monkeypatch):
    async def fake_retrieve(query, category=None, strategy="hybrid_rerank", top_k=10):
        assert strategy == "hybrid_rerank"
        if query == "原问题":
            return [{"chunk_id": "c1", "text": "七天退货", "score": 0.5}]
        if query == "扩写一":
            return [
                {"chunk_id": "c1", "text": "七天退货", "score": 0.9},
                {"chunk_id": "c2", "text": "商品完好", "score": 0.8},
            ]
        return [{"chunk_id": "c3", "text": "运费规则", "score": 0.7}]

    async def fake_paths(chunk_ids):
        return {chunk_id: f"路径/{chunk_id}" for chunk_id in chunk_ids}

    monkeypatch.setattr(rag_service, "retrieve", fake_retrieve)
    monkeypatch.setattr(rag_service, "load_chunk_section_paths", fake_paths)

    result = await rag_service.retrieve_multi_query(["原问题", "扩写一", "扩写二"])

    assert result["retrieval_query_count"] == 3
    assert result["failed_queries"] == []
    evidence = result["evidence"]
    assert evidence[0]["chunk_id"] == "c1"
    assert {item["chunk_id"] for item in evidence} == {"c1", "c2", "c3"}
    assert evidence[0]["score"] == 0.9
    assert evidence[0]["section_path"] == "路径/c1"


async def test_multi_query_retrieval_continues_after_one_failure(monkeypatch):
    async def fake_retrieve(query, category=None, strategy="hybrid_rerank", top_k=10):
        if query == "坏查询":
            raise RuntimeError("milvus down")
        return [{"chunk_id": "c1", "text": "有效证据", "score": 0.8}]

    async def fake_paths(chunk_ids):
        return {}

    monkeypatch.setattr(rag_service, "retrieve", fake_retrieve)
    monkeypatch.setattr(rag_service, "load_chunk_section_paths", fake_paths)

    result = await rag_service.retrieve_multi_query(["坏查询", "好查询"])

    assert result["retrieval_query_count"] == 2
    assert result["failed_queries"] == ["坏查询"]
    assert result["evidence"][0]["text"] == "有效证据"


async def test_policy_gate_accepts_relevant_policy_without_deciding_order(monkeypatch):
    monkeypatch.setattr(rag_service.settings, "knowledge_score_threshold", 0.35)

    result = await rag_service.assess_policy_evidence(
        "用户想确认这个订单是否可以退款",
        [{"citation_id": 1, "text": "签收后七天内商品完好可退。", "score": 0.91}],
        conversation_id="s-policy",
    )

    assert result["sufficient"] is True
    assert "context" in result
