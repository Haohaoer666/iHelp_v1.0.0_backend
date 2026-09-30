import json

import httpx
import pytest

from app.services import reranker


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

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(reranker, "_get_client", lambda: client)

    result = await reranker.rerank("智能猫砂盆", ["退款政策", "猫砂盆清理"])

    assert [item["index"] for item in result] == [1, 0]
    assert result[0]["score"] == 0.9
