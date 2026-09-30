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
    assert result == "退款到账时间 到款 打款 到账时间 退钱 退货退款"


@pytest.mark.asyncio
async def test_normalize_query_returns_structured_json(monkeypatch):
    class FakeModel:
        async def ainvoke(self, messages):
            return type(
                "Msg",
                (),
                {
                    "content": (
                        '{"normalized_query":"退款到账时间",'
                        '"category":null,"is_knowledge_question":true}'
                    )
                },
            )()

    monkeypatch.setattr(query_understanding, "get_chat_model", lambda **kwargs: FakeModel())

    result = await query_understanding.normalize_query("钱什么时候打回来")

    assert result["normalized_query"] == "退款到账时间"
    assert result["is_knowledge_question"] is True
