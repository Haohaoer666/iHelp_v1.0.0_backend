from langchain_core.messages import AIMessage, HumanMessage

from app.services.conversation_understanding import understand_query


class StaticModel:
    def __init__(self, content: str) -> None:
        self.content = content
        self.messages = []
        self.calls = 0

    async def ainvoke(self, messages, **kwargs):
        self.calls += 1
        self.messages = messages
        return AIMessage(content=self.content)


async def test_complete_question_passes_through():
    model = StaticModel('{"resolved_query":"退货政策是什么"}')

    result = await understand_query([], "退货政策是什么", model)

    assert result.resolved_query == "退货政策是什么"
    assert result.changed is False
    assert result.source == "passthrough"
    assert model.calls == 0


async def test_pronoun_uses_conversation_history():
    history = [
        HumanMessage("订单 1001 的物流到哪了"),
        AIMessage("订单 1001 正在运输中。"),
    ]
    model = StaticModel('{"resolved_query":"订单 1001 是否可以退款"}')

    result = await understand_query(history, "它能退吗", model)

    assert result.resolved_query == "订单 1001 是否可以退款"
    assert result.changed is True
    assert "订单 1001 的物流到哪了" in model.messages[0].content
    assert "它能退吗" in model.messages[0].content


async def test_invalid_json_preserves_original_message():
    model = StaticModel("不是 JSON")

    result = await understand_query([], "这个能退吗", model)

    assert result.resolved_query == "这个能退吗"
    assert result.changed is False
    assert result.source == "fallback"


async def test_blank_resolved_query_preserves_original_message():
    model = StaticModel('{"resolved_query":"   "}')

    result = await understand_query([], "这个能退吗", model)

    assert result.resolved_query == "这个能退吗"
    assert result.source == "fallback"


async def test_colloquial_question_still_uses_model_normalization():
    model = StaticModel('{"resolved_query":"退款到账时间"}')

    result = await understand_query([], "钱啥时候打回来", model)

    assert result.resolved_query == "退款到账时间"
    assert result.source == "model"
    assert model.calls == 1
