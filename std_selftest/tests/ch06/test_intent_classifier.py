from langchain_core.messages import AIMessage

from app.graph.intent import INTENT_TO_ROUTE, IntentDecision, classify_intent_text


class StaticModel:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0
        self.messages = []

    async def ainvoke(self, messages, **kwargs):
        self.calls += 1
        self.messages = messages
        return AIMessage(content=self.content)


async def test_intent_returns_only_business_intent_and_confidence():
    model = StaticModel('{"intent":"退款退货","confidence":0.91}')

    result = await classify_intent_text("退货政策是什么", model)

    assert result == IntentDecision(
        intent="退款退货",
        route="core_after_sales",
        confidence=0.91,
        model_used="primary",
    )
    prompt = model.messages[0].content
    for intent in (
        "物流",
        "订单",
        "商品咨询",
        "退款退货",
        "售后",
        "投诉",
        "闲聊",
        "转人工",
        "其他",
    ):
        assert intent in prompt
    assert '"intent"' in prompt
    assert '"confidence"' in prompt
    assert "以下九类" in prompt


async def test_unknown_question_falls_back_to_other():
    model = StaticModel("not json")

    result = await classify_intent_text("帮我看下那个奇怪的东西", model)

    assert result == IntentDecision(
        intent="其他",
        route="other",
        confidence=0.0,
        model_used="fallback",
    )


async def test_extra_json_fields_are_rejected():
    model = StaticModel('{"intent":"物流","confidence":0.9,"reason":"轨迹"}')

    result = await classify_intent_text("订单 1001 到哪了", model)

    assert result.intent == "其他"
    assert result.confidence == 0.0


async def test_low_confidence_escalates_to_primary_model():
    low = StaticModel('{"intent":"订单","confidence":0.31}')
    primary = StaticModel('{"intent":"物流","confidence":0.92}')

    result = await classify_intent_text(
        "订单 1001 到哪了",
        primary,
        low_cost_model=low,
        use_low_cost_first=True,
        confidence_threshold=0.75,
    )

    assert result.intent == "物流"
    assert result.confidence == 0.92
    assert result.model_used == "upgraded"
    assert low.calls == 1
    assert primary.calls == 1


async def test_high_confidence_low_cost_result_is_not_upgraded():
    low = StaticModel('{"intent":"物流","confidence":0.93}')
    primary = StaticModel('{"intent":"订单","confidence":0.99}')

    result = await classify_intent_text(
        "订单 1001 到哪了",
        primary,
        low_cost_model=low,
        use_low_cost_first=True,
        confidence_threshold=0.75,
    )

    assert result.intent == "物流"
    assert result.model_used == "low_cost"
    assert low.calls == 1
    assert primary.calls == 0


def test_intents_map_to_routes():
    assert INTENT_TO_ROUTE == {
        "物流": "business",
        "订单": "business",
        "商品咨询": "knowledge",
        "退款退货": "core_after_sales",
        "售后": "core_after_sales",
        "投诉": "complaint",
        "闲聊": "chitchat",
        "转人工": "human_transfer",
        "其他": "other",
    }
