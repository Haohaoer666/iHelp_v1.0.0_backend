import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.graph.intent import (
    INTENT_TO_ROUTE,
    IntentClassificationError,
    classify_intent_node,
    classify_intent_text,
    fallback_intent,
    is_after_sales_mcp_query,
    parse_intent_response,
)
from app.graph.state import ChatState


class StaticModel:
    def __init__(self, content: str) -> None:
        self.content = content
        self.received_messages = None

    async def ainvoke(self, messages, **kwargs):
        self.received_messages = messages
        return AIMessage(content=self.content)


def test_intents_map_to_workflow_routes():
    assert INTENT_TO_ROUTE == {
        "商品咨询": "knowledge",
        "退款退货": "core_after_sales",
        "物流": "business",
        "订单": "business",
        "售后": "core_after_sales",
        "投诉": "complaint",
        "闲聊": "chitchat",
        "转人工": "human_transfer",
        "其他": "other",
    }


def test_parse_intent_response_accepts_code_fence():
    decision = parse_intent_response(
        '```json\n{"intent":"物流","confidence":0.93}\n```'
    )

    assert decision.intent == "物流"
    assert decision.route == "business"
    assert decision.confidence == 0.93


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("订单 1001 的物流到哪了", "物流"),
        ("我的包裹现在在哪", "物流"),
        ("我的退款为什么还没到", "售后"),
        ("换货审核多久", "售后"),
        ("售后处理到哪一步了", "售后"),
        ("退货政策是什么", "退款退货"),
        ("怎么申请退款", "退款退货"),
        ("订单为什么还没发货", "订单"),
        ("这款显示器刷新率多少", "商品咨询"),
        ("我要投诉快递员", "投诉"),
        ("我要转人工", "转人工"),
        ("帮我找人工客服", "转人工"),
        ("你好呀", "闲聊"),
    ],
)
def test_fallback_intent_uses_keywords(text, expected):
    assert fallback_intent(text).intent == expected


def test_fallback_intent_returns_other_for_unknown_text():
    decision = fallback_intent("帮我看看这个")
    assert decision.intent == "其他"
    assert decision.route == "other"


def test_after_sales_mcp_query_uses_business_route():
    decision = fallback_intent("订单 1001 还在保修期内吗")

    assert decision.intent == "售后"
    assert decision.route == "business"
    assert is_after_sales_mcp_query("订单 1001 的退货进度") is True
    assert is_after_sales_mcp_query("订单 1001 的维修进度") is False


async def test_classify_intent_text_uses_model_json():
    model = StaticModel('{"intent":"退款退货","confidence":0.92}')

    decision = await classify_intent_text("退货政策是什么", model)

    assert decision.intent == "退款退货"
    assert decision.route == "core_after_sales"
    assert decision.confidence == 0.92
    assert isinstance(model.received_messages[0], HumanMessage)
    assert "退货政策是什么" in model.received_messages[0].content


async def test_explicit_ticket_request_bypasses_intent_model():
    class NoUseModel:
        async def ainvoke(self, *args, **kwargs):
            raise AssertionError("model should not be called")

    decision = await classify_intent_text(
        "帮我建个工单，快递一直不到",
        NoUseModel(),
    )

    assert decision.intent == "其他"
    assert decision.route == "other"
    assert decision.model_used == "ticket_gate"


async def test_explicit_transfer_cancel_bypasses_intent_model():
    class NoUseModel:
        async def ainvoke(self, *args, **kwargs):
            raise AssertionError("model should not be called")

    decision = await classify_intent_text(
        "取消转人工",
        NoUseModel(),
    )

    assert decision.intent == "转人工"
    assert decision.route == "human_transfer"
    assert decision.model_used == "human_transfer_cancel_gate"


async def test_classify_intent_text_falls_back_on_invalid_json():
    model = StaticModel("这不是 JSON")

    decision = await classify_intent_text("我要投诉快递员", model)

    assert decision.intent == "其他"
    assert decision.route == "other"


async def test_classify_intent_node_returns_state_updates():
    model = StaticModel('{"intent":"物流","confidence":0.95}')

    update = await classify_intent_node(
        {"resolved_query": "订单 1001 的物流到哪了"},
        model,
    )

    assert update["intent"] == "物流"
    assert update["route"] == "business"
    assert update["intent_confidence"] == 0.95
    assert update["intent_model_used"] == "primary"


async def test_classify_intent_node_routes_warranty_query_to_mcp_agent():
    model = StaticModel('{"intent":"售后","confidence":0.95}')

    update = await classify_intent_node(
        {"resolved_query": "订单 1001 还在保修期内吗"},
        model,
    )

    assert update["intent"] == "售后"
    assert update["route"] == "business"


def test_chat_state_has_required_fields():
    annotations = ChatState.__annotations__
    for field in [
        "messages",
        "agent_messages",
        "user_message",
        "resolved_query",
        "intent",
        "route",
        "evidence",
        "citations",
        "sufficient",
        "reply",
        "reply_options",
        "agent_steps",
        "tool_trace",
        "route_reason",
        "stop_reason",
        "error",
    ]:
        assert field in annotations
