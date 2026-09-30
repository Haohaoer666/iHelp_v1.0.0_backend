from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.context.assembler import build_model_messages
from app.context.models import ContextPack


def test_model_context_order_is_stable():
    pack = ContextPack(
        summaries_text="用户问过订单1001。",
        layer2_messages=[HumanMessage("旧用户话")],
        layer1_messages=[HumanMessage("新用户话"), AIMessage("新回答")],
        history_ctx=[],
        layer1_from_msg_id=2,
        summary_span=None,
        layer2_tokens=4,
        layer1_tokens=4,
        token_estimate=8,
        trigger_summary=False,
        budget_error="",
    )

    messages = build_model_messages(
        pack=pack,
        evidence=[{"citation_id": 1, "text": "七天无理由"}],
        order_data={},
        after_sales_action="",
        fixed_system="固定人设和红线",
    )

    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == "固定人设和红线"
    assert messages[1].content == "旧用户话"
    assert messages[2].content == "新用户话"
    assert messages[3].content == "新回答"
    assert "用户问过订单1001。" in messages[-1].content
    assert "[1] 七天无理由" in messages[-1].content
    assert not any(
        isinstance(item, SystemMessage) and "订单1001" in item.content
        for item in messages
    )


def test_current_user_precedes_dynamic_supplement():
    pack = ContextPack(
        summaries_text="用户问过订单1001。",
        layer2_messages=[HumanMessage("旧问题")],
        layer1_messages=[AIMessage("旧回答")],
        history_ctx=[],
        layer1_from_msg_id=2,
        summary_span=None,
        layer2_tokens=4,
        layer1_tokens=4,
        token_estimate=8,
        trigger_summary=False,
    )

    messages = build_model_messages(
        pack=pack,
        evidence=[{"citation_id": 1, "text": "七天无理由"}],
        order_data={},
        after_sales_action="",
        fixed_system="固定人设和红线",
        current_user=HumanMessage("当前问题"),
    )

    assert messages[-2].content == "当前问题"
    assert "用户问过订单1001。" in messages[-1].content
    assert "[1] 七天无理由" in messages[-1].content
