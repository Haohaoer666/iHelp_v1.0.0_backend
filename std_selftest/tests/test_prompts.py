from langchain_core.messages import HumanMessage

from app.core.prompts import CUSTOMER_SERVICE_PROMPT, EXTRACT_PROMPT


def test_customer_service_prompt_renders_with_history():
    messages = CUSTOMER_SERVICE_PROMPT.format_messages(
        history=[HumanMessage("你们卖猫粮吗？")]
    )

    assert messages[0].type == "system"
    assert messages[1].type == "human"
    assert "电商智能客服" in messages[0].content


def test_extract_prompt_renders_text():
    messages = EXTRACT_PROMPT.format_messages(text="订单 MH20260701123 到货损坏，要求退款")

    assert messages[0].type == "system"
    assert messages[1].content == "订单 MH20260701123 到货损坏，要求退款"
    assert "order_id" in messages[0].content
