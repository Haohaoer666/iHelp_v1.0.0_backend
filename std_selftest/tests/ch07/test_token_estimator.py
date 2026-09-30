from langchain_core.messages import AIMessage, HumanMessage

from app.context.tokens import estimate_messages_tokens, estimate_text_tokens


def test_cjk_is_not_folded_like_ascii():
    chinese = estimate_text_tokens("订单1001要退款", 0.80)
    ascii_text = estimate_text_tokens("order1001refund", 0.80)

    assert chinese > ascii_text


def test_message_overhead_is_included():
    one = estimate_messages_tokens([HumanMessage("你好")], 0.80)

    assert one > estimate_text_tokens("你好", 0.80)


def test_mixed_summary_estimate_is_stable():
    estimate = estimate_messages_tokens(
        [HumanMessage("订单1001"), AIMessage("已登记退款")],
        0.80,
    )

    assert 10 <= estimate <= 30

