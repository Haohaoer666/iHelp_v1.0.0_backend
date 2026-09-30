from app.context.budget import ContextBudget
from app.context.models import ContextMessage, SummaryRow, SummarySpan
from app.context.trimmer import build_context_pack


def budget(
    layer1: int,
    layer2: int,
    summary: int = 0,
) -> ContextBudget:
    return ContextBudget(
        model_context_window=10000,
        max_output_tokens=0,
        max_user_input_tokens=0,
        max_agent_steps=0,
        tool_result_max_tokens=0,
        rerank_top_k=0,
        evidence_chunk_token_budget=0,
        system_prompt_token_budget=0,
        summary_injection_token_budget=summary,
        safety_margin_tokens=0,
        desired_retained_turns=layer1 + layer2,
        steady_turn_token_estimate=1,
    )


def test_layer1_downgrade_only_moves_anchor():
    messages = [
        ContextMessage(1, "user", "问题一"),
        ContextMessage(2, "assistant", "回答一" * 20),
        ContextMessage(3, "user", "问题二"),
        ContextMessage(4, "assistant", "回答二"),
    ]

    pack = build_context_pack(
        messages=messages,
        summaries=[],
        layer1_from_msg_id=1,
        summary_upto_msg_id=0,
        budget=budget(layer1=35, layer2=15),
        chinese_tokens_per_char=1.0,
        assistant_layer2_chars=3,
    )

    assert pack.layer1_from_msg_id == 3
    assert [item.content for item in pack.layer1_messages] == ["问题二", "回答二"]
    assert [item.content for item in pack.layer2_messages] == [
        "问题一",
        "回答一...",
    ]
    assert pack.trigger_summary is False


def test_layer2_overflow_marks_summary_span_without_waiting():
    messages = [
        ContextMessage(1, "user", "订单1001要退款"),
        ContextMessage(2, "assistant", "已经登记退款" * 20),
        ContextMessage(3, "user", "后来呢"),
        ContextMessage(4, "assistant", "处理中"),
    ]

    pack = build_context_pack(
        messages=messages,
        summaries=[],
        layer1_from_msg_id=3,
        summary_upto_msg_id=0,
        budget=budget(layer1=21, layer2=8, summary=20),
        chinese_tokens_per_char=1.0,
        assistant_layer2_chars=2,
    )

    assert pack.summary_span == SummarySpan(1, 2)
    assert pack.trigger_summary is True
    assert pack.layer1_messages[-1].content == "处理中"


def test_existing_summary_is_background_only():
    from app.context.models import SummaryRow

    pack = build_context_pack(
        messages=[
            ContextMessage(1, "user", "旧问题"),
            ContextMessage(2, "assistant", "旧回答"),
            ContextMessage(3, "user", "新问题"),
        ],
        summaries=[SummaryRow(1, 2, "用户咨询过旧问题。")],
        layer1_from_msg_id=3,
        summary_upto_msg_id=2,
        budget=budget(layer1=21, layer2=8, summary=20),
        chinese_tokens_per_char=1.0,
        assistant_layer2_chars=4,
    )

    assert pack.summaries_text == "用户咨询过旧问题。"
    assert [item.message_id for item in pack.layer2_messages] == []
    assert [item.content for item in pack.layer1_messages] == ["新问题"]


def test_layer2_oversized_newest_item_is_omitted():
    pack = build_context_pack(
        messages=[
            ContextMessage(1, "user", "旧用户问题" * 20),
            ContextMessage(2, "user", "当前问题"),
        ],
        summaries=[],
        layer1_from_msg_id=2,
        summary_upto_msg_id=0,
        budget=budget(layer1=21, layer2=8),
        chinese_tokens_per_char=1.0,
        assistant_layer2_chars=4,
    )

    assert pack.trigger_summary is True
    assert pack.layer2_messages == []


def test_current_user_is_not_part_of_layer1():
    pack = build_context_pack(
        messages=[
            ContextMessage(1, "user", "第一轮问题"),
            ContextMessage(2, "assistant", "第一轮回答"),
            ContextMessage(3, "user", "当前问题"),
        ],
        summaries=[],
        layer1_from_msg_id=None,
        summary_upto_msg_id=0,
        budget=budget(layer1=100, layer2=100),
        chinese_tokens_per_char=0.8,
        assistant_layer2_chars=80,
        current_user_message_id=3,
    )

    assert pack.current_user_message is not None
    assert pack.current_user_message.content == "当前问题"
    assert all(
        "当前问题" not in message.content
        for message in pack.layer1_messages
    )


def test_summary_injection_keeps_latest_contiguous_segments():
    pack = build_context_pack(
        messages=[
            ContextMessage(1, "user", "旧问题"),
            ContextMessage(2, "assistant", "旧回答"),
            ContextMessage(3, "user", "新问题"),
            ContextMessage(4, "assistant", "新回答"),
            ContextMessage(5, "user", "当前问题"),
        ],
        summaries=[
            SummaryRow(1, 2, "旧摘要" * 50),
            SummaryRow(3, 4, "新摘要"),
        ],
        layer1_from_msg_id=None,
        summary_upto_msg_id=4,
        budget=budget(layer1=100, layer2=100, summary=20),
        chinese_tokens_per_char=0.8,
        assistant_layer2_chars=80,
        current_user_message_id=5,
    )

    assert "新摘要" in pack.summaries_text
    assert "旧摘要" not in pack.summaries_text
    assert pack.summary_injection_tokens <= 20
