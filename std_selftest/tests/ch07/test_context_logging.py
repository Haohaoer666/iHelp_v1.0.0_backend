import logging

from langchain_core.messages import HumanMessage

from app.context.logging import log_history_ctx, log_model_ctx
from app.context.models import PromptUsage


def test_model_context_log_contains_full_payload(caplog):
    with caplog.at_level(logging.INFO, logger="app.context"):
        log_model_ctx(
            "s1",
            [HumanMessage("订单1001")],
            token_estimate=12,
            summary_text="用户问过订单1001。",
        )

    record = next(item for item in caplog.records if "model_ctx" in item.message)
    rendered = record.getMessage()
    assert "订单1001" in rendered
    assert "用户问过订单1001。" in rendered
    assert "message_count=1" in rendered


def test_history_context_log_is_emitted_every_turn(caplog):
    with caplog.at_level(logging.INFO, logger="app.context"):
        log_history_ctx(
            "s1",
            [HumanMessage("历史问题")],
            token_estimate=8,
            summary_text="",
        )

    assert any("history_ctx" in item.message for item in caplog.records)


def test_model_context_usage_log_contains_component_breakdown(caplog):
    usage = PromptUsage(
        full_prompt_tokens=123,
        system_tokens=10,
        tool_definition_tokens=20,
        summary_tokens=30,
        evidence_tokens=40,
        layer1_tokens=5,
        layer2_tokens=4,
        current_user_tokens=6,
        react_peak_reserved=50,
        output_reserved=60,
        safety_margin_tokens=70,
        history_soft_budget=80,
        history_hard_budget=90,
    )

    with caplog.at_level(logging.INFO, logger="app.context"):
        log_model_ctx(
            "s1",
            [HumanMessage("订单1001")],
            token_estimate=12,
            usage=usage,
        )

    rendered = "\n".join(item.getMessage() for item in caplog.records)
    assert "model_ctx_usage" in rendered
    assert "full_prompt_tokens=123" in rendered
    assert "tool_definition_tokens=20" in rendered
    assert "history_soft_budget=80" in rendered
