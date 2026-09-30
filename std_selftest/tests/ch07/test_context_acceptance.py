from app.context.budget import ContextBudget
from app.context.models import ContextMessage
from app.context.trimmer import build_context_pack


def _acceptance_budget() -> ContextBudget:
    return ContextBudget(
        model_context_window=18000,
        max_output_tokens=2000,
        max_user_input_tokens=2000,
        max_agent_steps=3,
        tool_call_message_max_tokens=400,
        tool_result_max_tokens=1200,
        rerank_top_k=5,
        evidence_chunk_token_budget=400,
        evidence_total_max_tokens=None,
        system_prompt_token_budget=1800,
        summary_injection_token_budget=600,
        summary_segment_max_tokens=200,
        safety_margin_tokens=350,
        desired_retained_turns=24,
        steady_turn_token_estimate=800,
        legacy_acceptance=True,
    )


def _cost_optimized_budget() -> ContextBudget:
    return ContextBudget(
        model_context_window=65536,
        max_output_tokens=2000,
        max_user_input_tokens=2000,
        max_agent_steps=3,
        tool_call_message_max_tokens=400,
        tool_result_max_tokens=1200,
        rerank_top_k=10,
        evidence_chunk_token_budget=400,
        evidence_total_max_tokens=3000,
        system_prompt_token_budget=519,
        summary_injection_token_budget=600,
        summary_segment_max_tokens=200,
        safety_margin_tokens=350,
        desired_retained_turns=24,
        steady_turn_token_estimate=800,
    )


def test_cost_optimized_profile_keeps_history_soft_budget():
    budget = _cost_optimized_budget()

    assert budget.hard_history_budget == 52243
    assert budget.soft_history_budget == 19200
    assert budget.history_budget == 19200


def test_acceptance_sequence_moves_layer1_and_triggers_summary():
    messages = []
    for turn in range(60):
        messages.append(
            ContextMessage(
                turn * 2 + 1,
                "user",
                f"第{turn}轮订单{1000 + turn}的智能手表要求今天确认退款。"
                * 6,
            )
        )
        messages.append(
            ContextMessage(
                turn * 2 + 2,
                "assistant",
                "已记录，问题仍未解决。",
            )
        )

    pack = build_context_pack(
        messages=messages,
        summaries=[],
        layer1_from_msg_id=1,
        summary_upto_msg_id=0,
        budget=_acceptance_budget(),
        chinese_tokens_per_char=1.0,
        assistant_layer2_chars=8,
    )

    assert pack.layer1_from_msg_id is not None
    assert pack.layer1_from_msg_id > 1
    assert pack.trigger_summary is True
    assert pack.summary_span is not None
    assert pack.summary_span.from_msg_id == 1
