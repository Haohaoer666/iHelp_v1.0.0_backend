from app.context.budget import ContextBudget


def test_default_cost_profile_uses_soft_history_budget():
    budget = ContextBudget(
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

    assert budget.evidence_budget == 3000
    assert budget.react_peak == 4824
    assert budget.hard_history_budget == 52243
    assert budget.soft_history_budget == 19200
    assert budget.history_budget == 19200
    assert budget.layer1_budget == 13439
    assert budget.layer2_budget == 5760


def test_legacy_acceptance_profile_preserves_ch07_numbers():
    budget = ContextBudget(
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

    assert budget.evidence_budget == 2000
    assert budget.react_peak == 3600
    assert budget.hard_history_budget == 5650
    assert budget.soft_history_budget == 19200
    assert budget.history_budget == 5650


def test_acceptance_budget_matches_required_numbers():
    budget = ContextBudget(
        model_context_window=18000,
        max_output_tokens=2000,
        max_user_input_tokens=2000,
        max_agent_steps=3,
        tool_result_max_tokens=1200,
        rerank_top_k=5,
        evidence_chunk_token_budget=400,
        system_prompt_token_budget=1800,
        summary_injection_token_budget=600,
        safety_margin_tokens=350,
        desired_retained_turns=24,
        steady_turn_token_estimate=800,
        legacy_acceptance=True,
    )

    assert budget.fixed_overhead == 12350
    assert budget.history_budget == 5650
    assert budget.layer1_budget == 3954
    assert budget.layer2_budget == 1695


def test_only_shrinking_window_with_defaults_is_insufficient():
    budget = ContextBudget(
        model_context_window=18000,
        max_output_tokens=800,
        max_user_input_tokens=2000,
        max_agent_steps=5,
        tool_result_max_tokens=1200,
        rerank_top_k=20,
        evidence_chunk_token_budget=400,
        system_prompt_token_budget=1800,
        summary_injection_token_budget=600,
        safety_margin_tokens=350,
        desired_retained_turns=24,
        steady_turn_token_estimate=800,
    )

    assert budget.history_budget == 0
    assert budget.can_fit_one_turn is False
