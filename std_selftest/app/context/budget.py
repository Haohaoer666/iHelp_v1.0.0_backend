"""Derive history budgets from the configured model window and Agent limits."""

from dataclasses import dataclass

from app.config import settings


@dataclass(frozen=True)
class ContextBudget:
    """
        上下文预算数据类（不可变 frozen）
        核心：基于模型窗口，反向计算各类token额度，分层分配层1/层2历史预算
        frozen=True：实例一旦创建，所有字段不可修改，保证预算计算全程只读，避免运行时被意外篡改
    """
    model_context_window: int
    max_output_tokens: int
    max_user_input_tokens: int
    max_agent_steps: int
    tool_result_max_tokens: int
    rerank_top_k: int
    evidence_chunk_token_budget: int
    system_prompt_token_budget: int
    summary_injection_token_budget: int
    safety_margin_tokens: int
    desired_retained_turns: int
    steady_turn_token_estimate: int
    tool_call_message_max_tokens: int = 400
    evidence_total_max_tokens: int | None = None
    summary_segment_max_tokens: int = 200
    history_soft_budget_enabled: bool = True
    history_layer1_ratio: float = 0.70
    history_layer2_ratio: float = 0.30
    legacy_acceptance: bool = False

    @classmethod
    def from_settings(cls) -> "ContextBudget":
        legacy_acceptance = (
            settings.context_budget_profile == "legacy_acceptance"
        )
        return cls(
            model_context_window=settings.model_context_window,
            max_output_tokens=settings.agent_max_output_tokens,
            max_user_input_tokens=settings.max_user_input_tokens,
            max_agent_steps=settings.agent_max_steps,
            tool_result_max_tokens=settings.tool_result_max_tokens,
            rerank_top_k=settings.rerank_top_k,
            evidence_chunk_token_budget=settings.evidence_chunk_token_budget,
            summary_injection_token_budget=settings.summary_injection_token_budget,
            safety_margin_tokens=settings.safety_margin_tokens,
            desired_retained_turns=settings.desired_retained_turns,
            steady_turn_token_estimate=settings.steady_turn_token_estimate,
            tool_call_message_max_tokens=settings.tool_call_message_max_tokens,
            evidence_total_max_tokens=(
                None
                if legacy_acceptance
                else settings.evidence_total_max_tokens
            ),
            summary_segment_max_tokens=settings.summary_segment_max_tokens,
            history_soft_budget_enabled=settings.history_soft_budget_enabled,
            history_layer1_ratio=settings.history_layer1_ratio,
            history_layer2_ratio=settings.history_layer2_ratio,
            legacy_acceptance=legacy_acceptance,
            system_prompt_token_budget=(
                1800
                if legacy_acceptance
                else settings.system_prompt_token_budget
            ),
        )

    @property
    def evidence_budget(self) -> int:
        """Return the effective evidence reserve."""
        if self.evidence_total_max_tokens is not None:
            return self.evidence_total_max_tokens
        return self.rerank_top_k * self.evidence_chunk_token_budget

    @property
    def react_peak(self) -> int:
        """
            Agent工具调用瞬时峰值token
            最多max_agent_steps轮工具，每一步工具返回最大tool_result_max_tokens
            代表多步工具调用场景下，工具返回内容的最大瞬时开销
        """
        if self.legacy_acceptance:
            return self.max_agent_steps * self.tool_result_max_tokens
        return self.max_agent_steps * (
            self.tool_call_message_max_tokens
            + self.tool_result_max_tokens
            + 8
        )

    @property
    def fixed_overhead(self) -> int:
        """
                【固定总开销】
                所有LLM调用时，除常驻对话历史以外，一定会占用/可能瞬时占用的全部token总和
                包含：系统提示词、召回证据、摘要注入、模型输出预留、用户输入上限、Agent工具峰值、安全缓冲
        """
        return (
            self.system_prompt_token_budget
            + self.evidence_budget
            + self.summary_injection_token_budget
            + self.max_output_tokens
            + self.max_user_input_tokens
            + self.react_peak
            + self.safety_margin_tokens
        )

    @property
    def window_available(self) -> int:
        """
            扣除全部固定开销后，窗口剩余可以分配给【对话历史】的额度
            max(0, ...)：防止固定开销 > 模型窗口，出现负数
        """
        return max(0, self.model_context_window - self.fixed_overhead)

    @property
    def hard_history_budget(self) -> int:
        """Maximum history tokens after all mandatory reserves."""
        return self.window_available

    @property
    def desired_history(self) -> int:
        """
                目标历史预算：期望保留N轮，按单轮最大token估算出来的总需求
                desired_retained_turns × steady_turn_token_estimate
                只是理想值，不一定能满足
        """
        return self.desired_retained_turns * self.steady_turn_token_estimate

    @property
    def soft_history_budget(self) -> int:
        """Normal cost target for retained history."""
        return self.desired_history

    @property
    def history_budget(self) -> int:
        """
            最终生效的对话历史总预算：取【目标历史额度】和【窗口剩余可用额度】两者较小值
            窗口大：用期望轮数对应的额度；窗口小：自动压缩到window_available
        """
        if not self.history_soft_budget_enabled:
            return self.hard_history_budget
        return min(self.soft_history_budget, self.hard_history_budget)

    @property
    def layer1_budget(self) -> int:
        """
            层1预算：近期完整原始对话，占历史总预算70%
            layer1：原文对话，用于指代、意图理解；
            减1是做微小保护偏移，避免边界判断踩坑；
            history_budget<=1时预算直接置0，无法保存对话
        """
        if self.history_budget <= 1:
            return 0
        return int(
            (self.history_budget - 1) * self.history_layer1_ratio
        )

    @property
    def layer2_budget(self) -> int:
        """
            层2预算：远期压缩摘要，占历史总预算剩余30%
            layer2存放摘要，层1放不下的旧对话压缩后移入这里
        """
        if self.history_budget <= 1:
            return 0
        return self.history_budget - 1 - self.layer1_budget

    @property
    def can_fit_one_turn(self) -> bool:
        """
            启动自检标记：是否至少可以容纳一轮对话历史
            lifespan里用这个属性打critical告警：history_budget<=0说明配置完全不可用
        """
        return self.history_budget > 0
