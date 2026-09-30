"""Immutable data contracts shared by context-management components."""

from dataclasses import dataclass

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)


@dataclass(frozen=True)
class ContextMessage:
    message_id: int
    role: str
    content: str
    tool_name: str | None = None
    """
        会话消息的内部数据载体，承载单条消息，支持转换成LangChain消息对象
    """
    def to_langchain(self) -> BaseMessage:
        """把内部消息模型转为LangChain原生消息对象，用于拼接Prompt"""
        if self.role == "user":
            return HumanMessage(self.content)
        if self.role == "assistant":
            return AIMessage(self.content)
        return ToolMessage(
            self.content,
            tool_call_id=f"persisted-{self.message_id}",
            name=self.tool_name or "tool",
        )

    @classmethod
    def from_record(cls, row) -> "ContextMessage":
        """数据库ORM记录row，转为ContextMessage实例"""
        return cls(
            message_id=row.id,
            role=row.role,
            content=row.content or "",
            tool_name=row.tool_name,
        )


@dataclass(frozen=True)
class SummaryRow:
    """摘要记录数据类，记录一段消息区间对应的摘要文本"""
    covered_from_msg_id: int  # 被摘要覆盖的起始消息ID
    covered_to_msg_id: int  # 被摘要覆盖的结束消息ID
    content: str  # 这段消息生成的摘要文本


@dataclass(frozen=True)
class SummarySpan:
    from_msg_id: int
    to_msg_id: int


@dataclass(frozen=True)
class PromptUsage:
    full_prompt_tokens: int
    system_tokens: int
    tool_definition_tokens: int
    summary_tokens: int
    evidence_tokens: int
    layer1_tokens: int
    layer2_tokens: int
    current_user_tokens: int
    react_peak_reserved: int
    output_reserved: int
    safety_margin_tokens: int
    history_soft_budget: int
    history_hard_budget: int


@dataclass(frozen=True)
class ContextPack:
    summaries_text: str
    layer2_messages: list[BaseMessage]
    layer1_messages: list[BaseMessage]
    history_ctx: list[BaseMessage]
    layer1_from_msg_id: int | None
    summary_span: SummarySpan | None
    layer2_tokens: int
    layer1_tokens: int
    token_estimate: int
    trigger_summary: bool
    budget_error: str = ""
    layer2_trigger_tokens: int = 0
    current_user_message: BaseMessage | None = None
    summary_injection_tokens: int = 0
    prompt_usage: PromptUsage | None = None
