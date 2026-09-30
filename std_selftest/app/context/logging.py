"""One-line structured logs for context and summary observability."""

import json
import logging
from collections.abc import Sequence

from langchain_core.messages import BaseMessage

from app.context.models import PromptUsage, SummarySpan


logger = logging.getLogger("app.context")


def _message_payload(message: BaseMessage) -> dict[str, str]:
    """
        将 LangChain 的 BaseMessage 消息对象，转换成可序列化的字典
    """
    content = message.content
    return {
        "type": getattr(message, "type", "unknown"),
        "content": content if isinstance(content, str) else str(content),
    }


def _log_context(
    event: str,
    session_id: str,
    messages: Sequence[BaseMessage],
    token_estimate: int,
    summary_text: str,
) -> None:
    logger.info(
        "%s session_id=%s summary_json=%s messages_json=%s "
        "message_count=%d token_estimate=%d",
        event,
        session_id,
        json.dumps(summary_text, ensure_ascii=False),
        json.dumps(
            [_message_payload(message) for message in messages],
            ensure_ascii=False,
        ),
        len(messages),
        token_estimate,
    )


def log_model_ctx(
    session_id: str,
    messages: Sequence[BaseMessage],
    token_estimate: int,
    summary_text: str = "",
    *,
    usage: PromptUsage | None = None,
) -> None:
    _log_context(
        "model_ctx",
        session_id,
        messages,
        token_estimate,
        summary_text,
    )
    if usage is not None:
        logger.info(
            "model_ctx_usage session_id=%s full_prompt_tokens=%d "
            "system_tokens=%d tool_definition_tokens=%d summary_tokens=%d "
            "evidence_tokens=%d layer1_tokens=%d layer2_tokens=%d "
            "current_user_tokens=%d react_peak_reserved=%d output_reserved=%d "
            "safety_margin=%d history_soft_budget=%d history_hard_budget=%d",
            session_id,
            usage.full_prompt_tokens,
            usage.system_tokens,
            usage.tool_definition_tokens,
            usage.summary_tokens,
            usage.evidence_tokens,
            usage.layer1_tokens,
            usage.layer2_tokens,
            usage.current_user_tokens,
            usage.react_peak_reserved,
            usage.output_reserved,
            usage.safety_margin_tokens,
            usage.history_soft_budget,
            usage.history_hard_budget,
        )


def log_history_ctx(
    session_id: str,
    messages: Sequence[BaseMessage],
    token_estimate: int,
    summary_text: str = "",
) -> None:
    """
        记录上下文审计日志，把最终送入LangGraph的上下文信息打日志，用于排查调试。
        封装层，直接调用底层通用日志函数 _log_context。
    """
    _log_context(
        "history_ctx",
        session_id,
        messages,
        token_estimate,
        summary_text,
    )


def log_summary_event(
    event: str,
    session_id: str,
    *,
    span: SummarySpan | None = None,
    elapsed_ms: int | None = None,
    reason: str = "",
    error: str = "",
    sequence_no: int | None = None,
    layer2_tokens: int | None = None,
    budget: int | None = None,
    max_tokens: int | None = None,
) -> None:
    """
        输出对话链路的摘要日志（汇总事件日志，不是每一个流式chunk的日志）
    """
    parts = [f"summary {event}", f"session_id={session_id}"]
    if span is not None:
        parts.extend(
            (
                f"covered_from={span.from_msg_id}",
                f"covered_to={span.to_msg_id}",
            )
        )
    if sequence_no is not None:
        parts.append(f"sequence_no={sequence_no}")
    if elapsed_ms is not None:
        parts.append(f"elapsed_ms={elapsed_ms}")
    if layer2_tokens is not None:
        parts.append(f"layer2_tokens={layer2_tokens}")
    if budget is not None:
        parts.append(f"budget={budget}")
    if max_tokens is not None:
        parts.append(f"max_tokens={max_tokens}")
    if reason:
        parts.append(f"reason={json.dumps(reason, ensure_ascii=False)}")
    if error:
        parts.append(f"error={json.dumps(error, ensure_ascii=False)}")
    logger.info(" ".join(parts))
