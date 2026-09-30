"""
    构造上下文节点 和 路由节点
"""

from typing import Any

from langchain_core.messages import BaseMessage
from langgraph.config import get_stream_writer
from app.context.models import ContextPack
from app.graph.state import ChatState


def _legacy_pack(state: ChatState) -> ContextPack:
    messages: list[BaseMessage] = list(state.get("messages") or [])
    return ContextPack(
        summaries_text="",
        layer2_messages=[],
        layer1_messages=messages,
        history_ctx=messages,
        layer1_from_msg_id=None,
        summary_span=None,
        layer2_tokens=0,
        layer1_tokens=0,
        token_estimate=0,
        trigger_summary=False,
    )


def make_prepare_context_node(context_manager):
    """
        负责加载并裁剪会话历史，组装最终给大模型用的上下文包（ContextPack），存入图状态；并且向前端推送节点运行状态。
    """
    async def prepare_context(state: ChatState) -> dict[str, Any]:
        session_id = state["session_id"]
        if context_manager is None:
            pack = _legacy_pack(state)
        else:
            pack = await context_manager.prepare(
                session_id,
                state.get("current_user_message_id", 0),
            )

        get_stream_writer()(
            {
                "type": "node_status",
                "node": "prepare_context",
                "status": "success",
                "detail": {
                    "message_count": len(pack.history_ctx),
                    "token_estimate": pack.token_estimate,
                    "budget_error": pack.budget_error,
                },
            }
        )
        return {
            "context_pack": pack,
            "context_history_count": len(pack.history_ctx),
            "context_token_estimate": pack.token_estimate,
            "context_budget_error": pack.budget_error,
        }

    return prepare_context


def route_after_prepare_context(state: ChatState) -> str:
    """
        上下文准备完成后的路由判断：
        是否进入转人工节点
        如果预算超限，直接走兜底分支；否则进入意图理解节点
    """
    if state.get("pending_action") == "await_transfer_issue":
        return "transfer_issue"
    pack = state.get("context_pack")
    if pack is not None and pack.budget_error:
        return "fallback"
    return "understand"

