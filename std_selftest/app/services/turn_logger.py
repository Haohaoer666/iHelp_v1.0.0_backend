"""
    持久化并记录一轮对话工作流的最终状态
"""

import logging

from app.db.repository import (
    append_message,
    ensure_conversation,
)
from app.graph.state import ChatState


logger = logging.getLogger(__name__)


class TurnLogger:
    async def log(self, session_id: str, state: ChatState) -> dict:
        summary = {
            "session_id": session_id,
            "resolved_query": state.get("resolved_query", ""),
            "intent": state.get("intent", ""),
            "intent_confidence": state.get("intent_confidence", 0.0),
            "intent_model_used": state.get("intent_model_used", ""),
            "route": state.get("route", ""),
            "agent_steps": state.get("agent_steps", 0),
            "retrieval": bool(state.get("evidence")),
            "order_id": state.get("order_id", ""),
            "after_sales_action": state.get("after_sales_action", ""),
            "retrieval_query_count": state.get("retrieval_query_count", 0),
            "pending_action": state.get("pending_action", ""),
            "sufficient": state.get("sufficient", False),
            "stop_reason": state.get("stop_reason", ""),
        }

        try:
            await ensure_conversation(session_id)
            if not state.get("current_user_message_id"):
                await append_message(
                    session_id,
                    "user",
                    state.get("user_message", ""),
                )
            await append_message(session_id, "assistant", state.get("reply", ""))
        except Exception:
            summary["persistence_error"] = True
            logger.exception("轮次审计落库失败 session_id=%s", session_id)

        logger.info(
            "turn_complete session_id=%s resolved_query=%s intent=%s "
            "intent_confidence=%s intent_model_used=%s route=%s order_id=%s "
            "after_sales_action=%s agent_steps=%s retrieval_query_count=%s retrieval=%s "
            "pending_action=%s sufficient=%s stop_reason=%s",
            summary["session_id"],
            summary["resolved_query"],
            summary["intent"],
            summary["intent_confidence"],
            summary["intent_model_used"],
            summary["route"],
            summary["order_id"],
            summary["after_sales_action"],
            summary["agent_steps"],
            summary["retrieval_query_count"],
            summary["retrieval"],
            summary["pending_action"],
            summary["sufficient"],
            summary["stop_reason"],
        )
        return summary
