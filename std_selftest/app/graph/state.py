"""
    定义LangGraph总状态
"""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from app.context.models import ContextPack


class ChatState(TypedDict, total=False):
    session_id: str
    user_key: str
    current_user_message_id: int
    messages: Annotated[list[BaseMessage], add_messages]
    agent_messages: list[BaseMessage]
    user_message: str
    resolved_query: str
    query_history: list[BaseMessage]
    query_changed: bool
    query_understanding_source: str
    intent: str
    intent_confidence: float
    intent_model_used: str
    route: str
    evidence: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    sufficient: bool
    refusal: str
    reply: str
    reply_options: list[dict[str, str]]
    agent_steps: int
    tool_trace: list[dict[str, Any]]
    available_tool_names: list[str]
    tool_catalog_version: str
    tool_catalog_errors: list[dict[str, Any]]
    ticket_request_allowed: bool
    ticket_slots: dict[str, Any]
    route_reason: str
    stop_reason: str
    error: str
    turn_log: dict[str, Any]
    order_id: str
    order_options: list[dict[str, Any]]
    order_data: dict[str, Any]
    expanded_queries: list[str]
    retrieval_query_count: int
    pending_action: str
    after_sales_action: str
    preferred_intent: str
    last_intent: str
    last_order_id: str
    last_after_sales_action: str
    context_pack: ContextPack
    context_history_count: int
    context_token_estimate: int
    context_budget_error: str
