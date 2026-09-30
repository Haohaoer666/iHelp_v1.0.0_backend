"""Build the deterministic customer-service workflow graph."""

from functools import partial

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from app.agents.bare_react import AgentLimits
from app.config import settings
from app.context.budget import ContextBudget
from app.core.llm import get_chat_model
from app.graph.agent import build_react_agent
from app.graph.context_node import (
    make_prepare_context_node,
    route_after_prepare_context,
)
from app.graph.intent import classify_intent_text
from app.graph.nodes import (
    EvidenceAssessor,
    ExpansionService,
    IntentClassifier,
    MultiRetrievalService,
    RetrievalService,
    UnderstandingService,
    build_order_options,
    chitchat_response,
    complaint_response,
    ensure_refund_option,
    fallback_response,
    load_order,
    make_classify_node,
    make_expand_node,
    make_gate_node,
    make_human_transfer_node,
    make_log_node,
    make_policy_retrieve_node,
    make_prepare_tools_node,
    make_ticket_gate_node,
    make_retrieve_node,
    make_understanding_node,
    resolve_reference,
    route_after_classify,
    route_after_gate,
    route_after_load_order,
    route_after_order_options,
    route_after_ticket_gate,
    select_order,
)
from app.graph.state import ChatState
from app.services.conversation_understanding import understand_query
from app.services.rag_service import (
    assess_evidence,
    assess_policy_evidence,
    retrieve_knowledge_evidence,
    retrieve_multi_query,
)
from app.services.retrieval_queries import expand_retrieval_queries
from app.services.turn_logger import TurnLogger
from app.tools.executor import ToolExecutor
from app.tools.registry import (
    ToolRegistry,
    tool_registry as global_tool_registry,
)
from app.tools.ticket_gate import detect_ticket_request


async def _noop_refresh(registry):
    return []


def build_chat_graph(
    *,
    understanding_model: BaseChatModel | None = None,
    intent_model: BaseChatModel | None = None,
    expansion_model: BaseChatModel | None = None,
    agent_model: BaseChatModel | None = None,
    intent_low_cost_model: BaseChatModel | None = None,
    checkpointer,
    understanding_service: UnderstandingService = understand_query,
    intent_classifier: IntentClassifier = classify_intent_text,
    expansion_service: ExpansionService = expand_retrieval_queries,
    retrieval_service: RetrievalService = retrieve_knowledge_evidence,
    multi_retrieval_service: MultiRetrievalService = retrieve_multi_query,
    evidence_assessor: EvidenceAssessor = assess_evidence,
    policy_evidence_assessor: EvidenceAssessor = assess_policy_evidence,
    agent_builder=build_react_agent,
    turn_logger: TurnLogger | None = None,
    limits: AgentLimits | None = None,
    intent_use_low_cost_first: bool | None = None,
    intent_confidence_threshold: float | None = None,
    context_manager=None,
    context_budget: ContextBudget | None = None,
    tool_registry: ToolRegistry | None = None,
    tool_refresh_service=None,
    tool_executor: ToolExecutor | None = None,
    ticket_gate_model: BaseChatModel | None = None,
    ticket_gate_service=detect_ticket_request,
):
    intent_model = intent_model or get_chat_model(streaming=False)
    expansion_model = expansion_model or get_chat_model(
        streaming=False,
        model=settings.query_expansion_model,
    )
    agent_model = agent_model or get_chat_model(streaming=True)

    # 如果没传低成本意图模型，读取配置文件实例化
    if intent_low_cost_model is None and settings.intent_low_cost_model:
        intent_low_cost_model = get_chat_model(
            streaming=False,
            model=settings.intent_low_cost_model,
        )

    # 小模型优先开关：函数入参优先级 > 全局settings配置
    use_low_cost_first = (
        settings.intent_use_low_cost_first
        if intent_use_low_cost_first is None
        else intent_use_low_cost_first
    )

    # 意图置信阈值：函数入参优先级 > 全局settings配置
    confidence_threshold = (
        settings.intent_confidence_threshold
        if intent_confidence_threshold is None
        else intent_confidence_threshold
    )

    # 如果使用默认的classify_intent_text，用partial预绑定低成本模型、阈值参数
    if intent_classifier is classify_intent_text:
        intent_classifier = partial(
            classify_intent_text,
            low_cost_model=intent_low_cost_model,
            use_low_cost_first=use_low_cost_first,
            confidence_threshold=confidence_threshold,
        )

    turn_logger = turn_logger or TurnLogger()
    context_budget = context_budget or ContextBudget.from_settings()
    limits = limits or AgentLimits(
        max_steps=settings.agent_max_steps,
        max_output_tokens=settings.agent_max_output_tokens,
        token_budget=settings.agent_token_budget,
        full_prompt_token_budget=(
            context_budget.model_context_window
            - context_budget.max_output_tokens
            - context_budget.safety_margin_tokens
        ),
        tool_call_message_max_tokens=(
            context_budget.tool_call_message_max_tokens
        ),
        tool_result_max_tokens=context_budget.tool_result_max_tokens,
        history_soft_budget=context_budget.soft_history_budget,
        history_hard_budget=context_budget.hard_history_budget,
    )

    registry = tool_registry or global_tool_registry
    refresh_service = tool_refresh_service or _noop_refresh
    agent_kwargs = {
        "registry": registry,
        "checkpointer": checkpointer,
    }
    if tool_executor is not None:
        agent_kwargs["tool_runner"] = tool_executor.execute
    try:
        agent = agent_builder(
            agent_model,
            limits,
            **agent_kwargs,
        )
    except TypeError as exc:
        if "unexpected keyword" not in str(exc):
            raise
        agent = agent_builder(agent_model, limits)


    graph = StateGraph(ChatState)
    graph.add_node(
        "prepare_context",
        make_prepare_context_node(context_manager),
    )
    understanding_node = (
        resolve_reference
        if understanding_model is None
        else make_understanding_node(understanding_model, understanding_service)
    )
    graph.add_node("understand_query", understanding_node)
    graph.add_node(
        "classify_intent",
        make_classify_node(intent_model, intent_classifier),
    )
    graph.add_node(
        "retrieve_knowledge",
        make_retrieve_node(retrieval_service),
    )
    graph.add_node(
        "confidence_gate",
        make_gate_node(evidence_assessor, policy_evidence_assessor),
    )
    graph.add_node("build_order_options", build_order_options)
    graph.add_node("select_order", select_order)
    graph.add_node("load_order", load_order)
    graph.add_node(
        "expand_queries",
        make_expand_node(expansion_model, expansion_service),
    )
    graph.add_node(
        "retrieve_policy",
        make_policy_retrieve_node(multi_retrieval_service),
    )
    graph.add_node("agent", agent)
    graph.add_node("complaint_response", complaint_response)
    graph.add_node("chitchat_response", chitchat_response)
    graph.add_node("fallback", fallback_response)
    graph.add_node("ensure_refund_option", ensure_refund_option)
    graph.add_node(
        "prepare_tools",
        make_prepare_tools_node(registry, refresh_service),
    )
    graph.add_node(
        "ticket_request_gate",
        make_ticket_gate_node(ticket_gate_model, ticket_gate_service),
    )
    graph.add_node(
        "human_transfer_request",
        make_human_transfer_node(tool_executor),
    )
    graph.add_node("log_turn", make_log_node(turn_logger))
#=========================================================================
#---------------------------添加边-------------------------------------------
    graph.add_edge(START, "prepare_context")
    graph.add_conditional_edges(
        "prepare_context",
        route_after_prepare_context,
        {
            "understand": "understand_query",
            "transfer_issue": "human_transfer_request",
            "fallback": "fallback",
        },
    )
    graph.add_edge("understand_query", "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {
            "knowledge": "retrieve_knowledge",
            "business": "ticket_request_gate",
            "core_after_sales": "build_order_options",
            "complaint": "ticket_request_gate",
            "chitchat": "chitchat_response",
            "human_transfer": "human_transfer_request",
            "other": "ticket_request_gate",
        },
    )
    graph.add_conditional_edges(
        "ticket_request_gate",
        route_after_ticket_gate,
        {
            "agent": "prepare_tools",
            "order": "build_order_options",
            "complaint": "complaint_response",
            "fallback": "fallback",
        },
    )
    graph.add_conditional_edges(
        "build_order_options",
        route_after_order_options,
        {
            "load_order": "load_order",
            "select_order": "select_order",
        },
    )
    graph.add_edge("select_order", "load_order")
    graph.add_conditional_edges(
        "load_order",
        route_after_load_order,
        {
            "expand_queries": "expand_queries",
            "agent": "prepare_tools",
            "fallback": "fallback",
        },
    )
    graph.add_edge("expand_queries", "retrieve_policy")
    graph.add_edge("retrieve_policy", "confidence_gate")
    graph.add_edge("retrieve_knowledge", "confidence_gate")
    graph.add_conditional_edges(
        "confidence_gate",
        route_after_gate,
        {"agent": "prepare_tools", "fallback": "fallback"},
    )
    graph.add_edge("prepare_tools", "agent")
    graph.add_edge("agent", "ensure_refund_option")
    graph.add_edge("ensure_refund_option", "log_turn")
    graph.add_edge("complaint_response", "log_turn")
    graph.add_edge("chitchat_response", "log_turn")
    graph.add_edge("human_transfer_request", "log_turn")
    graph.add_edge("fallback", "log_turn")
    graph.add_edge("log_turn", END)
    return graph.compile(checkpointer=checkpointer)
