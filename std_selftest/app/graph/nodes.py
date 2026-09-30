"""
    定义graph中的节点
"""

import re
from collections.abc import Awaitable, Callable, Sequence
from datetime import date
from inspect import Parameter, signature
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.config import get_stream_writer
from langgraph.types import interrupt

from app.config import settings
from app.context.assembler import build_history_messages
from app.graph.intent import (
    INTENT_TO_ROUTE,
    IntentClassificationError,
    IntentDecision,
    is_after_sales_mcp_query,
    other_intent,
)
from app.graph.state import ChatState
from app.services.conversation_understanding import (
    QueryUnderstandingResult,
    understand_query,
)
from app.services.order_catalog import get_demo_order, list_demo_orders
from app.services.retrieval_queries import (
    QueryExpansion,
    expand_retrieval_queries,
)
from app.services.turn_logger import TurnLogger
from app.tools.executor import get_default_executor
from app.tools.models import (
    TicketSlots,
    ToolCaller,
    ToolExecutionContext,
)
from app.tools.ticket_gate import (
    TicketGateDecision,
    detect_ticket_request,
)
from app.tools.transfer_gate import (
    extract_transfer_issue,
    is_human_transfer_cancel,
    is_human_transfer_request,
    latest_problem_from_messages,
)


COMPLAINT_REPLY = (
    "很抱歉给您带来不好的体验。"
    "您可以转人工继续反馈，由人工客服协助处理。"
)
CHITCHAT_REPLY = "您好，我是 iHelp 智能客服，很高兴为您服务。"
FALLBACK_REPLY = "抱歉，我暂时无法可靠回答这个问题。"
OTHER_REPLY = "抱歉，我暂时无法判断您的具体需求，请补充更明确的购物或售后问题。"
COMPLAINT_OPTIONS = [
    {"id": "human_transfer", "label": "转人工"},
]
ORDER_ID_PATTERN = re.compile(r"(?<!\d)(\d{4,})(?!\d)")
AFTER_SALES_SLOT_MARKERS = (
    "没拆",
    "未拆",
    "拆封",
    "已使用",
    "使用过",
    "破损",
    "少件",
    "错件",
    "质量问题",
    "尺码",
    "大小",
    "颜色",
    "型号",
)

IntentClassifier = Callable[[str, Any], Awaitable[IntentDecision | str]]
UnderstandingService = Callable[
    [Sequence[BaseMessage], str, Any],
    Awaitable[QueryUnderstandingResult],
]
ExpansionService = Callable[[str, Any], Awaitable[QueryExpansion]]
RetrievalService = Callable[[str], Awaitable[dict]]
MultiRetrievalService = Callable[[list[str]], Awaitable[dict]]
EvidenceAssessor = Callable[[str, list[dict], str], Awaitable[dict]]


def emit_node(node: str, status: str, **detail: Any) -> None:
    get_stream_writer()(
        {"type": "node_status", "node": node, "status": status, "detail": detail}
    )


def extract_order_id(text: str) -> str:
    match = ORDER_ID_PATTERN.search(text or "")
    return match.group(1) if match else ""


def _normalize_order_id(order_id: str) -> str:
    """去掉字符串左侧所有连续的 `0`"""
    return order_id.lstrip("0") or "0"


def is_after_sales_slot_answer(message: str) -> bool:
    """
        判断用户当前输入是否属于【售后槽位补充回答】
        场景：多轮售后填信息环节，用户回复用来补充订单号、确认售后信息，不是全新的问题
    """
    text = message.strip()
    if not text:
        return False
    return bool(extract_order_id(text)) or any(
        marker in text for marker in AFTER_SALES_SLOT_MARKERS
    )


def infer_after_sales_action(text: str, fallback: str = "") -> str:
    """
        根据用户query关键词，推断售后具体动作类型
    """
    if any(marker in text for marker in ("换货", "换新", "我要换", "想换")):
        return "exchange"
    if any(marker in text for marker in ("退货", "退款", "退钱", "我要退", "想退")):
        return "refund"
    if any(marker in text for marker in ("维修", "修理", "修一下", "送修")):
        return "repair"
    return fallback


def resolve_reference(state: ChatState) -> dict[str, Any]:
    """
        查询解析兜底节点
        调用 LLM 做指代消解和 Query 改写,llm为空时，调用该兜底函数
        即，resolved_query=原来未处理的query
    """
    message = state.get("user_message") or ""
    if not message:     # 如果user_message为空：倒序遍历messages消息列表，找到最后一条用户消息HumanMessage
        for item in reversed(state.get("messages") or []):
            if isinstance(item, HumanMessage):
                message = str(item.content)
                break
    return {
        "user_message": message,
        "resolved_query": message,
        "query_history": [],
        "query_changed": False,
        "query_understanding_source": "fallback",
    }


def _reset_turn_update() -> dict[str, Any]:
    """
    生成【本轮对话临时状态重置字典】
    作用：每一轮用户新消息进来时，清空上一轮产生的临时业务状态，避免旧数据污染本轮图谱执行。
    返回的字典会和其他update字段合并，更新到ChatState
    """
    return {
        "agent_messages": [],          # agent工具调用消息列表，清空
        "intent": "",                  # 上一轮识别的意图，重置为空
        "intent_confidence": 0.0,      # 意图置信度，重置为0
        "intent_model_used": "",       # 识别意图使用的模型标记
        "route": "",                   # 路由分支（knowledge/business/complaint等）重置
        "evidence": [],                # 知识库召回的文档片段，清空
        "citations": [],               # 引用文献列表，清空
        "sufficient": False,           # 信息是否充足标记，重置为False
        "refusal": "",                 # 拒绝回答话术，清空
        "reply": "",                   # 上一轮最终回复文本
        "reply_options": [],           # 回复附带的前端按钮选项
        "agent_steps": 0,              # agent执行步骤计数归零
        "tool_trace": [],              # 工具调用追踪日志清空
        "route_reason": "",            # 路由选择原因
        "stop_reason": "",             # 图谱停止执行原因
        "error": "",                   # 上一轮的错误信息清空
        "turn_log": {},                # 本轮日志对象
        "resolved_query": "",          # 指代消解后的本轮 query
        "query_history": [],           # 指代消解使用的历史消息
        "order_id": "",                # 订单ID临时字段清空
        "order_options": [],           # 订单候选列表
        "order_data": {},              # 订单详情数据
        "expanded_queries": [],        # 扩展检索query列表
        "retrieval_query_count": 0,    # 检索调用次数归零
        "pending_action": "",          # 待确认中断动作（LangGraph interrupt用）
        "after_sales_action": "",      # 售后动作标记
        "preferred_intent": "",        # 优先继承意图（售后多轮上下文）重置为空
    }



def make_understanding_node(
    model: Any,
    understanding_service: UnderstandingService = understand_query,
):
    """
        调用 LLM 做指代消解,即指代的 Query 改写
    """
    async def understanding_node(state: ChatState) -> dict[str, Any]:
        emit_node("understand_query", "running")

        message = state.get("user_message") or ""       # 用户本轮原始消息
        last_intent = state.get("last_intent", "")      # 上一轮识别出的意图
        last_order_id = state.get("last_order_id", "")  # 上一轮的订单号（上下文记忆）
        last_after_sales_action = state.get("last_after_sales_action", "")  # 上一轮售后动作
        pack = state.get("context_pack")
        messages = (
            build_history_messages(pack)
            if pack is not None
            else list(state.get("messages") or [])
        )
        if (
            messages
            and isinstance(messages[-1], HumanMessage)
            and str(messages[-1].content) == message
        ):
            history = messages[:-1]     # 处理历史消息：如果messages最后一条刚好等于当前用户message，则把最后一条剔除，剩下的作为历史
        else:
            history = messages

        try:
            result = await understanding_service(history, message, model)
        except Exception as exc:
            result = QueryUnderstandingResult(message, False, "fallback")
            error = str(exc)
        else:
            error = ""

        get_stream_writer()(    #  SSE流式推送事件给前端
            {
                "type": "query_understood",
                "original": message,
                "resolved_query": result.resolved_query,
                "changed": result.changed,
                "source": result.source,
            }
        )
        emit_node("understand_query","success",changed=result.changed,source=result.source,)

        update = {
            **_reset_turn_update(),
            "user_message": message,
            "resolved_query": result.resolved_query,
            "query_history": history,
            "query_changed": result.changed,
            "query_understanding_source": result.source,
        }
        # ========== 售后场景特殊上下文继承逻辑 ==========
        if last_intent == "售后" and is_after_sales_slot_answer(message):     # 如果上一轮意图是【售后】，并且当前用户消息属于售后槽位回答（比如确认信息、补充材料）
            update["preferred_intent"] = last_intent                         # 优先沿用上次的售后意图，不重新走意图分类
            update["after_sales_action"] = last_after_sales_action             # 继承上一轮售后动作
            explicit_order_id = extract_order_id(result.resolved_query) or extract_order_id(    # 尝试从改写后的query / 原始消息提取订单号
                message
            )
            context_order_id = ""
            if explicit_order_id:
                normalized = _normalize_order_id(explicit_order_id)
                try:
                    get_demo_order(normalized)
                except KeyError:
                    context_order_id = last_order_id
                else:
                    context_order_id = normalized
            else:
                context_order_id = last_order_id
            if context_order_id:
                update["order_id"] = context_order_id
        if error:
            update["error"] = error
        return update

    return understanding_node


def make_classify_node(
    model: Any,
    classifier: IntentClassifier,
):
    """
    `ake_classify_node` 是工厂函数，生成 LangGraph 意图分类节点，
    调用分类器识别用户查询，支持模型识别为其他时继承 state 中的售后上下文覆盖意图，推断售后动作，流式推送识别结果并埋点，最终组装增量状态返回给图路由。
    """
    async def classify_node(state: ChatState) -> dict[str, Any]:
        emit_node("classify_intent", "running")

        try:
            history = state.get("query_history") or []
            # 【动态参数检测】 反射检查classifier函数签名，判断分类器是否支持传入history参数
            parameters = signature(classifier).parameters.values()
            accepts_history = any(
                parameter.name == "history"
                or parameter.kind == Parameter.VAR_KEYWORD
                for parameter in parameters
            )
            if accepts_history:
                decision = await classifier(
                    state["resolved_query"],
                    model,
                    history=history,
                )
            else:
                decision = await classifier(state["resolved_query"], model)
            if isinstance(decision, str):
                if decision not in INTENT_TO_ROUTE:
                    raise IntentClassificationError(f"未知意图: {decision}")
                decision = IntentDecision(
                    intent=decision,
                    route=INTENT_TO_ROUTE[decision],
                    confidence=1.0,
                    model_used="injected",
                )
        except Exception as exc:
            emit_node("classify_intent", "error", message=str(exc))
            decision = other_intent()
            state_error = str(exc)
        else:
            state_error = ""

        # 如果分类得到的意图是“其他”，并且state存在preferred_intent（售后继承意图）
        # 就覆盖本次分类结果，改用preferred_intent，标记来源 context_continuation
        if decision.intent == "其他" and state.get("preferred_intent"):
            preferred_intent = state["preferred_intent"]
            decision = IntentDecision(
                intent=preferred_intent,
                route=INTENT_TO_ROUTE[preferred_intent],
                confidence=0.7,
                model_used="context_continuation",
            )

        # 如果当前意图是售后/退款退货，推断售后子动作
        after_sales_action = ""
        if (
            decision.intent in {"退款退货", "售后"}
            and not is_after_sales_mcp_query(state["resolved_query"])
        ):
            after_sales_action = infer_after_sales_action(
                state["resolved_query"],
                state.get("after_sales_action", ""),
            )

        writer = get_stream_writer()
        writer(
            {
                "type": "intent",
                "intent": decision.intent,
                "confidence": decision.confidence,
                "model_used": decision.model_used,
                "route": decision.route,
            }
        )
        emit_node(
            "classify_intent",
            "success",
            intent=decision.intent,
            route=decision.route,
            confidence=decision.confidence,
        )
        update = {
            "intent": decision.intent,
            "route": decision.route,
            "route_reason": decision.reason,
            "intent_confidence": decision.confidence,
            "intent_model_used": decision.model_used,
        }
        if after_sales_action:
            update["after_sales_action"] = after_sales_action
        if state_error:
            update["error"] = state_error
        return update

    return classify_node


def route_after_classify(state: ChatState) -> str:
    """
        意图分类后的条件路由函数，读取state里的route，做合法性校验，非法值兜底返回other
    """
    route = state.get("route", "other")
    valid_routes = {
        "knowledge",
        "business",
        "core_after_sales",
        "complaint",
        "chitchat",
        "human_transfer",
        "other",
    }
    return route if route in valid_routes else "other"


def make_retrieve_node(retrieval_service: RetrievalService):
    async def retrieve_node(state: ChatState) -> dict[str, Any]:
        emit_node("retrieve_knowledge", "running")
        try:
            result = await retrieval_service(state["resolved_query"])
        except Exception as exc:
            emit_node("retrieve_knowledge", "error", message=str(exc))
            return {"evidence": [], "citations": [], "error": str(exc)}

        evidence = result.get("evidence") or []
        citations = _citation_payload(evidence)
        emit_node("retrieve_knowledge", "success", count=len(evidence))
        if citations:
            get_stream_writer()({"type": "citations", "citations": citations})
        return {"evidence": evidence, "citations": citations}

    return retrieve_node


def _citation_payload(evidence: list[dict]) -> list[dict]:
    """
        将检索召回的evidence证据列表，转换为前端约定的引用（citation）事件结构
    """
    return [
        {
            "id": item.get("citation_id"),
            "chunk_id": item.get("chunk_id"),
            "section_path": item.get("section_path", ""),
            "text": item.get("text", ""),
        }
        for item in evidence
    ]


def build_order_options(state: ChatState) -> dict[str, Any]:
    """
        售后子流程节点：尝试从state或用户query提取订单号；
        提取成功则直接带入order_id，无订单号则返回订单列表让用户选择
    """
    emit_node("build_order_options", "running")
    order_id = state.get("order_id") or extract_order_id(state.get("resolved_query", "")
    )
    if order_id:
        emit_node("build_order_options", "success", order_id=order_id, source="query")
        return {"order_id": order_id, "order_options": [], "pending_action": ""}

    orders = list_demo_orders()
    get_stream_writer()({"type": "order_selector", "orders": orders})
    emit_node("build_order_options", "success", count=len(orders))
    return {
        "order_id": "",
        "order_options": orders,
        "pending_action": "order_selection",
    }


def route_after_order_options(state: ChatState) -> str:
    """
        判断是加载订单 还是 选择订单
    """
    return "load_order" if state.get("order_id") else "select_order"


def route_after_load_order(state: ChatState) -> str:
    if not state.get("order_data"):
        return "fallback"
    if state.get("route") == "business":
        return "agent"
    return "expand_queries"


def select_order(state: ChatState) -> dict[str, Any]:
    """
        售后子流程节点：弹出人工选择订单中断，校验用户选中的订单ID合法后，写入order_id到状态
    """
    options = state.get("order_options") or []  # 获取订单候选列表
    offered_ids = {str(order.get("order_id")) for order in options}

    while True:
        selection = interrupt(
            {
                "type": "order_selection",
                "orders": options,
            }
        )
        if (
            isinstance(selection, dict)
            and selection.get("type") == "order_selected"
            and str(selection.get("order_id") or "") in offered_ids
        ):
            order_id = str(selection["order_id"])
            emit_node("select_order", "success", order_id=order_id)
            return {
                "order_id": order_id,
                "pending_action": "",
            }


def load_order(state: ChatState) -> dict[str, Any]:
    """
        售后子流程节点：根据state中的order_id加载订单详情，计算签收天数，流式推送订单信息；加载失败返回兜底提示
    """
    emit_node("load_order", "running")
    try:
        order = get_demo_order(state["order_id"])
        evaluation_date = date.today()
        order["evaluation_date"] = evaluation_date.isoformat()
        if order.get("signed_at"):
            signed_date = date.fromisoformat(order["signed_at"])
            order["days_since_signed"] = (evaluation_date - signed_date).days
        else:
            order["days_since_signed"] = None
    except Exception as exc:
        emit_node("load_order", "error", message=str(exc))
        return {
            "order_data": {},
            "refusal": "抱歉，没有找到这个订单，请重新选择。",
            "error": str(exc),
        }

    get_stream_writer()(
        {
            "type": "order_selected",
            "order_id": order["order_id"],
            "order": order,
        }
    )
    if state.get("after_sales_action") == "exchange":
        get_stream_writer()(
            {
                "type": "exchange_form",
                "order_id": order["order_id"],
                "reasons": settings.exchange_reason_list,
            }
        )
    emit_node("load_order", "success", order_id=order["order_id"])
    return {"order_data": order, "pending_action": ""}


def make_expand_node(
    model: Any,
    expansion_service: ExpansionService = expand_retrieval_queries,
):
    """
        生成LangGraph查询扩展节点expand_node
    """
    async def expand_node(state: ChatState) -> dict[str, Any]:
        emit_node("expand_queries", "running")
        try:
            result = await expansion_service(state["resolved_query"], model)
        except Exception as exc:
            result = QueryExpansion([state["resolved_query"]], "fallback")
            error = str(exc)
        else:
            error = ""

        get_stream_writer()(        # 向前端推送流式事件：扩展后的查询列表
            {
                "type": "expanded_queries",
                "queries": result.queries,
                "count": len(result.queries),
            }
        )
        emit_node("expand_queries", "success", count=len(result.queries))
        update = {
            "expanded_queries": result.queries,
            "retrieval_query_count": len(result.queries),
        }
        if error:
            update["error"] = error
        return update

    return expand_node


def make_policy_retrieve_node(
    multi_retrieval_service: MultiRetrievalService,
):
    """
        生成售后政策多查询检索节点 retrieve_policy_node
    """
    async def retrieve_policy_node(state: ChatState) -> dict[str, Any]:
        emit_node("retrieve_policy", "running")
        queries = state.get("expanded_queries") or [state["resolved_query"]]
        try:
            result = await multi_retrieval_service(queries)
        except Exception as exc:
            emit_node("retrieve_policy", "error", message=str(exc))
            return {
                "evidence": [],
                "citations": [],
                "error": str(exc),
            }

        evidence = result.get("evidence") or []
        citations = _citation_payload(evidence)
        update = {
            "evidence": evidence,
            "citations": citations,
            "retrieval_query_count": result.get(
                "retrieval_query_count",
                len(queries),
            ),
        }
        if result.get("failed_queries"):
            update["error"] = (
                "部分扩写检索失败: " + ",".join(result["failed_queries"])
            )
        emit_node("retrieve_policy", "success", count=len(evidence))
        if citations:
            get_stream_writer()({"type": "citations", "citations": citations})
        return update

    return retrieve_policy_node


def make_gate_node(
    evidence_assessor: EvidenceAssessor,
    policy_evidence_assessor: EvidenceAssessor | None = None,
):
    """
        生成置信度闸门节点 gate_node
        根据当前路由自动选择评估器；评估证据质量，异常时兜底判定证据不足。
    """
    async def gate_node(state: ChatState) -> dict[str, Any]:
        emit_node("confidence_gate", "running")

        assessor = evidence_assessor    # 默认使用通用证据评估器
        if state.get("route") == "core_after_sales" and policy_evidence_assessor:   # 如果当前路由是核心售后流程，并且传入了售后专用评估器，则切换为售后评估器
            assessor = policy_evidence_assessor
        try:
            result = await assessor(
                state["resolved_query"],
                state.get("evidence") or [],
                state["session_id"],
            )
        except Exception as exc:
            emit_node("confidence_gate", "error", message=str(exc))
            return {
                "sufficient": False,
                "refusal": FALLBACK_REPLY,
                "error": str(exc),
            }

        sufficient = bool(result.get("sufficient"))
        emit_node("confidence_gate", "success", sufficient=sufficient)
        return {
            "sufficient": sufficient,
            "refusal": result.get("refusal", ""),
        }

    return gate_node


def complaint_response(state: ChatState) -> dict[str, Any]:
    emit_node("retrieve_knowledge", "skipped", reason="complaint")
    emit_node("confidence_gate", "skipped", reason="complaint")
    writer = get_stream_writer()
    writer({"type": "delta", "text": COMPLAINT_REPLY})
    writer({"type": "reply_options", "options": COMPLAINT_OPTIONS})
    return {"reply": COMPLAINT_REPLY, "reply_options": COMPLAINT_OPTIONS}


def chitchat_response(state: ChatState) -> dict[str, Any]:
    emit_node("retrieve_knowledge", "skipped", reason="chitchat")
    emit_node("confidence_gate", "skipped", reason="chitchat")
    get_stream_writer()({"type": "delta", "text": CHITCHAT_REPLY})
    return {"reply": CHITCHAT_REPLY, "reply_options": []}


def fallback_response(state: ChatState) -> dict[str, Any]:
    """
        兜底分支节点：返回兜底回复文本（抱歉，无法恢复），向前端流式推送回答
    """
    emit_node("fallback", "success")
    default_reply = OTHER_REPLY if state.get("route") == "other" else FALLBACK_REPLY
    reply = (
        state.get("context_budget_error")
        or state.get("refusal")
        or default_reply
    )
    get_stream_writer()({"type": "delta", "text": reply})
    return {"reply": reply, "reply_options": []}


def route_after_gate(state: ChatState) -> str:
    return "agent" if state.get("sufficient") else "fallback"


def ensure_refund_option(state: ChatState) -> dict[str, Any]:
    """
        在核心售后判断完成后，保证对话回复带上【申请退款】按钮入口
    """
    if (
        state.get("intent") != "退款退货"
        or state.get("route") != "core_after_sales"
        or not state.get("sufficient")
    ):
        return {}

    options = list(state.get("reply_options") or [])    # 获取当前已有的回复按钮选项列表
    if not any(option.get("id") == "apply_refund" for option in options):   # 判断列表里是否已经存在 id = apply_refund 的按钮
        options.append({"id": "apply_refund", "label": "申请退款"})     # 追加申请退款按钮
        get_stream_writer()({"type": "reply_options", "options": options})
    return {"reply_options": options}


def make_prepare_tools_node(registry, refresh_service):
    """
        工厂函数，生成LangGraph节点：准备本轮Agent可用工具
        作用：刷新MCP工具目录 + 筛选出**本轮对话允许Agent看到、可以调用的工具列表**
    """
    async def prepare_tools(state: ChatState) -> dict[str, Any]:
        emit_node("prepare_tools", "running")
        errors = await refresh_service(registry)
        configured = state.get("available_tool_names")
        candidate_names = (
            list(configured)
            if configured is not None
            else [item.name for item in registry.list()]
        )
        slots = TicketSlots(
            **(
                state.get("ticket_slots")
                or {
                    "ticket_type": "",
                    "description": "",
                    "missing_fields": (),
                }
            )
        )
        names = []
        for name in candidate_names:
            definition = registry.get(name)
            if definition is None or not definition.agent_visible:
                continue
            if (
                name == "create_ticket"
                and not (
                    state.get("ticket_request_allowed", False)
                    and slots.complete
                )
            ):
                continue
            names.append(name)
        emit_node("prepare_tools", "success", count=len(names))
        return {
            "available_tool_names": names,
            "tool_catalog_version": registry.catalog_version,
            "tool_catalog_errors": errors,
        }

    return prepare_tools


def make_ticket_gate_node(
    ticket_gate_model,
    ticket_gate_service=detect_ticket_request,
):
    async def ticket_gate(state: ChatState) -> dict[str, Any]:
        """
            工单网关节点：判断用户对话里是否**明确提出建工单/转人工诉求**
            调用大模型，识别工单类型、问题描述、缺失字段；输出标记给后续工具执行器做权限校验
            输出状态更新：ticket_request_allowed 和 ticket_slots（工单槽位）
        """
        emit_node("ticket_request_gate", "running")
        if ticket_gate_model is None:
            decision = TicketGateDecision(explicit_request=False)
        else:
            try:
                decision = await ticket_gate_service(
                    state.get("resolved_query")
                    or state.get("user_message", ""),
                    state.get("query_history")
                    or state.get("messages")
                    or [],
                    ticket_gate_model,
                )
            except Exception as exc:
                emit_node(
                    "ticket_request_gate",
                    "error",
                    message=str(exc),
                )
                decision = TicketGateDecision(explicit_request=False)
        emit_node(
            "ticket_request_gate",
            "success",
            explicit_request=decision.explicit_request,
        )
        return {
            "ticket_request_allowed": decision.explicit_request,
            "ticket_slots": {
                "ticket_type": decision.ticket_type,
                "description": decision.description,
                "missing_fields": list(decision.missing_fields),
            },
        }

    return ticket_gate


def route_after_ticket_gate(state: ChatState) -> str:
    if state.get("ticket_request_allowed"):
        return "agent"
    if state.get("route") == "complaint":
        return "complaint"
    if state.get("route") == "other":
        return "fallback"
    if state.get("route") == "business":
        return "order"
    return "agent"


def make_human_transfer_node(tool_executor=None):
    """
        工厂函数：构造 LangGraph 节点函数【转人工节点】
        传入可选的 tool_executor，用于依赖注入
    """
    async def human_transfer_request(state: ChatState) -> dict[str, Any]:
        emit_node("human_transfer_request", "running")
        writer = get_stream_writer()
        current = (state.get("user_message") or "").strip()
        awaiting_issue = state.get("pending_action") == "await_transfer_issue"  # 判断状态标记：当前是否处于【等待用户补充转人工问题描述】的状态
        turn_reset = _reset_turn_update() if awaiting_issue else {} # 如果正在等待用户填写问题，则重置轮次状态；否则返回空字典，不修改轮次

        if is_human_transfer_cancel(current):
            reply = (
                "已取消转人工。"
                if awaiting_issue
                else "当前没有待取消的转人工请求。"
            )
            writer({"type": "delta", "text": reply})
            emit_node("human_transfer_request", "success", cancelled=True)
            return {
                **turn_reset,
                "intent": "转人工",
                "route": "human_transfer",
                "reply": reply,
                "reply_options": [],
                "pending_action": "",
            }

        if awaiting_issue and is_human_transfer_request(current):
            issue = extract_transfer_issue(current)
        else:
            issue = current if awaiting_issue else extract_transfer_issue(current)
        if not issue:
            issue = latest_problem_from_messages(
                state.get("messages") or [],
                current,
            )

        if not issue:
            reply = (
                "请问您遇到了什么问题？"
                "请说明具体情况，我会据此创建工单并转接人工客服。"
            )
            writer({"type": "delta", "text": reply})
            emit_node("human_transfer_request", "success", missing="issue")
            return {
                **turn_reset,
                "intent": "转人工",
                "route": "human_transfer",
                "reply": reply,
                "reply_options": [],
                "pending_action": "await_transfer_issue",
            }

        executor = tool_executor or get_default_executor()
        result = await executor.execute(
            "create_ticket",
            {
                "conversation_id": state["session_id"],
                "description": issue,
                "ticket_type": "转人工",
            },
            ToolExecutionContext(
                conversation_id=state["session_id"],
                tool_call_id=(
                    f"human-transfer-{state['session_id']}-"
                    f"{state.get('current_user_message_id') or 0}"
                ),
                caller=ToolCaller.AGENT,
                ticket_request_allowed=True,
                ticket_slots=TicketSlots(
                    ticket_type="转人工",
                    description=issue,
                ),
                emit=writer,
            ),
        )

        if result.success:
            reply = (
                "已介入客服。"
                f"您反馈的问题是：{issue}。现在为您解决。"
            )
        elif result.status == "permission_denied":
            reply = "已取消转人工。"
        else:
            reply = f"转人工失败：{result.error}"

        writer({"type": "delta", "text": reply})
        emit_node(
            "human_transfer_request",
            "success",
            result_status=result.status,
        )
        return {
            **turn_reset,
            "intent": "转人工",
            "route": "human_transfer",
            "reply": reply,
            "reply_options": [],
            "pending_action": "",
        }

    return human_transfer_request


def make_log_node(turn_logger: TurnLogger):
    async def log_turn_node(state: ChatState) -> dict[str, Any]:
        emit_node("log_turn", "running")
        reply = state.get("reply") or FALLBACK_REPLY
        turn_log = await turn_logger.log(state["session_id"], {**state, "reply": reply})
        emit_node("log_turn", "success")
        return {
            "messages": [AIMessage(content=reply)],
            "turn_log": turn_log,
            "last_intent": state.get("intent", ""),
            "last_order_id": state.get("order_id", ""),
            "last_after_sales_action": state.get("after_sales_action", ""),
        }

    return log_turn_node
