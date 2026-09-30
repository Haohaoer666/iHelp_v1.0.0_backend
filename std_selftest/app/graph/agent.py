"""
    基于 LangGraph 搭建了一个带 RAG 知识库、流式输出、工具调用步数与 Token 预算保护的客服 ReAct 智能体
"""

import json
import re
from inspect import signature
from typing import Any, Awaitable, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from app.agents.bare_react import AgentLimits
from app.config import settings
from app.context.assembler import build_model_messages
from app.context.logging import log_model_ctx
from app.context.models import ContextPack, PromptUsage
from app.context.tokens import (
    count_prompt_tokens,
    estimate_message_tokens,
    estimate_messages_tokens,
    estimate_text_tokens,
    truncate_text_to_tokens,
    truncate_tool_content,
)
from app.core.memory import trim_history
from app.core.prompts import CUSTOMER_SERVICE_SYSTEM
from app.graph.state import ChatState
from app.tools.executor import get_default_executor
from app.tools.models import (
    TicketSlots,
    ToolCaller,
    ToolExecutionContext,
    ToolExecutionResult,
)
from app.tools.registry import ToolRegistry, tool_registry


ToolRunner = Callable[
    [str, dict[str, Any], ToolExecutionContext],
    Awaitable[ToolExecutionResult],
]
SUGGESTION_PATTERN = re.compile(r"\[\[suggest:([^\]]+)\]\]")
LIMIT_REPLY = (
    "我已经查询到部分信息，但还需要更多信息才能继续确认。"
    "请补充订单号或具体问题。"
)


def split_suggestions(text: str) -> tuple[str, list[dict[str, str]]]:
    """
        解析模型输出文本，分离普通回答和前端快捷操作按钮配置
    """
    match = SUGGESTION_PATTERN.search(text)
    if not match:
        return text, []

    labels = {
        "human_transfer": "转人工",
        "create_ticket": "建工单",
        "apply_refund": "申请退款",
    }
    option_ids = [value for value in match.group(1).split("|") if value]
    options = [
        {"id": option_id, "label": labels[option_id]}
        for option_id in option_ids
        if option_id in labels
    ]
    return SUGGESTION_PATTERN.sub("", text).strip(), options


def _content_text(content: object) -> str:
    """将消息内容强制转为字符串，兼容非字符串类型"""
    return content if isinstance(content, str) else str(content)


def _tool_content(execution: ToolExecutionResult) -> str:
    return json.dumps(execution.model_payload(), ensure_ascii=False)


async def default_tool_runner(
    tool_name: str,
    tool_args: dict[str, Any],
    context: ToolExecutionContext,
) -> ToolExecutionResult:
    return await get_default_executor().execute(
        tool_name,
        tool_args,
        context,
    )


def _available_tool_names(
    state: ChatState,
    registry: ToolRegistry,
) -> list[str]:
    configured = state.get("available_tool_names")
    if configured is not None:
        return list(configured)
    return [
        definition.name
        for definition in registry.list()
        if definition.agent_visible
        and definition.name != "create_ticket"
    ]


def _build_evidence_system(state: ChatState) -> SystemMessage:
    """
        构建系统提示词消息
        将知识库检索出来的证据片段追加到系统prompt中，要求模型回答必须引用证据编号
    """
    lines = [CUSTOMER_SERVICE_SYSTEM]
    order_data = state.get("order_data") or {}
    if order_data:
        lines.append("\n\n本次订单事实：")
        lines.append(json.dumps(order_data, ensure_ascii=False))
        after_sales_action = state.get("after_sales_action", "")
        action_label = {
            "exchange": "换货",
            "refund": "退款退货",
            "repair": "维修售后",
        }.get(after_sales_action, "售后")
        lines.append(
            f"本次请求属于{action_label}核心子流程。"
            "只依据订单事实和下面的政策证据判断这一单是否符合条件，"
            "不得补造订单状态、签收日期、商品状态或政策条款。"
            "订单已由系统选中，禁止再次索要订单号；"
            "订单号、商品和签收信息必须直接使用本次订单事实。"
            "evaluation_date 和 days_since_signed 已按当前日期计算好，"
            "处理退货时效时必须直接使用，不要自行猜测当前日期。"
        )
        if after_sales_action == "refund":
            lines.append(
                "如果结论是符合退款条件，在回答末尾追加 "
                "[[suggest:apply_refund]]，供前端展示退款表单；"
                "不符合则不要追加该标记。"
            )
    evidence = state.get("evidence") or []
    if evidence:
        lines.append("\n\n本次知识库证据：")
        for item in evidence:
            lines.append(f"[{item['citation_id']}] {item['text']}")
        lines.append("回答知识问题时只使用上述证据，并保留引用编号。")
    return SystemMessage(content="\n".join(lines))


def _build_order_slot_message(state: ChatState) -> SystemMessage | None:
    order_data = state.get("order_data") or {}
    if not order_data:
        return None

    action_label = {
        "exchange": "换货",
        "refund": "退款退货",
        "repair": "维修售后",
    }.get(state.get("after_sales_action", ""), "售后")
    lines = [
        "【系统槽位已确认】",
        f"订单号：{order_data.get('order_id', '')}",
        f"商品：{order_data.get('product_name', '')}",
        f"订单状态：{order_data.get('status', '')}",
        f"签收日期：{order_data.get('signed_at') or '尚未签收'}",
        f"距签收天数：{order_data.get('days_since_signed')}",
        f"用户诉求：{action_label}",
        "该订单由用户通过订单选择器明确选择，不得再次询问订单号。",
        "直接基于上述槽位和本轮政策证据回答，不得要求用户重复提供订单信息。",
    ]
    return SystemMessage(content="\n".join(lines))


def _build_ticket_slot_message(state: ChatState) -> SystemMessage | None:
    if not state.get("ticket_request_allowed"):
        return None
    slots = TicketSlots(**(state.get("ticket_slots") or {}))
    lines = [
        "【建单槽位】",
        "用户已明确要求创建工单。",
        f"工单类型：{slots.ticket_type or '其他'}",
    ]
    if slots.missing_fields:
        lines.append(
            "当前缺少问题描述，必须先追问用户补充，"
            "本轮不得调用 create_ticket，也不得编造描述。"
        )
    else:
        lines.append(
            "槽位已完整；必须立即调用 create_ticket，"
            "不得再索要订单号或其他非必填信息。"
            "调用后必须等待前端确认，不得把调用本身当作已提交。"
        )
    return SystemMessage(content="\n".join(lines))


def _truncate_tool_call_message(
    message: AIMessage,
    max_tokens: int,
    chinese_tokens_per_char: float,
) -> AIMessage:
    if not message.tool_calls:
        return message
    truncated_calls = []
    for call in message.tool_calls:
        args_text = json.dumps(call.get("args") or {}, ensure_ascii=False)
        truncated_calls.append(
            {
                **call,
                "args": {
                    "_truncated": truncate_text_to_tokens(
                        args_text,
                        max_tokens,
                        chinese_tokens_per_char,
                    )
                },
            }
        )
    return message.model_copy(update={"tool_calls": truncated_calls})


def _build_prompt_usage(
    *,
    pack: ContextPack,
    messages: list[Any],
    evidence: list[dict],
    limits: AgentLimits,
    tools: list[BaseTool],
) -> PromptUsage:
    """
        统计整个Agent Prompt各组成部分的token用量，组装成PromptUsage结构体。
        用于：预算校验、埋点监控、日志上报、判断是否触达上下文裁剪阈值。
        全部是【预估算值】，不是LLM后端返回的真实token（前端/应用层估算）。
    """
    ratio = settings.chinese_tokens_per_char
    current_user = pack.current_user_message
    return PromptUsage(
        full_prompt_tokens=count_prompt_tokens(
            messages,
            tools=tools,
        ),
        system_tokens=(
            count_prompt_tokens(messages[:1])
            if messages
            else 0
        ),
        tool_definition_tokens=count_prompt_tokens(
            [],
            tools=tools,
        ),
        summary_tokens=estimate_text_tokens(pack.summaries_text, ratio),
        evidence_tokens=(
            estimate_text_tokens(
                json.dumps(evidence, ensure_ascii=False),
                ratio,
            )
            if evidence
            else 0
        ),
        layer1_tokens=estimate_messages_tokens(
            pack.layer1_messages,
            ratio,
        ),
        layer2_tokens=estimate_messages_tokens(
            pack.layer2_messages,
            ratio,
        ),
        current_user_tokens=(
            estimate_message_tokens(current_user, ratio)
            if current_user is not None
            else 0
        ),
        react_peak_reserved=(
            limits.max_steps
            * (
                limits.tool_call_message_max_tokens
                + limits.tool_result_max_tokens
                + 8
            )
        ),
        output_reserved=limits.max_output_tokens,
        safety_margin_tokens=settings.safety_margin_tokens,
        history_soft_budget=limits.history_soft_budget,
        history_hard_budget=limits.history_hard_budget,
    )




"""
       构建ReAct客服智能体LangGraph图
"""
def build_react_agent(
    model: BaseChatModel,
    limits: AgentLimits | None = None,
    tool_runner: ToolRunner = default_tool_runner,
    registry: ToolRegistry = tool_registry,
    checkpointer=None,
):

    limits = limits or AgentLimits()
    registry = registry


    def prepare_agent(state: ChatState) -> dict[str, list[Any]]:
        """
        组装agent内部推理消息栈，裁剪超长历史消息
        """
        if state.get("agent_messages"):
            return {}
        tools = registry.as_langchain_tools(
            _available_tool_names(state, registry)
        )
        pack = state.get("context_pack")    #prepare_context 节点生成并写入 ChatState 的一个已裁剪、已计费的上下文快照
        if pack is not None:
            agent_messages = build_model_messages(
                pack=pack,
                evidence=state.get("evidence") or [],
                order_data=state.get("order_data") or {},
                after_sales_action=state.get("after_sales_action", ""),
                fixed_system=CUSTOMER_SERVICE_SYSTEM,
                current_user=pack.current_user_message,
            )
            ticket_slot_message = _build_ticket_slot_message(state)
            if ticket_slot_message:
                agent_messages.insert(1, ticket_slot_message)
            token_estimate = count_prompt_tokens(
                agent_messages,
                tools=tools,
            )
            usage = _build_prompt_usage(
                pack=pack,
                messages=agent_messages,
                evidence=state.get("evidence") or [],
                limits=limits,
                tools=tools,
            )
        else:

            #当 Agent 没有 ContextPack 时，退回到旧的固定 token 裁剪方式，并把订单、证据和槽位信息作为 SystemMessage 加进模型上下文
            history = state.get("messages") or []
            trimmed = trim_history(history, max_tokens=settings.token_budget)   # 使用旧的固定 token 预算 settings.token_budget 裁剪历史
            agent_messages = [_build_evidence_system(state), *trimmed]  # 组装消息数组：先放证据相关的system提示词，再拼裁剪后的历史消息
            order_slot_message = _build_order_slot_message(state)       # 构造订单槽位提示消息（订单号收集、售后表单提示等）
            if order_slot_message:
                agent_messages.append(order_slot_message)
            ticket_slot_message = _build_ticket_slot_message(state)
            if ticket_slot_message:
                agent_messages.append(ticket_slot_message)
            token_estimate = count_prompt_tokens(
                agent_messages,
                tools=tools,
            )
            usage = None
        log_model_ctx(
            state.get("session_id", "direct-agent"),
            agent_messages,
            token_estimate,
            pack.summaries_text if pack is not None else "",
            usage=usage,
        )
        return {"agent_messages": agent_messages}

    async def call_agent(state: ChatState) -> dict[str, Any]:
        """
                图节点：agent，核心推理节点，调用大模型
                1. token预算校验，超限直接返回兜底回复
                2. 流式调用LLM，合并流式分片
                3. 判断模型输出是否需要调用工具
                4. 不需要工具：解析回答和前端按钮，流式推送结果
                返回：更新state的字段字典
        """

        writer = get_stream_writer()    # 获取流式输出写入器，用于向前端推送事件
        tools = registry.as_langchain_tools(
            _available_tool_names(state, registry)
        )
        bound_model = model.bind_tools(tools)
        full_prompt_tokens = count_prompt_tokens(
            state["agent_messages"],
            tools=tools,
        )
        prompt_limit = (
            limits.full_prompt_token_budget
            if limits.full_prompt_token_budget > 0
            else limits.token_budget
        )
        if full_prompt_tokens >= prompt_limit:
            writer({"type": "delta", "text": LIMIT_REPLY})
            return {
                "reply": LIMIT_REPLY,
                "stop_reason": "token_budget",
            }

        chunks: list[Any] = []
        async for chunk in bound_model.astream(
            state["agent_messages"],
            max_tokens=limits.max_output_tokens,
        ):
            chunks.append(chunk)
        if not chunks:
            raise RuntimeError("Agent model returned no chunks")

        combined = chunks[0]    # 合并所有流式chunk，得到完整模型输出消息对象
        for chunk in chunks[1:]:
            combined = combined + chunk

        steps = state.get("agent_steps", 0) + 1     # Agent执行轮次 +1
        tool_calls = getattr(combined, "tool_calls", [])

        if tool_calls:  # 如果模型输出工具调用，更新消息栈和步数，流转到tools节点
            return {
                "agent_messages": [*state["agent_messages"], combined],
                "agent_steps": steps,
                "stop_reason": "tool_calls",
            }

        reply, options = split_suggestions(_content_text(combined.content))
        if reply:
            writer({"type": "delta", "text": reply})
        if options:
            writer({"type": "reply_options", "options": options})
        return {
            "agent_messages": [*state["agent_messages"], combined],
            "agent_steps": steps,
            "reply": reply,
            "reply_options": options,
            "stop_reason": "completed",
        }

    async def run_tools(state: ChatState) -> dict[str, Any]:
        """
                图节点：tools，执行模型产生的工具调用
                支持一次多个工具并行调用；推送工具运行状态；记录工具调用日志trace
                返回：更新消息栈、工具调用日志
        """
        writer = get_stream_writer()
        last = state["agent_messages"][-1]
        trace = list(state.get("tool_trace") or [])     # 读取历史工具调用追踪记录
        tool_messages: list[ToolMessage] = []

        for index, call in enumerate(last.tool_calls):
            name = call["name"]
            args = call.get("args") or {}
            call_id = call.get("id") or f"agent-tool-{len(trace)}-{index}"
            batch_valid = not (
                any(
                    item.get("name") == "create_ticket"
                    for item in last.tool_calls
                )
                and len(last.tool_calls) != 1
            )
            writer({"type": "tool_status", "tool_name": name, "status": "running"}) # 向前端推送：工具开始运行
            context = ToolExecutionContext(
                conversation_id=state.get("session_id", ""),
                tool_call_id=call_id,
                caller=ToolCaller.AGENT,
                ticket_request_allowed=state.get(
                    "ticket_request_allowed",
                    False,
                ),
                ticket_batch_valid=batch_valid,
                ticket_slots=TicketSlots(
                    **(
                        state.get("ticket_slots")
                        or {
                            "ticket_type": "",
                            "description": "",
                            "missing_fields": (),
                        }
                    )
                ),
                emit=writer,
            )
            if len(signature(tool_runner).parameters) >= 3:
                execution = await tool_runner(name, args, context)
            else:
                execution = await tool_runner(name, args)
            content = truncate_tool_content(
                _tool_content(execution),
                limits.tool_result_max_tokens,
                settings.chinese_tokens_per_char,
            )
            writer( # 向前端推送:工具执行结果状态（成功/失败）
                {
                    "type": "tool_status",
                    "tool_name": name,
                    "status": "success" if execution.success else "error",
                    **({} if execution.success else {"message": execution.error}),
                }
            )
            trace.append(
                {
                    "name": name,
                    "args": args,
                    "call_id": call_id,
                    "success": execution.success,
                    "status": execution.status,
                    "retry_count": execution.retry_count,
                    "content": content,
                }
            )
            tool_messages.append(
                ToolMessage(
                    content=content,
                    tool_call_id=call_id,
                    name=name,
                )
            )
        truncated_last = _truncate_tool_call_message(
            last,
            limits.tool_call_message_max_tokens,
            settings.chinese_tokens_per_char,
        )
        return {
            "agent_messages": [
                *state["agent_messages"][:-1],
                truncated_last,
                *tool_messages,
            ],
            "tool_trace": trace,
        }

    def should_continue(state: ChatState) -> str:
        """
                条件路由函数：决定agent节点执行完毕后跳转到哪个节点
                返回字符串：tools / limit / end，对应graph条件分支
        """
        last = state["agent_messages"][-1]
        tool_calls = getattr(last, "tool_calls", [])
        if tool_calls and state.get("agent_steps", 0) < limits.max_steps:
            return "tools"
        if tool_calls:
            return "limit"
        return "end"

    def limit_response(state: ChatState) -> dict[str, Any]:
        """
                图节点：limit_response，达到最大执行步数兜底节点
                向前端推送limit提示文本，标记终止原因为max_steps
        """
        get_stream_writer()({"type": "delta", "text": LIMIT_REPLY})
        return {
            "reply": LIMIT_REPLY,
            "stop_reason": "max_steps",
        }

    graph = StateGraph(ChatState)
    graph.add_node("prepare_agent", prepare_agent)
    graph.add_node("agent", call_agent)
    graph.add_node("tools", run_tools)
    graph.add_node("limit_response", limit_response)

    graph.add_edge(START, "prepare_agent")
    graph.add_edge("prepare_agent", "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "limit": "limit_response", "end": END},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("limit_response", END)
    return graph.compile(checkpointer=checkpointer)
