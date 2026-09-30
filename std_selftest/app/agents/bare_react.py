"""用于scripts"""

import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import BaseTool

from app.core.tool_runner import ToolExecutionResult, execute_tool


ToolRunner = Callable[[str, dict[str, Any]], Awaitable[ToolExecutionResult]]
EventCallback = Callable[[dict[str, Any]], None]


#存放智能体（Agent）运行时的限制参数。
@dataclass(frozen=True)
class AgentLimits:
    max_steps: int = 5
    max_output_tokens: int = 800
    token_budget: int = 4000
    full_prompt_token_budget: int = 0
    tool_call_message_max_tokens: int = 400
    tool_result_max_tokens: int = 1200
    history_soft_budget: int = 0
    history_hard_budget: int = 0


@dataclass
class AgentRunResult:
    reply: str
    steps: int
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = "completed"
    token_usage: int = 0


def _text_content(message: AIMessage) -> str:
    return message.content if isinstance(message.content, str) else str(message.content)


def _tool_content(execution: ToolExecutionResult) -> str:
    if not execution.success:
        return json.dumps({"error": execution.error}, ensure_ascii=False)
    if isinstance(execution.result, str):
        return execution.result
    return json.dumps(execution.result, ensure_ascii=False)


async def run_bare_agent(
    model: BaseChatModel,
    tools: list[BaseTool],
    messages: list[BaseMessage],
    limits: AgentLimits,
    tool_runner: ToolRunner = execute_tool,
    on_event: EventCallback | None = None,
) -> AgentRunResult:
    """Run the minimal model/tool loop until convergence or a configured limit."""
    bound_model = model.bind_tools(tools) if tools else model
    working = list(messages)
    tool_trace: list[dict[str, Any]] = []
    steps = 0

    while steps < limits.max_steps:
        token_usage = count_tokens_approximately(working)
        if token_usage >= limits.token_budget:
            return AgentRunResult(
                reply="当前信息较多，暂时无法继续查询，请补充最关键的信息。",
                steps=steps,
                tool_trace=tool_trace,
                stop_reason="token_budget",
                token_usage=token_usage,
            )

        ai = await bound_model.ainvoke(
            working,
            max_tokens=limits.max_output_tokens,
        )
        if not isinstance(ai, AIMessage):
            raise TypeError("Agent model must return AIMessage")
        working.append(ai)
        steps += 1

        if on_event:
            on_event({"type": "agent_step", "step": steps, "tool_calls": ai.tool_calls})

        if not ai.tool_calls:
            return AgentRunResult(
                reply=_text_content(ai),
                steps=steps,
                tool_trace=tool_trace,
                token_usage=count_tokens_approximately(working),
            )

        for index, call in enumerate(ai.tool_calls):
            tool_name = call.get("name", "")
            tool_args = call.get("args") or {}
            tool_call_id = call.get("id") or f"call-{steps}-{index}"
            execution = await tool_runner(tool_name, tool_args)
            content = _tool_content(execution)
            tool_trace.append(
                {
                    "name": tool_name,
                    "args": tool_args,
                    "call_id": tool_call_id,
                    "success": execution.success,
                    "content": content,
                }
            )
            working.append(
                ToolMessage(
                    content=content,
                    tool_call_id=tool_call_id,
                    name=tool_name,
                )
            )

    return AgentRunResult(
        reply=(
            "我已经查询到部分信息，但还需要更多信息才能继续确认。"
            "请补充订单号或具体问题。"
        ),
        steps=steps,
        tool_trace=tool_trace,
        stop_reason="max_steps",
        token_usage=count_tokens_approximately(working),
    )
