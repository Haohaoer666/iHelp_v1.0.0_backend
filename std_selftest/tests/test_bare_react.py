import json

from langchain_core.messages import AIMessage, HumanMessage

from app.agents.bare_react import AgentLimits, run_bare_agent
from app.core.tool_runner import ToolExecutionResult


class ScriptedModel:
    def __init__(self, responses: list[AIMessage]) -> None:
        self.responses = responses
        self.bound_tools = None

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = tools
        return self

    async def ainvoke(self, messages, **kwargs):
        return self.responses.pop(0)


def make_tool_runner(*results: ToolExecutionResult):
    pending = list(results)

    async def runner(tool_name: str, tool_args: dict):
        if pending:
            return pending.pop(0)
        return ToolExecutionResult(
            tool_name=tool_name,
            tool_args=tool_args,
            result=json.dumps({"tool": tool_name, "args": tool_args}),
            success=True,
        )

    return runner


async def test_bare_agent_converges_without_tools():
    model = ScriptedModel(responses=[AIMessage(content="直接回答")])

    result = await run_bare_agent(
        model=model,
        tools=[],
        messages=[HumanMessage("你好")],
        limits=AgentLimits(),
    )

    assert result.reply == "直接回答"
    assert result.steps == 1
    assert result.stop_reason == "completed"
    assert result.tool_trace == []


async def test_bare_agent_executes_multiple_tool_steps():
    model = ScriptedModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "query_logistics",
                        "args": {"order_id": "1001"},
                        "id": "c2",
                    }
                ],
            ),
            AIMessage(content="订单已发货，物流运输中。"),
        ]
    )

    result = await run_bare_agent(
        model=model,
        tools=[],
        messages=[HumanMessage("先查订单再查物流")],
        limits=AgentLimits(max_steps=5),
        tool_runner=make_tool_runner(),
    )

    assert result.reply == "订单已发货，物流运输中。"
    assert [item["name"] for item in result.tool_trace] == [
        "query_order",
        "query_logistics",
    ]
    assert [item["success"] for item in result.tool_trace] == [True, True]
    assert result.steps == 3


async def test_bare_agent_feeds_tool_errors_back_to_model():
    model = ScriptedModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}
                ],
            ),
            AIMessage(content="订单服务暂时不可用，请稍后再试。"),
        ]
    )
    error = ToolExecutionResult(
        tool_name="query_order",
        tool_args={"order_id": "1001"},
        result=None,
        success=False,
        error="订单服务超时",
    )

    result = await run_bare_agent(
        model=model,
        tools=[],
        messages=[HumanMessage("查询订单 1001")],
        limits=AgentLimits(),
        tool_runner=make_tool_runner(error),
    )

    assert result.reply == "订单服务暂时不可用，请稍后再试。"
    assert result.tool_trace[0]["success"] is False
    assert "订单服务超时" in result.tool_trace[0]["content"]


async def test_bare_agent_stops_at_max_steps():
    model = ScriptedModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}
                ],
            )
        ]
    )

    result = await run_bare_agent(
        model=model,
        tools=[],
        messages=[HumanMessage("查询订单")],
        limits=AgentLimits(max_steps=1),
        tool_runner=make_tool_runner(),
    )

    assert result.stop_reason == "max_steps"
    assert result.steps == 1
    assert "还需要更多信息" in result.reply


async def test_bare_agent_stops_when_token_budget_is_already_exhausted():
    model = ScriptedModel(responses=[AIMessage(content="不应调用")])

    result = await run_bare_agent(
        model=model,
        tools=[],
        messages=[HumanMessage("你好")],
        limits=AgentLimits(token_budget=0),
    )

    assert result.stop_reason == "token_budget"
    assert result.steps == 0
    assert model.responses
