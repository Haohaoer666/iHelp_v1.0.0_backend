import json

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from app.agents.bare_react import AgentLimits
from app.core.tool_runner import ToolExecutionResult
from app.graph.agent import (
    _build_evidence_system,
    _build_ticket_slot_message,
    _truncate_tool_call_message,
    build_react_agent,
)
from app.tools.models import (
    SideEffect,
    ToolDefinition,
    ToolOrigin,
)
from app.tools.registry import ToolRegistry


class StreamingScriptedModel:
    def __init__(self, responses: list[AIMessageChunk]) -> None:
        self.responses = responses
        self.bound_tools = None
        self.messages = None

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = tools
        return self

    async def astream(self, messages, **kwargs):
        self.messages = messages
        yield self.responses.pop(0)


async def fake_tool_runner(tool_name: str, tool_args: dict):
    return ToolExecutionResult(
        tool_name=tool_name,
        tool_args=tool_args,
        result=json.dumps({"tool": tool_name, "args": tool_args}),
        success=True,
    )


def test_agent_prompt_treats_selected_order_as_authoritative():
    message = _build_evidence_system(
        {
            "order_data": {
                "order_id": "1003",
                "signed_at": "2026-08-20",
                "days_since_signed": 31,
            },
            "after_sales_action": "exchange",
        }
    )

    assert "订单已由系统选中，禁止再次索要订单号" in message.content
    assert "换货" in message.content


def test_agent_prompt_requires_follow_up_when_ticket_description_missing():
    message = _build_ticket_slot_message(
        {
            "ticket_request_allowed": True,
            "ticket_slots": {
                "ticket_type": "投诉",
                "description": "",
                "missing_fields": ["description"],
            },
        }
    )

    assert "必须先追问" in message.content
    assert "不得调用 create_ticket" in message.content


def test_agent_prompt_requires_immediate_call_when_ticket_slots_complete():
    message = _build_ticket_slot_message(
        {
            "ticket_request_allowed": True,
            "ticket_slots": {
                "ticket_type": "投诉",
                "description": "快递一直不到",
                "missing_fields": [],
            },
        }
    )

    assert "必须立即调用 create_ticket" in message.content
    assert "不得再索要订单号" in message.content


def test_tool_call_message_is_truncated_for_model_context():
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "query_order",
                "args": {"note": "x" * 10000},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )

    shortened = _truncate_tool_call_message(
        message,
        max_tokens=20,
        chinese_tokens_per_char=0.8,
    )

    assert len(
        json.dumps(
            shortened.tool_calls[0]["args"],
            ensure_ascii=False,
        )
    ) < 10000


async def test_react_agent_reasserts_selected_order_after_history():
    model = StreamingScriptedModel([AIMessageChunk(content="已确认换货条件。")])
    graph = build_react_agent(model=model, limits=AgentLimits())

    await graph.ainvoke(
        {
            "messages": [HumanMessage("衣服买大了 我要换")],
            "order_data": {
                "order_id": "1003",
                "signed_at": "2026-08-20",
                "days_since_signed": 31,
            },
            "after_sales_action": "exchange",
            "agent_steps": 0,
            "tool_trace": [],
        }
    )

    assert "系统槽位已确认" in model.messages[-1].content
    assert "不得再次询问订单号" in model.messages[-1].content


async def test_react_graph_converges_without_tools():
    graph = build_react_agent(
        model=StreamingScriptedModel([AIMessageChunk(content="你好")]),
        limits=AgentLimits(),
    )

    result = await graph.ainvoke(
        {
            "agent_messages": [HumanMessage("你好")],
            "agent_steps": 0,
            "tool_trace": [],
        }
    )

    assert result["reply"] == "你好"
    assert result["stop_reason"] == "completed"
    assert result["agent_steps"] == 1


async def test_react_graph_executes_tools_before_answering():
    chunks = [
        AIMessageChunk(
            content="",
            tool_calls=[
                {"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}
            ]
        ),
        AIMessageChunk(content="订单已发货。"),
    ]
    graph = build_react_agent(
        model=StreamingScriptedModel(chunks),
        limits=AgentLimits(),
        tool_runner=fake_tool_runner,
    )

    result = await graph.ainvoke(
        {
            "agent_messages": [HumanMessage("查询订单 1001")],
            "agent_steps": 0,
            "tool_trace": [],
        }
    )

    assert result["reply"] == "订单已发货。"
    assert result["tool_trace"][0]["name"] == "query_order"
    assert result["tool_trace"][0]["success"] is True
    assert result["agent_steps"] == 2


async def test_react_graph_emits_tool_and_delta_events():
    graph = build_react_agent(
        model=StreamingScriptedModel(
            [
                AIMessageChunk(
                    content="",
                    tool_calls=[
                        {
                            "name": "query_logistics",
                            "args": {"order_id": "1001"},
                            "id": "c1",
                        }
                    ],
                ),
                AIMessageChunk(content="物流运输中。"),
            ]
        ),
        limits=AgentLimits(),
        tool_runner=fake_tool_runner,
    )

    events = [
        event
        async for event in graph.astream(
            {
                "agent_messages": [HumanMessage("查询订单 1001 的物流")],
                "agent_steps": 0,
                "tool_trace": [],
            },
            stream_mode="custom",
        )
    ]

    assert {"type": "tool_status", "tool_name": "query_logistics", "status": "running"} in events
    assert {
        "type": "tool_status",
        "tool_name": "query_logistics",
        "status": "success",
    } in events
    assert {"type": "delta", "text": "物流运输中。"} in events


async def test_react_graph_runs_order_then_logistics():
    graph = build_react_agent(
        model=StreamingScriptedModel(
            [
                AIMessageChunk(
                    content="",
                    tool_calls=[
                        {
                            "name": "query_order",
                            "args": {"order_id": "1001"},
                            "id": "c1",
                        }
                    ],
                ),
                AIMessageChunk(
                    content="",
                    tool_calls=[
                        {
                            "name": "query_logistics",
                            "args": {"order_id": "1001"},
                            "id": "c2",
                        }
                    ],
                ),
                AIMessageChunk(content="订单已发货，物流运输中。"),
            ]
        ),
        limits=AgentLimits(),
        tool_runner=fake_tool_runner,
    )

    result = await graph.ainvoke(
        {
            "agent_messages": [HumanMessage("先查订单 1001，再查物流")],
            "agent_steps": 0,
            "tool_trace": [],
        }
    )

    assert [item["name"] for item in result["tool_trace"]] == [
        "query_order",
        "query_logistics",
    ]
    assert result["agent_steps"] == 3


async def test_react_graph_binds_only_agent_visible_catalog_tools():
    model = StreamingScriptedModel([AIMessageChunk(content="完成")])
    graph = build_react_agent(model=model, limits=AgentLimits())

    await graph.ainvoke(
        {
            "agent_messages": [HumanMessage("你好")],
            "agent_steps": 0,
            "tool_trace": [],
        }
    )

    assert [tool.name for tool in model.bound_tools] == [
        "query_order",
        "query_product",
    ]


async def test_react_agent_binds_current_catalog_names():
    async def fake_logistics(order_id: str) -> dict:
        return {"order_id": order_id}

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="query_order",
            description="查订单",
            input_schema={
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
            handler=fake_logistics,
            origin=ToolOrigin.BUILTIN,
            server_name=None,
            side_effect=SideEffect.READ,
        )
    )
    registry.register(
        ToolDefinition(
            name="query_logistics",
            description="查物流",
            input_schema={
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
            handler=fake_logistics,
            origin=ToolOrigin.MCP,
            server_name="logistics",
            side_effect=SideEffect.READ,
        )
    )
    model = StreamingScriptedModel([AIMessageChunk(content="完成")])
    graph = build_react_agent(
        model=model,
        limits=AgentLimits(),
        registry=registry,
    )

    await graph.ainvoke(
        {
            "agent_messages": [HumanMessage("查物流")],
            "agent_steps": 0,
            "tool_trace": [],
            "available_tool_names": [
                "query_order",
                "query_logistics",
            ],
        }
    )

    assert [tool.name for tool in model.bound_tools] == [
        "query_order",
        "query_logistics",
    ]


async def test_react_graph_extracts_suggestion_marker():
    graph = build_react_agent(
        model=StreamingScriptedModel(
            [
                AIMessageChunk(
                    content="建议转人工处理。[[suggest:human_transfer|create_ticket]]"
                )
            ]
        ),
        limits=AgentLimits(),
    )

    result = await graph.ainvoke(
        {
            "agent_messages": [HumanMessage("这个问题一直解决不了")],
            "agent_steps": 0,
            "tool_trace": [],
        }
    )

    assert result["reply"] == "建议转人工处理。"
    assert result["reply_options"] == [
        {"id": "human_transfer", "label": "转人工"},
        {"id": "create_ticket", "label": "建工单"},
    ]


async def test_react_graph_extracts_refund_form_suggestion():
    graph = build_react_agent(
        model=StreamingScriptedModel(
            [
                AIMessageChunk(
                    content="订单 1001 符合七天无理由条件。[[suggest:apply_refund]]"
                )
            ]
        ),
        limits=AgentLimits(),
    )

    result = await graph.ainvoke(
        {
            "agent_messages": [HumanMessage("订单 1001 能退吗")],
            "agent_steps": 0,
            "tool_trace": [],
        }
    )

    assert result["reply"] == "订单 1001 符合七天无理由条件。"
    assert result["reply_options"] == [
        {"id": "apply_refund", "label": "申请退款"}
    ]


async def test_react_graph_stops_at_max_steps_with_limit_message():
    graph = build_react_agent(
        model=StreamingScriptedModel(
            [
                AIMessageChunk(
                    content="",
                    tool_calls=[
                        {
                            "name": "query_order",
                            "args": {"order_id": "1001"},
                            "id": "c1",
                        }
                    ]
                )
            ]
        ),
        limits=AgentLimits(max_steps=1),
        tool_runner=fake_tool_runner,
    )

    result = await graph.ainvoke(
        {
            "agent_messages": [HumanMessage("查询订单 1001")],
            "agent_steps": 0,
            "tool_trace": [],
        }
    )

    assert result["stop_reason"] == "max_steps"
    assert "还需要更多信息" in result["reply"]


async def test_react_graph_stops_before_model_when_token_budget_is_exhausted():
    model = StreamingScriptedModel([AIMessageChunk(content="不应调用")])
    graph = build_react_agent(
        model=model,
        limits=AgentLimits(token_budget=0),
    )

    result = await graph.ainvoke(
        {
            "agent_messages": [HumanMessage("你好")],
            "agent_steps": 0,
            "tool_trace": [],
        }
    )

    assert result["stop_reason"] == "token_budget"
    assert model.responses
