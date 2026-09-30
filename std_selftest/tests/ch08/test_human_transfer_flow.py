from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.models import Ticket, ToolAuditLog
from app.graph import nodes as graph_nodes
from app.graph.builder import build_chat_graph
from app.tools import audit as audit_module
from app.tools.audit import AuditRecorder
from app.tools.builtin import create_ticket as ticket_module
from app.tools.builtin.create_ticket import TOOL_DEFINITION
from app.tools.executor import ToolExecutor
from app.tools.models import ToolExecutionResult
from app.tools.registry import ToolRegistry


class NoUseModel:
    def bind_tools(self, tools, **kwargs):
        return self

    async def astream(self, *args, **kwargs):
        raise AssertionError("agent should not run")
        yield


class FakeTurnLogger:
    async def log(self, session_id, state):
        return {"logged": True}


async def build_transfer_graph(monkeypatch):
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        poolclass=StaticPool,
    )
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    monkeypatch.setattr(ticket_module, "AsyncSessionLocal", sessions)
    monkeypatch.setattr(audit_module, "AsyncSessionLocal", sessions)

    registry = ToolRegistry()
    registry.register(TOOL_DEFINITION)
    executor = ToolExecutor(
        registry=registry,
        audit=AuditRecorder(),
    )

    async def classify(text, model):
        return "转人工"

    graph = build_chat_graph(
        intent_model=NoUseModel(),
        agent_model=NoUseModel(),
        intent_classifier=classify,
        agent_builder=lambda model, limits: RunnableLambda(
            lambda state: {
                "reply": "不应调用 Agent",
                "agent_steps": 1,
            }
        ),
        tool_registry=registry,
        tool_executor=executor,
        turn_logger=FakeTurnLogger(),
        checkpointer=InMemorySaver(),
    )
    return engine, sessions, graph


async def test_transfer_without_issue_asks_then_confirms(monkeypatch):
    engine, sessions, graph = await build_transfer_graph(monkeypatch)
    session_id = "human-transfer-confirm"
    config = {"configurable": {"thread_id": session_id}}
    try:
        first = await graph.ainvoke(
            {
                "session_id": session_id,
                "current_user_message_id": 1,
                "user_message": "转人工",
                "messages": [HumanMessage("转人工")],
            },
            config,
        )

        assert first["pending_action"] == "await_transfer_issue"
        assert "请问您遇到了什么问题" in first["reply"]
        assert "__interrupt__" not in first

        second = await graph.ainvoke(
            {
                "session_id": session_id,
                "current_user_message_id": 2,
                "user_message": "退款一直没到",
                "messages": [HumanMessage("退款一直没到")],
            },
            config,
        )
        assert second["__interrupt__"]

        third = await graph.ainvoke(
            Command(
                resume={
                    "type": "ticket_confirmed",
                    "tool_call_id": f"human-transfer-{session_id}-2",
                }
            ),
            config,
        )

        assert "已介入客服" in third["reply"]
        assert "退款一直没到" in third["reply"]
        async with sessions() as db:
            ticket = await db.scalar(select(Ticket))
            audit = await db.scalar(select(ToolAuditLog))
            assert ticket is not None
            assert ticket.ticket_type == "转人工"
            assert ticket.description == "退款一直没到"
            assert audit is not None
            assert audit.status == "success"
    finally:
        await engine.dispose()


async def test_transfer_cancel_does_not_create_ticket(monkeypatch):
    engine, sessions, graph = await build_transfer_graph(monkeypatch)
    session_id = "human-transfer-cancel"
    config = {"configurable": {"thread_id": session_id}}
    try:
        await graph.ainvoke(
            {
                "session_id": session_id,
                "current_user_message_id": 1,
                "user_message": "转人工",
                "messages": [HumanMessage("转人工")],
            },
            config,
        )
        second = await graph.ainvoke(
            {
                "session_id": session_id,
                "current_user_message_id": 2,
                "user_message": "退款一直没到",
                "messages": [HumanMessage("退款一直没到")],
            },
            config,
        )
        assert second["__interrupt__"]

        third = await graph.ainvoke(
            Command(
                resume={
                    "type": "ticket_cancelled",
                    "tool_call_id": f"human-transfer-{session_id}-2",
                }
            ),
            config,
        )

        assert "取消" in third["reply"]
        async with sessions() as db:
            assert await db.scalar(
                select(func.count()).select_from(Ticket)
            ) == 0
            audit = await db.scalar(select(ToolAuditLog))
            assert audit is not None
            assert audit.status == "permission_denied"
    finally:
        await engine.dispose()


async def test_transfer_awaiting_issue_can_cancel_and_clears_state(monkeypatch):
    engine, sessions, graph = await build_transfer_graph(monkeypatch)
    session_id = "human-transfer-await-cancel"
    config = {"configurable": {"thread_id": session_id}}
    try:
        first = await graph.ainvoke(
            {
                "session_id": session_id,
                "current_user_message_id": 1,
                "user_message": "转人工",
                "messages": [HumanMessage("转人工")],
                "order_data": {"stale": True},
            },
            config,
        )
        assert first["pending_action"] == "await_transfer_issue"

        second = await graph.ainvoke(
            {
                "session_id": session_id,
                "current_user_message_id": 2,
                "user_message": "算了",
                "messages": [HumanMessage("算了")],
            },
            config,
        )

        assert second["pending_action"] == ""
        assert "取消" in second["reply"]
        assert second["order_data"] == {}
        assert second["resolved_query"] == ""
        assert second["query_history"] == []
        assert "__interrupt__" not in second
    finally:
        await engine.dispose()


async def test_transfer_reuses_latest_problem_from_context(monkeypatch):
    engine, sessions, graph = await build_transfer_graph(monkeypatch)
    session_id = "human-transfer-context"
    try:
        result = await graph.ainvoke(
            {
                "session_id": session_id,
                "user_message": "转人工",
                "messages": [
                    HumanMessage("订单 1001 退款一直没到"),
                    HumanMessage("转人工"),
                ],
            },
            {"configurable": {"thread_id": session_id}},
        )

        assert result["__interrupt__"]
    finally:
        await engine.dispose()


async def test_transfer_call_ids_are_unique_per_user_message(monkeypatch):
    captured = []

    class CapturingExecutor:
        async def execute(self, tool_name, tool_args, context):
            captured.append(context.tool_call_id)
            return ToolExecutionResult(
                tool_name=tool_name,
                tool_args=tool_args,
                result={"ticket_id": "TK-TEST"},
                success=True,
                status="success",
            )

    monkeypatch.setattr(
        graph_nodes,
        "get_stream_writer",
        lambda: (lambda event: None),
    )
    node = graph_nodes.make_human_transfer_node(CapturingExecutor())

    await node(
        {
            "session_id": "transfer-id",
            "current_user_message_id": 101,
            "user_message": "转人工，退款没到",
        }
    )
    await node(
        {
            "session_id": "transfer-id",
            "current_user_message_id": 202,
            "user_message": "转人工，物流太慢",
        }
    )

    assert captured == [
        "human-transfer-transfer-id-101",
        "human-transfer-transfer-id-202",
    ]


async def test_cancel_without_pending_transfer_does_not_ask_issue(monkeypatch):
    class NoUseExecutor:
        async def execute(self, *args, **kwargs):
            raise AssertionError("executor should not run")

    writes = []
    monkeypatch.setattr(
        graph_nodes,
        "get_stream_writer",
        lambda: writes.append,
    )
    node = graph_nodes.make_human_transfer_node(NoUseExecutor())

    result = await node(
        {
            "session_id": "transfer-cancel-idle",
            "current_user_message_id": 11,
            "user_message": "取消转人工",
        }
    )

    assert result["pending_action"] == ""
    assert "没有待取消" in result["reply"]
    assert any(
        event.get("type") == "delta"
        and event.get("text") == result["reply"]
        for event in writes
    )
    assert any(
        event.get("type") == "node_status"
        and event.get("status") == "success"
        and event.get("detail", {}).get("cancelled") is True
        for event in writes
    )
