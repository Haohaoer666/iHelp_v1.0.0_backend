from langchain_core.messages import AIMessageChunk, HumanMessage
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
from app.graph.agent import build_react_agent
from app.tools import audit as audit_module
from app.tools.audit import AuditRecorder
from app.tools.builtin import create_ticket as ticket_module
from app.tools.builtin.create_ticket import TOOL_DEFINITION
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry


class ScriptedModel:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.bound_tools = None

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = tools
        return self

    async def astream(self, messages, **kwargs):
        yield self.chunks.pop(0)


async def build_test_graph(monkeypatch):
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
    model = ScriptedModel(
        [
            AIMessageChunk(
                content="",
                tool_calls=[
                    {
                        "name": "create_ticket",
                        "args": {
                            "conversation_id": "confirm-1",
                            "description": "快递一直不到",
                            "ticket_type": "投诉",
                        },
                        "id": "ticket-call-1",
                    }
                ],
            ),
            AIMessageChunk(content="已提交工单"),
        ]
    )
    executor = ToolExecutor(
        registry=registry,
        audit=AuditRecorder(),
    )
    graph = build_react_agent(
        model=model,
        tool_runner=executor.execute,
        registry=registry,
        checkpointer=InMemorySaver(),
    )
    return engine, sessions, graph


def ticket_inputs(session_id: str) -> dict:
    return {
        "session_id": session_id,
        "messages": [HumanMessage("帮我建个工单，快递一直不到")],
        "available_tool_names": ["create_ticket"],
        "ticket_request_allowed": True,
        "ticket_slots": {
            "ticket_type": "投诉",
            "description": "快递一直不到",
            "missing_fields": [],
        },
        "agent_steps": 0,
        "tool_trace": [],
    }


async def test_confirm_creates_ticket_and_audits_success(monkeypatch):
    engine, sessions, graph = await build_test_graph(monkeypatch)
    config = {"configurable": {"thread_id": "confirm-1"}}
    try:
        first = await graph.ainvoke(
            ticket_inputs("confirm-1"),
            config,
        )
        assert first["__interrupt__"]

        async with sessions() as db:
            assert await db.scalar(
                select(func.count()).select_from(Ticket)
            ) == 0
            assert await db.scalar(
                select(func.count()).select_from(ToolAuditLog)
            ) == 0

        await graph.ainvoke(
            Command(
                resume={
                    "type": "ticket_confirmed",
                    "tool_call_id": "ticket-call-1",
                }
            ),
            config,
        )

        async with sessions() as db:
            ticket = await db.scalar(select(Ticket))
            audit = await db.scalar(select(ToolAuditLog))
            assert ticket is not None
            assert ticket.description == "快递一直不到"
            assert audit is not None
            assert audit.status == "success"
    finally:
        await engine.dispose()


async def test_cancel_writes_no_ticket_and_audits_permission_denied(
    monkeypatch,
):
    engine, sessions, graph = await build_test_graph(monkeypatch)
    config = {"configurable": {"thread_id": "confirm-1"}}
    try:
        first = await graph.ainvoke(
            ticket_inputs("confirm-1"),
            config,
        )
        assert first["__interrupt__"]

        await graph.ainvoke(
            Command(
                resume={
                    "type": "ticket_cancelled",
                    "tool_call_id": "ticket-call-1",
                }
            ),
            config,
        )

        async with sessions() as db:
            assert await db.scalar(
                select(func.count()).select_from(Ticket)
            ) == 0
            audit = await db.scalar(select(ToolAuditLog))
            assert audit is not None
            assert audit.status == "permission_denied"
    finally:
        await engine.dispose()
