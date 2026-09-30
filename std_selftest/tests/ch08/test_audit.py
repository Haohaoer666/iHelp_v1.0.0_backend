from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.models import ToolAuditLog
from app.tools import audit as audit_module
from app.tools.audit import AuditRecorder


async def test_audit_record_persists_without_foreign_key(monkeypatch):
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
    monkeypatch.setattr(audit_module, "AsyncSessionLocal", sessions)

    await AuditRecorder().record(
        conversation_id="missing-conversation",
        tool_call_id="call-1",
        tool_name="query_order",
        origin="builtin",
        server_name=None,
        arguments={"order_id": "1001"},
        result_summary={"ok": True},
        status="success",
        error="",
        retry_count=0,
        duration_ms=12,
    )

    async with sessions() as db:
        rows = list((await db.scalars(select(ToolAuditLog))).all())
    assert len(rows) == 1
    assert rows[0].conversation_id == "missing-conversation"
    assert rows[0].origin == "builtin"
    await engine.dispose()


async def test_audit_failure_is_swallowed(monkeypatch):
    recorder = AuditRecorder()

    async def fail(**kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(recorder, "_write", fail)

    await recorder.record(
        conversation_id="s1",
        tool_call_id="c1",
        tool_name="query_order",
        origin="builtin",
        server_name=None,
        arguments={},
        result_summary={},
        status="success",
        error="",
        retry_count=0,
        duration_ms=1,
    )
