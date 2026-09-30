import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy import select
from sqlalchemy.pool import StaticPool

from app.db import repository
from app.db.base import Base
from app.db.models import Message, Ticket


@pytest.fixture
async def sqlite_repository(monkeypatch):
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        poolclass=StaticPool,
    )
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    monkeypatch.setattr(repository, "AsyncSessionLocal", maker)
    yield repository
    await engine.dispose()


async def test_append_message_returns_database_id(sqlite_repository):
    await sqlite_repository.ensure_conversation("s1", "u1")

    message_id = await sqlite_repository.append_message("s1", "user", "你好")

    assert message_id > 0


async def test_conversation_rejects_different_user_key(sqlite_repository):
    await sqlite_repository.ensure_conversation("s1", "u1")

    with pytest.raises(sqlite_repository.ConversationOwnershipError):
        await sqlite_repository.ensure_conversation("s1", "u2")


async def test_summary_append_advances_anchor_once(sqlite_repository):
    await sqlite_repository.ensure_conversation("s1", "u1")
    first = await sqlite_repository.append_message("s1", "user", "订单 1001")
    second = await sqlite_repository.append_message("s1", "assistant", "已查询")

    first_row = await sqlite_repository.append_summary_and_advance(
        "s1",
        first,
        second,
        "用户询问订单 1001。",
        8,
    )
    second_row = await sqlite_repository.append_summary_and_advance(
        "s1",
        first,
        second,
        "用户询问订单 1001。",
        8,
    )

    summaries = await sqlite_repository.load_summaries("s1")
    conversation = await sqlite_repository.get_conversation("s1")
    assert len(summaries) == 1
    assert first_row.id == second_row.id
    assert conversation.summary_upto_msg_id == second


async def test_conversation_list_has_preview_and_summary_marker(
    sqlite_repository,
):
    await sqlite_repository.ensure_conversation("s1", "u1")
    first = await sqlite_repository.append_message(
        "s1",
        "user",
        "最开始那个订单后来怎么说",
    )
    second = await sqlite_repository.append_message("s1", "assistant", "处理中")
    await sqlite_repository.append_summary_and_advance(
        "s1",
        first,
        second,
        "用户追问订单 1001。",
        10,
    )

    rows = await sqlite_repository.list_conversations("u1")

    assert rows[0]["id"] == "s1"
    assert rows[0]["preview"] == "最开始那个订单后来怎么说"
    assert rows[0]["summarized"] is True


async def test_message_read_filters_roles_and_enforces_owner(
    sqlite_repository,
):
    await sqlite_repository.ensure_conversation("s1", "u1")
    await sqlite_repository.append_message("s1", "user", "订单 1001")
    await sqlite_repository.append_message(
        "s1",
        "tool",
        "不应返回",
        tool_name="query_order",
        tool_call_id="c1",
    )
    await sqlite_repository.append_message("s1", "assistant", "已查询")

    rows = await sqlite_repository.load_conversation_messages("s1", "u1")
    denied = await sqlite_repository.load_conversation_messages("s1", "u2")

    assert [row["role"] for row in rows] == ["user", "assistant"]
    assert denied is None


async def test_delete_conversation_removes_chat_data_and_detaches_tickets(
    sqlite_repository,
):
    await sqlite_repository.ensure_conversation("s1", "u1")
    first = await sqlite_repository.append_message("s1", "user", "订单 1001")
    second = await sqlite_repository.append_message("s1", "assistant", "已查询")
    await sqlite_repository.append_summary_and_advance(
        "s1",
        first,
        second,
        "用户询问订单 1001。",
        10,
    )
    async with sqlite_repository.AsyncSessionLocal() as db:
        db.add(
            Ticket(
                ticket_id="TK1",
                conversation_id="s1",
                description="测试工单",
                ticket_type="投诉",
            )
        )
        await db.commit()

    deleted = await sqlite_repository.delete_conversation("s1", "u1")

    assert deleted is True
    assert await sqlite_repository.load_messages("s1") == []
    assert await sqlite_repository.load_summaries("s1") == []
    assert await sqlite_repository.get_conversation("s1") is None
    async with sqlite_repository.AsyncSessionLocal() as db:
        ticket = await db.get(Ticket, "TK1")
        assert ticket is not None
        assert ticket.conversation_id is None


async def test_delete_conversation_rejects_other_user(sqlite_repository):
    await sqlite_repository.ensure_conversation("s1", "u1")
    await sqlite_repository.append_message("s1", "user", "订单 1001")

    deleted = await sqlite_repository.delete_conversation("s1", "u2")

    assert deleted is False
    assert len(await sqlite_repository.load_messages("s1")) == 1
