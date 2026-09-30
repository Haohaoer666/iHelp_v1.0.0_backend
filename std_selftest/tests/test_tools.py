from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.core import tools
from app.tools.builtin import create_ticket as create_ticket_module
from app.tools.builtin import query_faq as query_faq_module
from app.db import models, session as db_session
from app.db.base import Base


async def setup_testing_session(monkeypatch):
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        poolclass=StaticPool,
        future=True,
    )
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    testing_session = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    monkeypatch.setattr(db_session, "AsyncSessionLocal", testing_session)
    monkeypatch.setattr(
        create_ticket_module,
        "AsyncSessionLocal",
        testing_session,
    )

    async with testing_session() as db:
        db.add(models.Conversation(id="s-tool"))
        db.add(
            models.FAQ(
                question="退货政策是什么",
                answer="支持 7 天无理由退货",
                category="售后",
            )
        )
        await db.commit()

    return engine


async def test_query_faq_hit(monkeypatch):
    engine = await setup_testing_session(monkeypatch)
    try:
        async def fake_answer(keyword, conversation_id=None, category=None):
            return {
                "matched": True,
                "items": [
                    {
                        "question": "退货政策是什么",
                        "answer": "支持 7 天无理由退货",
                        "category": "售后",
                    }
                ],
            }

        monkeypatch.setattr(
            query_faq_module,
            "answer_knowledge_question",
            fake_answer,
        )
        result = await tools.query_faq.ainvoke({"keyword": "退货"})

        assert "退货政策是什么" in result
    finally:
        await engine.dispose()


async def test_query_faq_miss(monkeypatch):
    engine = await setup_testing_session(monkeypatch)
    try:
        async def fake_answer(keyword, conversation_id=None, category=None):
            return {"matched": False, "message": "未找到相关常见问题"}

        monkeypatch.setattr(
            query_faq_module,
            "answer_knowledge_question",
            fake_answer,
        )
        result = await tools.query_faq.ainvoke({"keyword": "邮费"})

        assert '"matched": false' in result
    finally:
        await engine.dispose()


async def test_query_faq_passes_conversation_id(monkeypatch):
    engine = await setup_testing_session(monkeypatch)
    try:
        async def fake_answer(keyword, conversation_id=None, category=None):
            assert keyword == "明天股市"
            assert conversation_id == "s-tool"
            return {"matched": False, "sufficient": False, "refusal": "抱歉"}

        monkeypatch.setattr(
            query_faq_module,
            "answer_knowledge_question",
            fake_answer,
        )

        result = await tools.query_faq.ainvoke(
            {"keyword": "明天股市", "conversation_id": "s-tool"}
        )

        assert '"matched": false' in result
    finally:
        await engine.dispose()


async def test_create_ticket(monkeypatch):
    engine = await setup_testing_session(monkeypatch)
    try:
        result = await tools.create_ticket.ainvoke(
            {
                "conversation_id": "s-tool",
                "description": "快递员态度差",
                "ticket_type": "投诉",
            }
        )

        assert "TK" in result
        assert "投诉" in result
    finally:
        await engine.dispose()


async def test_tool_runner_unknown_tool():
    from app.core.tool_runner import execute_tool

    result = await execute_tool("not_exists", {})

    assert result.success is False
    assert "未知工具" in result.error
