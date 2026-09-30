from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.db import models, repository, session as db_session
from app.db.base import Base
from app.services import rag_service


def test_sufficiency_prompt_rejects_unseen_entities():
    prompt = rag_service.build_sufficiency_prompt(
        "iHao X99 支持多少刷新率",
        [{"citation_id": 1, "text": "iHao V1 支持 200Hz"}],
    )

    assert "不存在的型号" in prompt
    assert "sufficient=false" in prompt


async def test_retrieve_knowledge_evidence_returns_citations(monkeypatch):
    async def fake_retrieve(*args, **kwargs):
        return [
            {
                "chunk_id": "c1",
                "text": "支持 7 天退货",
                "score": 0.9,
                "category": "售后",
            }
        ]

    async def fake_paths(chunk_ids):
        assert chunk_ids == ["c1"]
        return {"c1": "售后政策 > 退货"}

    monkeypatch.setattr(rag_service, "retrieve", fake_retrieve)
    monkeypatch.setattr(rag_service, "load_chunk_section_paths", fake_paths)

    result = await rag_service.retrieve_knowledge_evidence("退货政策")

    assert result["matched"] is True
    assert result["evidence"][0]["chunk_id"] == "c1"
    assert result["evidence"][0]["section_path"] == "售后政策 > 退货"


async def test_assess_evidence_rejects_empty_and_records(monkeypatch):
    recorded = []

    async def fake_insert(question, reason, source_conversation_id, entry_point):
        recorded.append((question, reason, source_conversation_id, entry_point))

    monkeypatch.setattr(rag_service, "insert_low_confidence_question", fake_insert)

    result = await rag_service.assess_evidence(
        "未知政策",
        [],
        conversation_id="s1",
    )

    assert result["sufficient"] is False
    assert recorded == [
        ("未知政策", "未召回足够证据", "s1", "workflow_confidence_gate")
    ]


async def test_assess_evidence_accepts_sufficient_context(monkeypatch):
    class SufficiencyModel:
        async def ainvoke(self, messages, **kwargs):
            return type("Response", (), {"content": '{"sufficient":true,"reason":""}'})()

    monkeypatch.setattr(
        rag_service,
        "get_chat_model",
        lambda streaming=False: SufficiencyModel(),
    )

    result = await rag_service.assess_evidence(
        "退货政策",
        [{"citation_id": 1, "text": "支持 7 天退货"}],
    )

    assert result["sufficient"] is True
    assert result["context"] == "[1] 支持 7 天退货"


async def test_assess_evidence_rejects_low_score(monkeypatch):
    recorded = []

    async def fake_insert(question, reason, source_conversation_id, entry_point):
        recorded.append((question, reason, source_conversation_id, entry_point))

    monkeypatch.setattr(rag_service, "insert_low_confidence_question", fake_insert)

    result = await rag_service.assess_evidence(
        "退货政策",
        [{"citation_id": 1, "text": "弱相关证据", "score": 0.1}],
        conversation_id="s1",
    )

    assert result["sufficient"] is False
    assert result["reason"] == "检索置信度不足"
    assert recorded == [
        ("退货政策", "检索置信度不足", "s1", "workflow_confidence_gate")
    ]


async def test_answer_knowledge_question_composes_split_functions(monkeypatch):
    async def fake_retrieve(*args, **kwargs):
        return {"matched": True, "evidence": [{"citation_id": 1, "text": "证据"}]}

    async def fake_assess(*args, **kwargs):
        return {"sufficient": True, "context": "[1] 证据"}

    monkeypatch.setattr(rag_service, "retrieve_knowledge_evidence", fake_retrieve)
    monkeypatch.setattr(rag_service, "assess_evidence", fake_assess)

    result = await rag_service.answer_knowledge_question("问题")

    assert result["sufficient"] is True
    assert result["context"] == "[1] 证据"


async def make_session(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool, future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    monkeypatch.setattr(db_session, "AsyncSessionLocal", session)
    monkeypatch.setattr(repository, "AsyncSessionLocal", session)
    return engine, session


async def test_answer_knowledge_question_refuses_and_inserts_pool(monkeypatch):
    engine, session = await make_session(monkeypatch)
    try:
        async def fake_retrieve(*args, **kwargs):
            return []

        monkeypatch.setattr(rag_service, "retrieve", fake_retrieve)

        result = await rag_service.answer_knowledge_question(
            "明天股市",
            conversation_id="s-rag",
        )

        assert result["matched"] is False
        assert result["sufficient"] is False
        async with session() as db:
            rows = list(
                (await db.execute(select(models.LowConfidenceQuestion))).scalars()
            )
            assert len(rows) == 1
            assert rows[0].question == "明天股市"
            assert rows[0].source_conversation_id == "s-rag"
            assert rows[0].entry_point == "workflow_confidence_gate"
    finally:
        await engine.dispose()
