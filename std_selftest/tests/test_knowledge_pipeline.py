from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import json

from app.db.base import Base
from app.db.models import KnowledgeChunk
from app.services import knowledge_pipeline


class FakeVectorStore:
    async def reset(self):
        return None

    async def upsert(self, chunk_id, category, text, vector):
        return 1

    async def close(self):
        return None


async def make_testing_session():
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
    return engine, testing_session


async def test_insert_chunks_is_idempotent_without_rebuild(monkeypatch, tmp_path):
    engine, testing_session = await make_testing_session()
    try:
        monkeypatch.setattr(
            knowledge_pipeline,
            "AsyncSessionLocal",
            testing_session,
        )
        monkeypatch.setattr(knowledge_pipeline, "VectorStore", FakeVectorStore)

        markdown = tmp_path / "售后政策.md"
        markdown.write_text(
            "# 售后政策\n\n## 运费说明\n\n普通商品满 99 元包邮。",
            encoding="utf-8",
        )

        first = await knowledge_pipeline.insert_chunks_from_markdown(
            [markdown],
            rebuild=False,
        )
        second = await knowledge_pipeline.insert_chunks_from_markdown(
            [markdown],
            rebuild=False,
        )

        assert first > 0
        assert second == 0
    finally:
        await engine.dispose()


async def test_insert_chunks_normalizes_document_whitespace(monkeypatch, tmp_path):
    engine, testing_session = await make_testing_session()
    try:
        monkeypatch.setattr(
            knowledge_pipeline,
            "AsyncSessionLocal",
            testing_session,
        )
        monkeypatch.setattr(knowledge_pipeline, "VectorStore", FakeVectorStore)

        markdown = tmp_path / "售后政策.md"
        markdown.write_text(
            "# 售后政策\n\n## 运费说明\n\n普通商品   满 99 \u00a0 元包邮。\n   \n\n",
            encoding="utf-8",
        )

        await knowledge_pipeline.insert_chunks_from_markdown(
            [markdown],
            rebuild=False,
        )

        async with testing_session() as db:
            result = await db.scalars(select(KnowledgeChunk))
            answers = [row.answer for row in result.all()]

        assert any("普通商品 满 99 元包邮。" in answer for answer in answers)
    finally:
        await engine.dispose()


async def test_search_knowledge_embeds_query_with_embed_query(monkeypatch):
    class FakeStore:
        def __init__(self):
            self.search_calls = []
            self.closed = False

        async def search(self, vector, top_k=None):
            self.search_calls.append((vector, top_k))
            return [
                {
                    "distance": 0.9,
                    "chunk_id": "chunk-1",
                    "category": "售后",
                    "text": "答案",
                }
            ]

        async def close(self):
            self.closed = True

    store = FakeStore()
    monkeypatch.setattr(knowledge_pipeline, "VectorStore", lambda: store)

    async def fake_embed_query(query):
        return [1.0, 2.0]

    async def fail_embed_text(text):
        raise AssertionError("search_knowledge 不应再使用文档 embedding 路径")

    monkeypatch.setattr(knowledge_pipeline, "embed_query", fake_embed_query)
    monkeypatch.setattr(knowledge_pipeline, "embed_text", fail_embed_text)

    result = await knowledge_pipeline.search_knowledge("用户问题", top_k=5)

    assert result == [
        {
            "distance": 0.9,
            "chunk_id": "chunk-1",
            "category": "售后",
            "text": "答案",
        }
    ]
    assert store.search_calls == [([1.0, 2.0], 5)]
    assert store.closed is True


async def test_query_knowledge_uses_question_from_search_hit(monkeypatch):
    async def fake_search(keyword):
        return [
            {
                "question": "智能猫砂盆怎么清理",
                "text": "断电后拆下集便仓，用清水冲洗晾干。",
                "category": "商品FAQ",
            }
        ]

    monkeypatch.setattr(knowledge_pipeline, "search_knowledge", fake_search)

    payload = json.loads(await knowledge_pipeline.query_knowledge("猫砂盆"))

    assert payload["items"][0]["question"] == "智能猫砂盆怎么清理"
