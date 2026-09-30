
"""
MySQL + Milvus 双写流水

Markdown 文档
  -> 切块
  -> 写 MySQL，状态为 pending
  -> 调用 Embedding API
  -> 写入 Milvus
  -> 回填 milvus_id
  -> MySQL 状态改为 vectorized
"""

import asyncio
import hashlib
import json
import logging
from pathlib import Path

from sqlalchemy import delete, select

from app.config import settings
from app.db.models import KnowledgeChunk
from app.db.session import AsyncSessionLocal
from app.services.document_loader import (
    SUPPORTED_SUFFIXES,
    load_document,
    normalize_document_text,
)
from app.services.document_splitter import split_markdown_document
from app.services.embedding import embed_query, embed_text
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


def _text_for_embedding(category: str, questions: list[str], answer: str) -> str:
    question_text = "\n".join(questions) if questions else ""
    return f"category: {category}\nquestions: {question_text}\nanswer: {answer}"


def _make_chunk_id(source_name: str, index: int, text: str) -> str:     #文件名去掉后缀 - 切片顺序 - 切片内容 SHA1 前 10 位
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{Path(source_name).stem}-{index}-{digest}"


def _load_and_split(file_path: Path) -> tuple[str, list[dict]]:         #读取文件 → 文本规整 → Markdown 递归切片
    markdown = normalize_document_text(load_document(file_path).text)
    return file_path.name, split_markdown_document(markdown, file_path.name)


#该函数并发读取并切分多个 Markdown 文件，将文本块幂等写入数据库，维护块前后链表关系，支持清空重建，返回新增 chunk 数量，向量留待后续任务生成。
async def insert_chunks_from_markdown(files: list[Path], rebuild: bool = False) -> int:
    store = VectorStore()
    try:
        async with AsyncSessionLocal() as db:
            if rebuild:
                await db.execute(delete(KnowledgeChunk))
                await db.commit()
                await store.reset()

            chunks = []
            file_chunks = await asyncio.gather(     #asyncio.gather() 并发读取并切片文件。
                *(asyncio.to_thread(_load_and_split, file_path) for file_path in files)
            )
            for source_name, parts in file_chunks:
                chunks.extend((source_name, part) for part in parts)    #把所有文件的chunk展平到同一个列表，每个元素是 (源文件名, 块字典数据)


            existing_result = await db.scalars(select(KnowledgeChunk))
            existing = {row.chunk_id: row for row in existing_result.all()}
            rows_by_id = dict(existing)
            inserted = 0
            previous_chunk_id = None
            for index, (source_name, part) in enumerate(chunks):    ## 第一轮遍历：新增不存在的chunk；更新已有chunk的prev指针
                chunk_id = _make_chunk_id(source_name, index, part["answer"])
                questions = part.get("questions") or []
                text_for_embedding = _text_for_embedding(
                    part["category"], questions, part["answer"]
                )

                if chunk_id in rows_by_id:
                    row = rows_by_id[chunk_id]
                    row.prev_chunk_id = previous_chunk_id
                    previous_chunk_id = chunk_id
                    continue

                row = KnowledgeChunk(
                    chunk_id=chunk_id,
                    category=part["category"],
                    questions=questions,
                    answer=part["answer"],
                    section_path=part["section_path"],
                    content_type=part["content_type"],
                    is_critical=part["is_critical"],
                    prev_chunk_id=previous_chunk_id,
                    next_chunk_id=None,
                    vector_status="pending",
                    text_for_embedding=text_for_embedding,
                )
                db.add(row)
                rows_by_id[chunk_id] = row
                if previous_chunk_id:
                    rows_by_id[previous_chunk_id].next_chunk_id = chunk_id
                previous_chunk_id = chunk_id
                inserted += 1

            for index, (source_name, part) in enumerate(chunks):
                if index + 1 >= len(chunks):
                    continue
                next_source, next_part = chunks[index + 1]
                chunk_id = _make_chunk_id(source_name, index, part["answer"])
                next_chunk_id = _make_chunk_id(
                    next_source,
                    index + 1,
                    next_part["answer"],
                )
                row = rows_by_id.get(chunk_id)
                if row:
                    row.next_chunk_id = next_chunk_id

            await db.commit()
            return inserted
    finally:
        await store.close()

#pending / failed → embedding → Milvus → vectorized
async def sync_pending_chunks() -> dict:
    store = VectorStore()
    try:
        async with AsyncSessionLocal() as db:
            result = await db.scalars(
                select(KnowledgeChunk).where(
                    KnowledgeChunk.vector_status.in_(["pending", "failed"])
                )
            )
            rows = list(result.all())

        synced = 0
        failed = 0
        for row in rows:
            try:
                vector = await embed_text(row.text_for_embedding)
                milvus_id = await store.upsert(     # 将chunk信息和向量upsert写入向量库：存在则更新，不存在则新增
                    row.chunk_id,
                    row.category,
                    row.answer,
                    vector,
                    questions=row.questions if isinstance(row.questions, list) else [],
                )
                async with AsyncSessionLocal() as db:
                    item = await db.get(KnowledgeChunk, row.id)
                    if item:
                        item.vector_status = "vectorized"   # 修改状态为已向量化，保存向量库返回的milvus主键
                        item.milvus_id = milvus_id
                        await db.commit()
                synced += 1
            except Exception:
                logger.exception("向量化失败 chunk_id=%s", row.chunk_id)
                async with AsyncSessionLocal() as db:
                    item = await db.get(KnowledgeChunk, row.id)
                    if item:
                        item.vector_status = "failed"
                        await db.commit()
                failed += 1

        if synced:
            await store.flush()
        return {"synced": synced, "failed": failed}
    finally:
        await store.close()


#用户问题语义检索
async def search_knowledge(query: str, top_k: int | None = None) -> list[dict]:
    store = VectorStore()
    try:
        vector = await embed_query(query)
        hits = await store.search(vector, top_k=top_k or settings.knowledge_top_k)
        return [
            hit
            for hit in hits
            if hit["distance"] >= settings.knowledge_score_threshold
        ]
    finally:
        await store.close()


#前端“重建知识库”按钮最终调用的函数
async def build_knowledge_base(docs_dir: Path, rebuild: bool = False) -> dict:
    files = sorted(
        path
        for path in docs_dir.iterdir()
        if path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    inserted = await insert_chunks_from_markdown(files, rebuild=rebuild)
    sync = await sync_pending_chunks()
    return {"files": len(files), "inserted": inserted, **sync}


#给 LangChain 工具用的“知识查询格式化函数”。
async def query_knowledge(keyword: str) -> str:
    hits = await search_knowledge(keyword)
    if not hits:
        return json.dumps({"matched": False, "message": "未找到相关常见问题"}, ensure_ascii=False)
    return json.dumps(
        {
            "matched": True,
            "items": [
                {
                    "question": hit["question"],
                    "answer": hit["text"],
                    "category": hit["category"],
                }
                for hit in hits
            ],
        },
        ensure_ascii=False,
    )
