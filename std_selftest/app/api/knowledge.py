import logging
from pathlib import Path

from fastapi import APIRouter
from sqlalchemy import func, select

from app.db.models import KnowledgeChunk
from app.db.session import AsyncSessionLocal
from app.schemas.knowledge import KnowledgeBuildRequest, KnowledgeSearchRequest
from app.services.document_loader import SUPPORTED_SUFFIXES
from app.services.knowledge_pipeline import (
    build_knowledge_base,
    search_knowledge,
    sync_pending_chunks,
)
from app.services.qa_miner import mine_qa_from_messages

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])
logger = logging.getLogger(__name__)


#这个路由是知识库的“状态统计接口”，对应前端 RAG 页面顶部的 5 个统计卡片：文档\知识块\已向量化\待处理\失败
@router.get("/stats")
async def stats() -> dict:
    docs_dir = Path(__file__).resolve().parents[2] / "knowledge_docs"
    async with AsyncSessionLocal() as db:
        total = await db.scalar(select(func.count()).select_from(KnowledgeChunk)) or 0
        vectorized = (
            await db.scalar(
                select(func.count())
                .select_from(KnowledgeChunk)
                .where(KnowledgeChunk.vector_status == "vectorized")
            )
            or 0
        )
        pending = (
            await db.scalar(
                select(func.count())
                .select_from(KnowledgeChunk)
                .where(KnowledgeChunk.vector_status == "pending")
            )
            or 0
        )
        failed = (
            await db.scalar(
                select(func.count())
                .select_from(KnowledgeChunk)
                .where(KnowledgeChunk.vector_status == "failed")
            )
            or 0
        )
    return {
        "documents": len(
            [
                path
                for path in docs_dir.iterdir()
                if path.suffix.lower() in SUPPORTED_SUFFIXES
            ]
        ),
        "chunks": total,
        "vectorized": vectorized,
        "pending": pending,
        "failed": failed,
    }


#对应前端：展示最近__条知识块。
"""
从 MySQL 读取最近写入的知识块
按 id 倒序排列
最多返回 200 条
转成前端需要的 JSON 列表
"""
@router.get("/chunks")
async def chunks(limit: int = 50) -> list[dict]:
    async with AsyncSessionLocal() as db:
        result = await db.scalars(
            select(KnowledgeChunk)
            .order_by(KnowledgeChunk.id.desc())
            .limit(min(max(limit, 1), 200))
        )
        rows = list(result.all())
    return [
        {
            "chunk_id": row.chunk_id,
            "category": row.category,
            "questions": row.questions,
            "answer": row.answer[:240],
            "section_path": row.section_path,
            "content_type": row.content_type,
            "is_critical": row.is_critical,
            "vector_status": row.vector_status,
            "milvus_id": row.milvus_id,
        }
        for row in rows
    ]


#RAG 的“在线语义检索测试接口
"""
把用户输入的问题向量化
到 Milvus 里做 Top-K 相似度检索
返回命中的知识块
"""
@router.post("/search")
async def search(req: KnowledgeSearchRequest) -> list[dict]:
    return await search_knowledge(req.query, top_k=req.top_k)

#这个接口是 RAG 知识库的“建库/同步入口”，前端知识库页面里的按钮：重建知识库
@router.post("/build")
async def build(req: KnowledgeBuildRequest) -> dict:
    docs_dir = Path(__file__).resolve().parents[2] / "knowledge_docs"
    result = await build_knowledge_base(docs_dir, rebuild=req.rebuild)
    logger.info(
        "知识库构建完成 rebuild=%s files=%s inserted=%s synced=%s failed=%s",
        req.rebuild,
        result.get("files"),
        result.get("inserted"),
        result.get("synced"),
        result.get("failed"),
    )
    return result


#从历史客服对话中挖掘 QA 知识，并立即同步到 Milvus”的入口
@router.post("/mine")
async def mine() -> dict:
    result = await mine_qa_from_messages()
    result["sync"] = await sync_pending_chunks()
    return result
