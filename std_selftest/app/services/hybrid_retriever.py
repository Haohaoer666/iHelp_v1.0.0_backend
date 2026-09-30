"""
`retrieve`是检索入口函数：对用户 query 做标准化、改写扩写，根据策略执行向量检索 / BM25 检索 / 混合检索；
混合检索时会调用 rerank 重排并过滤低分文档，最后统一补齐相关性分数，返回召回文档列表。
"""

from app.config import settings
from app.services.embedding import embed_query
from app.services.query_understanding import build_retrieval_query, normalize_query
from app.services.reranker import rerank
from app.services.vector_store import VectorStore


async def retrieve(
    query: str,
    category: str | None = None,
    strategy: str = "hybrid_rerank",
    top_k: int = 10,
) -> list[dict]:
    normalized = await normalize_query(query)   # query改写
    retrieval_query = build_retrieval_query(normalized) # query改写-->query扩写(添加了词的近义词)
    category = category or normalized.get("category")   # 优先级：传入的category > 从标准化query中自动提取的category
    store = VectorStore()   # 实例化向量数据库存储对象
    try:
        if strategy == "dense":
            vector = await embed_query(retrieval_query)
            hits = await store.search(vector, top_k=top_k, category=category)
        elif strategy == "bm25":
            hits = await store.search_bm25(retrieval_query, top_k=top_k, category=category)
        else:
            vector = await embed_query(retrieval_query) # 生成dense向量
            hits = await store.hybrid_search(           # 执行混合检索：dense向量 + BM25多路召回
                vector,
                retrieval_query,
                top_k=top_k,
                category=category,
            )
            if strategy == "hybrid_rerank" and hits:    # 如果策略是hybrid_rerank并且召回结果不为空，执行重排模型二次打分
                ranked = await rerank(                  # 取出所有召回chunk的文本，送入rerank模型做相关性打分
                    retrieval_query,
                    [hit["text"] for hit in hits],
                    top_k=top_k,
                )
                hits = [hits[item["index"]] | {"score": item["score"]} for item in ranked]  # 根据rerank返回的索引，从原始召回结果中取出对应记录，并追加rerank的score分数
                hits = [hit for hit in hits if hit["score"] >= settings.rerank_score_threshold] # 过滤掉低于重排分数阈值的低相关文档
    finally:
        await store.close()
    # 遍历检索结果：如果没有score字段，则使用distance（向量相似度距离）作为score兜底
    for hit in hits:
        hit.setdefault("score", hit.get("distance", 0.0))
    return hits



#特殊重排,防止大模型lost in the middle
def assemble_evidence(hits: list[dict]) -> list[dict]:
    if not hits:
        return []
    ordered = sorted(hits, key=lambda item: item.get("score", 0), reverse=True)
    if len(ordered) >= 2:
        ordered = [ordered[0], *ordered[2:], ordered[1]]
    return [
        {
            "citation_id": index + 1,
            "chunk_id": hit["chunk_id"],
            "category": hit.get("category", ""),
            "text": hit.get("text", ""),
            "section_path": hit.get("section_path", ""),
            "score": float(hit.get("score", 0)),
        }
        for index, hit in enumerate(ordered)
    ]
