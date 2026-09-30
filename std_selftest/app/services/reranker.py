"""
实现了异步的文档重排（Rerank）功能，主要用于 RAG（检索增强生成）或搜索系统中。
它的核心作用是：调用外部的 Rerank API 服务，根据用户查询（query）与候选文档的相关性进行重新打分，并按分数从高到低排序返回结果。
"""


import httpx

from app.config import settings


_client: httpx.AsyncClient | None = None

"""
    获取httpx异步单例客户端
    return: httpx.AsyncClient 异步http客户端实例
"""
def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=30) #  创建一个异步 HTTP 客户端，所有请求默认超时时间 30 秒
    return _client

#  重排函数：调用外部Rerank重排接口，对候选文档列表按和query相关性重新打分排序
async def rerank(
    query: str,
    documents: list[str],
    top_k: int | None = None,
) -> list[dict]:
    if not documents:
        return []
    # 构造请求体：传入重排模型名称、查询、待排文档、返回条数
    payload = {
        "model": settings.rerank_model,
        "query": query,
        "documents": documents,
        "top_n": top_k or min(settings.rerank_top_k, len(documents)),
    }
    # POST请求调用远程Rerank服务接口，携带鉴权token，json传参
    response = await _get_client().post(
        f"{settings.rerank_base_url.rstrip('/')}/rerank",
        headers={"Authorization": f"Bearer {settings.rerank_api_key}"},
        json=payload,
    )
    response.raise_for_status()     # 如果HTTP状态码异常，直接抛出异常
    rows = response.json().get("results", [])
    results = [
        {
            "index": int(row["index"]),
            "score": float(row.get("relevance_score") or row.get("score") or 0),
            "document": documents[int(row["index"])],
        }
        for row in rows
    ]
    # 按相关性分数从高到低排序后返回
    return sorted(results, key=lambda item: item["score"], reverse=True)
