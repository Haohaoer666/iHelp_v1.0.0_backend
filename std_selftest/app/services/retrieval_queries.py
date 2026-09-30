"""
    为高风险售后流程创建仅用于检索的查询扩展.
"""

import json
import re
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from app.config import settings


@dataclass(frozen=True)
class QueryExpansion:
    queries: list[str]
    source: str = "model"


def _strip_code_fence(value: str) -> str:
    value = value.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    return value.strip()


def _deduplicate_queries(original: str, queries: list[str], max_queries: int) -> list[str]:
    """
        查询去重与截断工具函数
    """
    ordered = [original, *queries]
    deduplicated: list[str] = []
    seen: set[str] = set()
    for item in ordered:
        normalized = item.strip()
        key = normalized.casefold()     # casefold：转为小写归一化，大小写不敏感去重（比lower更适合多语言）
        if not normalized or key in seen:
            continue
        seen.add(key)
        deduplicated.append(normalized)
        if len(deduplicated) >= max_queries:
            break
    return deduplicated


def build_query_expansion_prompt(resolved_query: str) -> str:
    return f"""你是电商售后检索查询扩写器。围绕退款退货、售后政策，把原问题扩成侧重不同政策条款的多条检索查询。

要求：
1. 只输出 JSON，并且只包含 queries 一个字段。
2. queries 必须是非空字符串数组，推荐 2 到 4 条。
3. 每条查询要侧重不同条件，例如时效、商品状态、运费、拒绝场景或操作流程。
4. 不添加原问题中不存在的订单事实，不输出 eligibility 结论。

格式：
{{"queries":["检索查询 1","检索查询 2"]}}

原问题：{resolved_query}
输出："""


async def expand_retrieval_queries(
    resolved_query: str,
    model: BaseChatModel,
    *,
    max_queries: int | None = None,
) -> QueryExpansion:
    """
        异步查询扩展函数：接收消歧后的用户问句，调用大模型生成多条用于知识库检索的子查询；
        清洗模型输出的markdown代码块，解析并强校验JSON结构；
        一旦解析/校验失败，自动降级兜底，只使用原始查询；
        对原始查询+模型生成查询合并、去重、截断，封装为QueryExpansion对象返回。
    """
    limit = max_queries or settings.query_expansion_max_queries
    response = await model.ainvoke(
        [HumanMessage(content=build_query_expansion_prompt(resolved_query))]
    )
    content = getattr(response, "content", "") or ""
    if not isinstance(content, str):
        content = str(content)

    try:
        payload = json.loads(_strip_code_fence(content))
        if not isinstance(payload, dict) or set(payload) != {"queries"}:
            raise ValueError("query expansion must only contain queries")
        model_queries = payload["queries"]
        if not isinstance(model_queries, list) or not all(
            isinstance(item, str) for item in model_queries
        ):
            raise ValueError("queries must be a string array")
    except (TypeError, ValueError, KeyError, json.JSONDecodeError):
        return QueryExpansion(
            queries=_deduplicate_queries(resolved_query, [], limit),
            source="fallback",
        )

    return QueryExpansion(
        queries=_deduplicate_queries(resolved_query, model_queries, limit),     # 去重
        source="model",
    )
