"""
知识库问答前置校验：调用混合检索召回文档，无召回则记录低置信问题；
召回后组装证据，交由大模型判断证据是否足够回答问题，证据不足同样落库记录；校验通过则返回证据上下文供上层生成答案。
"""
import json

from langchain_core.messages import HumanMessage

from app.config import settings
from app.core.llm import get_chat_model
from app.db.repository import insert_low_confidence_question, load_chunk_section_paths
from app.services.hybrid_retriever import assemble_evidence, retrieve


async def retrieve_knowledge_evidence(
    keyword: str,
    category: str | None = None,
) -> dict:
    """
        检索知识库片段，只做召回，不做信息充足度判定
    """
    hits = await retrieve(
        keyword,
        category=category,
        strategy="hybrid_rerank",
        top_k=10,
    )
    if not hits:
        return {"matched": False, "evidence": []}

    section_paths = await load_chunk_section_paths([hit["chunk_id"] for hit in hits])
    for hit in hits:
        hit["section_path"] = section_paths.get(hit["chunk_id"], "")

    evidence = assemble_evidence(hits)
    return {"matched": True, "evidence": evidence}


async def retrieve_multi_query(
    queries: list[str],
    category: str | None = None,
) -> dict:
    """
        多查询并行检索：循环执行多条扩展query检索，合并召回结果；
        按chunk_id去重，保留最高分的片段；加载文档路径，组装证据返回。
    """
    merged: list[dict] = []
    failed_queries: list[str] = []

    for query in queries:
        try:
            merged.extend(
                await retrieve(
                    query,
                    category=category,
                    strategy="hybrid_rerank",
                    top_k=10,
                )
            )
        except Exception:
            failed_queries.append(query)

    by_chunk: dict[str, dict] = {}
    for hit in merged:
        chunk_id = hit.get("chunk_id")
        if not chunk_id:
            continue
        current = by_chunk.get(chunk_id)
        if current is None or float(hit.get("score", 0.0)) > float(
            current.get("score", 0.0)
        ):
            by_chunk[chunk_id] = hit

    hits = sorted(  # 将去重后的片段按分数降序排序
        by_chunk.values(),
        key=lambda item: float(item.get("score", 0.0)),
        reverse=True,
    )
    if not hits:
        return {
            "matched": False,
            "evidence": [],
            "retrieval_query_count": len(queries),
            "failed_queries": failed_queries,
        }

    section_paths = await load_chunk_section_paths(     # 批量查询每个chunk对应的文档章节路径
        [hit["chunk_id"] for hit in hits]
    )
    for hit in hits:    # 把章节路径回填到每条召回片段
        hit["section_path"] = section_paths.get(hit["chunk_id"], "")

    return {
        "matched": True,
        "evidence": assemble_evidence(hits),
        "retrieval_query_count": len(queries),
        "failed_queries": failed_queries,
    }


async def assess_policy_evidence(
    keyword: str,
    evidence: list[dict],
    conversation_id: str | None = None,
) -> dict:
    """
        评估召回的售后政策证据质量；**只做证据质量校验，不判定订单是否满足售后资格**
    """
    if not evidence:
        reason = "未召回政策证据"
    elif max(float(item.get("score", 1.0)) for item in evidence) < (
        settings.knowledge_score_threshold
    ):  # 取出所有片段的score最高分，如果最高分小于配置的置信度阈值
        reason = "政策检索置信度不足"
    else:
        reason = ""

    if reason:
        await insert_low_confidence_question(
            keyword,
            reason,
            conversation_id,
            "core_policy_confidence_gate",
        )
        return {
            "sufficient": False,
            "reason": reason,
            "refusal": "抱歉，我暂时没有检索到足够可靠的政策条款。",
        }

    return {
        "sufficient": True,
        "context": build_evidence_text(evidence),
    }


async def assess_evidence(
    keyword: str,
    evidence: list[dict],
    conversation_id: str | None = None,
) -> dict:
    """
        通用证据置信度评估函数：对召回的证据执行两层校验
        第一层：规则校验（有无证据、最高分是否达标）
        第二层：大模型校验，判断现有证据是否足够回答用户问题
    """
    if not evidence:
        reason = "未召回足够证据"
    elif max(float(item.get("score", 1.0)) for item in evidence) < (
        settings.knowledge_score_threshold
    ):
        reason = "检索置信度不足"
    else:
        reason = ""

    if reason:
        await insert_low_confidence_question(
            keyword,
            reason,
            conversation_id,
            "workflow_confidence_gate",
        )
        return {
            "sufficient": False,
            "reason": reason,
            "refusal": "抱歉，我暂时无法依据现有知识回答这个问题。",
        }

    check_prompt = build_sufficiency_prompt(keyword, evidence)
    model = get_chat_model(streaming=False)
    response = await model.ainvoke([HumanMessage(content=check_prompt)])
    try:
        check = json.loads(_extract_json(getattr(response, "content", "") or ""))
    except json.JSONDecodeError:
        check = {"sufficient": False, "reason": "证据判断失败"}

    if not check.get("sufficient"):
        reason = check.get("reason") or "证据不足"
        await insert_low_confidence_question(
            keyword,
            reason,
            conversation_id,
            "workflow_confidence_gate",
        )
        return {
            "sufficient": False,
            "reason": reason,
            "refusal": "抱歉，现有知识不足以可靠回答这个问题。",
        }

    return {
        "sufficient": True,
        "context": build_evidence_text(evidence),
    }


async def answer_knowledge_question(
    keyword: str,
    conversation_id: str | None = None,
    category: str | None = None,
) -> dict:
    """Compatibility wrapper used by the existing query_faq tool."""
    retrieved = await retrieve_knowledge_evidence(keyword, category=category)
    assessed = await assess_evidence(
        keyword,
        retrieved["evidence"],
        conversation_id=conversation_id,
    )
    return {**retrieved, **assessed}


# 构造「证据充足性判断」提示词，交给大模型判断现有证据能不能回答用户问题
def build_sufficiency_prompt(keyword: str, evidence: list[dict]) -> str:
    lines = [
        f"问题：{keyword}",
        "判断标准：只有证据能直接回答原问题，且问题中的型号、品类、实体都能在证据中找到时，才返回 sufficient=true。问题中出现证据里不存在的型号或实体时，必须返回 sufficient=false。",
        "证据：",
    ]
    # 循环把每一条证据带上引用编号拼入prompt
    for item in evidence:
        lines.append(f"[{item['citation_id']}] {item['text']}")
    lines.append('只输出 JSON：{"sufficient": true/false, "reason": "..."}')
    return "\n".join(lines)


#将结构化证据列表拼接成纯文本字符串，每条证据带上引用编号
def build_evidence_text(evidence: list[dict]) -> str:
    return "\n".join(
        f"[{item['citation_id']}] {item['text']}"
        for item in evidence
    )


def _extract_json(value: str) -> str:
    value = value.strip()
    if not value.startswith("```"):
        return value
    lines = value.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()
