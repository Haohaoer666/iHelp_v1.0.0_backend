"""
    消解上下文指代，标准化用户当前查询语句
"""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage


CONTEXT_MARKERS = (
    "它",
    "他",
    "这个",
    "这",
    "那个",
    "那",
    "这单",
    "那单",
    "该单",
    "上述",
    "刚才",
    "又到",
    "到哪一步",
    "到哪了",
    "怎么办",
)
COLLOQUIAL_MARKERS = (
    "啥",
    "咋",
    "打回来",
    "退钱",
    "能不能退",
    "能退吗",
    "什么时候打",
)


@dataclass(frozen=True)
class QueryUnderstandingResult:
    resolved_query: str
    changed: bool
    source: str = "model"


def needs_query_understanding(message: str) -> bool:
    """
        判断当前用户消息是否需要做查询理解
    """
    normalized = message.strip()
    return any(marker in normalized for marker in (*CONTEXT_MARKERS, *COLLOQUIAL_MARKERS))


def _content_text(content: object) -> str:
    """转字符串"""
    return content if isinstance(content, str) else str(content)


def _strip_code_fence(value: str) -> str:
    """
        清洗模型返回内容：剥离markdown ```json 代码块标记，防止json.loads解析失败
    """
    value = value.strip()
    if not value.startswith("```"):
        return value
    value = re.sub(r"^```(?:json)?\s*", "", value)
    value = re.sub(r"\s*```$", "", value)
    return value.strip()


def _history_text(history: Sequence[BaseMessage]) -> str:
    """
        把历史消息列表，拼接成可读文本，给到prompt
    """
    lines: list[str] = []
    for message in history:
        role = "用户" if isinstance(message, HumanMessage) else "客服"
        content = _content_text(message.content).strip()
        if content:
            lines.append(f"{role}：{content}")
    return "\n".join(lines) or "无"


def build_query_understanding_prompt(
    history: Sequence[BaseMessage],
    current_message: str,
) -> str:
    """
        # 构造查询理解Prompt，指导大模型完成指代消解、问句标准化
    """
    return f"""你是电商客服 Query 理解器。结合最近对话，把当前问题补全成不依赖上下文也能看懂的完整问题，同时把口语化问法归一成标准客服问法。

规则：
1. 指代必须结合最近对话消解，例如“它”“这个”“那单”要补成具体订单、商品或售后对象。
2. 完整且没有歧义的问题原样返回，不为了改写而改写。
3. 不新增对话中没有出现的订单号、型号、金额、时间或政策事实。
4. 只输出 JSON，并且只包含 resolved_query 一个字段：
{{"resolved_query":"完整问题"}}

最近对话：
{_history_text(history)}

当前问题：{current_message}
输出："""


async def understand_query(
    history: Sequence[BaseMessage] | None,
    current_message: str,
    model: BaseChatModel,
) -> QueryUnderstandingResult:
    if not needs_query_understanding(current_message):
        return QueryUnderstandingResult(
            resolved_query=current_message,
            changed=False,
            source="passthrough",
        )

    prompt = build_query_understanding_prompt(history or [], current_message)
    response = await model.ainvoke([HumanMessage(content=prompt)])
    content = _content_text(getattr(response, "content", "") or "")

    try:
        payload: dict[str, Any] = json.loads(_strip_code_fence(content))
        # 强约束：返回JSON只能有一个key "resolved_query"，防止模型输出多余字段，保证schema严格
        if set(payload) != {"resolved_query"}:
            raise ValueError("query understanding must only contain resolved_query")
        resolved_query = str(payload["resolved_query"]).strip()
        if not resolved_query:
            raise ValueError("resolved_query must not be empty")
    except (TypeError, ValueError, KeyError, json.JSONDecodeError):
        return QueryUnderstandingResult(
            resolved_query=current_message,
            changed=False,
            source="fallback",
        )

    return QueryUnderstandingResult(
        resolved_query=resolved_query,
        changed=resolved_query != current_message,
    )
