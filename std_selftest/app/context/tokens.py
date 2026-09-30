"""Conservative token estimation for Chinese customer-service messages."""

import math
import re
from collections.abc import Iterable, Sequence
import json
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.messages.utils import count_tokens_approximately


CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
MESSAGE_OVERHEAD = 4


def estimate_text_tokens(text: str, chinese_tokens_per_char: float) -> int:
    """
        文本token估算：区分中日韩字符(CJK)与英文字符
        CJK字符：按配置系数折算token；英文/符号：每4个字符折合1个token
        返回向上取整结果，最小保底1 token
    """
    if not text:
        return 0
    cjk_count = len(CJK_PATTERN.findall(text))
    other_count = len(text) - cjk_count
    estimate = cjk_count * chinese_tokens_per_char + other_count / 4
    return max(1, math.ceil(estimate))


def _content_text(content: object) -> str:
    return content if isinstance(content, str) else str(content)


def estimate_message_tokens(
    message: BaseMessage,
    chinese_tokens_per_char: float,
) -> int:
    """估算单条LangChain消息的token数量：消息固定开销 + 文本内容折算token"""
    return MESSAGE_OVERHEAD + estimate_text_tokens(
        _content_text(message.content),
        chinese_tokens_per_char,
    )


def estimate_messages_tokens(
    messages: Iterable[BaseMessage],
    chinese_tokens_per_char: float,
) -> int:
    """批量估算一组LangChain消息的总token数，累加单条消息的token估值"""
    return sum(
        estimate_message_tokens(message, chinese_tokens_per_char)
        for message in messages
    )


def count_prompt_tokens(
    messages: Sequence[BaseMessage],
    *,
    tools: Sequence[Any] | None = None,
) -> int:
    """Count the complete model request, including optional tool schemas."""
    try:
        return count_tokens_approximately(list(messages), tools=tools)
    except TypeError:
        base = count_tokens_approximately(list(messages))
        if not tools:
            return base
        from langchain_core.utils.function_calling import (
            convert_to_openai_tool,
        )

        tool_text = "\n".join(
            json.dumps(
                convert_to_openai_tool(tool),
                ensure_ascii=False,
            )
            for tool in tools
        )
        return base + count_tokens_approximately(
            [HumanMessage(tool_text)]
        )


def truncate_text_to_tokens(
    text: str,
    max_tokens: int,
    chinese_tokens_per_char: float,
) -> str:
    """Trim text to a conservative token budget, including the suffix."""
    if max_tokens <= 0:
        return ""
    if estimate_text_tokens(text, chinese_tokens_per_char) <= max_tokens:
        return text

    suffix = "..."
    content_budget = max(
        0,
        max_tokens - estimate_text_tokens(suffix, chinese_tokens_per_char),
    )
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_text_tokens(
            text[:middle],
            chinese_tokens_per_char,
        ) <= content_budget:
            low = middle
        else:
            high = middle - 1
    return text[:low].rstrip() + suffix


def truncate_tool_content(
    content: str,
    max_tokens: int,
    chinese_tokens_per_char: float,
) -> str:
    """
        将工具返回结果（tool payload）做截断，控制塞进模型上下文前的token上限。
    """
    return truncate_text_to_tokens(
        content,
        max_tokens,
        chinese_tokens_per_char,
    )
