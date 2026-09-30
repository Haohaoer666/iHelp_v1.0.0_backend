"""Generate facts-only summaries for an uncovered message span."""

import re
from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage

from app.context.models import ContextMessage
from app.context.tokens import truncate_text_to_tokens


def _content_text(content: object) -> str:
    return content if isinstance(content, str) else str(content)


def _strip_code_fence(value: str) -> str:
    value = value.strip()
    if not value.startswith("```"):
        return value
    value = re.sub(r"^```(?:text|markdown)?\s*", "", value)
    value = re.sub(r"\s*```$", "", value)
    return value.strip()


def build_summary_prompt(messages: Sequence[ContextMessage]) -> str:
    lines = [
        "你只提炼对话中明确出现的事实与诉求。",
        "允许保留：商品、订单号、手机号、明确诉求、未解决问题。",
        "不得编造、推测或补充对话中没有出现的订单、金额、时间、政策或承诺。",
        "寒暄、客套、感谢和闲聊必须删除。",
        "保留数字原样，不要改成示例号。",
        "只输出摘要正文，不要 JSON、标题或 Markdown。",
        "长度通常控制在 20 到 200 个中文字符。",
        "",
        "待压缩对话：",
    ]
    for message in messages:
        role = {
            "user": "用户",
            "assistant": "客服",
            "tool": f"工具[{message.tool_name or 'tool'}]",
        }.get(message.role, message.role)
        lines.append(f"{role}：{message.content}")
    lines.append("摘要：")
    return "\n".join(lines)


async def summarize_span(
    messages: Sequence[ContextMessage],
    model: BaseChatModel,
    *,
    min_chars: int,
    max_chars: int,
    max_tokens: int = 0,
    chinese_tokens_per_char: float = 0.80,
) -> str:
    """
        异步对一段连续消息区间生成纯事实摘要
        1. 组装prompt调用大模型
        2. 清洗返回结果，去掉代码块标记
        3. 超长则截断到max_chars，空结果抛异常
    """
    response = await model.ainvoke(
        [HumanMessage(content=build_summary_prompt(messages))]
    )
    body = _strip_code_fence(
        _content_text(getattr(response, "content", "") or "")
    )
    if not body:
        raise ValueError("summary model returned an empty body")
    if len(body) > max_chars:
        body = body[:max_chars].rstrip()
    if max_tokens > 0:
        body = truncate_text_to_tokens(
            body,
            max_tokens,
            chinese_tokens_per_char,
        )
    return body
