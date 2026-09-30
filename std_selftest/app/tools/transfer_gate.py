"""Detect human-transfer requests and extract the issue from context."""

from __future__ import annotations

import re
from collections.abc import Sequence

from langchain_core.messages import BaseMessage, HumanMessage


DIRECT_TRANSFER_PHRASES = (
    "转接人工",
    "转人工",
    "找人工",
    "联系人工",
    "接人工",
)

TRANSFER_MENTIONS = (
    *DIRECT_TRANSFER_PHRASES,
    "人工客服",
)

NEGATED_TRANSFER_PATTERN = re.compile(
    r"(?:(?<!要)不要|不用|不需要|无需|别|先不|暂不|不想|不(?!要)).{0,4}"
    r"(转接人工|转人工|人工客服|找人工|联系人工|接人工)"
)

TRANSFER_INFO_QUERY_PATTERN = re.compile(
    r"(几点|工作时间|上班时间|下班时间|服务时间)"
)

GENERIC_TRANSFER_REQUEST_PATTERN = re.compile(
    r"(我要|我想|我想要|想要|需要|请|麻烦|帮我|能否|可以|要|有没有)"
    r".{0,6}人工客服"
    r"|人工客服(?:在吗|在么|吗|吧|呢)?$"
)


def _normalize_transfer_text(text: str) -> str:
    return re.sub(r"[\s，,。！!？?；;：:、]+", "", text.strip().lower())


STANDALONE_TRANSFER_CANCEL_COMMANDS = frozenset(
    {
        "取消转人工",
        "不转人工",
        "不找人工",
        "不接人工",
        "不联系人工",
        "不想转人工",
    }
)


def is_standalone_transfer_cancel(text: str) -> bool:
    return _normalize_transfer_text(text) in STANDALONE_TRANSFER_CANCEL_COMMANDS


def is_human_transfer_request(text: str) -> bool:
    """
        判断用户输入是否是【发起转人工请求】
        返回True：是转人工请求；False：不是（或者是取消转人工/单纯查询信息）
    """
    normalized = _normalize_transfer_text(text)
    if not normalized or is_human_transfer_cancel(text):
        return False
    if TRANSFER_INFO_QUERY_PATTERN.search(normalized):
        return False
    if any(phrase in normalized for phrase in DIRECT_TRANSFER_PHRASES):
        return True
    if normalized in {"人工", "人工客服"}:
        return True
    return bool(GENERIC_TRANSFER_REQUEST_PATTERN.search(normalized))


def is_human_transfer_cancel(text: str) -> bool:
    """
        判断用户是否取消转人工
    """
    normalized = _normalize_transfer_text(text)
    if normalized in {
        "取消",
        "算了",
        "不了",
        "不用了",
        "没事了",
        "先不用了",
        "暂时不用了",
        "我不转了",
    }:
        return True
    if NEGATED_TRANSFER_PATTERN.search(normalized):
        return True
    return any(
        marker in normalized
        for marker in (
            "取消转人工",
            "不转人工",
            "不转了",
            "不用转人工",
            "先不转人工",
            "暂不转人工",
            "不想转人工",
        )
    )


def extract_transfer_issue(text: str) -> str:
    """从用户的转人工指令里，剥离“转人工”这类触发词，提取剩下的【问题描述】
    如果不是转人工请求，或者剥离后没有有效内容，返回空字符串 """
    if not is_human_transfer_request(text):
        return ""
    cleaned = text.strip()
    for phrase in sorted(TRANSFER_MENTIONS, key=len, reverse=True):
        cleaned = cleaned.replace(phrase, "")
    cleaned = re.sub(r"^[\s，,。；;：:、]+", "", cleaned)
    cleaned = re.sub(r"^(我要|我想|帮我|麻烦|请)+[\s，,。；;:、]*", "", cleaned)
    cleaned = re.sub(r"[\s，,。；;:、]+$", "", cleaned)
    if cleaned in {"", "客服", "人工"}:
        return ""
    return cleaned


def latest_problem_from_messages(
    messages: Sequence[BaseMessage] | None,
    current_message: str,
) -> str:
    """Return the latest previous user problem, excluding the transfer command."""
    for item in reversed(list(messages or [])):
        if not isinstance(item, HumanMessage):
            continue
        content = (
            item.content
            if isinstance(item.content, str)
            else str(item.content)
        ).strip()
        if not content or content == current_message:
            continue
        if is_human_transfer_request(content):
            continue
        return content
    return ""
