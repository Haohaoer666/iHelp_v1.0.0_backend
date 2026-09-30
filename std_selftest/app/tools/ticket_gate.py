"""
    优先用规则快速识别明确建单指令，识别不到再调用大模型做意图判断，严格校验 LLM 输出 JSON 格式，解析生成`TicketGateDecision`，
    输出是否允许建单、工单类型、问题描述和缺失字段，作为后续动态控制`create_ticket`工具可见性与权限校验的数据源。
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage

from app.tools.models import TicketSlots


ALLOWED_TICKET_TYPES = ("投诉", "咨询", "售后", "其他")


class TicketGateError(ValueError):
    """Raised when the ticket gate violates its JSON contract."""


@dataclass(frozen=True)
class TicketGateDecision:
    explicit_request: bool
    ticket_type: str = ""
    description: str = ""
    missing_fields: tuple[str, ...] = ()

    @property
    def slots(self) -> TicketSlots:
        return TicketSlots(
            ticket_type=self.ticket_type,
            description=self.description,
            missing_fields=self.missing_fields,
        )


def _clean_json(value: str) -> str:
    value = value.strip()
    if not value.startswith("```"):
        return value
    value = re.sub(r"^```(?:json)?\s*", "", value)
    value = re.sub(r"\s*```$", "", value)
    return value.strip()


def parse_ticket_gate_response(value: str) -> TicketGateDecision:
    try:
        payload = json.loads(_clean_json(value))
    except json.JSONDecodeError as exc:
        raise TicketGateError("建单判断不是合法 JSON") from exc
    required = {
        "explicit_request",
        "ticket_type",
        "description",
        "missing_fields",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise TicketGateError("建单判断 JSON 字段不合法")

    explicit = payload["explicit_request"]
    ticket_type = payload["ticket_type"]
    description = payload["description"]
    missing_fields = payload["missing_fields"]
    if not isinstance(explicit, bool):
        raise TicketGateError("explicit_request 必须是布尔值")
    if ticket_type not in ("", *ALLOWED_TICKET_TYPES):
        raise TicketGateError("ticket_type 不合法")
    if not isinstance(description, str):
        raise TicketGateError("description 必须是字符串")
    if (
        not isinstance(missing_fields, list)
        or not all(isinstance(item, str) for item in missing_fields)
    ):
        raise TicketGateError("missing_fields 必须是字符串数组")

    if not explicit:
        return TicketGateDecision(explicit_request=False)
    effective_type = ticket_type or "其他"
    effective_missing = (
        ("description",)
        if not description.strip() or "description" in missing_fields
        else ()
    )
    return TicketGateDecision(
        explicit_request=True,
        ticket_type=effective_type,
        description=description.strip(),
        missing_fields=effective_missing,
    )


def _history_text(history: Sequence[BaseMessage] | None) -> str:
    if not history:
        return "无"
    lines = []
    for message in history:
        role = "用户" if isinstance(message, HumanMessage) else "客服"
        content = (
            message.content
            if isinstance(message.content, str)
            else str(message.content)
        )
        if content.strip():
            lines.append(f"{role}：{content.strip()}")
    return "\n".join(lines) or "无"


def build_ticket_gate_prompt(
    message: str,
    history: Sequence[BaseMessage] | None = None,
) -> str:
    return f"""你是客服建单意图闸。只判断用户是否明确要求创建工单，不执行建单。

严格输出 JSON，不要 Markdown，不要解释：
{{"explicit_request":true,"ticket_type":"投诉|咨询|售后|其他",
"description":"用户原文中的问题描述","missing_fields":["description"]}}

规则：
- 只有用户明确说“建工单、提交工单、帮我记录并建单”等才是明确建单。
- 单纯投诉、询问进度、表达不满，不等于明确要求建单。
- description 只能摘录或概括对话中已经出现的事实，不得补造。
- 缺少问题描述时 description 留空，并在 missing_fields 填写 description。
- 没有明确建单时，其他字段留空，missing_fields 为空数组。

最近对话：
{_history_text(history)}

用户：{message}
输出："""


def deterministic_ticket_request(message: str) -> TicketGateDecision | None:
    """Recognize unambiguous ticket commands without an LLM call."""
    normalized = message.strip()
    if not normalized:
        return None
    if re.search(
        r"(别|不要|不用|先别|暂不).{0,4}(建|创建|提交).{0,3}工单",
        normalized,
    ) or "别建单" in normalized:
        return TicketGateDecision(explicit_request=False)
    markers = (
        "帮我建个工单",
        "帮我建工单",
        "创建工单",
        "创建一个工单",
        "提交工单",
        "建个工单",
        "建工单",
    )
    marker = next((item for item in markers if item in normalized), None)
    if marker is None:
        return None

    remainder = normalized.replace(marker, "", 1)
    remainder = re.sub(r"^[\s，,：:。]+", "", remainder)
    remainder = remainder.strip()
    ticket_type = "其他"
    if "投诉" in normalized:
        ticket_type = "投诉"
    elif any(word in normalized for word in ("退款", "退货", "换货", "售后")):
        ticket_type = "售后"
    missing_fields = () if remainder else ("description",)
    return TicketGateDecision(
        explicit_request=True,
        ticket_type=ticket_type,
        description=remainder,
        missing_fields=missing_fields,
    )


async def detect_ticket_request(
    message: str,
    history: Sequence[BaseMessage] | None,
    model: BaseChatModel,
) -> TicketGateDecision:
    deterministic = deterministic_ticket_request(message)
    if deterministic is not None:
        return deterministic
    try:
        response = await model.ainvoke(
            [HumanMessage(content=build_ticket_gate_prompt(message, history))]
        )
    except Exception:
        return TicketGateDecision(explicit_request=False)
    content = getattr(response, "content", "") or ""
    if not isinstance(content, str):
        content = str(content)
    return parse_ticket_gate_response(content)
