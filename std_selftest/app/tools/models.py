"""
    定义可插拔工具体系的共享数据契约、枚举和数据模型。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal


class ToolOrigin(StrEnum):  # 工具来源
    BUILTIN = "builtin"
    MCP = "mcp"


class SideEffect(StrEnum):  # 工具副作用类型
    READ = "read"
    WRITE = "write"


class ToolCaller(StrEnum):
    AGENT = "agent"             # LangGraph 中的模型 Agent 调用
    TRUSTED_UI = "trusted_ui"   # 用户点击前端按钮触发的可信 UI 调用
    SYSTEM = "system"           # 系统内部调用



@dataclass(frozen=True)
class TicketSlots:
    """
        它保存从用户对话中提取出的建单信息
    """
    ticket_type: str = ""       # 投诉、咨询、售后、其他
    description: str = ""       # 问题描述
    missing_fields: tuple[str, ...] = ()    # 还缺少哪些必填项

    @property
    def complete(self) -> bool:
        return not self.missing_fields and bool(self.description.strip())   #没有缺失字段 并且 问题描述不是空字符串


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Awaitable[Any]]
    origin: ToolOrigin
    server_name: str | None
    side_effect: SideEffect
    requires_ticket_confirmation: bool = False
    agent_visible: bool = True      #是否允许 Agent 看到并绑定


@dataclass(frozen=True)
class ToolExecutionContext:
    conversation_id: str
    tool_call_id: str
    caller: ToolCaller = ToolCaller.AGENT
    trusted_ui_confirmation: bool = False
    ticket_request_allowed: bool = False
    ticket_batch_valid: bool = True
    ticket_slots: TicketSlots | None = None
    emit: Callable[[dict[str, Any]], None] | None = None


@dataclass
class ToolExecutionResult:
    tool_name: str
    tool_args: dict[str, Any]
    result: Any = None
    success: bool = False
    error: str = ""
    status: Literal[
        "success",
        "failure",
        "timeout",
        "validation_rejected",
        "permission_denied",
        "not_found",
    ] = "failure"
    error_kind: str = ""
    retry_count: int = 0
    duration_ms: int = 0

    def model_payload(self) -> dict[str, Any]:
        """
            把它转换成模型能看懂的稳定 JSON。
        """
        if self.success:
            return {
                "ok": True,
                "status": self.status,
                "data": self.result,
            }
        return {
            "ok": False,
            "status": self.status,
            "error": {
                "category": self.error_kind or "failure",
                "message": self.error,
            },
        }
