import asyncio
from dataclasses import replace
from typing import Any

import httpx
import pytest

from app.config import settings
from app.tools.executor import ToolExecutor
from app.tools.executor import trusted_ui_context
from app.tools.builtin.create_ticket import TOOL_DEFINITION
from app.tools.models import (
    SideEffect,
    ToolCaller,
    ToolDefinition,
    ToolExecutionContext,
    ToolExecutionResult,
    ToolOrigin,
)
from app.tools.registry import ToolRegistry


class RecordingAudit:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    async def record(self, **kwargs: Any) -> None:
        self.rows.append(kwargs)


def build_executor(
    definition: ToolDefinition,
    *,
    interrupt_fn=None,
) -> tuple[ToolExecutor, RecordingAudit]:
    registry = ToolRegistry()
    registry.register(definition)
    audit = RecordingAudit()
    executor = ToolExecutor(
        registry=registry,
        audit=audit,
        interrupt_fn=interrupt_fn,
    )
    return executor, audit


async def test_schema_validation_is_returned_and_audited():
    async def handler(**kwargs):
        raise AssertionError("handler must not run")

    executor, audit = build_executor(
        ToolDefinition(
            name="query_order",
            description="查订单",
            input_schema={
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "minLength": 1},
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
            handler=handler,
            origin=ToolOrigin.BUILTIN,
            server_name=None,
            side_effect=SideEffect.READ,
        )
    )

    result = await executor.execute(
        "query_order",
        {},
        ToolExecutionContext("s1", "c1"),
    )

    assert result.status == "validation_rejected"
    assert result.error_kind == "invalid_arguments"
    assert audit.rows[0]["status"] == "validation_rejected"


async def test_agent_cannot_confirm_create_ticket():
    async def handler(**kwargs):
        raise AssertionError("handler must not run")

    executor, audit = build_executor(
        ToolDefinition(
            name="create_ticket",
            description="建工单",
            input_schema={
                "type": "object",
                "properties": {
                    "conversation_id": {"type": "string"},
                    "description": {"type": "string"},
                    "ticket_type": {"type": "string"},
                },
                "required": [
                    "conversation_id",
                    "description",
                    "ticket_type",
                ],
            },
            handler=handler,
            origin=ToolOrigin.BUILTIN,
            server_name=None,
            side_effect=SideEffect.WRITE,
            requires_ticket_confirmation=True,
        )
    )

    result = await executor.execute(
        "create_ticket",
        {
            "conversation_id": "s1",
            "description": "重复扣款",
            "ticket_type": "投诉",
        },
        ToolExecutionContext(
            "s1",
            "c1",
            caller=ToolCaller.AGENT,
            ticket_request_allowed=False,
        ),
    )

    assert result.status == "permission_denied"
    assert audit.rows[0]["status"] == "permission_denied"


async def test_read_timeout_retries_then_returns_timeout(monkeypatch):
    calls = 0

    async def slow(**kwargs):
        nonlocal calls
        calls += 1
        await asyncio.sleep(1)

    executor, audit = build_executor(
        ToolDefinition(
            name="slow_read",
            description="慢查询",
            input_schema={"type": "object", "properties": {}},
            handler=slow,
            origin=ToolOrigin.BUILTIN,
            server_name=None,
            side_effect=SideEffect.READ,
        )
    )
    monkeypatch.setattr(settings, "tool_timeout_seconds", 0.01)
    monkeypatch.setattr(settings, "tool_max_retries", 2)

    result = await executor.execute(
        "slow_read",
        {},
        ToolExecutionContext("s1", "c1"),
    )

    assert result.status == "timeout"
    assert result.retry_count == 1
    assert calls == 2
    assert audit.rows[-1]["retry_count"] == 1


async def test_write_timeout_does_not_retry(monkeypatch):
    calls = 0

    async def slow(**kwargs):
        nonlocal calls
        calls += 1
        await asyncio.sleep(1)

    executor, audit = build_executor(
        ToolDefinition(
            name="slow_write",
            description="慢写入",
            input_schema={"type": "object", "properties": {}},
            handler=slow,
            origin=ToolOrigin.BUILTIN,
            server_name=None,
            side_effect=SideEffect.WRITE,
        )
    )
    monkeypatch.setattr(settings, "tool_timeout_seconds", 0.01)

    result = await executor.execute(
        "slow_write",
        {},
        ToolExecutionContext(
            "s1",
            "c1",
            caller=ToolCaller.TRUSTED_UI,
            trusted_ui_confirmation=True,
        ),
    )

    assert result.status == "timeout"
    assert result.retry_count == 0
    assert calls == 1
    assert audit.rows[-1]["retry_count"] == 0


async def test_result_translates_internal_status():
    async def handler(**kwargs):
        return {"ticket_id": "TK1", "status": "open"}

    executor, _ = build_executor(
        ToolDefinition(
            name="query_ticket",
            description="查工单",
            input_schema={"type": "object", "properties": {}},
            handler=handler,
            origin=ToolOrigin.BUILTIN,
            server_name=None,
            side_effect=SideEffect.READ,
        )
    )

    result = await executor.execute(
        "query_ticket",
        {},
        ToolExecutionContext("s1", "c1"),
    )

    assert result.success is True
    assert result.result["status"] == "待处理"


def test_mcp_text_content_is_parsed_before_result_projection():
    from app.tools.executor import normalize_tool_result

    result = normalize_tool_result(
        [
            {
                "type": "text",
                "text": '{"order_id":"1001","status":"open"}',
            }
        ]
    )

    assert result == {
        "order_id": "1001",
        "status": "待处理",
    }


async def test_transient_read_failure_retries_once(monkeypatch):
    calls = 0

    async def flaky(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("temporary")
        return {"ok": True}

    executor, _ = build_executor(
        ToolDefinition(
            name="flaky_read",
            description="网络查询",
            input_schema={"type": "object", "properties": {}},
            handler=flaky,
            origin=ToolOrigin.BUILTIN,
            server_name=None,
            side_effect=SideEffect.READ,
        )
    )
    monkeypatch.setattr(settings, "tool_max_retries", 2)

    result = await executor.execute(
        "flaky_read",
        {},
        ToolExecutionContext("s1", "c1"),
    )

    assert result.success is True
    assert result.retry_count == 1
    assert calls == 2


async def test_unknown_tool_has_stable_error():
    executor = ToolExecutor(
        registry=ToolRegistry(),
        audit=RecordingAudit(),
        interrupt_fn=lambda value: value,
    )

    result = await executor.execute(
        "missing",
        {},
        ToolExecutionContext("s1", "c1"),
    )

    assert result == ToolExecutionResult(
        tool_name="missing",
        tool_args={},
        result=None,
        success=False,
        error="未知工具: missing",
        status="failure",
        error_kind="unknown_tool",
        retry_count=0,
        duration_ms=result.duration_ms,
    )


@pytest.mark.parametrize("ticket_type", ["退款", "换货"])
async def test_trusted_ui_ticket_types_pass_schema_validation(ticket_type):
    captured = {}

    async def handler(**kwargs):
        captured.update(kwargs)
        return {"ticket_id": "TK1", "status": "open"}

    registry = ToolRegistry()
    registry.register(replace(TOOL_DEFINITION, handler=handler))
    executor = ToolExecutor(
        registry=registry,
        audit=RecordingAudit(),
        interrupt_fn=lambda value: value,
    )

    result = await executor.execute(
        "create_ticket",
        {
            "conversation_id": "s-ui",
            "description": "前端表单提交",
            "ticket_type": ticket_type,
        },
        trusted_ui_context("s-ui", f"ui-{ticket_type}"),
    )

    assert result.success is True
    assert captured["ticket_type"] == ticket_type
