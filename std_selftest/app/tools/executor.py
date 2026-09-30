"""Single validation, permission, execution, and audit entry point."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from typing import Any

import httpx
from jsonschema import Draft202012Validator

from app.config import settings
from app.tools.audit import AuditRecorder
from app.tools.models import (
    SideEffect,
    ToolCaller,
    ToolDefinition,
    ToolExecutionContext,
    ToolExecutionResult,
)
from app.tools.registry import ToolRegistry


logger = logging.getLogger(__name__)


STATUS_LABELS = {
    "open": "待处理",
    "processing": "处理中",
    "closed": "已关闭",
    "refund": "退款",
    "exchange": "换货",
    "repair": "维修",
}

TRANSIENT_ERRORS = (
    TimeoutError,
    ConnectionError,
    OSError,
    httpx.TransportError,
)


def get_langgraph_interrupt():
    from langgraph.types import interrupt

    return interrupt


def normalize_tool_result(value: Any) -> Any:
    """
        将工具handler返回结果转为可序列化的JSON安全数据，同时做枚举/状态文本本地化翻译。
        递归处理：Pydantic对象、元组、列表、字典；对特定字段（status/ticket_type/request_type）做文本映射。
    """
    if hasattr(value, "model_dump"):
        return normalize_tool_result(value.model_dump(mode="json"))
    if isinstance(value, tuple):
        return [normalize_tool_result(item) for item in value]
    if isinstance(value, list):
        if (
            len(value) == 1
            and isinstance(value[0], dict)
            and value[0].get("type") == "text"
            and isinstance(value[0].get("text"), str)
        ):
            text = value[0]["text"]
            try:
                return normalize_tool_result(json.loads(text))
            except json.JSONDecodeError:
                return text
        return [normalize_tool_result(item) for item in value]
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = str(key)
            processed = normalize_tool_result(item)
            if (
                normalized_key
                in {"status", "ticket_type", "request_type"}
                and isinstance(processed, str)
            ):
                processed = STATUS_LABELS.get(processed, processed)
            normalized[normalized_key] = processed
        return normalized
    return value


class ToolExecutor:
    """
        工具执行器，是Agent调用工具的核心执行入口。
        负责工具全生命周期：查找工具、参数预处理、参数校验、权限校验、人工确认、带重试的实际执行、结果归一化、审计落库。
        内置重试逻辑：只读类工具遇到瞬时异常支持重试；写类操作不重试，防止重复提交。
    """
    def __init__(
        self,
        *,
        registry: ToolRegistry,
        audit: AuditRecorder,
        interrupt_fn: Callable[[Any], Any] | None = None,
    ) -> None:
        self.registry = registry
        self.audit = audit
        self.interrupt_fn = interrupt_fn or get_langgraph_interrupt()

    async def execute(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        """
            对外暴露的主入口：执行工具，返回统一封装的执行结果对象
        """
        started = time.perf_counter()
        definition = self.registry.get(tool_name)
        if definition is None:
            return await self._finish(
                ToolExecutionResult(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    success=False,
                    error=f"未知工具: {tool_name}",
                    status="failure",
                    error_kind="unknown_tool",
                ),
                definition=None,
                context=context,
                started=started,
            )

        effective_args = _effective_ticket_args(
            definition,
            tool_args,
            context,
        )

        # 参数校验：校验有效参数是否符合schema约束，有问题直接返回校验失败
        issues = _validate_arguments(definition, effective_args)
        if issues:
            return await self._finish(
                ToolExecutionResult(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    success=False,
                    error="；".join(issues),
                    status="validation_rejected",
                    error_kind="invalid_arguments",
                ),
                definition=definition,
                context=context,
                started=started,
            )
        # 权限校验：检查当前上下文是否拥有调用该工具的权限
        permission_error = _permission_error(definition, context)
        if permission_error:
            return await self._finish(
                ToolExecutionResult(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    success=False,
                    error=permission_error,
                    status="permission_denied",
                    error_kind="permission_denied",
                ),
                definition=definition,
                context=context,
                started=started,
            )

        # 如果工具需要人工确认（建单类高危操作），进入确认流程
        if definition.requires_ticket_confirmation:
            decision = await self._confirm_ticket(
                definition,
                effective_args,
                context,
            )
            if decision is not None:    # decision不为空，代表用户拒绝/取消，直接返回结果，不再执行handler
                return await self._finish(
                    decision,
                    definition=definition,
                    context=context,
                    started=started,
                )

        result = await self._execute_with_retry(
            definition,
            effective_args,
            started,
        )
        return await self._finish(
            result,
            definition=definition,
            context=context,
            started=started,
        )

    async def _confirm_ticket(
        self,
        definition: ToolDefinition,
        effective_args: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult | None:
        """
            人工确认建单弹窗逻辑。
            返回None=确认通过，可以继续执行工具；返回Result对象=用户取消，终止调用。
        """
        if (
            context.caller == ToolCaller.TRUSTED_UI
            and context.trusted_ui_confirmation
        ):
            return None

        preview = {     # 构造预览信息，传给前端展示确认弹窗
            "tool_call_id": context.tool_call_id,
            "ticket_type": effective_args.get("ticket_type", ""),
            "description": effective_args.get("description", ""),
        }
        if context.emit is not None:
            context.emit({"type": "ticket_confirmation", **preview})

        decision = self.interrupt_fn(preview)       # 调用LangGraph interrupt，暂停graph执行，等待人工反馈
        if (
            isinstance(decision, dict)
            and decision.get("type") == "ticket_confirmed"
            and decision.get("tool_call_id") == context.tool_call_id
        ):
            return None
        return ToolExecutionResult(
            tool_name=definition.name,
            tool_args=effective_args,
            success=False,
            error="用户取消或未确认建单",
            status="permission_denied",
            error_kind="permission_denied",
        )

    async def _execute_with_retry(
        self,
        definition: ToolDefinition,
        tool_args: dict[str, Any],
        started: float,
    ) -> ToolExecutionResult:
        """
            带超时、异常捕获、重试的工具真实执行逻辑。
            只对【只读工具】的瞬时异常进行重试；写类工具不重试，避免重复操作。
        """
        max_attempts = max(1, settings.tool_max_retries)
        retry_count = 0
        last_kind = "failure"
        last_error = ""

        while retry_count < max_attempts:
            try:
                raw = await asyncio.wait_for(
                    definition.handler(**tool_args),
                    timeout=settings.tool_timeout_seconds,
                )
            except asyncio.TimeoutError:
                last_kind = "timeout"
                last_error = (
                    f"工具执行超时({settings.tool_timeout_seconds}s)"
                )
            except KeyError as exc:
                # KeyError视为业务上“查无数据”，不再重试，直接返回not_found
                return ToolExecutionResult(
                    tool_name=definition.name,
                    tool_args=tool_args,
                    result=exc.args[0] if exc.args else None,
                    success=False,
                    error=f"未查询到数据: {exc}",
                    status="not_found",
                    error_kind="not_found",
                    retry_count=retry_count,
                    duration_ms=_elapsed_ms(started),
                )
            except TRANSIENT_ERRORS as exc:
                # TRANSIENT_ERRORS：配置的瞬时异常（网络抖动等），标记为可重试
                last_kind = "failure"
                last_error = str(exc)
            except Exception as exc:
                # 其他业务异常，不可重试，直接返回业务错误
                return ToolExecutionResult(
                    tool_name=definition.name,
                    tool_args=tool_args,
                    success=False,
                    error=str(exc),
                    status="failure",
                    error_kind="business_error",
                    retry_count=retry_count,
                    duration_ms=_elapsed_ms(started),
                )
            else:
                # handler执行成功，进入结果归一化
                normalized = normalize_tool_result(raw)
                if normalized is None or normalized in ({}, [], ""):
                    return ToolExecutionResult(     #  结果为空：{} [] ""，视为查无数据
                        tool_name=definition.name,
                        tool_args=tool_args,
                        result=normalized,
                        success=False,
                        error="未查询到数据",
                        status="not_found",
                        error_kind="not_found",
                        retry_count=retry_count,
                        duration_ms=_elapsed_ms(started),
                    )
                if (
                    isinstance(normalized, dict)
                    and normalized.get("matched") is False
                ):
                    return ToolExecutionResult(   # 返回体matched=false，代表没有匹配到数据
                        tool_name=definition.name,
                        tool_args=tool_args,
                        result=normalized,
                        success=False,
                        error="未查询到数据",
                        status="not_found",
                        error_kind="not_found",
                        retry_count=retry_count,
                        duration_ms=_elapsed_ms(started),
                    )

                # 全部校验通过 → 成功返回
                return ToolExecutionResult(
                    tool_name=definition.name,
                    tool_args=tool_args,
                    result=normalized,
                    success=True,
                    status="success",
                    retry_count=retry_count,
                    duration_ms=_elapsed_ms(started),
                )
            # 判断是否允许重试：只有只读工具，并且还没到最大次数，才重试
            can_retry = (
                definition.side_effect == SideEffect.READ
                and retry_count < max_attempts - 1
            )
            if not can_retry:
                break
            retry_count += 1
            await asyncio.sleep(0.2 * retry_count)

        return ToolExecutionResult(
            tool_name=definition.name,
            tool_args=tool_args,
            success=False,
            error=last_error,
            status=last_kind,
            error_kind=last_kind,
            retry_count=retry_count,
            duration_ms=_elapsed_ms(started),
        )

    async def _finish(
        self,
        result: ToolExecutionResult,
        *,
        definition: ToolDefinition | None,
        context: ToolExecutionContext,
        started: float,
    ) -> ToolExecutionResult:
        """
            收尾方法：填充耗时，调用审计Recorder记录日志；审计失败不影响主流程。
        """
        if result.duration_ms == 0:
            result.duration_ms = _elapsed_ms(started)
        try:
            await self.audit.record(
                conversation_id=context.conversation_id or None,
                tool_call_id=context.tool_call_id,
                tool_name=result.tool_name,
                origin=definition.origin.value if definition else "unknown",
                server_name=definition.server_name if definition else None,
                arguments=result.tool_args,
                result_summary=result.model_payload(),
                status=result.status,
                error=result.error,
                retry_count=result.retry_count,
                duration_ms=result.duration_ms,
            )
        except Exception:
            logger.exception(
                "工具审计调用失败 tool_name=%s tool_call_id=%s",
                result.tool_name,
                context.tool_call_id,
            )
        return result


def _validate_arguments(
    definition: ToolDefinition,
    tool_args: dict[str, Any],
) -> list[str]:
    """
        根据工具定义里的JSON Schema（Draft202012规范）校验工具入参。
        返回错误信息字符串列表；列表为空代表参数校验通过。
        使用jsonschema库的Draft202012Validator校验器，遍历所有校验错误，把底层校验异常翻译成业务友好的中文提示。
    """
    validator = Draft202012Validator(definition.input_schema)
    issues = []
    for error in sorted(
        validator.iter_errors(tool_args),
        key=lambda item: list(item.absolute_path),
    ):
        path = ".".join(str(item) for item in error.absolute_path)
        if error.validator == "required":
            missing = error.message.rsplit("'", 2)[1]
            issues.append(f"缺少必填字段: {missing}")
        elif error.validator == "type":
            expected = error.validator_value
            issues.append(f"字段 {path or '参数'} 类型应为 {expected}")
        elif error.validator == "minimum":
            issues.append(
                f"字段 {path} 不能小于 {error.validator_value}"
            )
        elif error.validator == "maximum":
            issues.append(
                f"字段 {path} 不能大于 {error.validator_value}"
            )
        elif error.validator == "minLength":
            issues.append(f"字段 {path} 长度不能小于 {error.validator_value}")
        elif error.validator == "enum":
            allowed = "、".join(str(item) for item in error.validator_value)
            issues.append(f"字段 {path} 只能是: {allowed}")
        else:
            issues.append(f"字段 {path or '参数'} 不合法")
    return issues


def _permission_error(
    definition: ToolDefinition,
    context: ToolExecutionContext,
) -> str:
    """
        写操作权限校验函数。
        只对有写副作用（SideEffect.WRITE）的工具做权限检查；读工具直接放行返回空字符串。
        返回空串代表权限校验通过；返回非空字符串，就是权限拒绝的错误提示。
    """
    if definition.side_effect != SideEffect.WRITE:
        return ""
    if context.caller == ToolCaller.TRUSTED_UI:
        if context.trusted_ui_confirmation:
            return ""
        return "写操作需要用户确认"
    if definition.requires_ticket_confirmation:
        if not context.ticket_batch_valid:
            return "建工单调用不能和其他工具混在同一批"
        if not context.ticket_request_allowed:
            return "只有客户明确要求建工单时才能发起"
        if context.ticket_slots is None or not context.ticket_slots.complete:
            return "工单信息不完整，请先补齐必填项"
        return ""
    return "模型不能绕过权限控制执行写操作"


def _effective_ticket_args(
    definition: ToolDefinition,
    tool_args: dict[str, Any],
    context: ToolExecutionContext,
) -> dict[str, Any]:
    """
        参数预处理函数：针对 create_ticket（创建工单）工具，在特定条件下，
        不用LLM输出的原始tool_args，改用会话上下文里已经收集齐全的工单槽位信息作为入参；
        其他场景直接返回LLM传过来的原始参数。
    """
    if (
        definition.name == "create_ticket"
        and context.caller == ToolCaller.AGENT
        and context.ticket_slots is not None
        and context.ticket_slots.complete
    ):
        return {
            "conversation_id": context.conversation_id,
            "description": context.ticket_slots.description,
            "ticket_type": context.ticket_slots.ticket_type,
        }
    return tool_args


def _elapsed_ms(started: float) -> int:
    return max(0, round((time.perf_counter() - started) * 1000))


_default_executor = ToolExecutor(
    registry=ToolRegistry(),
    audit=AuditRecorder(),
)


def configure_default_executor(executor: ToolExecutor) -> None:
    global _default_executor
    _default_executor = executor


def get_default_executor() -> ToolExecutor:
    return _default_executor


def trusted_ui_context(
    conversation_id: str,
    tool_call_id: str,
) -> ToolExecutionContext:
    return ToolExecutionContext(
        conversation_id=conversation_id,
        tool_call_id=tool_call_id,
        caller=ToolCaller.TRUSTED_UI,
        trusted_ui_confirmation=True,
    )
