"""
    工具调用审计记录器，负责把每一次工具调用的审计日志写入数据库。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.config import settings
from app.db.models import ToolAuditLog
from app.db.session import AsyncSessionLocal


logger = logging.getLogger(__name__)


class AuditRecorder:
    """
        工具调用审计记录器，负责把每一次工具调用的审计日志写入数据库。
        审计写入的异常不会向上抛出，只打日志，保证主业务（Agent工具调用流程）不受审计落库失败影响。
    """
    async def record(
        self,
        *,
        conversation_id: str | None,
        tool_call_id: str,
        tool_name: str,
        origin: str,
        server_name: str | None,
        arguments: dict[str, Any],
        result_summary: Any,
        status: str,
        error: str,
        retry_count: int,
        duration_ms: int,
    ) -> None:
        """Persist one audit row without allowing audit failures to escape."""
        try:
            await self._write(
                conversation_id=conversation_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                origin=origin,
                server_name=server_name,
                arguments=arguments,
                result_summary=json.dumps(
                    result_summary,
                    ensure_ascii=False,
                    default=str,
                )[: settings.audit_result_max_chars],
                status=status,
                error=error[: settings.audit_result_max_chars],
                retry_count=retry_count,
                duration_ms=duration_ms,
            )
        except Exception:
            logger.exception(
                "工具审计落库失败 tool_name=%s tool_call_id=%s",
                tool_name,
                tool_call_id,
            )

    async def _write(self, **kwargs: Any) -> None:
        async with AsyncSessionLocal() as db:
            db.add(ToolAuditLog(**kwargs))
            await db.commit()


audit_recorder = AuditRecorder()
