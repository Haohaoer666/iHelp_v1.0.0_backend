"""Compatibility wrapper around the unified Ch08 tool executor."""

from app.tools.executor import get_default_executor
from app.tools.models import (
    ToolCaller,
    ToolExecutionContext,
    ToolExecutionResult,
)


async def execute_tool(
    tool_name: str,
    tool_args: dict,
) -> ToolExecutionResult:
    """Execute a tool through the untrusted compatibility context."""
    executor = get_default_executor()
    return await executor.execute(
        tool_name,
        tool_args,
        ToolExecutionContext(
            conversation_id=str(
                tool_args.get("conversation_id") or ""
            ),
            tool_call_id=f"legacy-{tool_name}",
            caller=ToolCaller.AGENT,
        ),
    )
