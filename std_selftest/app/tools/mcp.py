"""
    多 MCP 服务客户端构建与工具目录刷新模块**。
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient

from app.config import settings
from app.tools.registry import ToolRegistry


MCP_SERVER_POLICIES = {
    "logistics": {"mode": "read_only"},
    "after_sales": {"mode": "read_only"},
}
_refresh_lock = asyncio.Lock()


def build_mcp_client() -> MultiServerMCPClient:
    return MultiServerMCPClient(
        {
            "logistics": {
                "transport": "streamable_http",
                "url": settings.mcp_logistics_url,
                "timeout": settings.mcp_transport_timeout_seconds,
                "sse_read_timeout": (
                    settings.mcp_sse_read_timeout_seconds
                ),
            },
            "after_sales": {
                "transport": "streamable_http",
                "url": settings.mcp_after_sales_url,
                "timeout": settings.mcp_transport_timeout_seconds,
                "sse_read_timeout": (
                    settings.mcp_sse_read_timeout_seconds
                ),
            },
        }
    )


async def refresh_mcp_catalog(
    registry: ToolRegistry,
    client: Any,
) -> list[dict[str, Any]]:
    """
        刷新全部MCP服务的工具目录。
        持有全局刷新锁，依次拉取每个MCP服务的工具列表，原子替换注册表内对应服务的工具；
        如果某个服务拉取/更新失败，则清理该服务已注册的工具，并收集错误信息，最后返回所有错误。
    """
    errors: list[dict[str, Any]] = []
    async with _refresh_lock:
        for server_name, policy in MCP_SERVER_POLICIES.items():
            try:
                tools = await client.get_tools(server_name=server_name)
                await registry.replace_mcp_server_tools(
                    server_name,
                    tools,
                    read_only=policy["mode"] == "read_only",
                )
            except Exception as exc:
                await registry.remove_mcp_server_tools(server_name)
                errors.append(
                    {
                        "server_name": server_name,
                        "error": str(exc),
                    }
                )
    return errors
