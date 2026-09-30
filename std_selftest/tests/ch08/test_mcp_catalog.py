from langchain_core.tools import StructuredTool

from app.tools.mcp import refresh_mcp_catalog
from app.tools.registry import ToolRegistry


async def fake_mcp_tool(value: str) -> dict:
    return {"value": value}


async def fake_get_tools(server_name=None):
    names = {
        "logistics": ["query_logistics"],
        "after_sales": ["query_warranty"],
    }
    return [
        StructuredTool.from_function(
            coroutine=fake_mcp_tool,
            name=name,
            description="MCP 工具",
        )
        for name in names[server_name]
    ]


class FakeClient:
    get_tools = staticmethod(fake_get_tools)


async def test_mcp_catalog_marks_trusted_server_tools_read_only():
    registry = ToolRegistry()
    errors = await refresh_mcp_catalog(registry, FakeClient())

    assert errors == []
    assert registry.get("query_logistics").origin == "mcp"
    assert registry.get("query_logistics").side_effect == "read"
    assert registry.get("query_logistics").input_schema["type"] == "object"
