from app.mcp_servers.after_sales import create_server as create_after_sales
from app.mcp_servers.logistics import create_server as create_logistics


async def test_logistics_server_exports_query_logistics():
    server = create_logistics()
    tools = await server.list_tools()
    assert [tool.name for tool in tools] == ["query_logistics"]


async def test_after_sales_server_exports_two_read_tools():
    server = create_after_sales()
    tools = await server.list_tools()
    assert {tool.name for tool in tools} == {
        "query_warranty",
        "query_return_progress",
    }
