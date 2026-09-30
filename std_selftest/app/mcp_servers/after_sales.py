"""Independent mock after-sales MCP server."""

import os
import random
from datetime import date, timedelta

from mcp.server.fastmcp import FastMCP


def create_server() -> FastMCP:
    mcp = FastMCP(
        "iHelp After Sales",
        host="127.0.0.1",
        port=int(os.getenv("MCP_AFTER_SALES_PORT", "8102")),
        streamable_http_path="/mcp",
        json_response=True,
    )

    @mcp.tool()
    def query_warranty(order_id: str) -> dict:
        """根据订单号查询模拟保修状态。"""
        in_warranty = random.choice([True, False])
        return {
            "order_id": order_id,
            "in_warranty": in_warranty,
            "warranty_status": "保修中" if in_warranty else "已过保",
            "warranty_until": (
                date.today() + timedelta(days=random.randint(30, 365))
            ).isoformat(),
        }

    @mcp.tool()
    def query_return_progress(order_id: str) -> dict:
        """根据订单号查询模拟退货进度。"""
        return {
            "order_id": order_id,
            "return_id": f"RT{random.randint(100000, 999999)}",
            "status": random.choice(
                ["待审核", "审核通过", "退货运输中", "已完成"]
            ),
            "latest_update": "售后系统已收到申请，正在按流程处理",
        }

    return mcp


def main() -> None:
    create_server().run(transport="streamable-http")


if __name__ == "__main__":
    main()
