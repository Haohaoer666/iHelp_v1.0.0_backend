"""Independent mock logistics MCP server."""

import os
import random
from datetime import datetime, timedelta

from mcp.server.fastmcp import FastMCP


def create_server() -> FastMCP:
    mcp = FastMCP(
        "iHelp Logistics",
        host="127.0.0.1",
        port=int(os.getenv("MCP_LOGISTICS_PORT", "8101")),
        streamable_http_path="/mcp",
        json_response=True,
    )

    @mcp.tool()
    def query_logistics(order_id: str) -> dict:
        """根据订单号查询模拟物流轨迹。"""
        return {
            "order_id": order_id,
            "carrier": random.choice(
                ["顺丰速运", "中通快递", "圆通速递", "京东物流"]
            ),
            "status": random.choice(
                ["已揽收", "运输中", "到达派送点", "派送中", "已签收"]
            ),
            "last_update": (
                datetime.now() - timedelta(hours=random.randint(1, 48))
            ).strftime("%Y-%m-%d %H:%M:%S"),
            "latest_event": "快件已到达本地中转场，正在安排下一站运输",
        }

    if os.getenv("MCP_DEMO_EXTRA_TOOL") == "true":
        @mcp.tool()
        def query_delivery_eta(order_id: str) -> dict:
            """查询模拟预计送达时间。"""
            return {
                "order_id": order_id,
                "estimated_delivery": (
                    datetime.now() + timedelta(days=random.randint(1, 3))
                ).strftime("%Y-%m-%d"),
                "status": "预计按期送达",
            }

    return mcp


def main() -> None:
    create_server().run(transport="streamable-http")


if __name__ == "__main__":
    main()
