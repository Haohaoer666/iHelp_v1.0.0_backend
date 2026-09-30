"""Order lookup built-in tool definition."""

from app.services.order_catalog import get_demo_order
from app.tools.models import SideEffect, ToolDefinition, ToolOrigin


async def query_order(order_id: str) -> dict:
    """Return demo order details for an order id."""
    return get_demo_order(order_id)


TOOL_DEFINITION = ToolDefinition(
    name="query_order",
    description="查询订单基础信息，包括订单状态、金额和创建时间。",
    input_schema={
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "minLength": 1},
        },
        "required": ["order_id"],
        "additionalProperties": False,
    },
    handler=query_order,
    origin=ToolOrigin.BUILTIN,
    server_name=None,
    side_effect=SideEffect.READ,
)
