"""Product lookup built-in tool definition."""

import random

from app.tools.models import SideEffect, ToolDefinition, ToolOrigin


async def query_product(product_name: str) -> dict:
    """Return mock product details for a product name."""
    return {
        "product_name": product_name,
        "price": round(random.uniform(30, 999), 2),
        "stock": random.randint(0, 500),
        "is_on_sale": random.choice([True, False]),
    }


TOOL_DEFINITION = ToolDefinition(
    name="query_product",
    description="查询商品价格、库存和销售状态。",
    input_schema={
        "type": "object",
        "properties": {
            "product_name": {"type": "string", "minLength": 1},
        },
        "required": ["product_name"],
        "additionalProperties": False,
    },
    handler=query_product,
    origin=ToolOrigin.BUILTIN,
    server_name=None,
    side_effect=SideEffect.READ,
)
