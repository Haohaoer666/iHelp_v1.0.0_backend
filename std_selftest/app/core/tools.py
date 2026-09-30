"""Legacy LangChain tool exports backed by the Ch08 built-in modules."""

import json

from langchain_core.tools import StructuredTool

from app.tools.builtin.create_ticket import (
    create_ticket as create_ticket_handler,
)
from app.tools.builtin.query_faq import query_faq as query_faq_handler
from app.tools.builtin.query_order import (
    query_order as query_order_handler,
)
from app.tools.builtin.query_product import (
    query_product as query_product_handler,
)


def _to_json(data: object) -> str:
    """将对象序列化为 UTF-8 不转义的 JSON 字符串。

    Args:
        data: 需要序列化的对象。

    Returns:
        序列化后的 JSON 字符串。
    """
    return json.dumps(data, ensure_ascii=False)


async def query_order(order_id: str) -> str:
    """查询订单基础信息。

    Args:
        order_id: 订单号。

    Returns:
        包含订单状态、金额和创建时间的 JSON 字符串。
    """
    return _to_json(await query_order_handler(order_id))


async def query_product(product_name: str) -> str:
    """查询商品信息。

    Args:
        product_name: 商品名称或关键词。

    Returns:
        包含商品价格、库存和销售状态的 JSON 字符串。
    """
    return _to_json(await query_product_handler(product_name))


async def query_faq(
    keyword: str,
    category: str | None = None,
    conversation_id: str | None = None,
) -> str:
    """从知识库中检索并判断证据是否足够回答。

    Args:
        keyword: 用户问题中的关键词或标准问法。
        category: 可选品类过滤条件。
        conversation_id: 当前会话 ID，可选。

    Returns:
        包含证据与引用元数据的 JSON；证据不足时返回拒答信息。
    """
    result = await query_faq_handler(
        keyword=keyword,
        category=category,
        conversation_id=conversation_id,
    )
    return _to_json(result)


async def create_ticket(
    conversation_id: str,
    description: str,
    ticket_type: str,
) -> str:
    """创建人工客服工单。

    Args:
        conversation_id: 当前会话 ID。
        description: 问题描述。
        ticket_type: 工单类型。

    Returns:
        新工单号、状态、类型和创建时间的 JSON 字符串。
    """
    return _to_json(
        await create_ticket_handler(
            conversation_id=conversation_id,
            description=description,
            ticket_type=ticket_type,
        )
    )


query_order = StructuredTool.from_function(
    coroutine=query_order,
    name="query_order",
    description="查询订单基础信息。",
)
query_product = StructuredTool.from_function(
    coroutine=query_product,
    name="query_product",
    description="查询商品信息。",
)
query_faq = StructuredTool.from_function(
    coroutine=query_faq,
    name="query_faq",
    description="查询知识库。",
)
create_ticket = StructuredTool.from_function(
    coroutine=create_ticket,
    name="create_ticket",
    description="创建人工客服工单。",
)

TOOLS = [
    query_order,
    query_product,
    query_faq,
    create_ticket,
]

TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}
