import json

from app.core.tools import query_order
from app.services.order_catalog import get_demo_order, list_demo_orders


def test_demo_order_catalog_is_stable():
    first = get_demo_order("1001")
    second = get_demo_order("1001")

    assert first == second
    assert first["order_id"] == "1001"


def test_list_demo_orders_has_selector_fields():
    orders = list_demo_orders()

    assert len(orders) == 3
    assert {"order_id", "product_name", "status", "amount", "signed_at"} <= set(
        orders[0]
    )


async def test_query_order_uses_deterministic_catalog():
    first = json.loads(await query_order.ainvoke({"order_id": "1001"}))
    second = json.loads(await query_order.ainvoke({"order_id": "1001"}))

    assert first == second
    assert first["product_name"] == "iHao 智能手表"
    assert first["condition"] == "未拆封"
