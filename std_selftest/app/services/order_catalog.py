"""
    列出几个订单demo，
    模拟订单查询、写出全部订单列表
"""

from copy import deepcopy
from typing import Any


DEMO_ORDERS: dict[str, dict[str, Any]] = {
    "1001": {
        "order_id": "1001",
        "product_name": "iHao 智能手表",
        "category": "智能穿戴",
        "amount": 899.0,
        "status": "已签收",
        "created_at": "2026-09-10",
        "signed_at": "2026-09-15",
        "condition": "未拆封",
    },
    "1002": {
        "order_id": "1002",
        "product_name": "iHao 蓝牙耳机",
        "category": "数码配件",
        "amount": 199.0,
        "status": "运输中",
        "created_at": "2026-09-17",
        "signed_at": None,
        "condition": "运输中",
    },
    "1003": {
        "order_id": "1003",
        "product_name": "定制手机壳",
        "category": "定制商品",
        "amount": 49.0,
        "status": "已完成",
        "created_at": "2026-08-12",
        "signed_at": "2026-08-20",
        "condition": "已拆封使用",
    },
}


def list_demo_orders() -> list[dict[str, Any]]:
    """
        返回**全部模拟订单列表**，用于查询用户名下所有订单场景
    """
    return [deepcopy(order) for order in DEMO_ORDERS.values()]


def get_demo_order(order_id: str) -> dict[str, Any]:
    """
        模拟订单查询函数
    """
    normalized = str(order_id).strip()
    if normalized not in DEMO_ORDERS:
        raise KeyError(f"订单不存在: {normalized}")
    return deepcopy(DEMO_ORDERS[normalized])
