"""
    退货接口
"""

import json

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.tools.executor import (
    get_default_executor,
    trusted_ui_context,
)
from app.schemas.chat import RefundRequest
from app.services.order_catalog import get_demo_order


router = APIRouter(prefix="/api/refunds")


@router.get("/reasons")
async def refund_reasons():
    return settings.refund_reason_list


@router.post("")
async def create_refund(req: RefundRequest):
    try:
        get_demo_order(req.order_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="订单不存在") from None

    if req.reason not in settings.refund_reason_list:
        raise HTTPException(status_code=422, detail="不支持的退款原因")

    description = f"订单 {req.order_id} 退款申请，原因：{req.reason}"
    result = await get_default_executor().execute(
        "create_ticket",
        {
            "conversation_id": req.session_id,
            "description": description,
            "ticket_type": "退款",
        },
        trusted_ui_context(
            req.session_id,
            f"refund-ui-{req.session_id}-{req.order_id}",
        ),
    )
    if not result.success:
        raise HTTPException(status_code=500, detail=result.error)
    if isinstance(result.result, str):
        return json.loads(result.result)
    return result.result
