import json

from fastapi import APIRouter, HTTPException

from app.tools.executor import (
    get_default_executor,
    trusted_ui_context,
)
from app.schemas.chat import TicketRequest


router = APIRouter()


@router.post("/api/tickets")
async def create_ticket_endpoint(req: TicketRequest):
    result = await get_default_executor().execute(
        "create_ticket",
        {
            "conversation_id": req.session_id,
            "description": req.description,
            "ticket_type": req.ticket_type,
        },
        trusted_ui_context(
            req.session_id,
            f"ticket-ui-{req.session_id}",
        ),
    )
    if not result.success:
        raise HTTPException(status_code=500, detail=result.error)
    if isinstance(result.result, str):
        return json.loads(result.result)
    return result.result
