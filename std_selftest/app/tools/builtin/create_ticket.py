"""Ticket creation built-in tool definition."""

import random
import time

from app.db.models import Ticket
from app.db.session import AsyncSessionLocal
from app.tools.models import SideEffect, ToolDefinition, ToolOrigin


async def create_ticket(
    conversation_id: str,
    description: str,
    ticket_type: str,
) -> dict:
    """Persist a ticket and return its public fields."""
    ticket_id = f"TK{int(time.time())}{random.randint(100, 999)}"
    async with AsyncSessionLocal() as db:
        ticket = Ticket(
            ticket_id=ticket_id,
            conversation_id=conversation_id,
            description=description,
            ticket_type=ticket_type,
            status="open",
        )
        db.add(ticket)
        await db.commit()
        await db.refresh(ticket)

    return {
        "ticket_id": ticket.ticket_id,
        "status": ticket.status,
        "ticket_type": ticket.ticket_type,
        "created_at": ticket.created_at.isoformat(),
    }


TOOL_DEFINITION = ToolDefinition(
    name="create_ticket",
    description="创建人工客服工单。",
    input_schema={
        "type": "object",
        "properties": {
            "conversation_id": {"type": "string", "minLength": 1},
            "description": {"type": "string", "minLength": 1},
            "ticket_type": {
                "type": "string",
                "enum": [
                    "投诉",
                    "咨询",
                    "售后",
                    "其他",
                    "退款",
                    "换货",
                    "转人工",
                ],
            },
        },
        "required": [
            "conversation_id",
            "description",
            "ticket_type",
        ],
        "additionalProperties": False,
    },
    handler=create_ticket,
    origin=ToolOrigin.BUILTIN,
    server_name=None,
    side_effect=SideEffect.WRITE,
    requires_ticket_confirmation=True,
)
