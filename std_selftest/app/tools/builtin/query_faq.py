"""Knowledge-base built-in tool definition."""

from app.services.rag_service import answer_knowledge_question
from app.tools.models import SideEffect, ToolDefinition, ToolOrigin


async def query_faq(
    keyword: str,
    category: str | None = None,
    conversation_id: str | None = None,
) -> dict:
    """Retrieve evidence for a knowledge-base question."""
    return await answer_knowledge_question(
        keyword,
        conversation_id=conversation_id,
        category=category,
    )


TOOL_DEFINITION = ToolDefinition(
    name="query_faq",
    description="从知识库检索并判断证据是否足够回答。",
    input_schema={
        "type": "object",
        "properties": {
            "keyword": {"type": "string", "minLength": 1},
            "category": {"type": ["string", "null"]},
            "conversation_id": {"type": ["string", "null"]},
        },
        "required": ["keyword"],
        "additionalProperties": False,
    },
    handler=query_faq,
    origin=ToolOrigin.BUILTIN,
    server_name=None,
    side_effect=SideEffect.READ,
    agent_visible=False,
)
