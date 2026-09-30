"""Read-only conversation list and history APIs."""

from fastapi import APIRouter, HTTPException, Query, Request, Response, status

from app.db.repository import (
    delete_conversation as delete_conversation_data,
    get_conversation_for_user,
    list_conversations,
    load_conversation_messages,
)
from app.schemas.chat import (
    ConversationListItem,
    ConversationMessageItem,
)


router = APIRouter()


@router.get(
    "/api/conversations",
    response_model=list[ConversationListItem],
)
async def get_conversations(
    user_key: str = Query(min_length=1, max_length=64),
):
    return await list_conversations(user_key)


@router.get(
    "/api/conversations/{conversation_id}/messages",
    response_model=list[ConversationMessageItem],
)
async def get_conversation_messages(
    conversation_id: str,
    user_key: str = Query(min_length=1, max_length=64),
):
    rows = await load_conversation_messages(conversation_id, user_key)
    if rows is None:
        raise HTTPException(
            status_code=404,
            detail="conversation not found",
        )
    return rows


@router.delete(
    "/api/conversations/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_conversation(
    conversation_id: str,
    request: Request,
    user_key: str = Query(min_length=1, max_length=64),
):
    conversation = await get_conversation_for_user(
        conversation_id,
        user_key,
    )
    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail="conversation not found",
        )

    checkpointer = getattr(request.app.state, "checkpointer", None)
    if checkpointer is not None:
        await checkpointer.adelete_thread(conversation_id)

    deleted = await delete_conversation_data(conversation_id, user_key)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="conversation not found",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
