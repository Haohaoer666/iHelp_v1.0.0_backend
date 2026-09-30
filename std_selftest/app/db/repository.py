"""
创建对话会话、保存用户 / 助手 / 工具调用、工具返回结果等聊天记录，并可读取会话完整历史消息，实现对话持久化。
"""

import json
from datetime import datetime, timezone

from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError

from app.db.models import (
    Conversation,
    ConversationSummary,
    KnowledgeChunk,
    LowConfidenceQuestion,
    Message,
    Ticket,
)
from app.db.session import AsyncSessionLocal


class ConversationOwnershipError(ValueError):
    """Raised when a session is used with a different user key."""


async def ensure_conversation(
    session_id: str,
    user_key: str | None = None,
) -> Conversation:
    """
        创建会话，或者校验当前用户是否是会话的拥有者
    """
    resolved_user_key = user_key or "访客"
    async with AsyncSessionLocal() as db:
        conversation = await db.get(Conversation, session_id)
        if conversation is None:
            conversation = Conversation(
                id=session_id,
                user_name=resolved_user_key,
            )
            db.add(conversation)
            await db.commit()
            await db.refresh(conversation)
            return conversation
        if user_key is not None and conversation.user_name != resolved_user_key:
            raise ConversationOwnershipError(
                f"session {session_id} belongs to another user"
            )
        return conversation


async def get_conversation(session_id: str) -> Conversation | None:
    async with AsyncSessionLocal() as db:
        return await db.get(Conversation, session_id)


async def get_conversation_for_user(
    session_id: str,
    user_key: str,
) -> Conversation | None:
    async with AsyncSessionLocal() as db:
        conversation = await db.get(Conversation, session_id)
        if conversation is None or conversation.user_name != user_key:
            return None
        return conversation


async def append_message(     #向 messages 表新增一条消息记录
    session_id: str,
    role: str,
    content: str,
    *,
    tool_name: str | None = None,
    tool_args: dict | None = None,
    tool_call_id: str | None = None,
) -> int:
    async with AsyncSessionLocal() as db:
        conversation = await db.get(Conversation, session_id)
        if conversation is not None:
            conversation.updated_at = datetime.now(timezone.utc)
        message = Message(
            conversation_id=session_id,
            role=role,
            content=content,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_call_id=tool_call_id,
        )
        db.add(message)
        await db.commit()
        await db.refresh(message)
        return message.id


async def append_tool_call_message(       #记录大模型输出的工具调用请求
    session_id: str,
    tool_name: str,
    tool_args: dict,
    tool_call_id: str,
) -> None:
    await append_message(
        session_id,
        "assistant",
        "",
        tool_name=tool_name,
        tool_args=tool_args,
        tool_call_id=tool_call_id,
    )


async def append_tool_result_message(     #保存工具执行返回结果
    session_id: str,
    tool_name: str,
    tool_args: dict,
    tool_call_id: str,
    result: object,
) -> None:
    content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    await append_message(
        session_id,
        "tool",
        content,
        tool_name=tool_name,
        tool_args=tool_args,
        tool_call_id=tool_call_id,
    )


async def load_messages(session_id: str) -> list[Message]:        #读取指定会话的全部历史消息
    async with AsyncSessionLocal() as db:
        result = await db.scalars(
            select(Message)
            .where(Message.conversation_id == session_id)
            .order_by(Message.id)
        )
        return list(result.all())


async def load_summaries(session_id: str) -> list[ConversationSummary]:
    async with AsyncSessionLocal() as db:
        result = await db.scalars(
            select(ConversationSummary)
            .where(ConversationSummary.conversation_id == session_id)
            .order_by(ConversationSummary.sequence_no)
        )
        return list(result.all())


async def update_layer1_anchor(
    session_id: str,
    message_id: int | None,
) -> None:
    async with AsyncSessionLocal() as db:
        conversation = await db.get(Conversation, session_id)
        if conversation is None:
            return
        conversation.layer1_from_msg_id = message_id
        await db.commit()


async def append_summary_and_advance(
    session_id: str,
    from_id: int,
    to_id: int,
    content: str,
    token_estimate: int,
) -> ConversationSummary:
    """
        新增一段对话摘要记录，并更新主会话表的摘要标记位点。
        用于长对话压缩：把 [from_id, to_id] 区间内多条原始消息合并成摘要存入ConversationSummary。
        做幂等保护：同一区间的摘要已经存在就直接返回已有记录，避免重复插入。
    """
    try:
        async with AsyncSessionLocal() as db:
            async with db.begin():
                existing = await db.scalar(
                    select(ConversationSummary).where(
                        ConversationSummary.conversation_id == session_id,
                        ConversationSummary.covered_from_msg_id == from_id,
                        ConversationSummary.covered_to_msg_id == to_id,
                    )
                )
                if existing is not None:
                    return existing

                max_sequence = await db.scalar(
                    select(
                        func.coalesce(
                            func.max(ConversationSummary.sequence_no),
                            0,
                        )
                    ).where(ConversationSummary.conversation_id == session_id)
                )
                row = ConversationSummary(
                    conversation_id=session_id,
                    sequence_no=int(max_sequence or 0) + 1,
                    covered_from_msg_id=from_id,
                    covered_to_msg_id=to_id,
                    content=content,
                    token_estimate=token_estimate,
                )
                conversation = await db.get(Conversation, session_id)
                if conversation is None:
                    raise ConversationOwnershipError(
                        f"conversation {session_id} does not exist"
                    )
                conversation.summary_upto_msg_id = to_id
                conversation.updated_at = datetime.now(timezone.utc)
                db.add(row)
                await db.flush()
            await db.refresh(row)
            return row
    except IntegrityError:
        async with AsyncSessionLocal() as db:
            existing = await db.scalar(
                select(ConversationSummary).where(
                    ConversationSummary.conversation_id == session_id,
                    ConversationSummary.covered_from_msg_id == from_id,
                    ConversationSummary.covered_to_msg_id == to_id,
                )
            )
            if existing is not None:
                return existing
            raise


async def list_conversations(user_key: str) -> list[dict]:
    """
        查询指定用户的会话列表
        返回：会话id、首条用户消息预览、创建/更新时间、是否已生成会话摘要标记
    """
    async with AsyncSessionLocal() as db:
        first_user_id = (
            select(func.min(Message.id))
            .where(
                Message.conversation_id == Conversation.id,
                Message.role == "user",
            )
            .correlate(Conversation)
            .scalar_subquery()
        )
        preview = (
            select(Message.content)
            .where(Message.id == first_user_id)
            .scalar_subquery()
        )
        summarized = (
            select(ConversationSummary.id)
            .where(ConversationSummary.conversation_id == Conversation.id)
            .exists()
        )
        result = await db.execute(
            select(
                Conversation.id,
                preview.label("preview"),
                Conversation.created_at,
                Conversation.updated_at,
                summarized.label("summarized"),
            )
            .where(Conversation.user_name == user_key)
            .order_by(
                Conversation.updated_at.desc(),
                Conversation.created_at.desc(),
            )
        )
        return [
            {
                "id": row.id,
                "preview": (row.preview or "")[:120],
                "created_at": row.created_at,
                "updated_at": row.updated_at or row.created_at,
                "summarized": bool(row.summarized),
            }
            for row in result.all()
        ]


async def load_conversation_messages(
    conversation_id: str,
    user_key: str,
) -> list[dict] | None:
    """
        根据会话ID加载会话内消息列表
    """
    async with AsyncSessionLocal() as db:
        conversation = await db.get(Conversation, conversation_id)
        if conversation is None or conversation.user_name != user_key:
            return None
        result = await db.scalars(
            select(Message)
            .where(
                Message.conversation_id == conversation_id,
                Message.role.in_(("user", "assistant")),
            )
            .order_by(Message.id)
        )
        return [
            {
                "id": row.id,
                "role": row.role,
                "content": row.content,
                "created_at": row.created_at,
            }
            for row in result.all()
        ]


async def delete_conversation(
    session_id: str,
    user_key: str,
) -> bool:
    """Delete chat data while retaining tickets as detached audit records."""
    async with AsyncSessionLocal() as db:
        async with db.begin():
            conversation = await db.get(Conversation, session_id)
            if conversation is None or conversation.user_name != user_key:
                return False
            await db.execute(
                update(Ticket)
                .where(Ticket.conversation_id == session_id)
                .values(conversation_id=None)
            )
            await db.execute(
                delete(Message).where(Message.conversation_id == session_id)
            )
            await db.execute(
                delete(ConversationSummary).where(
                    ConversationSummary.conversation_id == session_id
                )
            )
            await db.delete(conversation)
        return True


#写入低置信问答记录，用于后续人工复盘知识库
async def insert_low_confidence_question(
    question: str,
    reason: str,
    source_conversation_id: str | None,
    entry_point: str = "query_faq_self_check",
) -> None:
    async with AsyncSessionLocal() as db:
        db.add(
            LowConfidenceQuestion(
                question=question,
                source_conversation_id=source_conversation_id,
                entry_point=entry_point,
                reason=reason,
            )
        )
        await db.commit()


# 批量查出每个切片所属文档章节路径，挂载到 hit 里用于溯源
async def load_chunk_section_paths(chunk_ids: list[str]) -> dict[str, str]:
    if not chunk_ids:
        return {}
    async with AsyncSessionLocal() as db:
        rows = await db.scalars(
            select(KnowledgeChunk).where(KnowledgeChunk.chunk_id.in_(chunk_ids))
        )
        return {row.chunk_id: row.section_path or "" for row in rows}
