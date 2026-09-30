"""
读取 messages 表
  -> 按 user / assistant 配对成对话轮次
  -> 分批交给 LLM 抽取问答对
  -> 先写入 extracted_qa_pairs 暂存表
  -> 去重后写入 knowledge_chunks
"""

import json

from sqlalchemy import select

from app.core.llm import get_chat_model
from app.db.models import ExtractedQAPair, KnowledgeChunk, Message
from app.db.session import AsyncSessionLocal
from app.services.knowledge_pipeline import _text_for_embedding


PROMPT = """你是客服知识挖掘助手。请从下面的历史对话中抽取可复用的问答对。
只输出 JSON 数组，每个元素格式为：
{{"question": "用户问题", "answer": "客服答案"}}
不要输出 JSON 以外的内容。

对话：
{dialogue}
"""


def _build_turns(messages: list[Message]) -> list[dict]:
    turns: list[dict] = []
    pending_user: str | None = None
    for message in messages:
        if message.role == "user":
            pending_user = message.content.strip()
        elif message.role == "assistant" and pending_user:
            turns.append({"question": pending_user, "answer": message.content.strip()})
            pending_user = None
    return turns


async def mine_qa_from_messages(batch_size: int = 8, limit: int = 200) -> dict:
    async with AsyncSessionLocal() as db:
        result = await db.scalars(
                select(Message)
                .where(Message.role.in_(["user", "assistant"]))
                .order_by(Message.id)
                .limit(limit)
        )
        messages = list(result.all())

    turns = _build_turns(messages)
    model = get_chat_model()
    extracted: list[dict] = []

    for index in range(0, len(turns), batch_size):
        batch = turns[index : index + batch_size]
        dialogue = "\n".join(f"问：{turn['question']}\n答：{turn['answer']}" for turn in batch)
        result = await model.ainvoke(PROMPT.format(dialogue=dialogue))
        content = result.content if hasattr(result, "content") else str(result)
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                extracted.extend(parsed)
        except json.JSONDecodeError:
            continue

    inserted = 0
    async with AsyncSessionLocal() as db:
        for item in extracted:
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            if not question or not answer:
                continue
            existing = await db.scalar(
                select(ExtractedQAPair).where(ExtractedQAPair.question == question)
            )
            if existing:
                continue
            db.add(
                ExtractedQAPair(
                    question=question,
                    answer=answer,
                    status="staging",
                )
            )
            db.add(
                KnowledgeChunk(
                    chunk_id=f"mined-{question[:20]}-{len(question)}",
                    category="挖掘知识",
                    questions=[question],
                    answer=answer,
                    section_path="历史对话 > 挖掘知识",
                    content_type="mined_qa",
                    is_critical=False,
                    vector_status="pending",
                    text_for_embedding=_text_for_embedding("挖掘知识", [question], answer),
                )
            )
            inserted += 1
        await db.commit()

    return {"turns": len(turns), "extracted": len(extracted), "inserted": inserted}
