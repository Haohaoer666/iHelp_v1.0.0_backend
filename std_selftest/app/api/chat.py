import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app.db.repository import (
    ConversationOwnershipError,
    append_message,
    ensure_conversation,
)
from app.schemas.chat import ChatRequest


logger = logging.getLogger(__name__)
router = APIRouter()


# 从当前 FastAPI 应用对象里取出已经编译好的 LangGraph 图，交给聊天接口使用。创建chat_graph 依赖，路由使用时注入
def get_chat_graph(request: Request):
    return request.app.state.chat_graph


async def persist_user_message(
    session_id: str,
    user_key: str,
    message: str,
) -> int:
    """
        持久化用户输入消息，对外暴露的入口函数
    """
    await ensure_conversation(session_id, user_key)
    return await append_message(session_id, "user", message)


@router.post("/api/chat")
async def chat(
    req: ChatRequest,
    graph=Depends(get_chat_graph),
):
    """
        对话接口 POST /api/chat
        FastAPI 接口，接收前端聊天请求，调用LangGraph智能图，SSE流式返回结果
        :param req: ChatRequest，Pydantic模型，前端传的请求体
        :param graph: LangGraph实例，由依赖注入get_chat_graph获取（整个服务复用同一个graph）
        :return StreamingResponse: SSE长流式响应，向前端逐块推送模型输出
    """
    if req.resume is not None:  # 如果resume不为空，代表【恢复中断的会话】，不用新增用户消息，直接下发恢复指令给LangGraph
        graph_input = Command(resume=req.resume)
    else:
        try:
            user_message_id = await persist_user_message(
                req.session_id,
                req.user_key,
                req.message or "",
            )
        except ConversationOwnershipError as exc:
            raise HTTPException(
                status_code=409,
                detail="session belongs to another user",
            ) from exc
        graph_input = {
            "session_id": req.session_id,
            "user_key": req.user_key,
            "current_user_message_id": user_message_id,
            "user_message": req.message,
            "messages": [HumanMessage(req.message or "")],
        }

    async def event_stream() -> AsyncIterator[str]:

        try:
            async for chunk in graph.astream(       # graph.astream：LangGraph异步流式接口，逐块返回图执行的事件/数据
                graph_input,
                {"configurable": {"thread_id": req.session_id}},    #这是传给 LangGraph 运行时的配置，不是 ChatState 的字段。
                stream_mode="custom",
                subgraphs=True,
            ):

                # 如果是长度为2的tuple，取第二个元素（真正事件数据）；否则直接拿chunk本身作为事件
                event = (
                    chunk[1]
                    if isinstance(chunk, tuple) and len(chunk) == 2
                    else chunk
                )
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"  # SSE标准格式：data: {json}\n\n，推送给前端
        except Exception:
            logger.exception("客服图执行失败 session_id=%s", req.session_id)
            yield "event: error\n"
            yield (
                'data: {"message":"上游服务暂时不可用，请稍后重试"}\n\n'
            )
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")    # StreamingResponse：FastAPI SSE响应，media_type指定SSE类型
