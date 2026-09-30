from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ChatRequest(BaseModel):
    """
        对话接口入参模型，用于接口接收用户对话请求，做参数校验
    """
    session_id: str = Field(min_length=1, description="会话 ID，同一会话多轮复用")
    user_key: str = Field(
        default="demo-user",
        min_length=1,
        max_length=64,
        description="演示用户键，用于认领会话列表",
    )
    message: str | None = Field(
        default=None,
        min_length=1,
        description="用户本轮消息；与 resume 二选一",
    )
    resume: dict[str, Any] | None = Field(
        default=None,
        description="LangGraph 中断恢复值；与 message 二选一",
    )

    @model_validator(mode="after")
    def validate_message_or_resume(self):
        if (self.message is None) == (self.resume is None):
            raise ValueError("message 和 resume 必须且只能提供一个")
        if self.resume is not None and not self.resume:
            raise ValueError("resume 不能为空对象")
        return self


class ConversationListItem(BaseModel):
    id: str
    preview: str
    created_at: datetime
    updated_at: datetime
    summarized: bool


class ConversationMessageItem(BaseModel):
    id: int
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime


class TicketRequest(BaseModel):
    session_id: str = Field(min_length=1, description="会话 ID")
    description: str = Field(min_length=1, description="问题描述")
    ticket_type: str = Field(default="投诉", min_length=1, description="工单类型")


class RefundRequest(BaseModel):
    session_id: str = Field(min_length=1)
    order_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class ExchangeRequest(BaseModel):
    session_id: str = Field(min_length=1)
    order_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)
