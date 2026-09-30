from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import trim_messages

from app.config import settings
from app.context.tokens import estimate_messages_tokens


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, list[BaseMessage]] = {}

    def get(self, session_id: str) -> list[BaseMessage]:
        return self._sessions.get(session_id, [])

    def append(self, session_id: str, *messages: BaseMessage) -> None:
        self._sessions.setdefault(session_id, []).extend(messages)


def trim_history(messages: list[BaseMessage], max_tokens: int) -> list[BaseMessage]:
    """
        裁剪对话历史，控制消息总token不超过max_tokens
        采用保留最新消息的策略；裁剪时以Human消息为边界，不会把单轮问答切到一半。
    """
    def token_counter(items: list[BaseMessage]) -> int: # 内部函数：计算消息列表的预估token数量
        return estimate_messages_tokens(
            items,
            settings.chinese_tokens_per_char,
        )

    return trim_messages(   # 调用LangChain内置trim_messages工具做裁剪
        messages,
        strategy="last",
        token_counter=token_counter,
        max_tokens=max_tokens,
        start_on="human",
        allow_partial=False,
    )
