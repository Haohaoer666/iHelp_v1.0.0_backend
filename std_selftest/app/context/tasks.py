"""
    用于调度会话后台异步摘要任务，防止同会话并发执行摘要，管理任务生命周期并在服务关闭时优雅回收任务。
"""

import asyncio
import time
from collections.abc import Callable, Sequence

from langchain_core.language_models import BaseChatModel

from app.context.logging import log_summary_event
from app.context.models import ContextMessage, SummarySpan
from app.context.summary import summarize_span


class SummaryTaskManager:
    """
        会话摘要任务管理器
        职责：管理后台异步摘要生成任务，避免同一个会话并发跑多个摘要；
        触发裁剪时，把旧对话丢给LLM压缩成摘要，写入存储；服务关闭优雅等待/取消正在跑的摘要任务
    """
    def __init__(
        self,
        *,
        model: BaseChatModel,
        store,
        estimator: Callable[[str], int],
        min_chars: int,
        max_chars: int,
        max_tokens: int = 0,
        chinese_tokens_per_char: float = 0.80,
    ) -> None:
        self._model = model
        self._store = store
        self._estimator = estimator
        self._min_chars = min_chars
        self._max_chars = max_chars
        self._max_tokens = max_tokens
        self._chinese_tokens_per_char = chinese_tokens_per_char
        self._tasks: dict[str, asyncio.Task] = {}

    def schedule(
        self,
        session_id: str,
        span: SummarySpan,
        messages: Sequence[ContextMessage],
        *,
        layer2_tokens: int | None = None,
        budget: int | None = None,
    ) -> str:
        """
            调度摘要任务：对外入口，尝试为一个会话启动摘要任务
            如果该会话已有摘要任务正在运行 → 跳过，防止并发重复压缩
            返回状态字符串 "running"
        """
        current = self._tasks.get(session_id)
        if current is not None and not current.done():
            log_summary_event(
                "skip",
                session_id,
                span=span,
                reason="already_running",
            )
            return "running"

        log_summary_event(
            "trigger",
            session_id,
            span=span,
            layer2_tokens=layer2_tokens,
            budget=budget,
        )
        task = asyncio.create_task(
            self._run(session_id, span, list(messages))
        )
        self._tasks[session_id] = task

        def clear_finished(done: asyncio.Task) -> None:
            if self._tasks.get(session_id) is done:
                self._tasks.pop(session_id, None)

        task.add_done_callback(clear_finished)
        return "running"

    def has_active(self, session_id: str) -> bool:
        """判断该会话是否存在正在执行的摘要任务"""
        task = self._tasks.get(session_id)
        return bool(task is not None and not task.done())

    async def _run(
        self,
        session_id: str,
        span: SummarySpan,
        messages: list[ContextMessage],
    ) -> None:
        """
                【私有异步任务主体】真正执行摘要的逻辑
                1. 调用LLM把一段消息压缩为摘要文本
                2. 估算摘要token
                3. 写入存储层
                4. 记录成功/失败日志，埋耗时
        """
        started = time.perf_counter()
        log_summary_event("start", session_id, span=span)
        try:
            content = await summarize_span(
                messages,
                self._model,
                min_chars=self._min_chars,
                max_chars=self._max_chars,
                max_tokens=self._max_tokens,
                chinese_tokens_per_char=self._chinese_tokens_per_char,
            )
            row = await self._store.append_summary_and_advance(
                session_id=session_id,
                from_id=span.from_msg_id,
                to_id=span.to_msg_id,
                content=content,
                token_estimate=self._estimator(content),
            )
            sequence_no = getattr(row, "sequence_no", None)
            log_summary_event(
                "done",
                session_id,
                span=span,
                sequence_no=sequence_no,
                elapsed_ms=int((time.perf_counter() - started) * 1000),
                max_tokens=self._max_tokens or None,
            )
        except Exception as exc:  # noqa: BLE001
            log_summary_event(
                "fail",
                session_id,
                span=span,
                elapsed_ms=int((time.perf_counter() - started) * 1000),
                error=str(exc),
            )

    async def wait_for(self, session_id: str) -> None:
        """阻塞等待该会话的摘要任务执行完成（可选调用，一般主流程不等待，摘要后台异步跑）"""
        task = self._tasks.get(session_id)
        if task is not None:
            await task

    async def shutdown(self) -> None:
        """
            服务优雅关闭时调用（在lifespan finally里执行）
            等待正在运行的摘要任务：最多等待2秒，还没跑完就cancel，防止服务卡住无法退出
        """
        tasks = list(self._tasks.values())
        if not tasks:
            return
        done, pending = await asyncio.wait(tasks, timeout=2)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
