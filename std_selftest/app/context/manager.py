"""
读取会话消息与历史摘要，调用裁剪函数生成分层上下文包，校验用户输入 token，持久化 layer1 锚点，必要时调度后台异步摘要任务，返回组装完成的上下文包供大模型调用。
"""

from collections.abc import Callable
from dataclasses import replace
import logging

from app.context.budget import ContextBudget
from app.context.logging import log_history_ctx
from app.context.models import (
    ContextMessage,
    ContextPack,
    SummaryRow,
)
from app.context.tasks import SummaryTaskManager
from app.context.tokens import estimate_text_tokens
from app.context.trimmer import build_context_pack


logger = logging.getLogger("app.context")


class ContextManager:
    def __init__(
        self,
        *,
        store,
        budget: ContextBudget,
        tasks: SummaryTaskManager,
        chinese_tokens_per_char: float,
        assistant_layer2_chars: int,
        estimator: Callable[[str], int],
    ) -> None:
        """
            上下文管理器核心类
            职责：读取会话消息+历史摘要，调用build_context_pack完成分层上下文裁剪；
            校验用户输入token上限；更新层1锚点；超限则调度后台摘要任务；最终产出不可变ContextPack，供LLM拼装prompt
        """
        self._store = store
        self._budget = budget
        self._tasks = tasks
        self._chinese_tokens_per_char = chinese_tokens_per_char
        self._assistant_layer2_chars = assistant_layer2_chars
        self._estimator = estimator

    async def prepare(
        self,
        session_id: str,
        current_user_message_id: int,
    ) -> ContextPack:
        """
            【上下文准备主入口】
            加载会话数据、原始消息、历史摘要，调用build_context_pack做分层裁剪，生成ContextPack。
            同时做输入校验、layer1锚点持久化、调度后台摘要任务。
            这个方法在每一轮对话开始前执行，给LangGraph图准备好分层上下文包（layer2摘要 + layer1原始消息）。
        """
        conversation = await self._store.get_conversation(session_id)   #生产环境注入的是 repository，见 main.py (line 78),调用的是repository.py 中的get_conversation函数
        if conversation is None:
            pack = ContextPack(
                summaries_text="",
                layer2_messages=[],
                layer1_messages=[],
                history_ctx=[],
                layer1_from_msg_id=None,
                summary_span=None,
                layer2_tokens=0,
                layer1_tokens=0,
                token_estimate=0,
                trigger_summary=False,
                budget_error=(
                    ""
                    if self._budget.can_fit_one_turn
                    else "上下文预算不足"
                ),
            )
            log_history_ctx(session_id, [], 0, "")
            return pack

        rows = await self._store.load_messages(session_id)      #  加载该会话全部原始消息、历史摘要记录
        summaries = await self._store.load_summaries(session_id)
        context_messages = [        # 数据库记录转为内部ContextMessage模型
            ContextMessage.from_record(row)
            for row in rows
        ]
        pack = build_context_pack(      # 上下文裁剪
            messages=context_messages,
            summaries=[
                SummaryRow(
                    row.covered_from_msg_id,
                    row.covered_to_msg_id,
                    row.content,
                )
                for row in summaries
            ],
            layer1_from_msg_id=conversation.layer1_from_msg_id,
            summary_upto_msg_id=conversation.summary_upto_msg_id,
            budget=self._budget,
            chinese_tokens_per_char=self._chinese_tokens_per_char,
            assistant_layer2_chars=self._assistant_layer2_chars,
            current_user_message_id=current_user_message_id,
        )
        #   定位当前用户消息，校验用户输入是否超出单条token上限
        current_message = next(
            (
                item
                for item in context_messages
                if item.message_id == current_user_message_id
            ),
            None,
        )
        # 找不到指定消息时，取会话里最后一条用户消息兜底
        if current_message is None:
            current_message = next(
                (
                    item
                    for item in reversed(context_messages)
                    if item.role == "user"
                ),
                None,
            )
        # 校验当前用户消息token，如果超限，标记budget_error
        if (
            current_message is not None
            and self._estimator(current_message.content) + 4
            > self._budget.max_user_input_tokens
        ):
            pack = replace(pack, budget_error="上下文预算不足")
            log_history_ctx(
                session_id,
                pack.history_ctx,
                pack.token_estimate,
                pack.summaries_text,
            )
            return pack

        #  如果本次计算得到的layer1锚点和数据库会话记录不一致，更新会话layer1锚点
        if pack.layer1_from_msg_id != conversation.layer1_from_msg_id:
            if (
                conversation.layer1_from_msg_id is not None
                and pack.layer1_from_msg_id is not None
                and pack.layer1_from_msg_id
                > conversation.layer1_from_msg_id
            ):
                logger.info(
                    "layer1 downgrade %s->%s session_id=%s",
                    conversation.layer1_from_msg_id,
                    pack.layer1_from_msg_id,
                    session_id,
                )
            await self._store.update_layer1_anchor(
                session_id,
                pack.layer1_from_msg_id,
            )

        #  build_context_pack标记trigger_summary=True，调度后台异步摘要任务
        if pack.trigger_summary and pack.summary_span is not None:
            span_messages = [       # 取出摘要区间对应的完整原始消息
                item
                for item in context_messages
                if pack.summary_span.from_msg_id
                <= item.message_id
                <= pack.summary_span.to_msg_id
            ]
            self._tasks.schedule(       #schedule函数压缩上下文
                session_id,
                pack.summary_span,
                span_messages,
                layer2_tokens=(
                    pack.layer2_trigger_tokens or pack.layer2_tokens
                ),
                budget=self._budget.layer2_budget,
            )
        #   打印日志，返回最终组装好的ContextPack，送入LLM生成回答
        log_history_ctx(
            session_id,
            pack.history_ctx,
            pack.token_estimate,
            pack.summaries_text,
        )
        return pack


def make_text_estimator(chinese_tokens_per_char: float) -> Callable[[str], int]:
    """
        创建一个固定中文换算系数的token估算闭包函数
    """
    return lambda text: estimate_text_tokens(
        text,
        chinese_tokens_per_char,
    )
