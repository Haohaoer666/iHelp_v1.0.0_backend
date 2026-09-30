"""Run a deterministic three-layer context and summary demonstration."""

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings
from app.context.budget import ContextBudget
from app.context.manager import ContextManager, make_text_estimator
from app.context.tasks import SummaryTaskManager
from app.core.llm import get_chat_model
from app.db import repository
from app.graph.builder import build_chat_graph
from app.services.turn_logger import TurnLogger


class NoModel:
    def bind_tools(self, tools, **kwargs):
        return self

    async def ainvoke(self, *args, **kwargs):
        raise AssertionError("unexpected model call")

    async def astream(self, *args, **kwargs):
        raise AssertionError("unexpected model call")
        yield


class SummaryAwareModel:
    def bind_tools(self, tools, **kwargs):
        return self

    async def astream(self, messages, **kwargs):
        combined = "\n".join(
            message.content
            if isinstance(message.content, str)
            else str(message.content)
            for message in messages
        )
        if "订单1001" in combined:
            reply = (
                "根据早期梗概，最开始订单1001咨询智能手表配送，"
                "希望今天确认，问题仍未解决。"
            )
        else:
            reply = "根据上下文未找到订单1001。"
        yield AIMessageChunk(content=reply)


def _budget(profile: str) -> ContextBudget:
    if profile in ("default", "cost_optimized"):
        return ContextBudget(
            model_context_window=settings.model_context_window,
            max_output_tokens=settings.agent_max_output_tokens,
            max_user_input_tokens=settings.max_user_input_tokens,
            max_agent_steps=settings.agent_max_steps,
            tool_call_message_max_tokens=(
                settings.tool_call_message_max_tokens
            ),
            tool_result_max_tokens=settings.tool_result_max_tokens,
            rerank_top_k=settings.rerank_top_k,
            evidence_chunk_token_budget=(
                settings.evidence_chunk_token_budget
            ),
            evidence_total_max_tokens=settings.evidence_total_max_tokens,
            system_prompt_token_budget=settings.system_prompt_token_budget,
            summary_injection_token_budget=(
                settings.summary_injection_token_budget
            ),
            summary_segment_max_tokens=settings.summary_segment_max_tokens,
            safety_margin_tokens=settings.safety_margin_tokens,
            desired_retained_turns=settings.desired_retained_turns,
            steady_turn_token_estimate=(
                settings.steady_turn_token_estimate
            ),
            history_soft_budget_enabled=(
                settings.history_soft_budget_enabled
            ),
            history_layer1_ratio=settings.history_layer1_ratio,
            history_layer2_ratio=settings.history_layer2_ratio,
        )
    return ContextBudget(
        model_context_window=18000,
        max_output_tokens=2000,
        max_user_input_tokens=2000,
        max_agent_steps=3,
        tool_call_message_max_tokens=400,
        tool_result_max_tokens=1200,
        rerank_top_k=5,
        evidence_chunk_token_budget=400,
        evidence_total_max_tokens=None,
        system_prompt_token_budget=1800,
        summary_injection_token_budget=600,
        summary_segment_max_tokens=200,
        safety_margin_tokens=350,
        desired_retained_turns=24,
        steady_turn_token_estimate=800,
        legacy_acceptance=True,
    )


def _configure_logging(console_level: str) -> None:
    log_path = Path(settings.context_log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, console_level.upper()))
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(console_handler)


async def _classifier(text, model, *, history=None):
    return "物流" if "最开始" in text else "闲聊"


async def run_demo(
    *,
    profile: str,
    turns: int,
    session_id: str,
    user_key: str,
    ask_start_order: bool,
) -> None:
    budget = _budget(profile)
    estimator = make_text_estimator(settings.chinese_tokens_per_char)
    summary_tasks = SummaryTaskManager(
        model=get_chat_model(streaming=False),
        store=repository,
        estimator=estimator,
        min_chars=settings.summary_target_min_chars,
        max_chars=settings.summary_target_max_chars,
        max_tokens=settings.summary_segment_max_tokens,
        chinese_tokens_per_char=settings.chinese_tokens_per_char,
    )
    manager = ContextManager(
        store=repository,
        budget=budget,
        tasks=summary_tasks,
        chinese_tokens_per_char=settings.chinese_tokens_per_char,
        assistant_layer2_chars=settings.assistant_layer2_chars,
        estimator=estimator,
    )
    await repository.ensure_conversation(session_id, user_key)
    graph = build_chat_graph(
        understanding_model=None,
        intent_model=NoModel(),
        agent_model=SummaryAwareModel(),
        context_manager=manager,
        context_budget=budget,
        intent_classifier=_classifier,
        turn_logger=TurnLogger(),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": session_id}}

    for turn in range(1, turns + 1):
        message = (
            f"第{turn}轮：用户咨询订单{1000 + turn}的智能手表配送情况，"
            "并补充了商品使用场景和希望今天确认的问题。"
            + "需要如实记录商品、订单号、明确诉求和未解决问题。" * 8
            + "需要如实记录商品、订单号、明确诉求和未解决问题。" * 8
        )
        message_id = await repository.append_message(
            session_id,
            "user",
            message,
        )
        await graph.ainvoke(
            {
                "session_id": session_id,
                "user_key": user_key,
                "current_user_message_id": message_id,
                "user_message": message,
                "messages": [HumanMessage(message)],
            },
            config,
        )

    while summary_tasks.has_active(session_id):
        await summary_tasks.wait_for(session_id)
        await asyncio.sleep(0.1)

    if ask_start_order:
        question = "最开始那个订单后来怎么说"
        message_id = await repository.append_message(
            session_id,
            "user",
            question,
        )
        result = await graph.ainvoke(
            {
                "session_id": session_id,
                "user_key": user_key,
                "current_user_message_id": message_id,
                "user_message": question,
                "messages": [HumanMessage(question)],
            },
            config,
        )
        print("final_reply=", result.get("reply", ""))
        while summary_tasks.has_active(session_id):
            await summary_tasks.wait_for(session_id)
            await asyncio.sleep(0.1)

    rows = await repository.load_summaries(session_id)
    print("session_id=", session_id)
    print(
        "budget=",
        budget.history_budget,
        budget.layer1_budget,
        budget.layer2_budget,
    )
    print(
        "history_hard_budget=",
        budget.hard_history_budget,
        "history_soft_budget=",
        budget.soft_history_budget,
        "full_prompt_limit=",
        budget.model_context_window
        - budget.max_output_tokens
        - budget.safety_margin_tokens,
    )
    print("summary_count=", len(rows))
    for row in rows:
        print(
            "summary",
            row.sequence_no,
            row.covered_from_msg_id,
            row.covered_to_msg_id,
            row.content,
        )
    await summary_tasks.shutdown()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=(
            "cost_optimized",
            "legacy_acceptance",
            "default",
            "acceptance",
        ),
        default="cost_optimized",
    )
    parser.add_argument("--turns", type=int, default=20)
    parser.add_argument("--session-id", default="")
    parser.add_argument("--user-key", default="访客")
    parser.add_argument("--ask-start-order", action="store_true")
    parser.add_argument(
        "--console-level",
        choices=("debug", "info", "warning", "error"),
        default="warning",
    )
    args = parser.parse_args()

    _configure_logging(args.console_level)
    session_id = args.session_id or (
        f"demo-ch07-{args.profile}-{os.getpid()}-{int(time.time())}"
    )
    await run_demo(
        profile=args.profile,
        turns=args.turns,
        session_id=session_id,
        user_key=args.user_key,
        ask_start_order=args.ask_start_order,
    )


if __name__ == "__main__":
    asyncio.run(main())
