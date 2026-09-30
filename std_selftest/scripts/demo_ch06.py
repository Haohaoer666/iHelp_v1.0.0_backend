import argparse
import asyncio
import json
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings
from app.core.llm import get_chat_model
from app.graph.builder import build_chat_graph


def _print_result(result: dict) -> None:
    print(
        json.dumps(
            {
                "resolved_query": result.get("resolved_query"),
                "intent": result.get("intent"),
                "intent_confidence": result.get("intent_confidence"),
                "route": result.get("route"),
                "order_id": result.get("order_id"),
                "order_options": result.get("order_options"),
                "expanded_queries": result.get("expanded_queries"),
                "evidence_count": len(result.get("evidence") or []),
                "reply": result.get("reply"),
                "pending_action": result.get("pending_action"),
                "interrupt": bool(result.get("__interrupt__")),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("message")
    parser.add_argument("session_id")
    parser.add_argument("--resume-order", default="")
    args = parser.parse_args()

    graph = build_chat_graph(
        understanding_model=get_chat_model(
            streaming=False,
            model=settings.query_understanding_model,
        ),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": args.session_id}}
    result = await graph.ainvoke(
        {
            "session_id": args.session_id,
            "user_message": args.message,
            "messages": [HumanMessage(args.message)],
        },
        config,
    )
    _print_result(result)

    if result.get("__interrupt__") and args.resume_order:
        result = await graph.ainvoke(
            Command(
                resume={
                    "type": "order_selected",
                    "order_id": args.resume_order,
                }
            ),
            config,
        )
        _print_result(result)


if __name__ == "__main__":
    asyncio.run(main())
