import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.graph.builder import build_chat_graph


async def main() -> None:
    question = sys.argv[1]
    session_id = sys.argv[2]
    graph = build_chat_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": session_id}}

    async for chunk in graph.astream(
        {
            "session_id": session_id,
            "user_message": question,
            "messages": [HumanMessage(question)],
        },
        config,
        stream_mode="custom",
        subgraphs=True,
    ):
        namespace, event = (
            chunk if isinstance(chunk, tuple) and len(chunk) == 2 else ((), chunk)
        )
        print(
            json.dumps(
                {"namespace": list(namespace), "event": event},
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
