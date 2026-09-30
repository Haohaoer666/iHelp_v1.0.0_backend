import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_core.messages import HumanMessage

from app.agents.bare_react import AgentLimits, run_bare_agent
from app.core.llm import get_chat_model
from app.core.tools import TOOLS_BY_NAME


async def main() -> None:
    model = get_chat_model(streaming=False)
    result = await run_bare_agent(
        model=model,
        tools=[TOOLS_BY_NAME["query_order"], TOOLS_BY_NAME["query_logistics"]],
        messages=[HumanMessage("先查订单 1001 的状态，再查物流")],
        limits=AgentLimits(),
        on_event=print,
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
