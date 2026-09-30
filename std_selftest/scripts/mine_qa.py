"""Mine QA pairs from historical messages and stage them into the knowledge base."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.qa_miner import mine_qa_from_messages


async def main() -> int:
    result = await mine_qa_from_messages()
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
