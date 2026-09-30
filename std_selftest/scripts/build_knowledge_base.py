"""Build the Markdown knowledge base and sync it to Milvus."""

import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.services.knowledge_pipeline import build_knowledge_base


async def main() -> int:
    docs_dir = pathlib.Path("knowledge_docs")
    result = await build_knowledge_base(docs_dir, rebuild=True)
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
