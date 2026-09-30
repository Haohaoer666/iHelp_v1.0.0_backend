"""Chapter 3 knowledge base acceptance evaluation."""

import asyncio
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.knowledge_pipeline import sync_pending_chunks

BASE = "http://127.0.0.1:8000"


async def stream_chat(message: str):
    tools = []
    final_text = []
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream(
            "POST",
            f"{BASE}/api/chat",
            json={"session_id": "eval-ch03", "message": message},
        ) as response:
            response.raise_for_status()
            buffer = ""
            async for chunk in response.aiter_bytes():
                buffer += chunk.decode("utf-8", errors="ignore")
                while "\n\n" in buffer:
                    frame, buffer = buffer.split("\n\n", 1)
                    data = ""
                    for line in frame.split("\n"):
                        if line.startswith("data:"):
                            data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    payload = json.loads(data)
                    if payload.get("type") == "tool_status":
                        tools.append(payload["tool_name"])
                    elif "delta" in payload:
                        final_text.append(payload["delta"])
    return list(dict.fromkeys(tools)), "".join(final_text)


async def main() -> int:
    tools, answer = await stream_chat("邮费是多少")
    recall_ok = "query_faq" in tools and ("运费" in answer or "包邮" in answer)
    retry_result = await sync_pending_chunks()
    retry_ok = retry_result == {"synced": 0, "failed": 0}

    print(f"[{'PASS' if recall_ok else 'FAIL'}] 邮费是多少")
    print(f"         tools={tools}")
    print(f"         answer={answer[:100]}")
    print(f"[{'PASS' if retry_ok else 'FAIL'}] 重跑无遗漏")
    print(f"         sync={retry_result}")
    return 0 if recall_ok and retry_ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
