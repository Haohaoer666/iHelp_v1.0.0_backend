"""Chapter 2 acceptance evaluation against the running FastAPI backend."""

import asyncio
import json
import sys

import httpx

BASE = "http://127.0.0.1:8000"
CASES = [
    ("订单 1001 的物流到哪了", "query_logistics"),
    ("退货政策是什么", "query_faq"),
    ("邮费是多少", None),
]


async def stream_chat(client: httpx.AsyncClient, session_id: str, message: str):
    tool_names = []
    final_text = []
    async with client.stream(
        "POST",
        f"{BASE}/api/chat",
        json={"session_id": session_id, "message": message},
        timeout=60,
    ) as response:
        response.raise_for_status()
        buffer = ""
        async for chunk in response.aiter_bytes():
            buffer += chunk.decode("utf-8", errors="ignore")
            while "\n\n" in buffer:
                frame, buffer = buffer.split("\n\n", 1)
                data = ""
                event_type = "message"
                for line in frame.split("\n"):
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    if line.startswith("data:"):
                        data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if payload.get("type") == "tool_status":
                    tool_names.append(payload["tool_name"])
                elif event_type != "error" and "delta" in payload:
                    final_text.append(payload["delta"])
    return tool_names, "".join(final_text)


async def main() -> int:
    failures = 0
    async with httpx.AsyncClient(timeout=60) as client:
        for index, (message, expected_tool) in enumerate(CASES, 1):
            tools, answer = await stream_chat(
                client,
                f"eval-ch02-{index}",
                message,
            )
            if expected_tool is None:
                ok = expected_tool not in tools
            else:
                ok = expected_tool in tools
            print(f"[{'PASS' if ok else 'FAIL'}] {message}")
            print(f"         tools={tools}")
            print(f"         answer={answer[:80]}")
            failures += not ok
    print(f"\n{len(CASES) - failures}/{len(CASES)} 通过")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
