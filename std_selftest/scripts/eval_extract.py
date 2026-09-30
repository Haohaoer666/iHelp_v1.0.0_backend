"""Run annotated samples against /api/extract."""
import asyncio
import json
import pathlib
import sys

import httpx

SAMPLES = pathlib.Path(__file__).parent.parent / "tests/data/extract_samples.json"
BASE = "http://localhost:8000"


async def main() -> int:
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
    failures = 0
    async with httpx.AsyncClient(timeout=60) as client:
        for index, sample in enumerate(samples, 1):
            resp = await client.post(
                f"{BASE}/api/extract",
                json={"text": sample["text"]},
            )
            resp.raise_for_status()
            got = resp.json()
            expected = sample["expected"]
            ok = (
                got["order_id"] == expected["order_id"]
                and got["request_type"] == expected["request_type"]
            )
            failures += not ok
            status = "PASS" if ok else "FAIL"
            print(f"[{status}] #{index} {sample['text'][:24]}...")
            print(
                f"       期望 order_id={expected['order_id']} "
                f"type={expected['request_type']}"
            )
            print(
                f"       实际 order_id={got['order_id']} "
                f"type={got['request_type']} "
                f"方案={got['expected_solution']}"
            )
    print(f"\n{len(samples) - failures}/{len(samples)} 通过")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
