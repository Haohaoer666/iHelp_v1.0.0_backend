import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.llm import get_chat_model
from app.graph.intent import classify_intent_text


async def main() -> int:
    rows = json.loads(
        Path("eval_data/ch05_intent_eval_set.json").read_text(encoding="utf-8")
    )
    model = get_chat_model(streaming=False)
    intent_correct = 0
    route_correct = 0
    confusion: Counter[tuple[str, str]] = Counter()

    for row in rows:
        decision = await classify_intent_text(row["text"], model)
        intent_correct += decision.intent == row["intent"]
        route_correct += decision.route == row["route"]
        confusion[(row["intent"], decision.intent)] += 1

    total = len(rows)
    intent_accuracy = intent_correct / total
    route_accuracy = route_correct / total
    print(f"intent_accuracy={intent_accuracy:.4f}")
    print(f"route_accuracy={route_accuracy:.4f}")
    for (expected, actual), count in sorted(confusion.items()):
        print(f"{expected}->{actual}: {count}")
    return 0 if min(intent_accuracy, route_accuracy) >= 0.85 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
