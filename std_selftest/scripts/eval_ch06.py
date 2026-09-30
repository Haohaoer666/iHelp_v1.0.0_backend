import asyncio
import json
import re
import sys
from collections import Counter
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings
from app.core.llm import get_chat_model
from app.graph.intent import classify_intent_text
from app.services.conversation_understanding import understand_query


def _normalize_query(value: str) -> str:
    return re.sub(r"[\s，。！？、,.!?：:；;]+", "", value).lower()


def _resolved_matches(expected: dict, resolved_query: str) -> bool:
    if expected.get("expected_resolved_query") is not None:
        return _normalize_query(resolved_query) == _normalize_query(
            expected["expected_resolved_query"]
        )
    contains = expected.get("expected_resolved_contains") or []
    normalized = _normalize_query(resolved_query)
    return all(_normalize_query(item) in normalized for item in contains)


def _message_history(rows: list[dict]):
    messages = []
    for row in rows:
        message = HumanMessage(row["content"]) if row["role"] == "user" else AIMessage(row["content"])
        messages.append(message)
    return messages


async def main() -> int:
    dataset = json.loads(
        Path("eval_data/ch06_query_intent_eval_set.json").read_text(
            encoding="utf-8"
        )
    )
    understanding_model = get_chat_model(
        streaming=False,
        model=settings.query_understanding_model,
    )
    intent_model = get_chat_model(streaming=False)

    intent_correct = 0
    resolved_correct = 0
    json_parse_count = 0
    other_expected = 0
    other_correct = 0
    confusion: Counter[tuple[str, str]] = Counter()

    for row in dataset:
        understanding = await understand_query(
            _message_history(row.get("history") or []),
            row["current"],
            understanding_model,
        )
        decision = await classify_intent_text(
            understanding.resolved_query,
            intent_model,
        )

        resolved_correct += _resolved_matches(row, understanding.resolved_query)
        intent_correct += decision.intent == row["expected_intent"]
        json_parse_count += decision.model_used != "fallback"
        confusion[(row["expected_intent"], decision.intent)] += 1
        if row["expected_intent"] == "其他":
            other_expected += 1
            other_correct += decision.intent == "其他"

    total = len(dataset)
    json_parse_rate = json_parse_count / total
    intent_accuracy = intent_correct / total
    resolved_rate = resolved_correct / total
    other_rate = other_correct / other_expected if other_expected else 1.0

    print(f"json_parse_rate={json_parse_rate:.4f}")
    print(f"intent_accuracy={intent_accuracy:.4f}")
    print(f"resolved_query_match_rate={resolved_rate:.4f}")
    print(f"other_fallback_rate={other_rate:.4f}")
    for (expected, actual), count in sorted(confusion.items()):
        print(f"{expected}->{actual}: {count}")

    return 0 if (
        json_parse_rate >= 0.98
        and intent_accuracy >= 0.90
        and resolved_rate >= 0.80
    ) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
