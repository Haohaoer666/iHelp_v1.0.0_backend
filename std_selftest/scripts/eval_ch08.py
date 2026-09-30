"""Evaluate explicit ticket-request detection against the Ch08 label set."""

import argparse
import asyncio
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.llm import get_chat_model
from app.tools.ticket_gate import (
    TicketGateDecision,
    detect_ticket_request,
    deterministic_ticket_request,
)


DATASET_PATH = ROOT / "eval_data" / "ch08_ticket_gate_eval_set.json"
REPORT_PATH = ROOT / "reports" / "ch08_ticket_gate_failures.json"


async def evaluate(online: bool) -> dict[str, float]:
    samples = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    model = get_chat_model(streaming=False) if online else None
    rows = []
    for sample in samples:
        decision = deterministic_ticket_request(sample["text"])
        if decision is None:
            if model is None:
                decision = TicketGateDecision(explicit_request=False)
            else:
                decision = await detect_ticket_request(
                    sample["text"],
                    [],
                    model,
                )
        expected_missing = list(sample["missing_fields"])
        actual = {
            "explicit_request": decision.explicit_request,
            "ticket_type": decision.ticket_type,
            "description": decision.description,
            "missing_fields": list(decision.missing_fields),
        }
        expected = {
            "explicit_request": sample["explicit_request"],
            "ticket_type": sample["ticket_type"],
            "description": sample["description"],
            "missing_fields": expected_missing,
        }
        rows.append(
            {
                "text": sample["text"],
                "expected": expected,
                "actual": actual,
                "passed": actual == expected,
            }
        )

    total = len(rows)
    metrics = {
        "explicit_request_accuracy": sum(
            row["actual"]["explicit_request"]
            == row["expected"]["explicit_request"]
            for row in rows
        )
        / total,
        "ticket_type_accuracy": sum(
            row["actual"]["ticket_type"]
            == row["expected"]["ticket_type"]
            for row in rows
        )
        / total,
        "description_exact_match_rate": sum(
            row["actual"]["description"]
            == row["expected"]["description"]
            for row in rows
        )
        / total,
        "missing_fields_accuracy": sum(
            row["actual"]["missing_fields"]
            == row["expected"]["missing_fields"]
            for row in rows
        )
        / total,
    }
    failures = [row for row in rows if not row["passed"]]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(failures, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--online",
        action="store_true",
        help="Use the configured model for non-deterministic gate samples.",
    )
    args = parser.parse_args()
    metrics = asyncio.run(evaluate(online=args.online))
    for name, value in metrics.items():
        print(f"{name}={value:.4f}")
    thresholds = {
        "explicit_request_accuracy": 0.90,
        "missing_fields_accuracy": 0.85,
        "description_exact_match_rate": 0.80,
    }
    if any(metrics[name] < value for name, value in thresholds.items()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
