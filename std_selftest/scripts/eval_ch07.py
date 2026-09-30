"""Evaluate the facts-only summary prompt on the Ch07 labeled set."""

import argparse
import asyncio
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings
from app.context.models import ContextMessage
from app.context.summary import summarize_span
from app.context.tokens import estimate_text_tokens
from app.core.llm import get_chat_model


def _load_cases(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


async def run_evaluation(
    *,
    data_path: Path,
    report_path: Path,
) -> dict[str, float]:
    cases = _load_cases(data_path)
    model = get_chat_model(streaming=False)
    key_total = 0
    key_hits = 0
    unsupported = 0
    small_talk_leaks = 0
    length_pass = 0
    failures = []

    for case in cases:
        messages = [
            ContextMessage(
                index + 1,
                item["role"],
                item["content"],
            )
            for index, item in enumerate(case["span"])
        ]
        summary = await summarize_span(
            messages,
            model,
            min_chars=settings.summary_target_min_chars,
            max_chars=settings.summary_target_max_chars,
            max_tokens=settings.summary_segment_max_tokens,
            chinese_tokens_per_char=settings.chinese_tokens_per_char,
        )
        missing = [
            fact
            for fact in case["must_include"]
            if fact not in summary
        ]
        unsupported_terms = [
            fact
            for fact in case["must_not_include"]
            if fact in summary
        ]
        leaked = [
            phrase
            for phrase in case["small_talk"]
            if phrase in summary
        ]
        length_ok = (
            settings.summary_target_min_chars
            <= len(summary)
            <= settings.summary_target_max_chars
            and estimate_text_tokens(
                summary,
                settings.chinese_tokens_per_char,
            )
            <= settings.summary_segment_max_tokens
        )

        key_total += len(case["must_include"])
        key_hits += len(case["must_include"]) - len(missing)
        unsupported += bool(unsupported_terms)
        small_talk_leaks += bool(leaked)
        length_pass += int(length_ok)

        if missing or unsupported_terms or leaked or not length_ok:
            failures.append(
                {
                    "id": case["id"],
                    "summary": summary,
                    "missing": missing,
                    "unsupported": unsupported_terms,
                    "small_talk_leak": leaked,
                    "length": len(summary),
                }
            )

    metrics = {
        "key_fact_recall": key_hits / key_total if key_total else 1.0,
        "unsupported_fact_rate": unsupported / len(cases),
        "small_talk_leak_rate": small_talk_leaks / len(cases),
        "length_pass_rate": length_pass / len(cases),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(failures, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=str(ROOT / "eval_data/ch07_context_summary_eval_set.json"),
    )
    parser.add_argument(
        "--report",
        default=str(ROOT / "reports/ch07_summary_failures.json"),
    )
    args = parser.parse_args()

    metrics = await run_evaluation(
        data_path=Path(args.data),
        report_path=Path(args.report),
    )
    for name, value in metrics.items():
        print(f"{name}={value:.4f}")

    failed = (
        metrics["key_fact_recall"] < 0.90
        or metrics["unsupported_fact_rate"] > 0
        or metrics["small_talk_leak_rate"] > 0
        or metrics["length_pass_rate"] < 0.90
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
