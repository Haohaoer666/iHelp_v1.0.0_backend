"""Chapter 4 retrieval strategy evaluation."""

import argparse
import asyncio
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.llm import get_chat_model
from app.services.hybrid_retriever import retrieve


STRATEGIES = ("dense", "bm25", "hybrid", "hybrid_rerank")

#计算 Recall@K 检索召回率
def recall_at_k(
    retrieved: list[str],
    ground_truth: list[str],
    k: int,
) -> float:
    if not ground_truth:
        return 1.0 if not retrieved else 0.0
    hits = set(retrieved[:k]) & set(ground_truth)
    return len(hits) / len(set(ground_truth))


#MRR 能衡量 rerank 是否真的把正确结果提前了，这是 Recall@K 无法反映的
def mrr(retrieved: list[str], ground_truth: list[str]) -> float:
    truth = set(ground_truth)
    if not truth:
        return 1.0 if not retrieved else 0.0
    for index, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in truth:
            return 1.0 / index
    return 0.0


def average(values: list[float]) -> float:
    return round(statistics.fmean(values), 4) if values else 0.0


#证据充分性判断
async def judge_faithfulness(
    query: str,
    answer: str,
    evidence: str,
) -> float | None:
    if not answer:
        return None
    prompt = f"""判断参考答案是否能被给定证据支持，只输出 JSON。
问题：{query}
参考答案：{answer}
证据：{evidence}
输出格式：{{"faithful": true/false}}
"""
    try:
        response = await get_chat_model(streaming=False).ainvoke(prompt)
        payload = json.loads(_extract_json(getattr(response, "content", "")))
        return 1.0 if payload.get("faithful") else 0.0      # 解析json结果，true返回1.0，false返回0.0
    except Exception:
        return None


def _extract_json(value: str) -> str:
    value = value.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        value = "\n".join(lines)
    return value.strip()


async def evaluate_item(
    item: dict,
    strategy: str,
    with_faithfulness: bool,
) -> dict:
    hits = await retrieve(      #在reranker之后，特殊重排（lost in the middle）之前
        item["query"],
        category=item.get("category"),
        strategy=strategy,
        top_k=10,
    )
    retrieved_ids = [hit["chunk_id"] for hit in hits]
    truth_ids = item.get("ground_truth_chunk_ids") or []
    result = {
        "id": item["id"],
        "type": item["type"],
        "difficulty": item["difficulty"],
        "retrieved": retrieved_ids,
        "recall@3": recall_at_k(retrieved_ids, truth_ids, 3),
        "recall@5": recall_at_k(retrieved_ids, truth_ids, 5),
        "recall@10": recall_at_k(retrieved_ids, truth_ids, 10),
        "mrr": mrr(retrieved_ids, truth_ids),
    }
    if with_faithfulness and strategy == "hybrid_rerank":       #faithfulness 是**后面 if 分支里面，动态追加进 result 字典**
        result["faithfulness"] = await judge_faithfulness(
            item["query"],
            item.get("ground_truth_answer", ""),
            "\n".join(hit.get("text", "") for hit in hits),
        )
    return result


#聚合所有单条样本结果，计算各项指标的整体平均值，生成汇总指标
def aggregate(rows: list[dict]) -> dict:
    return {
        "count": len(rows),
        "recall@3": average([row["recall@3"] for row in rows]),
        "recall@5": average([row["recall@5"] for row in rows]),
        "recall@10": average([row["recall@10"] for row in rows]),
        "mrr": average([row["mrr"] for row in rows]),
        "faithfulness": average(
            [
                row["faithfulness"]
                for row in rows
                if row.get("faithfulness") is not None
            ]
        ),
    }


def grouped_aggregate(rows: list[dict], field: str) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row[field]].append(row)
    return {name: aggregate(items) for name, items in grouped.items()}


#评估主入口函数：读取评估数据集，遍历多个检索策略，执行评估、分组聚合，输出JSON和Markdown报表
async def run(args: argparse.Namespace) -> int:
    items = json.loads(Path(args.data).read_text(encoding="utf-8"))
    report = {"strategies": {}, "by_type": {}, "by_difficulty": {}}
    for strategy in args.strategies:
        rows = [
            await evaluate_item(item, strategy, args.with_faithfulness)
            for item in items
        ]
        report["strategies"][strategy] = aggregate(rows)
        report["by_type"][strategy] = grouped_aggregate(rows, "type")
        report["by_difficulty"][strategy] = grouped_aggregate(rows, "difficulty")

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "ch04_eval_report.json"
    md_path = report_dir / "ch04_eval_report.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Ch04 Evaluation Report",
        "",
        "| Strategy | Count | Recall@3 | Recall@5 | Recall@10 | MRR | Faithfulness |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy, metrics in report["strategies"].items():
        lines.append(
            f"| {strategy} | {metrics['count']} | {metrics['recall@3']} | "
            f"{metrics['recall@5']} | {metrics['recall@10']} | "
            f"{metrics['mrr']} | {metrics['faithfulness']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report["strategies"], ensure_ascii=False, indent=2))
    print(f"report_json={json_path}")
    print(f"report_md={md_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="eval_data/ch04_eval_set.json")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES))
    parser.add_argument("--with-faithfulness", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))
