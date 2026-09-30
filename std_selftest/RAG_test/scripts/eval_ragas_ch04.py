"""Optional RAGAS generation-quality evaluation for Chapter 4."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_core.messages import HumanMessage, SystemMessage
from openai import AsyncOpenAI
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from app.config import settings
from app.core.llm import get_chat_model
from app.services.hybrid_retriever import retrieve


REFUSAL_MARKERS = ("抱歉", "无法", "没有相关", "不足以", "不能可靠回答")
METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)

#7. 功能是判断一段模型输出是否属于拒答。
def is_refusal_response(text: str) -> bool:
    return any(marker in text for marker in REFUSAL_MARKERS)


#5. 从完整评估行里筛选出真正能进入 RAGAS 的样本。
def build_ragas_rows(rows: list[dict]) -> list[dict]:
    samples: list[dict] = []
    for row in rows:
        if row["expect_refusal"]:   # 过滤标记为 expect_refusal=True 的样本（这类样本只用来算拒答准确率，不参与RAGAS打分）
            continue
        if not row["contexts"]:     # 过滤没有检索上下文的样本，没有上下文无法计算RAGAS指标
            continue

        # 组装RAGAS固定字段
        samples.append(
            {
                "user_input": row["query"],
                "retrieved_contexts": row["contexts"],
                "response": row["response"],
                "reference": row["ground_truth_answer"],
            }
        )
    return samples


#6. 计算知识库外问题的拒答 **准确率**。
def refusal_accuracy(rows: list[dict]) -> float:
    refusal_rows = [row for row in rows if row["expect_refusal"]]
    if not refusal_rows:
        return 0.0
    correct = sum(1 for row in refusal_rows if is_refusal_response(row["response"]))
    return round(correct / len(refusal_rows), 4)


#2. 根据检索到的上下文生成一个待评估回答。
async def generate_answer(query: str, contexts: list[str]) -> str:
    if not contexts:
        return "抱歉，现有知识不足以可靠回答这个问题。"
    evidence = "\n".join(
        f"[{index}] {context}"
        for index, context in enumerate(contexts, start=1)
    )
    response = await get_chat_model(streaming=False).ainvoke(
        [
            SystemMessage(
                content=(
                    "你是RAG评估回答生成器。只能依据给定证据回答，"
                    "不得编造事实或来源；证据不足时必须明确拒答。"
                )
            ),
            HumanMessage(
                content=f"用户问题：{query}\n\n证据：\n{evidence}\n\n请生成最终回答。"
            ),
        ]
    )
    return str(getattr(response, "content", "") or "")


#1. 准备 RAGAS 评估所需的每条原始记录。
async def collect_rows(
    items: list[dict],
    strategy: str,
    limit: int | None,
    top_k: int,
) -> list[dict]:
    selected = items[:limit] if limit else items
    rows = []
    for item in selected:
        print(f"[ragas] collect {strategy}: {item['id']}", flush=True)
        hits = await retrieve(
            item["query"],
            category=item.get("category"),
            strategy=strategy,
            top_k=top_k,
        )
        contexts = [hit["text"] for hit in hits]
        response = await generate_answer(item["query"], contexts)
        rows.append(
            {
                "id": item["id"],
                "query": item["query"],
                "type": item["type"],
                "difficulty": item["difficulty"],
                "expect_refusal": item.get("expect_refusal", False),
                "ground_truth_answer": item.get("ground_truth_answer", ""),
                "contexts": contexts,
                "response": response,
            }
        )
    return rows

#4. 构造 RAGAS 可评分样本
async def score_rows(
    rows: list[dict],
    judge_model: str,
) -> tuple[dict, list[dict]]:
    samples = build_ragas_rows(rows)
    if not samples:
        return {}, []
    evaluator_llm = llm_factory(        # 初始化RAGAS评估LLM（裁判模型），AsyncOpenAI异步客户端
        judge_model,
        client=AsyncOpenAI(
            api_key=settings.chat_api_key,
            base_url=settings.chat_base_url,
        ),
        max_tokens=16384,
    )
    evaluator_embeddings = embedding_factory(   # 初始化RAGAS评估用Embedding（AnswerRelevancy需要向量）
        "openai",
        settings.embedding_model,
        client=AsyncOpenAI(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
        ),
    )
    #   实例化4个RAGAS指标对象，绑定对应的LLM/Embedding
    metric_objects = {
        "faithfulness": Faithfulness(llm=evaluator_llm),
        "answer_relevancy": AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        ),
        "context_precision": ContextPrecision(llm=evaluator_llm),
        "context_recall": ContextRecall(llm=evaluator_llm),
    }
    #  准备容器：保存每个指标所有样本的分数；errors存放异常记录
    values: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    errors: list[dict] = []

    for sample in samples:  #   遍历每一条待评估样本
        print(f"[ragas] score: {sample['user_input'][:40]}", flush=True)
        metric_calls = {
            "faithfulness": metric_objects["faithfulness"].ascore(
                user_input=sample["user_input"],
                response=sample["response"],
                retrieved_contexts=sample["retrieved_contexts"],
            ),
            "answer_relevancy": metric_objects["answer_relevancy"].ascore(
                user_input=sample["user_input"],
                response=sample["response"],
            ),
            "context_precision": metric_objects["context_precision"].ascore(
                user_input=sample["user_input"],
                reference=sample["reference"],
                retrieved_contexts=sample["retrieved_contexts"],
            ),
            "context_recall": metric_objects["context_recall"].ascore(
                user_input=sample["user_input"],
                retrieved_contexts=sample["retrieved_contexts"],
                reference=sample["reference"],
            ),
        }
        # 逐个执行指标打分
        for metric_name, metric_call in metric_calls.items():
            try:
                metric_result = await metric_call
                values[metric_name].append(float(metric_result.value))
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "user_input": sample["user_input"],
                        "metric": metric_name,
                        "error": str(exc),
                    }
                )
    #  计算每个指标的平均分，保留4位小数；无有效分数则填None
    summary = {
        name: round(sum(scores) / len(scores), 4) if scores else None
        for name, scores in values.items()
    }
    return summary, errors

# 主函数入口
async def run(args: argparse.Namespace) -> int:
    items = json.loads(Path(args.data).read_text(encoding="utf-8"))
    strategies = args.strategies or ["hybrid_rerank"]
    report = {}
    all_rows = {}
    for strategy in strategies:
        rows = await collect_rows(
            items,
            strategy,
            args.limit,
            args.ragas_top_k,
        )
        all_rows[strategy] = rows
        metrics, errors = await score_rows(rows, args.judge_model)

        # 组装当前策略的报告信息
        report[strategy] = {
            "count": len(rows),
            "ragas_count": len(build_ragas_rows(rows)),
            "refusal_accuracy": refusal_accuracy(rows),
            "metrics": metrics,
            "errors": errors,
            "judge_model": args.judge_model,
            "ragas_top_k": args.ragas_top_k,
        }

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "ch04_ragas_report.json"
    rows_path = report_dir / "ch04_ragas_rows.json"
    md_path = report_dir / "ch04_ragas_report.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    rows_path.write_text(
        json.dumps(all_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Ch04 RAGAS Evaluation Report",
        "",
        "| Strategy | Count | RAGAS Count | Refusal Accuracy | Faithfulness | Answer Relevancy | Context Precision | Context Recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy, item in report.items():
        metrics = item["metrics"]
        lines.append(
            f"| {strategy} | {item['count']} | {item['ragas_count']} | "
            f"{item['refusal_accuracy']} | "
            f"{metrics.get('faithfulness', '-')} | "
            f"{metrics.get('answer_relevancy', '-')} | "
            f"{metrics.get('context_precision', '-')} | "
            f"{metrics.get('context_recall', '-')} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report_json={json_path}")
    print(f"report_md={md_path}")
    print(f"rows_json={rows_path}")
    return 0

# 负责解析命令行参数
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="eval_data/ch04_eval_set.json")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["hybrid_rerank"],
        choices=["dense", "bm25", "hybrid", "hybrid_rerank"],
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N samples.",
    )
    parser.add_argument(
        "--judge-model",
        default=settings.chat_model,
        help="Model used by RAGAS as judge.",
    )
    parser.add_argument(
        "--ragas-top-k",
        type=int,
        default=5,
        help="Number of retrieved contexts passed to RAGAS.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))
