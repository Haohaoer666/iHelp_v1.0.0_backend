from langchain_core.messages import AIMessage

from app.context.models import ContextMessage
from app.context.summary import build_summary_prompt, summarize_span


class StaticSummaryModel:
    def __init__(self, content: str) -> None:
        self.content = content
        self.messages = []

    async def ainvoke(self, messages):
        self.messages = messages
        return AIMessage(self.content)


def test_summary_prompt_forbids_invention():
    prompt = build_summary_prompt(
        [
            ContextMessage(1, "user", "订单1001要退款"),
            ContextMessage(2, "assistant", "已登记"),
        ]
    )

    assert "不得编造" in prompt
    assert "订单1001" in prompt
    assert "寒暄" in prompt


async def test_summarize_span_returns_body_only():
    model = StaticSummaryModel("用户咨询订单1001退款，客服已登记，问题仍未解决。")

    result = await summarize_span(
        [ContextMessage(1, "user", "订单1001要退款")],
        model,
        min_chars=5,
        max_chars=100,
    )

    assert result.startswith("用户咨询订单1001")


async def test_summarize_span_strips_markdown_and_limits_length():
    model = StaticSummaryModel("```text\n" + "事实" * 40 + "\n```")

    result = await summarize_span(
        [ContextMessage(1, "user", "订单1001")],
        model,
        min_chars=5,
        max_chars=20,
    )

    assert result == "事实" * 10


async def test_summarize_span_enforces_token_cap():
    model = StaticSummaryModel("订单1001要退款。" * 100)

    result = await summarize_span(
        [ContextMessage(1, "user", "订单1001要退款")],
        model,
        min_chars=5,
        max_chars=500,
        max_tokens=30,
        chinese_tokens_per_char=0.8,
    )

    from app.context.tokens import estimate_text_tokens

    assert estimate_text_tokens(result, 0.8) <= 30
