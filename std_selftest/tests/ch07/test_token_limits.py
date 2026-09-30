from langchain_core.messages import HumanMessage, SystemMessage

from app.context.tokens import (
    count_prompt_tokens,
    truncate_text_to_tokens,
    truncate_tool_content,
)


def test_full_prompt_count_includes_system_and_messages():
    prompt = [
        SystemMessage("固定客服人设"),
        HumanMessage("订单1001要退款"),
    ]
    assert count_prompt_tokens(prompt) > 0


def test_full_prompt_count_includes_tool_schema():
    prompt = [HumanMessage("查询订单 1001")]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "query_order",
                "description": "查询订单状态",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string"},
                    },
                    "required": ["order_id"],
                },
            },
        }
    ]

    assert count_prompt_tokens(prompt, tools=tools) > count_prompt_tokens(
        prompt
    )


def test_text_truncation_respects_budget():
    text = "订单1001要退款" * 100
    shortened = truncate_text_to_tokens(text, 20, 0.8)
    assert len(shortened) < len(text)
    assert shortened.endswith("...")


def test_tool_content_truncation_respects_budget():
    content = '{"order_id":"1001","items":["' + "x" * 2000 + '"]}'
    shortened = truncate_tool_content(content, 40, 0.8)
    assert len(shortened) < len(content)
    assert shortened.endswith("...")
