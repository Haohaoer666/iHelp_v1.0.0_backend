from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.graph.builder import build_chat_graph


class NoModel:
    async def ainvoke(self, *args, **kwargs):
        raise AssertionError("model must not be called")

    async def astream(self, *args, **kwargs):
        raise AssertionError("model must not be called")
        yield

    def bind_tools(self, tools, **kwargs):
        return self


class FakeContextManager:
    def __init__(self, pack):
        self.pack = pack

    async def prepare(self, session_id, current_user_message_id):
        return self.pack


class FakeTurnLogger:
    async def log(self, session_id, state):
        return {"logged": True}


async def test_understanding_and_intent_share_history_context():
    from app.context.models import ContextPack
    from app.graph.intent import IntentDecision

    captured = {}

    async def classify(text, model, *, history=None):
        captured["history"] = history
        return IntentDecision("闲聊", "chitchat", 0.9, "test")

    pack = ContextPack(
        summaries_text="早期订单 1001 已经登记退款。",
        layer2_messages=[],
        layer1_messages=[
            HumanMessage("旧问题"),
            HumanMessage("你好"),
        ],
        history_ctx=[
            HumanMessage("历史摘要：早期订单 1001 已经登记退款。"),
            HumanMessage("旧问题"),
            HumanMessage("你好"),
        ],
        layer1_from_msg_id=1,
        summary_span=None,
        layer2_tokens=0,
        layer1_tokens=4,
        token_estimate=4,
        trigger_summary=False,
    )
    graph = build_chat_graph(
        understanding_model=NoModel(),
        intent_model=NoModel(),
        agent_model=NoModel(),
        context_manager=FakeContextManager(pack),
        intent_classifier=classify,
        turn_logger=FakeTurnLogger(),
        checkpointer=InMemorySaver(),
    )

    await graph.ainvoke(
        {
            "session_id": "s-history",
            "current_user_message_id": 3,
            "user_message": "你好",
            "messages": [HumanMessage("你好")],
        },
        {"configurable": {"thread_id": "s-history"}},
    )

    assert [item.content for item in captured["history"]] == [
        "【历史摘要】\n早期订单 1001 已经登记退款。",
        "旧问题",
    ]
class FailingContextManager:
    async def prepare(self, session_id, current_user_message_id):
        from app.context.models import ContextPack

        return ContextPack(
            summaries_text="",
            layer2_messages=[],
            layer1_messages=[],
            history_ctx=[],
            layer1_from_msg_id=None,
            summary_span=None,
            layer2_tokens=0,
            layer1_tokens=0,
            token_estimate=0,
            trigger_summary=False,
            budget_error="上下文预算不足",
        )


async def test_budget_failure_returns_before_any_model_call():
    graph = build_chat_graph(
        intent_model=NoModel(),
        agent_model=NoModel(),
        context_manager=FailingContextManager(),
        checkpointer=InMemorySaver(),
    )

    result = await graph.ainvoke(
        {
            "session_id": "s-budget",
            "current_user_message_id": 1,
            "user_message": "你好",
            "messages": [HumanMessage("你好")],
        },
        {"configurable": {"thread_id": "s-budget"}},
    )

    assert result["reply"] == "上下文预算不足"
