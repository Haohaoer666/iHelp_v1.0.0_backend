from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from app.config import settings
from app.context.budget import ContextBudget
from app.graph.builder import build_chat_graph


class NoUseModel:
    async def ainvoke(self, *args, **kwargs):
        raise AssertionError("model should not be called")

    async def astream(self, *args, **kwargs):
        raise AssertionError("model should not be called")
        yield

    def bind_tools(self, tools, **kwargs):
        return self


class FakeTurnLogger:
    def __init__(self) -> None:
        self.calls = []

    async def log(self, session_id, state):
        self.calls.append((session_id, state))
        return {"logged": True}


def fixed_agent(reply: str):
    def builder(model, limits):
        return RunnableLambda(lambda state: {"reply": reply, "agent_steps": 1})

    return builder


def test_build_chat_graph_uses_agent_limit_settings(monkeypatch):
    captured = {}

    monkeypatch.setattr(settings, "agent_max_steps", 2)
    monkeypatch.setattr(settings, "agent_max_output_tokens", 321)
    monkeypatch.setattr(settings, "agent_token_budget", 654)

    def builder(model, limits):
        captured["limits"] = limits
        return RunnableLambda(lambda state: {"reply": "ok", "agent_steps": 1})

    build_chat_graph(
        intent_model=NoUseModel(),
        agent_model=NoUseModel(),
        intent_classifier=lambda text, model: None,
        agent_builder=builder,
        turn_logger=FakeTurnLogger(),
        checkpointer=InMemorySaver(),
    )

    assert captured["limits"].max_steps == 2
    assert captured["limits"].max_output_tokens == 321
    assert captured["limits"].token_budget == 654


def test_build_chat_graph_uses_full_prompt_budget(monkeypatch):
    captured = {}

    monkeypatch.setattr(settings, "model_context_window", 65536)
    monkeypatch.setattr(settings, "agent_max_output_tokens", 2000)
    monkeypatch.setattr(settings, "safety_margin_tokens", 350)

    def builder(model, limits):
        captured["limits"] = limits
        return RunnableLambda(lambda state: {"reply": "ok", "agent_steps": 1})

    build_chat_graph(
        intent_model=NoUseModel(),
        agent_model=NoUseModel(),
        intent_classifier=lambda text, model: None,
        agent_builder=builder,
        turn_logger=FakeTurnLogger(),
        checkpointer=InMemorySaver(),
    )

    assert captured["limits"].full_prompt_token_budget == 63186


def test_build_chat_graph_accepts_explicit_context_budget():
    captured = {}
    legacy_budget = ContextBudget(
        model_context_window=18000,
        max_output_tokens=2000,
        max_user_input_tokens=2000,
        max_agent_steps=3,
        tool_call_message_max_tokens=400,
        tool_result_max_tokens=1200,
        rerank_top_k=5,
        evidence_chunk_token_budget=400,
        evidence_total_max_tokens=None,
        system_prompt_token_budget=1800,
        summary_injection_token_budget=600,
        summary_segment_max_tokens=200,
        safety_margin_tokens=350,
        desired_retained_turns=24,
        steady_turn_token_estimate=800,
        legacy_acceptance=True,
    )

    def builder(model, limits):
        captured["limits"] = limits
        return RunnableLambda(lambda state: {"reply": "ok", "agent_steps": 1})

    build_chat_graph(
        intent_model=NoUseModel(),
        agent_model=NoUseModel(),
        intent_classifier=lambda text, model: None,
        agent_builder=builder,
        turn_logger=FakeTurnLogger(),
        checkpointer=InMemorySaver(),
        context_budget=legacy_budget,
    )

    assert captured["limits"].history_hard_budget == 5650
    assert captured["limits"].history_soft_budget == 19200


async def test_complaint_does_not_call_model_and_returns_independent_options():
    async def classify(text, model):
        return "投诉"

    logger = FakeTurnLogger()
    graph = build_chat_graph(
        intent_model=NoUseModel(),
        agent_model=NoUseModel(),
        intent_classifier=classify,
        agent_builder=fixed_agent("不应执行"),
        turn_logger=logger,
        checkpointer=InMemorySaver(),
    )

    result = await graph.ainvoke(
        {
            "session_id": "s1",
            "user_message": "我要投诉",
            "messages": [HumanMessage("我要投诉")],
        },
        {"configurable": {"thread_id": "s1"}},
    )

    assert result["reply_options"] == [
        {"id": "human_transfer", "label": "转人工"},
    ]
    assert "建工单" not in result["reply"]
    assert "创建工单" not in result["reply"]
    assert len(result["messages"]) == 2
    assert logger.calls[0][0] == "s1"


async def test_complaint_emits_fixed_reply_and_options_events():
    async def classify(text, model):
        return "投诉"

    graph = build_chat_graph(
        intent_model=NoUseModel(),
        agent_model=NoUseModel(),
        intent_classifier=classify,
        agent_builder=fixed_agent("不应执行"),
        turn_logger=FakeTurnLogger(),
        checkpointer=InMemorySaver(),
    )

    events = [
        event
        async for event in graph.astream(
            {
                "session_id": "s-complaint-events",
                "user_message": "我要投诉",
                "messages": [HumanMessage("我要投诉")],
            },
            {"configurable": {"thread_id": "s-complaint-events"}},
            stream_mode="custom",
        )
    ]

    assert any(
        event.get("type") == "delta" and "很抱歉" in event.get("text", "")
        for event in events
    )
    assert any(
        event.get("type") == "reply_options"
        and event.get("options")
        == [
            {"id": "human_transfer", "label": "转人工"},
        ]
        for event in events
    )


async def test_knowledge_path_always_runs_retrieval_and_gate():
    seen = []

    async def classify(text, model):
        return "商品咨询"

    async def retrieve(query):
        seen.append("retrieve")
        return {
            "matched": True,
            "evidence": [
                {
                    "citation_id": 1,
                    "chunk_id": "c1",
                    "text": "七天退货",
                    "score": 0.9,
                }
            ],
        }

    async def assess(query, evidence, conversation_id):
        seen.append("gate")
        return {"sufficient": True, "context": "[1] 七天退货"}

    graph = build_chat_graph(
        intent_model=NoUseModel(),
        agent_model=NoUseModel(),
        intent_classifier=classify,
        retrieval_service=retrieve,
        evidence_assessor=assess,
        agent_builder=fixed_agent("按政策处理"),
        turn_logger=FakeTurnLogger(),
        checkpointer=InMemorySaver(),
    )

    result = await graph.ainvoke(
        {
            "session_id": "s1",
            "user_message": "退款政策是什么",
            "messages": [HumanMessage("退款政策是什么")],
        },
        {"configurable": {"thread_id": "s1"}},
    )

    assert seen == ["retrieve", "gate"]
    assert result["reply"] == "按政策处理"


async def test_knowledge_path_emits_retrieval_and_gate_node_events():
    async def classify(text, model):
        return "商品咨询"

    async def retrieve(query):
        return {
            "matched": True,
            "evidence": [{"citation_id": 1, "text": "七天退货", "score": 0.9}],
        }

    async def assess(query, evidence, conversation_id):
        return {"sufficient": True, "context": "[1] 七天退货"}

    graph = build_chat_graph(
        intent_model=NoUseModel(),
        agent_model=NoUseModel(),
        intent_classifier=classify,
        retrieval_service=retrieve,
        evidence_assessor=assess,
        agent_builder=fixed_agent("按政策处理"),
        turn_logger=FakeTurnLogger(),
        checkpointer=InMemorySaver(),
    )

    events = [
        event
        async for event in graph.astream(
            {
                "session_id": "s-events",
                "user_message": "退款政策是什么",
                "messages": [HumanMessage("退款政策是什么")],
            },
            {"configurable": {"thread_id": "s-events"}},
            stream_mode="custom",
        )
    ]

    assert any(
        event.get("type") == "node_status"
        and event.get("node") == "retrieve_knowledge"
        and event.get("status") == "running"
        for event in events
    )
    assert any(
        event.get("type") == "node_status"
        and event.get("node") == "confidence_gate"
        and event.get("status") == "running"
        for event in events
    )


async def test_business_path_skips_retrieval_and_calls_agent():
    seen = []

    async def classify(text, model):
        return "物流"

    async def retrieve(query):
        raise AssertionError("business path must not retrieve")

    def agent_builder(model, limits):
        async def run(state):
            seen.append("agent")
            return {"reply": "物流运输中", "agent_steps": 1}

        return RunnableLambda(run)

    graph = build_chat_graph(
        intent_model=NoUseModel(),
        agent_model=NoUseModel(),
        intent_classifier=classify,
        retrieval_service=retrieve,
        agent_builder=agent_builder,
        turn_logger=FakeTurnLogger(),
        checkpointer=InMemorySaver(),
    )

    result = await graph.ainvoke(
        {
            "session_id": "s1",
            "user_message": "订单 1001 的物流到哪了",
            "messages": [HumanMessage("订单 1001 的物流到哪了")],
        },
        {"configurable": {"thread_id": "s1"}},
    )

    assert seen == ["agent"]
    assert result["reply"] == "物流运输中"


async def test_business_without_order_id_interrupts_with_order_selector():
    captured = {}

    async def classify(text, model):
        return "物流"

    def agent_builder(model, limits):
        async def run(state):
            captured.update(state)
            return {"reply": "物流运输中", "agent_steps": 1}

        return RunnableLambda(run)

    graph = build_chat_graph(
        intent_model=NoUseModel(),
        agent_model=NoUseModel(),
        intent_classifier=classify,
        agent_builder=agent_builder,
        turn_logger=FakeTurnLogger(),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "business-order-select"}}

    first = await graph.ainvoke(
        {
            "session_id": "business-order-select",
            "user_message": "我的包裹到哪了",
            "messages": [HumanMessage("我的包裹到哪了")],
        },
        config,
    )

    assert first["__interrupt__"]
    assert [item["order_id"] for item in first["order_options"]] == [
        "1001",
        "1002",
        "1003",
    ]
    assert captured == {}

    second = await graph.ainvoke(
        Command(
            resume={
                "type": "order_selected",
                "order_id": "1002",
            }
        ),
        config,
    )

    assert captured["order_id"] == "1002"
    assert captured["order_data"]["product_name"] == "iHao 蓝牙耳机"
    assert second["reply"] == "物流运输中"


async def test_business_with_explicit_order_id_skips_order_selector():
    captured = {}

    async def classify(text, model):
        return "订单"

    def agent_builder(model, limits):
        async def run(state):
            captured.update(state)
            return {"reply": "订单已加载", "agent_steps": 1}

        return RunnableLambda(run)

    graph = build_chat_graph(
        intent_model=NoUseModel(),
        agent_model=NoUseModel(),
        intent_classifier=classify,
        agent_builder=agent_builder,
        turn_logger=FakeTurnLogger(),
        checkpointer=InMemorySaver(),
    )

    result = await graph.ainvoke(
        {
            "session_id": "business-order-explicit",
            "user_message": "订单 1002 是什么状态",
            "messages": [HumanMessage("订单 1002 是什么状态")],
        },
        {"configurable": {"thread_id": "business-order-explicit"}},
    )

    assert "__interrupt__" not in result
    assert captured["order_id"] == "1002"
    assert captured["order_data"]["status"] == "运输中"
    assert result["reply"] == "订单已加载"


async def test_weak_evidence_returns_fallback_without_calling_agent():
    async def classify(text, model):
        return "商品咨询"

    async def retrieve(query):
        return {
            "matched": True,
            "evidence": [{"citation_id": 1, "text": "弱相关证据", "score": 0.1}],
        }

    async def assess(query, evidence, conversation_id):
        return {
            "sufficient": False,
            "reason": "检索置信度不足",
            "refusal": "抱歉，现有知识不足以可靠回答这个问题。",
        }

    def agent_builder(model, limits):
        def run(state):
            raise AssertionError("weak evidence must not call agent")

        return RunnableLambda(run)

    graph = build_chat_graph(
        intent_model=NoUseModel(),
        agent_model=NoUseModel(),
        intent_classifier=classify,
        retrieval_service=retrieve,
        evidence_assessor=assess,
        agent_builder=agent_builder,
        turn_logger=FakeTurnLogger(),
        checkpointer=InMemorySaver(),
    )

    result = await graph.ainvoke(
        {
            "session_id": "s1",
            "user_message": "退款政策是什么",
            "messages": [HumanMessage("退款政策是什么")],
        },
        {"configurable": {"thread_id": "s1"}},
    )

    assert result["reply"] == "抱歉，现有知识不足以可靠回答这个问题。"


async def test_chitchat_returns_fixed_reply_without_retrieval_or_agent():
    async def classify(text, model):
        return "闲聊"

    graph = build_chat_graph(
        intent_model=NoUseModel(),
        agent_model=NoUseModel(),
        intent_classifier=classify,
        retrieval_service=lambda query: (_ for _ in ()).throw(
            AssertionError("chitchat must not retrieve")
        ),
        agent_builder=fixed_agent("不应执行"),
        turn_logger=FakeTurnLogger(),
        checkpointer=InMemorySaver(),
    )

    result = await graph.ainvoke(
        {
            "session_id": "s1",
            "user_message": "你好",
            "messages": [HumanMessage("你好")],
        },
        {"configurable": {"thread_id": "s1"}},
    )

    assert result["reply"] == "您好，我是 iHelp 智能客服，很高兴为您服务。"


async def test_inmemory_checkpointer_restores_messages():
    async def classify(text, model):
        return "闲聊"

    graph = build_chat_graph(
        intent_model=NoUseModel(),
        agent_model=NoUseModel(),
        intent_classifier=classify,
        agent_builder=fixed_agent("不应执行"),
        turn_logger=FakeTurnLogger(),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "persist-1"}}

    await graph.ainvoke(
        {
            "session_id": "persist-1",
            "user_message": "你好",
            "messages": [HumanMessage("你好")],
        },
        config,
    )
    result = await graph.ainvoke(
        {
            "session_id": "persist-1",
            "user_message": "继续",
            "messages": [HumanMessage("继续")],
        },
        config,
    )

    assert len(result["messages"]) == 4


async def test_sqlite_checkpointer_restores_after_reopen(tmp_path):
    async def classify(text, model):
        return "闲聊"

    config = {"configurable": {"thread_id": "persist-2"}}
    path = tmp_path / "checkpoints.db"

    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        graph = build_chat_graph(
            intent_model=NoUseModel(),
            agent_model=NoUseModel(),
            intent_classifier=classify,
            agent_builder=fixed_agent("不应执行"),
            turn_logger=FakeTurnLogger(),
            checkpointer=saver,
        )
        await graph.ainvoke(
            {
                "session_id": "persist-2",
                "user_message": "你好",
                "messages": [HumanMessage("你好")],
            },
            config,
        )

    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        graph = build_chat_graph(
            intent_model=NoUseModel(),
            agent_model=NoUseModel(),
            intent_classifier=classify,
            agent_builder=fixed_agent("不应执行"),
            turn_logger=FakeTurnLogger(),
            checkpointer=saver,
        )
        snapshot = await graph.aget_state(config)

    assert len(snapshot.values["messages"]) == 2
