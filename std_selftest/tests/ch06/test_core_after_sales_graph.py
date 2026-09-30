from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.graph.builder import build_chat_graph


class StaticModel:
    def __init__(self, content: str) -> None:
        self.content = content

    async def ainvoke(self, messages, **kwargs):
        return AIMessage(content=self.content)


class NoUseAgentModel:
    def bind_tools(self, tools, **kwargs):
        return self

    async def astream(self, *args, **kwargs):
        raise AssertionError("agent model should not be used in these tests")
        yield


class FakeTurnLogger:
    async def log(self, session_id, state):
        return {"logged": True}


async def fake_multi_retrieve(queries, category=None):
    return {
        "matched": True,
        "evidence": [
            {
                "citation_id": 1,
                "chunk_id": "policy-1",
                "text": "签收后七天内商品完好可退。",
                "score": 0.95,
            }
        ],
        "failed_queries": [],
        "retrieval_query_count": len(queries),
    }


async def sufficient_evidence(query, evidence, conversation_id):
    return {"sufficient": True, "context": "[1] 签收后七天内商品完好可退。"}


async def expansion_model_response():
    return None


def fixed_agent(captured):
    def builder(model, limits):
        async def run(state):
            captured.update(state)
            return {
                "reply": "订单 1001 签收未超七天且商品完好，可以申请退款。",
                "agent_steps": 1,
            }

        return RunnableLambda(run)

    return builder


def make_graph(*, captured, understand_content, intent_content, expansion_content):
    return build_chat_graph(
        understanding_model=StaticModel(understand_content),
        intent_model=StaticModel(intent_content),
        expansion_model=StaticModel(expansion_content),
        agent_model=NoUseAgentModel(),
        retrieval_service=lambda query: None,
        multi_retrieval_service=fake_multi_retrieve,
        evidence_assessor=sufficient_evidence,
        agent_builder=fixed_agent(captured),
        checkpointer=InMemorySaver(),
        turn_logger=FakeTurnLogger(),
    )


async def test_missing_order_interrupts_and_resume_completes_core_subflow():
    captured = {}
    graph = make_graph(
        captured=captured,
        understand_content='{"resolved_query":"用户想确认这个订单是否可以退款"}',
        intent_content='{"intent":"退款退货","confidence":0.96}',
        expansion_content='{"queries":["七天无理由退货条件","退款退货政策"]}',
    )
    config = {"configurable": {"thread_id": "ch06-resume"}}

    first = await graph.ainvoke(
        {
            "session_id": "ch06-resume",
            "user_message": "这个能退吗",
            "messages": [HumanMessage("这个能退吗")],
        },
        config,
    )

    assert first["__interrupt__"]
    assert first["order_options"]

    second = await graph.ainvoke(
        Command(resume={"type": "order_selected", "order_id": "1001"}),
        config,
    )

    assert second["order_id"] == "1001"
    assert second["order_data"]["product_name"] == "iHao 智能手表"
    assert second["expanded_queries"] == [
        "用户想确认这个订单是否可以退款",
        "七天无理由退货条件",
        "退款退货政策",
    ]
    assert captured["order_data"]["order_id"] == "1001"
    assert captured["expanded_queries"][0] == "用户想确认这个订单是否可以退款"
    assert second["reply_options"] == [
        {"id": "apply_refund", "label": "申请退款"}
    ]


async def test_resolved_query_with_order_skips_selector():
    captured = {}
    graph = make_graph(
        captured=captured,
        understand_content='{"resolved_query":"订单 1001 是否可以退款"}',
        intent_content='{"intent":"退款退货","confidence":0.96}',
        expansion_content='{"queries":["退款政策"]}',
    )

    result = await graph.ainvoke(
        {
            "session_id": "ch06-order-present",
            "user_message": "订单 1001 能退吗",
            "messages": [HumanMessage("订单 1001 能退吗")],
        },
        {"configurable": {"thread_id": "ch06-order-present"}},
    )

    assert "__interrupt__" not in result
    assert result["order_id"] == "1001"
    assert result["pending_action"] == ""
    assert result["order_data"]["evaluation_date"]
    assert isinstance(result["order_data"]["days_since_signed"], int)
    assert result["order_data"]["days_since_signed"] >= 0
    assert captured["order_id"] == "1001"


async def test_unknown_explicit_order_does_not_reach_agent():
    captured = {}
    graph = make_graph(
        captured=captured,
        understand_content='{"resolved_query":"订单 9999 是否可以退款"}',
        intent_content='{"intent":"退款退货","confidence":0.96}',
        expansion_content='{"queries":["退款政策"]}',
    )

    result = await graph.ainvoke(
        {
            "session_id": "ch06-order-missing",
            "user_message": "订单 9999 能退吗",
            "messages": [HumanMessage("订单 9999 能退吗")],
        },
        {"configurable": {"thread_id": "ch06-order-missing"}},
    )

    assert result["order_data"] == {}
    assert "没有找到这个订单" in result["reply"]
    assert captured == {}


async def test_after_sales_does_not_add_refund_form_option():
    captured = {}
    graph = make_graph(
        captured=captured,
        understand_content='{"resolved_query":"订单 1001 的维修进度"}',
        intent_content='{"intent":"售后","confidence":0.95}',
        expansion_content='{"queries":["维修进度政策"]}',
    )

    result = await graph.ainvoke(
        {
            "session_id": "ch06-after-sales",
            "user_message": "订单 1001 的维修进度",
            "messages": [HumanMessage("订单 1001 的维修进度")],
        },
        {"configurable": {"thread_id": "ch06-after-sales"}},
    )

    assert result["reply_options"] == []


async def test_exchange_order_selection_emits_exchange_form():
    captured = {}
    graph = make_graph(
        captured=captured,
        understand_content='{"resolved_query":"衣服买大了，想换货"}',
        intent_content='{"intent":"售后","confidence":0.96}',
        expansion_content='{"queries":["换货条件和流程"]}',
    )
    config = {"configurable": {"thread_id": "ch06-exchange"}}

    first = await graph.ainvoke(
        {
            "session_id": "ch06-exchange",
            "user_message": "衣服买大了 我要换",
            "messages": [HumanMessage("衣服买大了 我要换")],
        },
        config,
    )
    assert first["__interrupt__"]

    events = [
        event
        async for event in graph.astream(
            Command(resume={"type": "order_selected", "order_id": "1003"}),
            config,
            stream_mode="custom",
        )
    ]
    snapshot = await graph.aget_state(config)

    assert snapshot.values["intent"] == "售后"
    assert snapshot.values["after_sales_action"] == "exchange"
    assert snapshot.values["order_id"] == "1003"
    assert any(
        event.get("type") == "exchange_form"
        and event.get("order_id") == "1003"
        and "尺码不合适" in event.get("reasons", [])
        for event in events
    )


async def test_slot_answer_reuses_previous_after_sales_context():
    captured = {}
    graph = make_graph(
        captured=captured,
        understand_content='{"resolved_query":"00100，没拆封"}',
        intent_content='{"intent":"其他","confidence":0.88}',
        expansion_content='{"queries":["换货条件和流程"]}',
    )

    result = await graph.ainvoke(
        {
            "session_id": "ch06-slot-answer",
            "user_message": "00100，没拆封",
            "messages": [HumanMessage("00100，没拆封")],
            "last_intent": "售后",
            "last_order_id": "1003",
            "last_after_sales_action": "exchange",
        },
        {"configurable": {"thread_id": "ch06-slot-answer"}},
    )

    assert "__interrupt__" not in result
    assert result["intent"] == "售后"
    assert result["after_sales_action"] == "exchange"
    assert result["order_id"] == "1003"
    assert "无法判断" not in result["reply"]


async def test_invalid_order_resume_keeps_selector_pending():
    captured = {}
    graph = make_graph(
        captured=captured,
        understand_content='{"resolved_query":"用户想确认这个订单是否可以退款"}',
        intent_content='{"intent":"退款退货","confidence":0.96}',
        expansion_content='{"queries":["退款政策"]}',
    )
    config = {"configurable": {"thread_id": "ch06-invalid-resume"}}
    first = await graph.ainvoke(
        {
            "session_id": "ch06-invalid-resume",
            "user_message": "这个能退吗",
            "messages": [HumanMessage("这个能退吗")],
        },
        config,
    )
    assert first["__interrupt__"]

    second = await graph.ainvoke(
        Command(resume={"type": "order_selected", "order_id": "9999"}),
        config,
    )

    assert second["__interrupt__"]
    assert second["pending_action"] == "order_selection"
    assert captured == {}


async def test_other_intent_uses_controlled_fallback():
    captured = {}
    graph = make_graph(
        captured=captured,
        understand_content='{"resolved_query":"帮我看看那个奇怪的东西"}',
        intent_content='{"intent":"其他","confidence":0.9}',
        expansion_content='{"queries":["不应调用"]}',
    )

    result = await graph.ainvoke(
        {
            "session_id": "ch06-other",
            "user_message": "帮我看看那个奇怪的东西",
            "messages": [HumanMessage("帮我看看那个奇怪的东西")],
        },
        {"configurable": {"thread_id": "ch06-other"}},
    )

    assert result["intent"] == "其他"
    assert "无法判断" in result["reply"]
    assert captured == {}
