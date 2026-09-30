import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from app.graph.builder import build_chat_graph
from app.tools.ticket_gate import (
    TicketGateError,
    detect_ticket_request,
    parse_ticket_gate_response,
)


class StaticModel:
    def __init__(self, content: str) -> None:
        self.content = content

    async def ainvoke(self, messages, **kwargs):
        return AIMessage(content=self.content)


class FakeTurnLogger:
    async def log(self, session_id, state):
        return {"logged": True}


def test_gate_accepts_explicit_request_with_description():
    decision = parse_ticket_gate_response(
        '{"explicit_request":true,'
        '"ticket_type":"投诉",'
        '"description":"快递一直不到",'
        '"missing_fields":[]}'
    )

    assert decision.explicit_request is True
    assert decision.ticket_type == "投诉"
    assert decision.description == "快递一直不到"
    assert decision.missing_fields == ()
    assert decision.slots.complete is True


def test_gate_marks_missing_description():
    decision = parse_ticket_gate_response(
        '{"explicit_request":true,'
        '"ticket_type":"投诉",'
        '"description":"",'
        '"missing_fields":["description"]}'
    )

    assert decision.slots.missing_fields == ("description",)
    assert decision.slots.complete is False


def test_gate_accepts_json_code_fence():
    decision = parse_ticket_gate_response(
        '```json\n'
        '{"explicit_request":false,"ticket_type":"",'
        '"description":"","missing_fields":[]}'
        "\n```"
    )

    assert decision.explicit_request is False


def test_gate_rejects_extra_keys():
    with pytest.raises(TicketGateError, match="字段不合法"):
        parse_ticket_gate_response(
            '{"explicit_request":false,"ticket_type":"",'
            '"description":"","missing_fields":[],"reason":"x"}'
        )


async def test_explicit_ticket_request_bypasses_complaint_response():
    graph = build_chat_graph(
        intent_model=StaticModel(
            '{"intent":"投诉","confidence":0.95}'
        ),
        ticket_gate_model=StaticModel(
            '{"explicit_request":true,"ticket_type":"投诉",'
            '"description":"快递一直不到","missing_fields":[]}'
        ),
        agent_builder=lambda model, limits: RunnableLambda(
            lambda state: {
                "reply": "准备确认",
                "agent_steps": 1,
            }
        ),
        turn_logger=FakeTurnLogger(),
        checkpointer=InMemorySaver(),
    )

    result = await graph.ainvoke(
        {
            "session_id": "ticket-1",
            "user_message": "帮我建个工单，快递一直不到",
            "messages": [HumanMessage("帮我建个工单，快递一直不到")],
        },
        {"configurable": {"thread_id": "ticket-1"}},
    )

    assert result["reply"] == "准备确认"
    assert result["ticket_request_allowed"] is True
    assert result["ticket_slots"]["description"] == "快递一直不到"


async def test_missing_description_is_marked_for_follow_up():
    graph = build_chat_graph(
        intent_model=StaticModel(
            '{"intent":"投诉","confidence":0.95}'
        ),
        ticket_gate_model=StaticModel(
            '{"explicit_request":true,"ticket_type":"投诉",'
            '"description":"","missing_fields":["description"]}'
        ),
        agent_builder=lambda model, limits: RunnableLambda(
            lambda state: {
                "reply": "请补充具体问题描述",
                "agent_steps": 1,
            }
        ),
        turn_logger=FakeTurnLogger(),
        checkpointer=InMemorySaver(),
    )

    result = await graph.ainvoke(
        {
            "session_id": "ticket-missing",
            "user_message": "帮我建个工单",
            "messages": [HumanMessage("帮我建个工单")],
        },
        {"configurable": {"thread_id": "ticket-missing"}},
    )

    assert result["ticket_request_allowed"] is True
    assert result["ticket_slots"]["missing_fields"] == ["description"]
    assert "补充" in result["reply"]


async def test_explicit_phrase_bypasses_model_without_losing_description():
    class MustNotRunModel:
        async def ainvoke(self, *args, **kwargs):
            raise AssertionError("model should not be needed")

    decision = await detect_ticket_request(
        "帮我建个工单，快递一直不到",
        [],
        MustNotRunModel(),
    )

    assert decision.explicit_request is True
    assert decision.ticket_type == "其他"
    assert decision.description == "快递一直不到"
    assert decision.missing_fields == ()
