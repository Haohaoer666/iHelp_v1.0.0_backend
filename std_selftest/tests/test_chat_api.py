import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.types import Command

from app.api import chat as chat_api


api_app = FastAPI()
api_app.include_router(chat_api.router)
api_app.state.chat_graph = None


@pytest.fixture(autouse=True)
def fake_user_persistence(monkeypatch):
    calls = []

    async def fake_persist_user_message(session_id, user_key, message):
        calls.append((session_id, user_key, message))
        return 101

    monkeypatch.setattr(
        chat_api,
        "persist_user_message",
        fake_persist_user_message,
    )
    return calls


class FakeGraph:
    def __init__(self, events=None, error: Exception | None = None) -> None:
        self.events = events or []
        self.error = error
        self.calls = []

    async def astream(self, inputs, config, stream_mode, subgraphs=False):
        self.calls.append((inputs, config, stream_mode, subgraphs))
        for event in self.events:
            yield event
        if self.error:
            raise self.error


def collect_data_lines(response) -> list[str]:
    return [
        line
        for line in response.iter_lines()
        if line.startswith("data: ")
    ]


def test_chat_streams_graph_custom_events(fake_user_persistence):
    fake_graph = FakeGraph(
        events=[
            {"type": "node_status", "node": "resolve_reference", "status": "success"},
            {"type": "tool_status", "tool_name": "query_logistics", "status": "running"},
            {"type": "delta", "text": "物流运输中。"},
            {
                "type": "reply_options",
                "options": [{"id": "create_ticket", "label": "建工单"}],
            },
        ]
    )
    api_app.dependency_overrides[chat_api.get_chat_graph] = lambda: fake_graph
    try:
        with TestClient(api_app) as client:
            with client.stream(
                "POST",
                "/api/chat",
                json={"session_id": "s1", "message": "订单 1001 的物流到哪了"},
            ) as response:
                lines = collect_data_lines(response)
    finally:
        api_app.dependency_overrides.clear()

    payloads = [json.loads(line[6:]) for line in lines[:-1]]
    assert payloads[0]["type"] == "node_status"
    assert payloads[2] == {"type": "delta", "text": "物流运输中。"}
    assert payloads[3]["type"] == "reply_options"
    assert lines[-1] == "data: [DONE]"
    assert fake_graph.calls[0][0]["user_message"] == "订单 1001 的物流到哪了"
    assert fake_graph.calls[0][0]["user_key"] == "demo-user"
    assert fake_graph.calls[0][0]["current_user_message_id"] == 101
    assert fake_graph.calls[0][1] == {"configurable": {"thread_id": "s1"}}
    assert fake_graph.calls[0][2] == "custom"
    assert fake_graph.calls[0][3] is True


def test_chat_emits_error_event_when_graph_fails():
    api_app.dependency_overrides[chat_api.get_chat_graph] = lambda: FakeGraph(
        error=RuntimeError("upstream unavailable")
    )
    try:
        with TestClient(api_app) as client:
            with client.stream(
                "POST",
                "/api/chat",
                json={"session_id": "s2", "message": "你好"},
            ) as response:
                frames = list(response.iter_lines())
    finally:
        api_app.dependency_overrides.clear()

    content_frames = [frame for frame in frames if frame]
    assert "event: error" in content_frames
    assert any("上游服务暂时不可用" in frame for frame in content_frames)
    assert content_frames[-1] == "data: [DONE]"


def test_chat_forwards_order_resume_as_command(fake_user_persistence):
    fake_graph = FakeGraph(events=[{"type": "delta", "text": "继续处理"}])
    api_app.dependency_overrides[chat_api.get_chat_graph] = lambda: fake_graph
    try:
        with TestClient(api_app) as client:
            with client.stream(
                "POST",
                "/api/chat",
                json={
                    "session_id": "s3",
                    "resume": {"type": "order_selected", "order_id": "1001"},
                },
            ) as response:
                lines = collect_data_lines(response)
    finally:
        api_app.dependency_overrides.clear()

    graph_input = fake_graph.calls[0][0]
    assert isinstance(graph_input, Command)
    assert graph_input.resume == {"type": "order_selected", "order_id": "1001"}
    assert fake_user_persistence == []
    assert lines[-1] == "data: [DONE]"


def test_chat_rejects_message_and_resume_together():
    with TestClient(api_app) as client:
        response = client.post(
            "/api/chat",
            json={
                "session_id": "s4",
                "message": "这个能退吗",
                "resume": {"type": "order_selected", "order_id": "1001"},
            },
        )

    assert response.status_code == 422


def test_chat_rejects_missing_message_and_resume():
    with TestClient(api_app) as client:
        response = client.post("/api/chat", json={"session_id": "s5"})

    assert response.status_code == 422


def test_chat_rejects_empty_resume():
    with TestClient(api_app) as client:
        response = client.post(
            "/api/chat",
            json={"session_id": "s6", "resume": {}},
        )

    assert response.status_code == 422


def test_chat_rejects_session_owned_by_another_user(monkeypatch):
    from app.db.repository import ConversationOwnershipError

    async def reject(*args, **kwargs):
        raise ConversationOwnershipError("owned")

    monkeypatch.setattr(chat_api, "persist_user_message", reject)
    with TestClient(api_app) as client:
        response = client.post(
            "/api/chat",
            json={
                "session_id": "s7",
                "user_key": "u2",
                "message": "你好",
            },
        )

    assert response.status_code == 409
