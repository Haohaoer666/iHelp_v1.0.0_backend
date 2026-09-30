from datetime import datetime

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import conversations as conversations_api


app = FastAPI()
app.include_router(conversations_api.router)


class FakeCheckpointer:
    def __init__(self):
        self.deleted_threads = []

    async def adelete_thread(self, thread_id):
        self.deleted_threads.append(thread_id)


def test_conversation_list_requires_user_key():
    with TestClient(app) as client:
        response = client.get("/api/conversations")

    assert response.status_code == 422


def test_conversation_list_returns_sidebar_contract(monkeypatch):
    async def fake_list_conversations(user_key):
        assert user_key == "u1"
        return [
            {
                "id": "s1",
                "preview": "订单1001后来怎么说",
                "created_at": datetime(2026, 9, 21, 10, 0, 0),
                "updated_at": datetime(2026, 9, 21, 10, 5, 0),
                "summarized": True,
            }
        ]

    monkeypatch.setattr(
        conversations_api,
        "list_conversations",
        fake_list_conversations,
    )
    with TestClient(app) as client:
        response = client.get("/api/conversations?user_key=u1")

    assert response.status_code == 200
    assert response.json()[0]["preview"] == "订单1001后来怎么说"
    assert response.json()[0]["summarized"] is True


def test_message_api_returns_original_turns(monkeypatch):
    async def fake_load_conversation_messages(conversation_id, user_key):
        assert (conversation_id, user_key) == ("s1", "u1")
        return [
            {
                "id": 1,
                "role": "user",
                "content": "订单1001",
                "created_at": datetime(2026, 9, 21, 10, 0, 0),
            },
            {
                "id": 2,
                "role": "assistant",
                "content": "已查询",
                "created_at": datetime(2026, 9, 21, 10, 0, 1),
            },
        ]

    monkeypatch.setattr(
        conversations_api,
        "load_conversation_messages",
        fake_load_conversation_messages,
    )
    with TestClient(app) as client:
        response = client.get(
            "/api/conversations/s1/messages?user_key=u1"
        )

    assert response.status_code == 200
    assert [item["role"] for item in response.json()] == [
        "user",
        "assistant",
    ]


def test_message_api_returns_404_for_other_user(monkeypatch):
    async def fake_load_conversation_messages(conversation_id, user_key):
        return None

    monkeypatch.setattr(
        conversations_api,
        "load_conversation_messages",
        fake_load_conversation_messages,
    )
    with TestClient(app) as client:
        response = client.get(
            "/api/conversations/s1/messages?user_key=other"
        )

    assert response.status_code == 404


def test_delete_conversation_removes_db_and_checkpoint(monkeypatch):
    calls = []
    checkpointer = FakeCheckpointer()
    app.state.checkpointer = checkpointer

    async def fake_get_conversation(conversation_id, user_key):
        return {"id": conversation_id, "user_name": user_key}

    async def fake_delete_conversation(conversation_id, user_key):
        calls.append((conversation_id, user_key))
        return True

    monkeypatch.setattr(
        conversations_api,
        "get_conversation_for_user",
        fake_get_conversation,
    )
    monkeypatch.setattr(
        conversations_api,
        "delete_conversation_data",
        fake_delete_conversation,
    )
    with TestClient(app) as client:
        response = client.delete("/api/conversations/s1?user_key=u1")

    assert response.status_code == 204
    assert calls == [("s1", "u1")]
    assert checkpointer.deleted_threads == ["s1"]


def test_delete_conversation_returns_404_for_other_user(monkeypatch):
    async def fake_get_conversation(conversation_id, user_key):
        return None

    monkeypatch.setattr(
        conversations_api,
        "get_conversation_for_user",
        fake_get_conversation,
    )
    with TestClient(app) as client:
        response = client.delete("/api/conversations/s1?user_key=u2")

    assert response.status_code == 404
