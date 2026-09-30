from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import tickets as tickets_api
from app.core.tool_runner import ToolExecutionResult
from app.tools.models import ToolCaller


app = FastAPI()
app.include_router(tickets_api.router)


def test_ticket_endpoint_calls_create_ticket(monkeypatch):
    class FakeExecutor:
        async def execute(self, tool_name, tool_args, context):
            assert tool_name == "create_ticket"
            assert tool_args == {
                "conversation_id": "s1",
                "description": "我要投诉",
                "ticket_type": "投诉",
            }
            assert context.caller == ToolCaller.TRUSTED_UI
            assert context.trusted_ui_confirmation is True
            return ToolExecutionResult(
                tool_name=tool_name,
                tool_args=tool_args,
                result={"ticket_id": "TK1", "status": "待处理"},
                success=True,
                status="success",
            )

    monkeypatch.setattr(
        tickets_api,
        "get_default_executor",
        lambda: FakeExecutor(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/tickets",
            json={
                "session_id": "s1",
                "description": "我要投诉",
                "ticket_type": "投诉",
            },
        )

    assert response.status_code == 200
    assert response.json()["ticket_id"] == "TK1"


def test_ticket_endpoint_returns_error(monkeypatch):
    class FakeExecutor:
        async def execute(self, tool_name, tool_args, context):
            return ToolExecutionResult(
                tool_name=tool_name,
                tool_args=tool_args,
                result=None,
                success=False,
                error="ticket db unavailable",
                status="failure",
            )

    monkeypatch.setattr(
        tickets_api,
        "get_default_executor",
        lambda: FakeExecutor(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/tickets",
            json={"session_id": "s1", "description": "我要投诉"},
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "ticket db unavailable"
