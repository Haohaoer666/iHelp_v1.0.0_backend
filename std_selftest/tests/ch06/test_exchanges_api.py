from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import exchanges as exchanges_api
from app.core.tool_runner import ToolExecutionResult
from app.tools.models import ToolCaller


app = FastAPI()
app.include_router(exchanges_api.router)


def test_exchange_reasons_returns_fixed_categories():
    with TestClient(app) as client:
        response = client.get("/api/exchanges/reasons")

    assert response.status_code == 200
    assert response.json() == [
        "尺码不合适",
        "颜色或款式不喜欢",
        "商品破损",
        "少件或错件",
        "质量问题",
        "其他",
    ]


def test_exchange_endpoint_creates_exchange_ticket(monkeypatch):
    class FakeExecutor:
        async def execute(self, tool_name, tool_args, context):
            assert tool_name == "create_ticket"
            assert tool_args["conversation_id"] == "s-exchange"
            assert tool_args["ticket_type"] == "换货"
            assert "1003" in tool_args["description"]
            assert "尺码不合适" in tool_args["description"]
            assert context.caller == ToolCaller.TRUSTED_UI
            return ToolExecutionResult(
                tool_name=tool_name,
                tool_args=tool_args,
                result={"ticket_id": "EX1", "status": "待处理"},
                success=True,
                status="success",
            )

    monkeypatch.setattr(
        exchanges_api,
        "get_default_executor",
        lambda: FakeExecutor(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/exchanges",
            json={
                "session_id": "s-exchange",
                "order_id": "1003",
                "reason": "尺码不合适",
            },
        )

    assert response.status_code == 200
    assert response.json()["ticket_id"] == "EX1"


def test_exchange_endpoint_rejects_unknown_reason():
    with TestClient(app) as client:
        response = client.post(
            "/api/exchanges",
            json={
                "session_id": "s-exchange",
                "order_id": "1003",
                "reason": "随便写的理由",
            },
        )

    assert response.status_code == 422
