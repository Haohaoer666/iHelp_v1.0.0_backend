from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import refunds as refunds_api
from app.core.tool_runner import ToolExecutionResult
from app.tools.models import ToolCaller


app = FastAPI()
app.include_router(refunds_api.router)


def test_refund_reasons_returns_fixed_categories():
    with TestClient(app) as client:
        response = client.get("/api/refunds/reasons")

    assert response.status_code == 200
    assert response.json() == [
        "商品质量问题",
        "七天无理由",
        "尺寸或型号不合适",
        "收到商品破损",
        "少件或错件",
        "其他",
    ]


def test_refund_endpoint_creates_refund_ticket(monkeypatch):
    class FakeExecutor:
        async def execute(self, tool_name, tool_args, context):
            assert tool_name == "create_ticket"
            assert tool_args["conversation_id"] == "s-refund"
            assert tool_args["ticket_type"] == "退款"
            assert "1001" in tool_args["description"]
            assert "七天无理由" in tool_args["description"]
            assert context.caller == ToolCaller.TRUSTED_UI
            return ToolExecutionResult(
                tool_name=tool_name,
                tool_args=tool_args,
                result={"ticket_id": "RF1", "status": "待处理"},
                success=True,
                status="success",
            )

    monkeypatch.setattr(
        refunds_api,
        "get_default_executor",
        lambda: FakeExecutor(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/refunds",
            json={
                "session_id": "s-refund",
                "order_id": "1001",
                "reason": "七天无理由",
            },
        )

    assert response.status_code == 200
    assert response.json()["ticket_id"] == "RF1"


def test_refund_endpoint_rejects_unknown_reason():
    with TestClient(app) as client:
        response = client.post(
            "/api/refunds",
            json={
                "session_id": "s-refund",
                "order_id": "1001",
                "reason": "随便写的理由",
            },
        )

    assert response.status_code == 422
