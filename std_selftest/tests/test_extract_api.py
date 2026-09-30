from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableLambda

from app.api import extract as extract_api
from app.schemas.extract import AfterSalesTicket


app = FastAPI()
app.include_router(extract_api.router)


def override(runnable):
    app.dependency_overrides[extract_api.get_extractor] = lambda: runnable
    return TestClient(app)


def teardown_function():
    app.dependency_overrides.clear()


def test_extract_returns_structured_json():
    ticket = AfterSalesTicket(
        order_id="MH20260701123",
        request_type="退款",
        expected_solution="到货损坏要求退款",
    )
    client = override(RunnableLambda(lambda _: ticket))
    with client:
        resp = client.post(
            "/api/extract",
            json={"text": "订单 MH20260701123 散架了，退款"},
        )

    assert resp.status_code == 200
    assert resp.json() == {
        "order_id": "MH20260701123",
        "request_type": "退款",
        "expected_solution": "到货损坏要求退款",
    }


def test_extract_upstream_failure_returns_502():
    def boom(_):
        raise RuntimeError("upstream unavailable")

    client = override(RunnableLambda(boom))
    with client:
        resp = client.post(
            "/api/extract",
            json={"text": "订单号缺失，我要投诉"},
        )

    assert resp.status_code == 502
    assert "上游模型" in resp.json()["detail"]
