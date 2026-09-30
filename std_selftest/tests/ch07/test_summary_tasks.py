import asyncio

from langchain_core.messages import AIMessage

from app.context.models import ContextMessage, SummarySpan
from app.context.tasks import SummaryTaskManager


class SlowModel:
    async def ainvoke(self, messages):
        await asyncio.sleep(0.05)
        return AIMessage("用户询问订单1001。")


class Store:
    def __init__(self) -> None:
        self.calls = []

    async def append_summary_and_advance(self, **kwargs):
        self.calls.append(kwargs)


async def test_schedule_does_not_wait_for_model(monkeypatch):
    logs = []
    monkeypatch.setattr(
        "app.context.tasks.log_summary_event",
        lambda *args, **kwargs: logs.append((args, kwargs)),
    )
    store = Store()
    manager = SummaryTaskManager(
        model=SlowModel(),
        store=store,
        estimator=lambda text: len(text),
        min_chars=2,
        max_chars=100,
    )

    status = manager.schedule(
        "s1",
        SummarySpan(1, 2),
        [ContextMessage(1, "user", "订单1001")],
    )

    assert status == "running"
    assert store.calls == []
    await manager.wait_for("s1")
    assert store.calls[0]["from_id"] == 1
    assert any(args[0] == "done" for args, _ in logs)


async def test_schedule_skips_when_same_session_is_running():
    manager = SummaryTaskManager(
        model=SlowModel(),
        store=Store(),
        estimator=lambda text: len(text),
        min_chars=2,
        max_chars=100,
    )
    span = SummarySpan(1, 2)
    messages = [ContextMessage(1, "user", "订单1001")]

    assert manager.schedule("s1", span, messages) == "running"
    assert manager.schedule("s1", span, messages) == "running"
    await manager.wait_for("s1")

