from app.services.turn_logger import TurnLogger


async def test_turn_logger_does_not_persist_tools(monkeypatch):
    calls = []

    async def fake_ensure_conversation(session_id):
        calls.append(("conversation", session_id))

    async def fake_append_message(session_id, role, content, **kwargs):
        calls.append(("message", session_id, role, content))
        return 1

    monkeypatch.setattr(
        "app.services.turn_logger.ensure_conversation",
        fake_ensure_conversation,
    )
    monkeypatch.setattr(
        "app.services.turn_logger.append_message",
        fake_append_message,
    )
    await TurnLogger().log(
        "s1",
        {
            "current_user_message_id": 7,
            "user_message": "订单1001",
            "reply": "已查询",
            "tool_trace": [
                {
                    "name": "query_order",
                    "args": {"order_id": "1001"},
                    "call_id": "c1",
                    "content": "{}",
                }
            ],
        },
    )

    assert calls == [
        ("conversation", "s1"),
        ("message", "s1", "assistant", "已查询"),
    ]
