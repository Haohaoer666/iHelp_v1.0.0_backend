from langchain_core.messages import AIMessage, HumanMessage

from app.core.memory import SessionStore, trim_history


def test_store_get_unknown_session_returns_empty():
    assert SessionStore().get("nope") == []


def test_store_append_and_get_isolated_by_session():
    store = SessionStore()
    store.append("a", HumanMessage("hi"), AIMessage("hello"))
    store.append("b", HumanMessage("嗨"))

    assert len(store.get("a")) == 2
    assert len(store.get("b")) == 1


def test_trim_keeps_recent_and_starts_on_human():
    messages = []
    for i in range(20):
        messages.append(HumanMessage(f"问题{i}:" + "喵" * 50))
        messages.append(AIMessage(f"回答{i}:" + "喵" * 50))

    trimmed = trim_history(messages, max_tokens=200)
    assert 0 < len(trimmed) < len(messages)
    assert trimmed[0].type == "human"
