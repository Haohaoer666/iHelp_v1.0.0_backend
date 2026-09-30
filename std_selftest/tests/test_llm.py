from langchain_openai import ChatOpenAI

from app.core.llm import get_chat_model


def test_get_chat_model_returns_chat_openai(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "chat_model", "test-model")
    monkeypatch.setattr(settings, "chat_base_url", "http://localhost:9000")
    monkeypatch.setattr(settings, "chat_api_key", "test-key")

    model = get_chat_model()
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "test-model"
    assert model.openai_api_base == "http://localhost:9000"


def test_get_chat_model_streaming(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "chat_model", "test-model")
    monkeypatch.setattr(settings, "chat_base_url", "http://localhost:9000")
    monkeypatch.setattr(settings, "chat_api_key", "test-key")

    model = get_chat_model(streaming=True)
    assert model.streaming is True


def test_get_chat_model_disables_deepseek_thinking(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "chat_model", "deepseek-v4-flash")
    monkeypatch.setattr(settings, "chat_base_url", "https://api.deepseek.com/v1")
    monkeypatch.setattr(settings, "chat_api_key", "test-key")

    model = get_chat_model()
    assert model.extra_body == {"thinking": {"type": "disabled"}}
