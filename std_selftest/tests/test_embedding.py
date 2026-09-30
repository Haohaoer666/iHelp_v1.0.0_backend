import httpx
import pytest

from app.config import settings
from app.services import embedding


class FakeEmbeddings:
    def __init__(self) -> None:
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls.append(texts)
        return [[float(index), float(index + 1)] for index, _ in enumerate(texts)]

    async def aembed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return [0.5, 0.6]


async def test_embed_texts_delegates_to_langchain_embeddings(monkeypatch):
    fake = FakeEmbeddings()
    monkeypatch.setattr(embedding, "_embeddings", fake, raising=False)
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *args, **kwargs: pytest.fail("embed_texts 不应再直接调用 httpx"),
        raising=False,
    )

    assert await embedding.embed_texts(["A", "B"]) == [[0.0, 1.0], [1.0, 2.0]]
    assert fake.document_calls == [["A", "B"]]


def test_get_embeddings_uses_custom_openai_compatible_config(monkeypatch):
    captured = {}
    fake = object()

    def fake_init_embeddings(model, **kwargs):
        captured["model"] = model
        captured.update(kwargs)
        return fake

    monkeypatch.setattr(settings, "embedding_model", "test-embedding-model")
    monkeypatch.setattr(settings, "embedding_base_url", "http://localhost:9999/v1")
    monkeypatch.setattr(settings, "embedding_api_key", "test-embedding-key")
    monkeypatch.setattr(
        embedding,
        "init_embeddings",
        fake_init_embeddings,
        raising=False,
    )
    monkeypatch.setattr(embedding, "_embeddings", None, raising=False)

    assert embedding._get_embeddings() is fake
    assert captured == {
        "model": "test-embedding-model",
        "provider": "openai",
        "api_key": "test-embedding-key",
        "base_url": "http://localhost:9999/v1",
        "timeout": 60,
        "check_embedding_ctx_length": False,
        "model_kwargs": {"encoding_format": "float"},
    }


async def test_embed_text_reuses_document_embedding_path(monkeypatch):
    fake = FakeEmbeddings()
    monkeypatch.setattr(embedding, "_embeddings", fake, raising=False)
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *args, **kwargs: pytest.fail("embed_text 不应再直接调用 httpx"),
        raising=False,
    )

    assert await embedding.embed_text("A") == [0.0, 1.0]
    assert fake.document_calls == [["A"]]


async def test_embed_query_delegates_to_langchain_query(monkeypatch):
    fake = FakeEmbeddings()
    monkeypatch.setattr(embedding, "_embeddings", fake, raising=False)

    assert await embedding.embed_query("用户问题") == [0.5, 0.6]
    assert fake.query_calls == ["用户问题"]
