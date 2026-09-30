"""
使用 LangChain 的 OpenAI-compatible Embeddings 接口生成文本向量。
"""

from typing import cast

from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings

from app.config import settings


_embeddings: Embeddings | None = None


def _get_embeddings() -> Embeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = cast(
            Embeddings,
            init_embeddings(
                model=settings.embedding_model,
                provider="openai",
                api_key=settings.embedding_api_key,
                base_url=settings.embedding_base_url,
                timeout=60,
                check_embedding_ctx_length=False,
                model_kwargs={"encoding_format": "float"},
            ),
        )
    return _embeddings


def get_embeddings() -> Embeddings:
    """Return the configured LangChain embeddings client."""
    return _get_embeddings()


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return vectors for each input text using the configured embedding API."""
    if not texts:
        return []

    return await _get_embeddings().aembed_documents(texts)


async def embed_text(text: str) -> list[float]:
    return (await embed_texts([text]))[0]


async def embed_query(text: str) -> list[float]:
    """Embed a user query through the query-specific LangChain path."""
    return await _get_embeddings().aembed_query(text)
