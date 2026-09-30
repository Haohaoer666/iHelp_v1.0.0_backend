from langchain_openai import ChatOpenAI

from app.config import settings


def get_chat_model(
    streaming: bool = False,
    model: str | None = None,
) -> ChatOpenAI:
    """Create a ChatOpenAI client using the configured upstream endpoint."""
    model_name = model or settings.chat_model
    model_kwargs = {}
    if "deepseek" in model_name.lower():
        model_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    return ChatOpenAI(
        model=model_name,
        base_url=settings.chat_base_url,
        api_key=settings.chat_api_key,
        streaming=streaming,
        temperature=0.3,
        **model_kwargs,
    )
