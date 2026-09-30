from app.config import Settings

import pytest
from pydantic import ValidationError


def test_account_fields_required(monkeypatch):
    for key in ("CHAT_MODEL", "CHAT_BASE_URL", "CHAT_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_env_override(monkeypatch):
    values = {
        "CHAT_MODEL": "test-model",
        "CHAT_BASE_URL": "http://localhost:9000",
        "CHAT_API_KEY": "test-key",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("TOKEN_BUDGET", "500")

    loaded = Settings(_env_file=None)
    assert loaded.token_budget == 500
    assert loaded.chat_model == "test-model"
    assert loaded.async_database_url.startswith("mysql+aiomysql://")
