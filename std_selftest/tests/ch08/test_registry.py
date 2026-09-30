import pytest

from app.tools.models import (
    SideEffect,
    ToolDefinition,
    ToolOrigin,
)
from app.tools.registry import ToolRegistry


async def echo_value(value: str) -> dict:
    return {"value": value}


def definition(name: str) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description="测试工具",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        handler=echo_value,
        origin=ToolOrigin.BUILTIN,
        server_name=None,
        side_effect=SideEffect.READ,
        requires_ticket_confirmation=False,
        agent_visible=True,
    )


def test_registry_rejects_duplicate_name():
    registry = ToolRegistry()
    registry.register(definition("echo"))

    with pytest.raises(ValueError, match="重复工具名"):
        registry.register(definition("echo"))


def test_registry_exposes_stable_catalog_version():
    first = ToolRegistry()
    first.register(definition("echo"))
    second = ToolRegistry()
    second.register(definition("echo"))

    assert first.catalog_version == second.catalog_version
    assert len(first.catalog_version) == 64
