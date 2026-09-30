import pkgutil
from types import SimpleNamespace

from app.tools.models import SideEffect, ToolDefinition, ToolOrigin
from app.tools.registry import ToolRegistry, discover_builtin_tools


def test_builtin_discovery_registers_expected_tools():
    registry = ToolRegistry()
    discover_builtin_tools(registry)

    by_name = {item.name: item for item in registry.list()}
    assert set(by_name) == {
        "create_ticket",
        "query_faq",
        "query_order",
        "query_product",
    }
    assert "query_logistics" not in by_name
    assert by_name["query_faq"].agent_visible is False
    assert by_name["create_ticket"].requires_ticket_confirmation is True


def test_new_builtin_module_is_discovered_without_registry_edit(
    monkeypatch,
):
    async def sample_tool(value: str) -> dict:
        return {"value": value}

    sample_module = SimpleNamespace(
        TOOL_DEFINITION=ToolDefinition(
            name="sample_tool",
            description="示例",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
            handler=sample_tool,
            origin=ToolOrigin.BUILTIN,
            server_name=None,
            side_effect=SideEffect.READ,
        )
    )

    def fake_import(name):
        if name == "new_builtin_package":
            return SimpleNamespace(__path__=[])
        if name == "new_builtin_package.sample":
            return sample_module
        raise AssertionError(name)

    monkeypatch.setattr("app.tools.registry.importlib.import_module", fake_import)
    monkeypatch.setattr(
        "app.tools.registry.pkgutil.iter_modules",
        lambda paths: iter([pkgutil.ModuleInfo(None, "sample", False)]),
    )

    registry = ToolRegistry()
    discover_builtin_tools(
        registry,
        package_name="new_builtin_package",
        package_path=[],
    )

    assert registry.get("sample_tool") is not None
