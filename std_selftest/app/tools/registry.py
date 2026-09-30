"""
    **统一工具注册表**：自动扫描加载内置工具，支持动态注册 / 原子替换 / 删除 MCP 远程工具，可生成仅用于 LLM function-calling 的 LangChain 空壳工具，
    提供工具目录哈希版本号，内置 MCP 调用包装器与 Schema 标准化提取函数，统一管理内置和 MCP 两类工具定义。
"""

from __future__ import annotations

import hashlib
import importlib
import json
import pkgutil
import asyncio
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from app.tools.models import SideEffect, ToolDefinition, ToolOrigin


class ToolRegistry:
    """
        存放不可变工具定义，使用唯一工具名作为索引表。
        管理内置工具、MCP工具的注册、查询、替换与删除；支持生成LangChain工具包装，提供目录版本哈希。
    """

    def __init__(self) -> None:
        self._definitions: dict[str, ToolDefinition] = {}   # 工具定义存储字典
        self._lock = asyncio.Lock()                         # 异步锁，用于并发修改MCP工具时保证原子操作（异步上下文下互斥）

    def register(self, definition: ToolDefinition) -> None:
        """
            注册单个工具定义（同步方法）
            如果同名工具已存在，直接抛异常禁止覆盖，保证工具名全局唯一。
        """
        if definition.name in self._definitions:
            raise ValueError(f"重复工具名: {definition.name}")
        self._definitions[definition.name] = definition

    def get(self, name: str) -> ToolDefinition | None:
        """根据工具名查询工具定义，找不到返回None"""
        return self._definitions.get(name)

    def list(self) -> list[ToolDefinition]:
        """根据工具名查询工具定义，找不到返回None"""
        return [
            self._definitions[name]
            for name in sorted(self._definitions)
        ]

    def as_langchain_tools(self, names: list[str]) -> list[BaseTool]:
        """
            根据传入的工具名列表，生成仅用于模型绑定的LangChain StructuredTool
        """
        tools = []
        for name in names:
            definition = self._definitions.get(name)
            if definition is None:
                continue

            async def not_called(**kwargs: Any) -> Any:
                raise RuntimeError(
                    "tool binding object must not execute directly"
                )

            tools.append(
                StructuredTool(     # 构造一个 LangChain 的工具描述对象，专门传给大模型，做工具绑定
                    name=definition.name,
                    description=definition.description,
                    args_schema=definition.input_schema,
                    coroutine=not_called,
                )
            )
        return tools

    def register_mcp_tools(
        self,
        tools: list[BaseTool],
        *,
        server_name: str,
    ) -> None:
        """
            注册一批MCP工具（同步）
        """
        for source in tools:
            self.register(
                ToolDefinition(
                    name=source.name,
                    description=source.description or "",
                    input_schema=_tool_input_schema(source),
                    handler=_mcp_handler(source),
                    origin=ToolOrigin.MCP,
                    server_name=server_name,
                    side_effect=SideEffect.READ,
                    requires_ticket_confirmation=False,
                    agent_visible=True,
                )
            )

    async def replace_mcp_server_tools(
        self,
        server_name: str,
        tools: list[BaseTool],
        *,
        read_only: bool,
    ) -> None:
        """
            原子替换【单个MCP服务】下的全部工具。
            场景：MCP服务重连/刷新时，先剔除该server旧工具，再新增当前拿到的工具；
        """
        # 这里保留所有：1. 内置工具2. 其他 MCP Server 的工具
        # 只排除当前 server_name 对应的旧 MCP 工具
        replacement = {
            name: definition
            for name, definition in self._definitions.items()
            if not (
                definition.origin == ToolOrigin.MCP
                and definition.server_name == server_name
            )
        }

        # 第二步：把 MCP 客户端返回的 BaseTool 转换成项目内部统一的 ToolDefinition。
        # 这样内置工具和 MCP 工具在 Agent、权限、执行器看来使用的是同一套数据结构。
        for source in tools:
            definition = ToolDefinition(
                name=source.name,
                description=source.description or "",
                input_schema=_tool_input_schema(source),
                handler=_mcp_handler(source),
                origin=ToolOrigin.MCP,
                server_name=server_name,
                side_effect=(
                    SideEffect.READ if read_only else SideEffect.WRITE
                ),
                requires_ticket_confirmation=not read_only,
                agent_visible=True,
            )
            if definition.name in replacement:      # 第三步：检查新工具名是否和保留下来的工具重名
                raise ValueError(f"重复工具名: {definition.name}")
            replacement[definition.name] = definition
        async with self._lock:
            self._definitions = replacement

    async def remove_mcp_server_tools(self, server_name: str) -> None:
        """
            删除某个 MCP Server 下的全部工具。
        """
        # 加锁后构造新的字典并一次性替换。保留条件：不是“当前 server_name 的 MCP 工具”; 因此会删除：origin == MCP且 server_name == 当前 server_name 的所有工具。
        async with self._lock:
            self._definitions = {
                name: definition
                for name, definition in self._definitions.items()
                if not (
                    definition.origin == ToolOrigin.MCP
                    and definition.server_name == server_name
                )
            }

    #用该装饰器---->只读实例属性
    @property
    def catalog_version(self) -> str:
        """
            工具目录版本哈希值（sha256）。
            把全部工具的元信息序列化（排序key，保证顺序稳定），算出哈希。
            用途：判断工具目录是否发生变更；可用于缓存、前端同步、版本校验。
            只要任意工具名称/描述/入参schema/来源等元信息变了，哈希就会变。
        """
        payload = [
            {
                "name": item.name,
                "description": item.description,
                "input_schema": item.input_schema,
                "origin": item.origin.value,
                "server_name": item.server_name,
                "side_effect": item.side_effect.value,
                "requires_ticket_confirmation": (
                    item.requires_ticket_confirmation
                ),
                "agent_visible": item.agent_visible,
            }
            for item in self.list()
        ]
        raw = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _mcp_handler(tool: BaseTool):
    """
        调用MCP BaseTool的ainvoke，把参数传给MCP工具，执行远程MCP调用
    """
    async def invoke(**kwargs: Any) -> Any:
        return await tool.ainvoke(kwargs)

    return invoke


def _tool_input_schema(tool: BaseTool) -> dict[str, Any]:
    """
        把不同来源、不同格式的 LangChain 工具参数 Schema，统一转换成标准 JSON Schema，并且要求最外层必须是 type: "object"
    """
    raw_schema = getattr(tool, "args_schema", None)
    if isinstance(raw_schema, dict) and raw_schema.get("type") == "object":
        return raw_schema
    if hasattr(raw_schema, "model_json_schema"):
        schema = raw_schema.model_json_schema()
        if schema.get("type") == "object":
            return schema
    schema = tool.get_input_schema().model_json_schema()
    if schema.get("type") == "object":
        return schema
    raise ValueError(f"工具 {tool.name} 的输入 Schema 不是 object")


tool_registry = ToolRegistry()


def discover_builtin_tools(
    registry: ToolRegistry,
    package_name: str = "app.tools.builtin",
    package_path: list[str] | None = None,
) -> None:
    """
        导入所有内置工具模块，并将模块中定义的 TOOL_DEFINITION 注册到工具注册器。
        自动扫描指定包下所有子模块，发现工具定义并完成注册。
    """
    package = importlib.import_module(package_name)
    paths = package_path or list(package.__path__)
    for module_info in pkgutil.iter_modules(paths):
        module = importlib.import_module(
            f"{package_name}.{module_info.name}"
        )
        definition = getattr(module, "TOOL_DEFINITION", None)
        if (
            definition is not None
            and registry.get(definition.name) is None
        ):
            registry.register(definition)


discover_builtin_tools(tool_registry)
