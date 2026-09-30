# iHelp Ch08 Pluggable Tool System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fixed tool dictionary with one validated, permissioned, audited tool system shared by auto-discovered built-ins and MCP tools, then add the two Streamable HTTP business MCP servers and the ticket confirmation flow.

**Architecture:** `app/tools` owns one immutable `ToolDefinition` catalog and one `ToolExecutor`. Built-ins self-register from `app/tools/builtin`; MCP definitions are refreshed at the start of each Agent turn through `MultiServerMCPClient`. The existing LangGraph parent graph gains a ticket-intent gate and a tool-preparation node, while the ReAct subgraph binds the current catalog and sends every call through the executor. `create_ticket` pauses through `interrupt()` before any write and resumes from a trusted frontend confirmation.

**Tech Stack:** Python 3.12, FastAPI, LangChain 1.4.2, LangGraph 1.2.12, SQLAlchemy 2 async, MySQL, MCP Python SDK 1.30.0, langchain-mcp-adapters 0.3.2, jsonschema 4.26.0, pytest, Vite.

**Spec:** `docs/superpowers/specs/2026-09-25-ch08-pluggable-tool-system-design.md`

## Global Constraints

- Preserve the current uncommitted Ch07 comments in the working tree.
- Do not replace MCP Python SDK, `langchain-mcp-adapters`, LangGraph, FastAPI, SQLAlchemy, or MySQL.
- MCP Streamable HTTP is mandatory; do not use stdio or legacy SSE for the two business servers.
- MCP server annotations never grant permission. Only local Server policy is authoritative.
- `logistics` and `after_sales` are local `read_only` servers.
- Unknown MCP servers are omitted from the Agent catalog.
- Refresh MCP tools at the start of every Agent-bound user turn.
- Reject duplicate tool names; do not silently rename or shadow.
- Every model-facing JSON is serialized with `ensure_ascii=False`.
- `create_ticket` is the only local write tool and requires explicit user intent plus frontend confirmation.
- `create_ticket` must be the only tool call in its model batch.
- No ticket write or audit write may happen before `interrupt()` returns confirmation.
- Cancelled or permission-denied ticket attempts are audited as `permission_denied`.
- Audit failures never block or alter tool execution.
- Retry only read-only transient failures; write tools always use zero retries.
- Keep `query_logistics` out of built-ins; the MCP logistics server owns that name.
- Preserve the Ch05 complaint button, refund form, and exchange form behavior with a trusted-UI context.
- Frontend uses the Vibe Coding exception; verify it through build and browser interaction, not TDD.
- Re-check installed dependency signatures through Context7 before invoking MCP or LangGraph APIs in code.
- Append the current task to `dev-notes/ch08.md` before every task commit.

---

## File Structure

Create:

- `app/tools/__init__.py`
- `app/tools/models.py`
- `app/tools/registry.py`
- `app/tools/audit.py`
- `app/tools/executor.py`
- `app/tools/ticket_gate.py`
- `app/tools/mcp.py`
- `app/tools/builtin/__init__.py`
- `app/tools/builtin/query_order.py`
- `app/tools/builtin/query_product.py`
- `app/tools/builtin/query_faq.py`
- `app/tools/builtin/create_ticket.py`
- `app/mcp_servers/__init__.py`
- `app/mcp_servers/logistics.py`
- `app/mcp_servers/after_sales.py`
- `tests/ch08/__init__.py`
- `tests/ch08/test_registry.py`
- `tests/ch08/test_builtin_tools.py`
- `tests/ch08/test_audit.py`
- `tests/ch08/test_executor.py`
- `tests/ch08/test_ticket_gate.py`
- `tests/ch08/test_confirmation_graph.py`
- `tests/ch08/test_mcp_servers.py`
- `tests/ch08/test_mcp_catalog.py`
- `eval_data/ch08_ticket_gate_eval_set.json`
- `scripts/eval_ch08.py`
- `scripts/demo_ch08.py`

Modify:

- `pyproject.toml`
- `.env.example`
- `app/config.py`
- `app/core/tools.py`
- `app/graph/state.py`
- `app/graph/agent.py`
- `app/graph/nodes.py`
- `app/graph/builder.py`
- `app/api/tickets.py`
- `app/api/refunds.py`
- `app/api/exchanges.py`
- `app/main.py`
- `app/db/models.py`
- `tests/test_tools.py`
- `tests/test_graph_agent.py`
- `tests/test_graph_workflow.py`
- `tests/test_tickets_api.py`
- `tests/ch06/test_refunds_api.py`
- `tests/ch06/test_exchanges_api.py`
- `README.md`
- `dev-notes/ch08.md`

Modify outside this Git repository:

- `D:\std_selftest_frontend\src\main.js`
- `D:\std_selftest_frontend\src\styles.css`

---

### Task 1: Tool Contracts, Configuration, and Registry

**Files:**

- Modify: `pyproject.toml`
- Modify: `.env.example`
- Modify: `app/config.py`
- Create: `app/tools/__init__.py`
- Create: `app/tools/models.py`
- Create: `app/tools/registry.py`
- Create: `tests/ch08/__init__.py`
- Create: `tests/ch08/test_registry.py`
- Modify: `dev-notes/ch08.md`

**Interfaces:**

- Consumes: `BaseTool` from LangChain and the resolved MCP adapter tools.
- Produces:
  - `ToolOrigin`, `SideEffect`, `ToolCaller`
  - `TicketSlots`
  - `ToolDefinition`
  - `ToolRegistry.register`, `ToolRegistry.register_mcp_tools`,
    `ToolRegistry.get`, `ToolRegistry.list`, `ToolRegistry.catalog_version`
  - `tool_registry`

- [ ] **Step 1: Install the pinned runtime dependencies**

Run:

```powershell
uv add "langchain==1.4.2" "langchain-core==1.6.5" \
  "langchain-openai==1.6.6" "langchain-text-splitters==1.1.2" \
  "langgraph==1.2.12" "langgraph-checkpoint-sqlite==3.1.1" \
  "mcp==1.30.0" "langchain-mcp-adapters==0.3.2" \
  "jsonschema==4.26.0"
```

Expected: `uv.lock` and `pyproject.toml` include the three exact versions.

- [ ] **Step 2: Re-check the installed API signatures through Context7**

Query and record:

```text
MCP Python SDK: FastMCP, @mcp.tool(), run(transport="streamable-http")
langchain-mcp-adapters: MultiServerMCPClient.get_tools(server_name=...)
LangGraph: interrupt(value), Command(resume=value)
```

Use the installed versions `mcp==1.30.0`,
`langchain-mcp-adapters==0.3.2`, `langchain==1.4.2`, and
`langgraph==1.2.12`. If a required signature differs, stop and ask before
changing the plan.

- [ ] **Step 3: Write the failing registry tests**

Create `tests/ch08/test_registry.py`:

```python
import pytest

from app.tools.models import (
    SideEffect,
    ToolDefinition,
    ToolOrigin,
)
from app.tools.registry import ToolRegistry


async def echo_value(value: str) -> dict:
    return {"value": value}


def definition(name: str, origin: ToolOrigin = ToolOrigin.BUILTIN):
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
        origin=origin,
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
```

- [ ] **Step 4: Run the registry tests and verify they fail**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/ch08/test_registry.py -q -p no:cacheprovider
```

Expected: FAIL because `app.tools.models` and `app.tools.registry` do not
exist.

- [ ] **Step 5: Implement tool data contracts**

Create `app/tools/models.py` with:

```python
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal


class ToolOrigin(StrEnum):
    BUILTIN = "builtin"
    MCP = "mcp"


class SideEffect(StrEnum):
    READ = "read"
    WRITE = "write"


class ToolCaller(StrEnum):
    AGENT = "agent"
    TRUSTED_UI = "trusted_ui"
    SYSTEM = "system"


@dataclass(frozen=True)
class TicketSlots:
    ticket_type: str = ""
    description: str = ""
    missing_fields: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return not self.missing_fields and bool(self.description.strip())


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Awaitable[Any]]
    origin: ToolOrigin
    server_name: str | None
    side_effect: SideEffect
    requires_ticket_confirmation: bool = False
    agent_visible: bool = True


@dataclass(frozen=True)
class ToolExecutionContext:
    conversation_id: str
    tool_call_id: str
    caller: ToolCaller = ToolCaller.AGENT
    trusted_ui_confirmation: bool = False
    ticket_request_allowed: bool = False
    ticket_slots: TicketSlots | None = None
    emit: Callable[[dict[str, Any]], None] | None = None


@dataclass
class ToolExecutionResult:
    tool_name: str
    tool_args: dict[str, Any]
    result: Any = None
    success: bool = False
    error: str = ""
    status: Literal[
        "success",
        "failure",
        "timeout",
        "validation_rejected",
        "permission_denied",
        "not_found",
    ] = "failure"
    error_kind: str = ""
    retry_count: int = 0
    duration_ms: int = 0

    def model_payload(self) -> dict[str, Any]:
        if self.success:
            return {
                "ok": True,
                "status": self.status,
                "data": self.result,
            }
        return {
            "ok": False,
            "status": self.status,
            "error": {
                "category": self.error_kind or "failure",
                "message": self.error,
            },
        }
```

Create `app/tools/__init__.py` with:

```python
from app.tools.models import (
    SideEffect,
    TicketSlots,
    ToolCaller,
    ToolDefinition,
    ToolExecutionContext,
    ToolExecutionResult,
    ToolOrigin,
)

__all__ = [
    "SideEffect",
    "TicketSlots",
    "ToolCaller",
    "ToolDefinition",
    "ToolExecutionContext",
    "ToolExecutionResult",
    "ToolOrigin",
]
```

- [ ] **Step 6: Implement the registry**

Create `app/tools/registry.py` with:

```python
from __future__ import annotations

import hashlib
import json
from typing import Any

from langchain_core.tools import BaseTool

from app.tools.models import SideEffect, ToolDefinition, ToolOrigin


class ToolRegistry:
    def __init__(self) -> None:
        self._definitions: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        if definition.name in self._definitions:
            raise ValueError(f"重复工具名: {definition.name}")
        self._definitions[definition.name] = definition

    def register_mcp_tools(
        self,
        tools: list[BaseTool],
        *,
        server_name: str,
    ) -> None:
        for source in tools:
            definition = ToolDefinition(
                name=source.name,
                description=source.description or "",
                input_schema=source.get_input_schema().model_json_schema(),
                handler=_mcp_handler(source),
                origin=ToolOrigin.MCP,
                server_name=server_name,
                side_effect=SideEffect.READ,
                requires_ticket_confirmation=False,
                agent_visible=True,
            )
            self.register(definition)

    def get(self, name: str) -> ToolDefinition | None:
        return self._definitions.get(name)

    def list(self) -> list[ToolDefinition]:
        return [
            self._definitions[name]
            for name in sorted(self._definitions)
        ]

    @property
    def catalog_version(self) -> str:
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
    async def invoke(**kwargs: Any) -> Any:
        return await tool.ainvoke(kwargs)

    return invoke


tool_registry = ToolRegistry()
```

The later MCP task replaces MCP definitions atomically through a dedicated
registry method; it must never clear built-ins.

- [ ] **Step 7: Add Ch08 configuration**

Add to `app/config.py`:

```python
ticket_gate_model: str | None = None
ticket_gate_confidence_threshold: float = 0.75
mcp_transport_timeout_seconds: float = 5.0
mcp_sse_read_timeout_seconds: float = 30.0
mcp_logistics_url: str = "http://127.0.0.1:8101/mcp"
mcp_after_sales_url: str = "http://127.0.0.1:8102/mcp"
mcp_logistics_port: int = 8101
mcp_after_sales_port: int = 8102
audit_result_max_chars: int = 1000
```

Add the same keys with sample values to `.env.example`.

- [ ] **Step 8: Run the focused tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/ch08/test_registry.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 9: Append Task 1 to `dev-notes/ch08.md`**

Record the installed versions, Context7 signature checks, test count, any API
mismatch, and the decision not to replace dependencies.

- [ ] **Step 10: Commit**

```powershell
git add pyproject.toml uv.lock .env.example app/config.py app/tools tests/ch08 dev-notes/ch08.md
git commit -m "feat: add unified tool registry contracts"
```

---

### Task 2: Built-In Modules and Legacy Compatibility

**Files:**

- Create: `app/tools/builtin/__init__.py`
- Create: `app/tools/builtin/query_order.py`
- Create: `app/tools/builtin/query_product.py`
- Create: `app/tools/builtin/query_faq.py`
- Create: `app/tools/builtin/create_ticket.py`
- Modify: `app/tools/registry.py`
- Modify: `app/core/tools.py`
- Create: `tests/ch08/test_builtin_tools.py`
- Modify: `tests/test_tools.py`
- Modify: `dev-notes/ch08.md`

**Interfaces:**

- Consumes: `ToolDefinition`, `SideEffect`, `ToolRegistry`.
- Produces:
  - `discover_builtin_tools(registry) -> None`
  - one `TOOL_DEFINITION` per built-in module
  - legacy `query_order`, `query_product`, `query_faq`, and `create_ticket`
    BaseTool exports from `app.core.tools`
  - no `query_logistics` built-in

- [ ] **Step 1: Write failing discovery and observability tests**

Create `tests/ch08/test_builtin_tools.py`:

```python
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
    tmp_path,
):
    package = tmp_path / "new_builtin_package"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "sample.py").write_text(
        "from app.tools.models import SideEffect, ToolDefinition, ToolOrigin\n"
        "async def run(value: str): return {'value': value}\n"
        "TOOL_DEFINITION = ToolDefinition(\n"
        "    name='sample_tool', description='示例',\n"
        "    input_schema={'type':'object','properties':{}},\n"
        "    handler=run, origin=ToolOrigin.BUILTIN, server_name=None,\n"
        "    side_effect=SideEffect.READ,\n"
        ")\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    registry = ToolRegistry()
    discover_builtin_tools(
        registry,
        package_name="new_builtin_package",
        package_path=[str(package)],
    )

    assert registry.get("sample_tool") is not None
```

The production signature remains:

```python
def discover_builtin_tools(
    registry: ToolRegistry,
    package_name: str = "app.tools.builtin",
    package_path: list[str] | None = None,
) -> None:
    ...
```

- [ ] **Step 2: Run the discovery test and verify it fails**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/ch08/test_builtin_tools.py -q -p no:cacheprovider
```

Expected: FAIL because the built-in package and discovery function do not
exist.

- [ ] **Step 3: Implement built-in discovery**

Add to `app/tools/registry.py`:

```python
import importlib
import pkgutil


def discover_builtin_tools(
    registry: ToolRegistry,
    package_name: str = "app.tools.builtin",
    package_path: list[str] | None = None,
) -> None:
    package = importlib.import_module(package_name)
    paths = package_path or list(package.__path__)
    for module_info in pkgutil.iter_modules(paths):
        module = importlib.import_module(
            f"{package_name}.{module_info.name}"
        )
        definition = getattr(module, "TOOL_DEFINITION", None)
        if definition is not None:
            registry.register(definition)
```

- [ ] **Step 4: Implement the four built-in modules**

Each module exports one raw async handler and one `TOOL_DEFINITION`.
`query_order.py`:

```python
from app.services.order_catalog import get_demo_order
from app.tools.models import SideEffect, ToolDefinition, ToolOrigin


async def query_order(order_id: str):
    return get_demo_order(order_id)


TOOL_DEFINITION = ToolDefinition(
    name="query_order",
    description="查询订单基础信息，包括状态、金额和创建时间。",
    input_schema={
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "minLength": 1}
        },
        "required": ["order_id"],
        "additionalProperties": False,
    },
    handler=query_order,
    origin=ToolOrigin.BUILTIN,
    server_name=None,
    side_effect=SideEffect.READ,
)
```

`query_product.py` uses the existing random product response with schema
`{"product_name": {"type": "string", "minLength": 1}}`.

`query_faq.py` calls `answer_knowledge_question` and declares
`keyword`, nullable `category`, and nullable `conversation_id`. It sets
`agent_visible=False` so the deterministic parent graph remains the only
knowledge entry point.

`create_ticket.py` contains the existing ticket insert, schema requiring
`conversation_id`, non-empty `description`, and `ticket_type`; it sets:

```python
side_effect=SideEffect.WRITE
requires_ticket_confirmation=True
```

- [ ] **Step 5: Rebuild `app/core/tools.py` as a compatibility facade**

Remove the built-in `query_logistics` function. Import each module's raw
handler and build the legacy `StructuredTool` wrappers in this facade:

```python
from langchain_core.tools import StructuredTool

from app.tools.builtin.create_ticket import create_ticket
from app.tools.builtin.query_faq import query_faq
from app.tools.builtin.query_order import query_order
from app.tools.builtin.query_product import query_product


query_order = StructuredTool.from_function(
    coroutine=query_order,
    name="query_order",
    description="查询订单基础信息。",
)
```

Apply the same wrapper pattern to the other three handlers and keep:

```python
TOOLS = [
    query_order,
    query_product,
    query_faq,
    create_ticket,
]
TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}
```

The wrappers continue to return UTF-8 JSON strings so old direct-call tests
remain valid. They are compatibility objects, not the production execution
path.

- [ ] **Step 6: Update old tool tests**

Update `tests/test_tools.py` so it asserts `query_logistics` is absent from
`TOOLS_BY_NAME`, while the other four tools remain. Keep the database ticket
test but change its assertion to compare parsed ticket data rather than
requiring the old implementation module location.

- [ ] **Step 7: Run focused compatibility tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/ch08/test_builtin_tools.py tests/test_tools.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 8: Append Task 2 and commit**

```powershell
git add app/tools app/core/tools.py tests/ch08/test_builtin_tools.py tests/test_tools.py dev-notes/ch08.md
git commit -m "feat: discover built-in tools from modules"
```

---

### Task 3: Tool Audit Persistence

**Files:**

- Modify: `app/db/models.py`
- Create: `app/tools/audit.py`
- Create: `tests/ch08/test_audit.py`
- Modify: `dev-notes/ch08.md`

**Interfaces:**

- Consumes: `settings.audit_result_max_chars`.
- Produces:
  - ORM `ToolAuditLog`
  - `AuditRecorder.record(...) -> None`
  - `audit_recorder`

- [ ] **Step 1: Write failing audit tests**

Create `tests/ch08/test_audit.py` with an in-memory SQLite setup mirroring
`tests/test_tools.py`:

```python
from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from app.db import session as db_session
from app.db.base import Base
from app.db.models import ToolAuditLog
from app.tools import audit as audit_module
from app.tools.audit import AuditRecorder


async def test_audit_record_persists_without_foreign_key(monkeypatch):
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        poolclass=StaticPool,
    )
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    monkeypatch.setattr(audit_module, "AsyncSessionLocal", sessions)

    await AuditRecorder().record(
        conversation_id="missing-conversation",
        tool_call_id="call-1",
        tool_name="query_order",
        origin="builtin",
        server_name=None,
        arguments={"order_id": "1001"},
        result_summary={"ok": True},
        status="success",
        error="",
        retry_count=0,
        duration_ms=12,
    )

    async with sessions() as db:
        rows = list((await db.scalars(select(ToolAuditLog))).all())
    assert len(rows) == 1
    assert rows[0].conversation_id == "missing-conversation"
    assert rows[0].origin == "builtin"
    await engine.dispose()


async def test_audit_failure_is_swallowed(monkeypatch):
    recorder = AuditRecorder()

    async def fail(**kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(recorder, "_write", fail)

    await recorder.record(
        conversation_id="s1",
        tool_call_id="c1",
        tool_name="query_order",
        origin="builtin",
        server_name=None,
        arguments={},
        result_summary={},
        status="success",
        error="",
        retry_count=0,
        duration_ms=1,
    )
```

- [ ] **Step 2: Run the audit tests and verify they fail**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/ch08/test_audit.py -q -p no:cacheprovider
```

Expected: FAIL because the audit model and recorder do not exist.

- [ ] **Step 3: Add the ORM model**

Add to `app/db/models.py`:

```python
class ToolAuditLog(Base):
    __tablename__ = "tool_audit_logs"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    conversation_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
    )
    tool_call_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
    )
    tool_name: Mapped[str] = mapped_column(String(128), index=True)
    origin: Mapped[str] = mapped_column(String(16), index=True)
    server_name: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    arguments: Mapped[dict] = mapped_column(JSON, default=dict)
    result_summary: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(32), index=True)
    error: Mapped[str] = mapped_column(Text, default="")
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        index=True,
    )
```

Do not add `ForeignKey`.

- [ ] **Step 4: Implement the audit recorder**

Create `app/tools/audit.py`:

```python
import json
import logging
from typing import Any

from app.config import settings
from app.db.models import ToolAuditLog
from app.db.session import AsyncSessionLocal


logger = logging.getLogger(__name__)


class AuditRecorder:
    async def record(
        self,
        *,
        conversation_id: str | None,
        tool_call_id: str,
        tool_name: str,
        origin: str,
        server_name: str | None,
        arguments: dict[str, Any],
        result_summary: Any,
        status: str,
        error: str,
        retry_count: int,
        duration_ms: int,
    ) -> None:
        try:
            await self._write(
                conversation_id=conversation_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                origin=origin,
                server_name=server_name,
                arguments=arguments,
                result_summary=json.dumps(
                    result_summary,
                    ensure_ascii=False,
                    default=str,
                )[: settings.audit_result_max_chars],
                status=status,
                error=error[: settings.audit_result_max_chars],
                retry_count=retry_count,
                duration_ms=duration_ms,
            )
        except Exception:
            logger.exception(
                "工具审计落库失败 tool_name=%s tool_call_id=%s",
                tool_name,
                tool_call_id,
            )

    async def _write(self, **kwargs: Any) -> None:
        async with AsyncSessionLocal() as db:
            db.add(ToolAuditLog(**kwargs))
            await db.commit()


audit_recorder = AuditRecorder()
```

- [ ] **Step 5: Run audit tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/ch08/test_audit.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 6: Append Task 3 and commit**

```powershell
git add app/db/models.py app/tools/audit.py tests/ch08/test_audit.py dev-notes/ch08.md
git commit -m "feat: persist tool audit logs independently"
```

---

### Task 4: Unified Validation, Permission, Retry, and Result Execution

**Files:**

- Create: `app/tools/executor.py`
- Create: `tests/ch08/test_executor.py`
- Modify: `app/core/tool_runner.py`
- Modify: `dev-notes/ch08.md`

**Interfaces:**

- Consumes: `ToolRegistry`, `ToolExecutionContext`, `AuditRecorder`,
  `settings.tool_timeout_seconds`, and `settings.tool_max_retries`.
- Produces:
  - `ToolExecutor.execute(name, args, context) -> ToolExecutionResult`
  - `ToolExecutor` constructor injection for registry, audit, and interrupt
  - compatibility `execute_tool(name, args)`

- [ ] **Step 1: Write failing validation and permission tests**

Create `tests/ch08/test_executor.py`:

```python
from app.tools.executor import ToolExecutor
from app.tools.models import (
    SideEffect,
    ToolCaller,
    ToolDefinition,
    ToolExecutionContext,
    ToolOrigin,
)
from app.tools.registry import ToolRegistry


class RecordingAudit:
    def __init__(self):
        self.rows = []

    async def record(self, **kwargs):
        self.rows.append(kwargs)


def build_executor(definition, *, interrupt_fn=None):
    registry = ToolRegistry()
    registry.register(definition)
    audit = RecordingAudit()
    executor = ToolExecutor(
        registry=registry,
        audit=audit,
        interrupt_fn=interrupt_fn,
    )
    return executor, audit


async def test_schema_validation_is_returned_and_audited():
    async def handler(**kwargs):
        raise AssertionError("handler must not run")

    executor, audit = build_executor(
        ToolDefinition(
            name="query_order",
            description="查订单",
            input_schema={
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
                "additionalProperties": False,
            },
            handler=handler,
            origin=ToolOrigin.BUILTIN,
            server_name=None,
            side_effect=SideEffect.READ,
        )
    )

    result = await executor.execute(
        "query_order",
        {},
        ToolExecutionContext("s1", "c1"),
    )

    assert result.status == "validation_rejected"
    assert result.error_kind == "invalid_arguments"
    assert audit.rows[0]["status"] == "validation_rejected"


async def test_agent_cannot_confirm_create_ticket():
    async def handler(**kwargs):
        raise AssertionError("handler must not run")

    executor, audit = build_executor(
        ToolDefinition(
            name="create_ticket",
            description="建工单",
            input_schema={
                "type": "object",
                "properties": {
                    "conversation_id": {"type": "string"},
                    "description": {"type": "string"},
                    "ticket_type": {"type": "string"},
                },
                "required": [
                    "conversation_id",
                    "description",
                    "ticket_type",
                ],
            },
            handler=handler,
            origin=ToolOrigin.BUILTIN,
            server_name=None,
            side_effect=SideEffect.WRITE,
            requires_ticket_confirmation=True,
        )
    )

    result = await executor.execute(
        "create_ticket",
        {
            "conversation_id": "s1",
            "description": "重复扣款",
            "ticket_type": "投诉",
        },
        ToolExecutionContext(
            "s1",
            "c1",
            caller=ToolCaller.AGENT,
            ticket_request_allowed=False,
        ),
    )

    assert result.status == "permission_denied"
    assert audit.rows[0]["status"] == "permission_denied"
```

- [ ] **Step 2: Run executor tests and verify they fail**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/ch08/test_executor.py -q -p no:cacheprovider
```

Expected: FAIL because `app.tools.executor` does not exist.

- [ ] **Step 3: Implement schema and permission checks**

Create `app/tools/executor.py` with `Draft202012Validator`. The class accepts:

```python
def get_langgraph_interrupt():
    from langgraph.types import interrupt

    return interrupt


class ToolExecutor:
    def __init__(
        self,
        *,
        registry: ToolRegistry,
        audit: AuditRecorder,
        interrupt_fn: Callable[[Any], Any] | None = None,
    ) -> None:
        self.registry = registry
        self.audit = audit
        self.interrupt_fn = interrupt_fn or get_langgraph_interrupt()
```

Validation returns all `best_match` issues but never raises. Permission rules:

- agent calls `create_ticket` only when `ticket_request_allowed` is true,
  slots are complete, and the caller is `AGENT`;
- trusted UI calls `create_ticket` with `trusted_ui_confirmation=True` and
  bypasses the model-intent gate but not schema validation;
- unknown origin/server combinations are denied;
- writes are denied unless one of those two trusted paths applies.

- [ ] **Step 4: Add failing timeout and retry tests**

Add to `tests/ch08/test_executor.py`:

```python
import asyncio

import pytest

from app.config import settings


async def test_read_timeout_retries_then_returns_timeout():
    calls = 0

    async def slow(**kwargs):
        nonlocal calls
        calls += 1
        await asyncio.sleep(1)

    executor, audit = build_executor(
        ToolDefinition(
            name="slow_read",
            description="慢查询",
            input_schema={"type": "object", "properties": {}},
            handler=slow,
            origin=ToolOrigin.BUILTIN,
            server_name=None,
            side_effect=SideEffect.READ,
        )
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(settings, "tool_timeout_seconds", 0.01)
    monkeypatch.setattr(settings, "tool_max_retries", 2)
    try:
        result = await executor.execute(
            "slow_read",
            {},
            ToolExecutionContext("s1", "c1"),
        )
    finally:
        monkeypatch.undo()

    assert result.status == "timeout"
    assert result.retry_count == 1
    assert calls == 2
    assert audit.rows[-1]["retry_count"] == 1


async def test_write_timeout_does_not_retry():
    calls = 0

    async def slow(**kwargs):
        nonlocal calls
        calls += 1
        await asyncio.sleep(1)

    executor, audit = build_executor(
        ToolDefinition(
            name="slow_write",
            description="慢写入",
            input_schema={"type": "object", "properties": {}},
            handler=slow,
            origin=ToolOrigin.BUILTIN,
            server_name=None,
            side_effect=SideEffect.WRITE,
        )
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(settings, "tool_timeout_seconds", 0.01)
    try:
        result = await executor.execute(
            "slow_write",
            {},
            ToolExecutionContext(
                "s1",
                "c1",
                caller=ToolCaller.TRUSTED_UI,
                trusted_ui_confirmation=True,
            ),
        )
    finally:
        monkeypatch.undo()

    assert result.status == "timeout"
    assert result.retry_count == 0
    assert calls == 1
    assert audit.rows[-1]["retry_count"] == 0
```

The write test uses a trusted-UI context to reach the timeout path without
changing the production `create_ticket` gate.

- [ ] **Step 5: Implement timeout and retry**

Execute with:

```python
attempts = 0
retry_count = 0
while True:
    attempts += 1
    try:
        raw = await asyncio.wait_for(
            definition.handler(**effective_args),
            timeout=settings.tool_timeout_seconds,
        )
        return success
    except asyncio.TimeoutError:
        transient = definition.side_effect == SideEffect.READ
        ...
    except (ConnectionError, OSError) as exc:
        transient = definition.side_effect == SideEffect.READ
        ...
```

Use `tool_max_retries` as the maximum retry count. Empty business results do
not raise and therefore are not retried. Classify `KeyError` and explicit
empty/miss payloads as `not_found` without retry. Classify reset/connection
errors as `failure`; only read-only transient errors enter the retry loop.

- [ ] **Step 6: Implement result formatting**

Add:

```python
def normalize_tool_result(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, tuple):
        return [normalize_tool_result(item) for item in value]
    if isinstance(value, list):
        return [normalize_tool_result(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): normalize_tool_result(item)
            for key, item in value.items()
        }
    return value
```

Translate known status enums:

```python
STATUS_LABELS = {
    "open": "待处理",
    "processing": "处理中",
    "closed": "已关闭",
    "refund": "退款",
    "exchange": "换货",
    "repair": "维修",
}
```

Apply the mapping to result keys named `status`, `ticket_type`, or
`request_type`. Serialization happens in the graph tool-content adapter, not
inside the executor object.

If `raw is None`, `raw == {}`, or `raw == {"matched": False}`, set
`status="not_found"` while preserving the returned data in `result`.

- [ ] **Step 7: Replace `app/core/tool_runner.py` with a compatibility wrapper**

Keep `ToolExecutionResult` import compatibility:

```python
from app.tools.models import ToolExecutionResult


async def execute_tool(
    tool_name: str,
    tool_args: dict,
) -> ToolExecutionResult:
    from app.tools.executor import get_default_executor

    executor = get_default_executor()
    return await executor.execute(
        tool_name,
        tool_args,
        ToolExecutionContext(
            conversation_id=str(tool_args.get("conversation_id") or ""),
            tool_call_id=f"legacy-{tool_name}",
            caller=ToolCaller.AGENT,
        ),
    )
```

`app/tools/executor.py` owns a process-level default executor accessor:

```python
_default_executor: ToolExecutor | None = None


def configure_default_executor(executor: ToolExecutor) -> None:
    global _default_executor
    _default_executor = executor


def get_default_executor() -> ToolExecutor:
    if _default_executor is None:
        raise RuntimeError("tool executor is not configured")
    return _default_executor
```

The observable contract is that no compatibility caller can set
`trusted_ui_confirmation=True`.

- [ ] **Step 8: Run focused executor tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/ch08/test_executor.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 9: Append Task 4 and commit**

```powershell
git add app/tools/executor.py app/core/tool_runner.py tests/ch08/test_executor.py dev-notes/ch08.md
git commit -m "feat: validate and execute tools through one engine"
```

---

### Task 5: MCP Servers and Per-Turn Catalog Refresh

**Files:**

- Create: `app/mcp_servers/__init__.py`
- Create: `app/mcp_servers/logistics.py`
- Create: `app/mcp_servers/after_sales.py`
- Create: `app/tools/mcp.py`
- Modify: `app/tools/registry.py`
- Create: `tests/ch08/test_mcp_servers.py`
- Create: `tests/ch08/test_mcp_catalog.py`
- Modify: `dev-notes/ch08.md`

**Interfaces:**

- Consumes: `MultiServerMCPClient`, settings MCP URLs/timeouts.
- Produces:
  - two independent `FastMCP` processes
  - `build_mcp_client() -> MultiServerMCPClient`
  - `refresh_mcp_catalog(registry, client) -> list[dict[str, Any]]`
  - `MCP_SERVER_POLICIES`

- [ ] **Step 1: Write failing server tests**

Create `tests/ch08/test_mcp_servers.py`:

```python
from app.mcp_servers.after_sales import create_server as create_after_sales
from app.mcp_servers.logistics import create_server as create_logistics


async def test_logistics_server_exports_query_logistics():
    server = create_logistics()
    tools = await server.list_tools()
    assert [tool.name for tool in tools] == ["query_logistics"]


async def test_after_sales_server_exports_two_read_tools():
    server = create_after_sales()
    tools = await server.list_tools()
    assert {tool.name for tool in tools} == {
        "query_warranty",
        "query_return_progress",
    }
```

`FastMCP.list_tools()` is asynchronous in the installed 1.30.0 SDK. Keep the
same asserted tool names while adapting the test to the installed signature.

- [ ] **Step 2: Run server tests and verify they fail**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/ch08/test_mcp_servers.py -q -p no:cacheprovider
```

Expected: FAIL because the servers do not exist.

- [ ] **Step 3: Implement the two servers**

`app/mcp_servers/logistics.py`:

```python
import os
import random
from datetime import datetime, timedelta

from mcp.server.fastmcp import FastMCP


def create_server() -> FastMCP:
    mcp = FastMCP(
        "iHelp Logistics",
        host="127.0.0.1",
        port=int(os.getenv("MCP_LOGISTICS_PORT", "8101")),
        streamable_http_path="/mcp",
        json_response=True,
    )

    @mcp.tool()
    def query_logistics(order_id: str) -> dict:
        """根据订单号查询模拟物流轨迹。"""
        return {
            "order_id": order_id,
            "carrier": random.choice(
                ["顺丰速运", "中通快递", "圆通速递", "京东物流"]
            ),
            "status": random.choice(
                ["已揽收", "运输中", "到达派送点", "派送中", "已签收"]
            ),
            "last_update": (
                datetime.now() - timedelta(hours=random.randint(1, 48))
            ).strftime("%Y-%m-%d %H:%M:%S"),
            "latest_event": "快件已到达本地中转场，正在安排下一站运输",
        }

    return mcp


def main() -> None:
    create_server().run(transport="streamable-http")


if __name__ == "__main__":
    main()
```

`after_sales.py` follows the same structure and returns random warranty and
return-progress dictionaries from `query_warranty` and
`query_return_progress`.

- [ ] **Step 4: Write failing catalog policy tests**

Create `tests/ch08/test_mcp_catalog.py`:

```python
from langchain_core.tools import StructuredTool

from app.tools.registry import ToolRegistry


async def fake_mcp_tool(value: str) -> dict:
    return {"value": value}


async def fake_get_tools(server_name=None):
    names = {
        "logistics": ["query_logistics"],
        "after_sales": ["query_warranty"],
    }
    return [
        StructuredTool.from_function(
            coroutine=fake_mcp_tool,
            name=name,
            description="MCP 工具",
        )
        for name in names[server_name]
    ]


class FakeClient:
    get_tools = staticmethod(fake_get_tools)


async def test_mcp_catalog_marks_trusted_server_tools_read_only():
    from app.tools.mcp import refresh_mcp_catalog

    registry = ToolRegistry()
    errors = await refresh_mcp_catalog(registry, FakeClient())

    assert errors == []
    assert registry.get("query_logistics").origin == "mcp"
    assert registry.get("query_logistics").side_effect == "read"
```

- [ ] **Step 5: Implement the MCP client and refresh**

Create `app/tools/mcp.py`:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.config import settings


MCP_SERVER_POLICIES = {
    "logistics": {"mode": "read_only"},
    "after_sales": {"mode": "read_only"},
}


def build_mcp_client() -> MultiServerMCPClient:
    return MultiServerMCPClient(
        {
            "logistics": {
                "transport": "streamable_http",
                "url": settings.mcp_logistics_url,
                "timeout": settings.mcp_transport_timeout_seconds,
                "sse_read_timeout": (
                    settings.mcp_sse_read_timeout_seconds
                ),
            },
            "after_sales": {
                "transport": "streamable_http",
                "url": settings.mcp_after_sales_url,
                "timeout": settings.mcp_transport_timeout_seconds,
                "sse_read_timeout": (
                    settings.mcp_sse_read_timeout_seconds
                ),
            },
        }
    )


async def refresh_mcp_catalog(registry, client):
    errors = []
    for server_name, policy in MCP_SERVER_POLICIES.items():
        try:
            tools = await client.get_tools(server_name=server_name)
            registry.replace_mcp_server_tools(
                server_name,
                tools,
                read_only=policy["mode"] == "read_only",
            )
        except Exception as exc:
            registry.remove_mcp_server_tools(server_name)
            errors.append(
                {
                    "server_name": server_name,
                    "error": str(exc),
                }
            )
    return errors
```

Add `replace_mcp_server_tools` and `remove_mcp_server_tools` to
`ToolRegistry`. Replacement removes only definitions whose
`origin=MCP` and `server_name` matches. Duplicate names across servers raise
and leave the previous complete catalog intact.

- [ ] **Step 6: Run MCP-focused tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/ch08/test_mcp_servers.py tests/ch08/test_mcp_catalog.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 7: Start real servers and verify discovery manually**

Run in two terminals:

```powershell
.venv\Scripts\python.exe -m app.mcp_servers.logistics
.venv\Scripts\python.exe -m app.mcp_servers.after_sales
```

Run:

```powershell
.venv\Scripts\python.exe -c "import asyncio; from app.tools.mcp import build_mcp_client; async def main():\n client=build_mcp_client(); print([(t.name,t.description) for t in await client.get_tools()])\nasyncio.run(main())"
```

Expected: `query_logistics`, `query_warranty`, and
`query_return_progress` appear.

- [ ] **Step 8: Append Task 5 and commit**

```powershell
git add app/mcp_servers app/tools/mcp.py app/tools/registry.py tests/ch08/test_mcp_servers.py tests/ch08/test_mcp_catalog.py dev-notes/ch08.md
git commit -m "feat: add streamable http business mcp servers"
```

---

### Task 6: Dynamic Agent Tool Binding

**Files:**

- Modify: `app/graph/state.py`
- Modify: `app/graph/agent.py`
- Modify: `app/graph/builder.py`
- Modify: `app/main.py`
- Modify: `tests/test_graph_agent.py`
- Modify: `tests/test_graph_workflow.py`
- Modify: `dev-notes/ch08.md`

**Interfaces:**

- Consumes: `tool_registry`, `refresh_mcp_catalog`, `ToolExecutor`.
- Produces:
  - `ChatState.available_tool_names`
  - `ChatState.tool_catalog_version`
  - `ChatState.tool_catalog_errors`
  - `make_prepare_tools_node(registry, refresh_service)`
  - Agent `bind_tools()` from the current catalog instead of
    `AGENT_TOOL_NAMES`

- [ ] **Step 1: Write failing dynamic binding tests**

Add to `tests/test_graph_agent.py`:

```python
async def test_react_agent_binds_current_catalog_names():
    model = StreamingScriptedModel([AIMessageChunk(content="完成")])
    graph = build_react_agent(
        model=model,
        limits=AgentLimits(),
        available_tool_names=["query_order", "query_logistics"],
    )

    await graph.ainvoke(
        {
            "agent_messages": [HumanMessage("查物流")],
            "agent_steps": 0,
            "tool_trace": [],
            "available_tool_names": [
                "query_order",
                "query_logistics",
            ],
        }
    )

    assert [tool.name for tool in model.bound_tools] == [
        "query_order",
        "query_logistics",
    ]
```

Update the legacy assertion that always expected exactly three built-in tools.

- [ ] **Step 2: Run the dynamic binding test and verify it fails**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_graph_agent.py -q -p no:cacheprovider
```

Expected: FAIL because the Agent constructor has no catalog input.

- [ ] **Step 3: Extend graph state**

Add:

```python
available_tool_names: list[str]
tool_catalog_version: str
tool_catalog_errors: list[dict[str, Any]]
ticket_request_allowed: bool
ticket_slots: dict[str, Any]
```

- [ ] **Step 4: Bind tools from State in `call_agent`**

Delete the module-level `AGENT_TOOL_NAMES` and `AGENT_TOOLS`. Before each
model stream call:

```python
tools = registry.as_langchain_tools(
    state.get("available_tool_names") or []
)
bound_model = model.bind_tools(tools)
```

Use that bound model in the stream call. Pass the same current tool list into
`count_prompt_tokens`.

- [ ] **Step 5: Add a `prepare_tools` node**

In `app/graph/nodes.py`, add:

```python
def make_prepare_tools_node(registry, refresh_service):
    async def prepare_tools(state):
        emit_node("prepare_tools", "running")
        errors = await refresh_service(registry)
        slots = TicketSlots(**state.get("ticket_slots", {}))
        names = [
            item.name
            for item in registry.list()
            if item.agent_visible
            and (
                item.name != "create_ticket"
                or (
                    state.get("ticket_request_allowed", False)
                    and slots.complete
                )
            )
        ]
        emit_node("prepare_tools", "success", count=len(names))
        return {
            "available_tool_names": names,
            "tool_catalog_version": registry.catalog_version,
            "tool_catalog_errors": errors,
        }

    return prepare_tools
```

Only agent-bound routes pass through this node.

- [ ] **Step 6: Wire production resources in `main.py`**

In lifespan:

```python
discover_builtin_tools(tool_registry)
mcp_client = build_mcp_client()
tool_executor = ToolExecutor(
    registry=tool_registry,
    audit=audit_recorder,
)
app.state.mcp_client = mcp_client
app.state.tool_registry = tool_registry
app.state.tool_executor = tool_executor
configure_default_executor(tool_executor)
```

Pass `registry=tool_registry`,
`refresh_service=lambda registry: refresh_mcp_catalog(registry, mcp_client)`,
and `tool_executor=tool_executor` into `build_chat_graph`.

No MCP connection failure may abort application startup.

- [ ] **Step 7: Run graph and Agent tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_graph_agent.py tests/test_graph_workflow.py -q -p no:cacheprovider
```

Expected: PASS after legacy tests are updated to supply the current catalog.

- [ ] **Step 8: Append Task 6 and commit**

```powershell
git add app/graph/state.py app/graph/agent.py app/graph/nodes.py app/graph/builder.py app/main.py tests/test_graph_agent.py tests/test_graph_workflow.py dev-notes/ch08.md
git commit -m "feat: bind mcp and builtin tools dynamically"
```

---

### Task 7: Ticket Request Gate and Graph Routing

**Files:**

- Create: `app/tools/ticket_gate.py`
- Modify: `app/graph/nodes.py`
- Modify: `app/graph/builder.py`
- Create: `tests/ch08/test_ticket_gate.py`
- Modify: `dev-notes/ch08.md`

**Interfaces:**

- Consumes: `BaseChatModel`, resolved query, recent history.
- Produces:
  - `TicketGateDecision`
  - `parse_ticket_gate_response(value: str) -> TicketGateDecision`
  - `detect_ticket_request(message, history, model) -> TicketGateDecision`
  - `route_after_ticket_gate(state) -> str`

- [ ] **Step 1: Write failing gate parsing tests**

Create `tests/ch08/test_ticket_gate.py`:

```python
import pytest

from app.tools.ticket_gate import parse_ticket_gate_response


def test_gate_accepts_explicit_request_with_description():
    decision = parse_ticket_gate_response(
        '{"explicit_request":true,'
        '"ticket_type":"投诉",'
        '"description":"快递一直不到",'
        '"missing_fields":[]}'
    )

    assert decision.explicit_request is True
    assert decision.ticket_type == "投诉"
    assert decision.description == "快递一直不到"
    assert decision.missing_fields == ()


def test_gate_marks_missing_description():
    decision = parse_ticket_gate_response(
        '{"explicit_request":true,'
        '"ticket_type":"投诉",'
        '"description":"",'
        '"missing_fields":["description"]}'
    )

    assert decision.slots.missing_fields == ("description",)


def test_gate_accepts_json_code_fence():
    decision = parse_ticket_gate_response(
        '```json\n'
        '{"explicit_request":false,"ticket_type":"",'
        '"description":"","missing_fields":[]}'
        '\n```'
    )

    assert decision.explicit_request is False


def test_gate_rejects_extra_keys():
    with pytest.raises(ValueError, match="字段不合法"):
        parse_ticket_gate_response(
            '{"explicit_request":false,"ticket_type":"",'
            '"description":"","missing_fields":[],"reason":"x"}'
        )
```

The parser must reject unknown keys and invalid ticket types rather than
guessing.

- [ ] **Step 2: Run the gate parser tests and verify they fail**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/ch08/test_ticket_gate.py -q -p no:cacheprovider
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement the structured gate**

Create `app/tools/ticket_gate.py`:

```python
ALLOWED_TICKET_TYPES = ("投诉", "咨询", "售后", "其他")


@dataclass(frozen=True)
class TicketGateDecision:
    explicit_request: bool
    ticket_type: str = ""
    description: str = ""
    missing_fields: tuple[str, ...] = ()

    @property
    def slots(self) -> TicketSlots:
        return TicketSlots(
            ticket_type=self.ticket_type,
            description=self.description,
            missing_fields=self.missing_fields,
        )


def parse_ticket_gate_response(value: str) -> TicketGateDecision:
    payload = json.loads(_clean_json(value))
    required = {
        "explicit_request",
        "ticket_type",
        "description",
        "missing_fields",
    }
    if set(payload) != required:
        raise ValueError("建单判断 JSON 字段不合法")
    ...
```

The prompt must instruct the model to copy only facts from the conversation,
not infer a description beyond the user's words, and not treat a general
complaint as an explicit ticket request.

- [ ] **Step 4: Write failing graph routing tests**

Add to `tests/ch08/test_ticket_gate.py`:

```python
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from app.graph.builder import build_chat_graph


class StaticModel:
    def __init__(self, content):
        self.content = content

    async def ainvoke(self, messages, **kwargs):
        from langchain_core.messages import AIMessage
        return AIMessage(content=self.content)


async def test_explicit_ticket_request_bypasses_complaint_response():
    graph = build_chat_graph(
        intent_model=StaticModel(
            '{"intent":"投诉","confidence":0.95}'
        ),
        ticket_gate_model=StaticModel(
            '{"explicit_request":true,"ticket_type":"投诉",'
            '"description":"快递一直不到","missing_fields":[]}'
        ),
        agent_model=StaticModel("not used"),
        agent_builder=lambda model, limits: RunnableLambda(
            lambda state: {
                "reply": "准备确认",
                "agent_steps": 1,
                "ticket_request_allowed": True,
            }
        ),
        checkpointer=InMemorySaver(),
    )

    result = await graph.ainvoke(
        {
            "session_id": "ticket-1",
            "user_message": "帮我建个工单，快递一直不到",
            "messages": [HumanMessage("帮我建个工单，快递一直不到")],
        },
        {"configurable": {"thread_id": "ticket-1"}},
    )

    assert result["reply"] == "准备确认"
    assert result["ticket_request_allowed"] is True
```

- [ ] **Step 5: Add the gate node and conditional routing**

Add node `ticket_request_gate` after `classify_intent` for
`business`, `complaint`, and `other` routes. If `explicit_request` is false,
return to the original route. If true, set:

```python
{
    "ticket_request_allowed": True,
    "ticket_slots": {
        "ticket_type": decision.ticket_type,
        "description": decision.description,
        "missing_fields": decision.missing_fields,
    },
}
```

Route true results to `prepare_tools` and then `agent`. Route false
complaints to the existing `complaint_response`.

- [ ] **Step 6: Run gate and workflow tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/ch08/test_ticket_gate.py tests/test_graph_workflow.py tests/test_graph_intent.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 7: Append Task 7 and commit**

```powershell
git add app/tools/ticket_gate.py app/graph/nodes.py app/graph/builder.py tests/ch08/test_ticket_gate.py dev-notes/ch08.md
git commit -m "feat: gate ticket calls behind explicit user intent"
```

---

### Task 8: Ticket Interrupt, Resume, and Trusted UI Paths

**Files:**

- Modify: `app/tools/executor.py`
- Modify: `app/graph/agent.py`
- Modify: `app/graph/builder.py`
- Modify: `app/api/tickets.py`
- Modify: `app/api/refunds.py`
- Modify: `app/api/exchanges.py`
- Create: `tests/ch08/test_confirmation_graph.py`
- Modify: `tests/test_tickets_api.py`
- Modify: `tests/ch06/test_refunds_api.py`
- Modify: `tests/ch06/test_exchanges_api.py`
- Modify: `dev-notes/ch08.md`

**Interfaces:**

- Consumes: `interrupt`, `Command`, `ToolExecutor`, trusted UI context.
- Produces:
  - `ticket_confirmation` custom event
  - confirmation/cancellation resume handling
  - trusted UI ticket creation
  - one audit row for cancel as `permission_denied`

- [ ] **Step 1: Write the failing interrupt tests**

Create `tests/ch08/test_confirmation_graph.py` using a real
`InMemorySaver`, a scripted model that emits one `create_ticket` call, and the
production executor:

```python
async def test_confirm_writes_ticket_and_agent_returns_number(...):
    graph = build_confirmation_test_graph(...)
    config = {"configurable": {"thread_id": "confirm-1"}}

    first = await graph.ainvoke(
        {
            "session_id": "confirm-1",
            "user_message": "帮我建个工单，快递一直不到",
            "ticket_request_allowed": True,
            "ticket_slots": {
                "ticket_type": "投诉",
                "description": "快递一直不到",
                "missing_fields": [],
            },
            "available_tool_names": ["create_ticket"],
        },
        config,
    )
    assert first["__interrupt__"]
    assert ticket_count() == 0
    assert audit_count() == 0

    second = await graph.ainvoke(
        Command(
            resume={
                "type": "ticket_confirmed",
                "tool_call_id": "ticket-call-1",
            }
        ),
        config,
    )
    assert "TK" in second["reply"]
    assert ticket_count() == 1
    assert latest_audit()["status"] == "success"


async def test_cancel_does_not_write_ticket_and_audits_permission_denied(...):
    ...
    second = await graph.ainvoke(
        Command(
            resume={
                "type": "ticket_cancelled",
                "tool_call_id": "ticket-call-1",
            }
        ),
        config,
    )
    assert ticket_count() == 0
    assert latest_audit()["status"] == "permission_denied"
```

Implement helpers in the test module to create an in-memory SQLite session,
monkeypatch `Ticket` writes and `AuditRecorder`, and count rows. Assert the
first invocation audit count is zero so replay safety is explicit.

- [ ] **Step 2: Run confirmation tests and verify they fail**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/ch08/test_confirmation_graph.py -q -p no:cacheprovider
```

Expected: FAIL because `create_ticket` has no interrupt gate.

- [ ] **Step 3: Implement confirmation in `ToolExecutor`**

After validation and permission checks:

```python
preview = {
    "tool_call_id": context.tool_call_id,
    "ticket_type": slots.ticket_type,
    "description": slots.description,
}
if context.emit is not None:
    context.emit({"type": "ticket_confirmation", **preview})
decision = self.interrupt_fn(preview)
if (
    decision.get("type") != "ticket_confirmed"
    or decision.get("tool_call_id") != context.tool_call_id
):
    return permission_denied_result("用户取消或未确认建单")
```

Only after this point call the handler. Use the authoritative slots to build
the handler arguments for Agent calls:

```python
effective_args = {
    "conversation_id": context.conversation_id,
    "description": slots.description,
    "ticket_type": slots.ticket_type,
}
```

Do not trust the model's description or type when the gate has authoritative
slots.

- [ ] **Step 4: Emit the preview from the ReAct tool node**

In `app/graph/agent.py`, construct the executor context with:

```python
ToolExecutionContext(
    conversation_id=state.get("session_id", ""),
    tool_call_id=call_id,
    caller=ToolCaller.AGENT,
    ticket_request_allowed=state.get("ticket_request_allowed", False),
    ticket_slots=TicketSlots(**state.get("ticket_slots", {})),
    emit=writer,
)
```

Reject a call batch containing `create_ticket` plus any other tool with a
permission-denied ToolMessage, without calling `interrupt()`.

- [ ] **Step 5: Add trusted UI execution to the three existing endpoints**

Create a helper in `app/tools/executor.py`:

```python
def trusted_ui_context(
    conversation_id: str,
    tool_call_id: str,
) -> ToolExecutionContext:
    return ToolExecutionContext(
        conversation_id=conversation_id,
        tool_call_id=tool_call_id,
        caller=ToolCaller.TRUSTED_UI,
        trusted_ui_confirmation=True,
    )
```

Update `/api/tickets`, `/api/refunds`, and `/api/exchanges` to call the
dependency-injected or process `ToolExecutor.execute` with this context.
Their public request and response payloads do not change.

- [ ] **Step 6: Update endpoint tests**

Change the old fake `execute_tool` monkeypatches to fake
`ToolExecutor.execute` with signature:

```python
async def fake_execute(tool_name, tool_args, context):
    assert context.caller == ToolCaller.TRUSTED_UI
    assert context.trusted_ui_confirmation is True
    ...
```

Keep the same status code and JSON assertions.

- [ ] **Step 7: Run focused confirmation and API tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/ch08/test_confirmation_graph.py tests/test_tickets_api.py tests/ch06/test_refunds_api.py tests/ch06/test_exchanges_api.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 8: Append Task 8 and commit**

```powershell
git add app/tools/executor.py app/graph/agent.py app/graph/builder.py app/api tests/ch08/test_confirmation_graph.py tests/test_tickets_api.py tests/ch06/test_refunds_api.py tests/ch06/test_exchanges_api.py dev-notes/ch08.md
git commit -m "feat: require confirmation before creating agent tickets"
```

---

### Task 9: Ticket Preview Frontend

**Files:**

- Modify: `D:\std_selftest_frontend\src\main.js`
- Modify: `D:\std_selftest_frontend\src\styles.css`
- Modify: `dev-notes/ch08.md`

**Interfaces:**

- Consumes: `ticket_confirmation` SSE events and `/api/chat` resume values.
- Produces: a preview card with confirmation and cancellation buttons.

- [ ] **Step 1: Implement the SSE branch**

Add:

```javascript
} else if (payload.type === "ticket_confirmation") {
  renderTicketConfirmation(
    currentAssistantStack,
    payload,
  );
}
```

`renderTicketConfirmation` must deduplicate by `tool_call_id`, show
`ticket_type` and `description`, and append two buttons.

- [ ] **Step 2: Implement resume actions**

Confirmation sends:

```javascript
{
  session_id: elements.sessionId.value.trim() || state.sessionId,
  resume: {
    type: "ticket_confirmed",
    tool_call_id: payload.tool_call_id,
  },
}
```

Cancellation sends the same shape with `ticket_cancelled`.

Both actions lock the card immediately, call the existing chat stream helper,
and append the resumed assistant response. Network failure unlocks the buttons
and shows the existing inline chat error.

- [ ] **Step 3: Add restrained card styles**

Use the existing design tokens and an 8px radius. The card contains:

- a compact heading `工单预览`;
- a type badge;
- the full description with `overflow-wrap: anywhere`;
- a right-aligned action row with `取消` and `确认提交`.

Add a mobile rule where the action row stacks or stretches without covering
the description.

- [ ] **Step 4: Build the frontend**

Run:

```powershell
npm run build
```

Workdir: `D:\std_selftest_frontend`

Expected: Vite build succeeds.

- [ ] **Step 5: Verify with the in-app browser**

Start the backend, the two MCP servers, and the frontend. Test:

```text
帮我建个工单，快递一直不到
```

Confirm the preview card appears. Click `确认提交`; assert the assistant reply
contains a `TK...` ticket number and the card is locked. Repeat in a new
session and click `取消`; assert no ticket number appears.

Check desktop and a 390x844 viewport for overlap and horizontal overflow.

- [ ] **Step 6: Append Task 9 and commit backend-tracked files**

The frontend directory is not a Git repository. Record its file hashes in
`dev-notes/ch08.md`. Commit only the note:

```powershell
git add dev-notes/ch08.md
git commit -m "docs: verify ch08 ticket confirmation frontend"
```

---

### Task 10: Evaluation, Demo, Acceptance, and Final Verification

**Files:**

- Create: `eval_data/ch08_ticket_gate_eval_set.json`
- Create: `scripts/eval_ch08.py`
- Create: `scripts/demo_ch08.py`
- Modify: `app/mcp_servers/logistics.py`
- Modify: `README.md`
- Modify: `dev-notes/ch08.md`

- [ ] **Step 1: Create the ticket gate evaluation set**

Create `eval_data/ch08_ticket_gate_eval_set.json` with at least 20 entries.
Each entry has:

```json
{
  "text": "帮我建个工单，快递一直不到",
  "explicit_request": true,
  "ticket_type": "投诉",
  "description": "快递一直不到",
  "missing_fields": []
}
```

Cover:

- explicit ticket requests with and without descriptions;
- complaints that only ask for help;
- the Ch05 button wording `建工单`;
- follow-ups supplying a missing description;
- negations such as `先别建工单`;
- unrelated shopping questions and chit-chat.

- [ ] **Step 2: Implement the evaluation script**

`scripts/eval_ch08.py` loads the JSON, calls
`detect_ticket_request` with the configured model, and prints:

```text
explicit_request_accuracy
ticket_type_accuracy
description_exact_match_rate
missing_fields_accuracy
```

Write failures to `reports/ch08_ticket_gate_failures.json`. Fail the command
unless:

```text
explicit_request_accuracy >= 0.90
missing_fields_accuracy >= 0.85
description_exact_match_rate >= 0.80
```

- [ ] **Step 3: Implement the acceptance demo**

`scripts/demo_ch08.py` supports:

```text
--scenario registry
--scenario mcp
--scenario ticket-confirm
--scenario ticket-cancel
--scenario timeout-read
--scenario timeout-write
--scenario all
```

Behavior:

- `registry` registers one `demo_current_time` tool through
  `ToolRegistry.register` and invokes the Agent catalog path.
- `mcp` calls the two real servers and prints discovered tool names.
- `ticket-confirm` drives graph preview plus confirmation resume.
- `ticket-cancel` drives preview plus cancellation resume.
- `timeout-read` registers a test read handler that sleeps, sets a short
  timeout, and prints the timeout audit fields.
- `timeout-write` does the same with a write definition and asserts retry
  count zero.

The timeout scenarios may use an isolated in-memory audit store so they do not
pollute MySQL.

- [ ] **Step 4: Run the prompt evaluation**

Run:

```powershell
.venv\Scripts\python.exe scripts\eval_ch08.py
```

Expected: all three thresholds pass and the failures report contains only
passing rows or is absent.

- [ ] **Step 5: Run the functional demo**

Run the two MCP servers in separate terminals, then:

```powershell
.venv\Scripts\python.exe scripts\demo_ch08.py --scenario all
```

Expected: registry, MCP, ticket confirm/cancel, and both timeout scenarios
produce explicit pass lines.

- [ ] **Step 6: Perform dynamic MCP new-tool acceptance**

Start the logistics server normally and confirm the current tool list.
Restart only the logistics process with:

```powershell
$env:MCP_DEMO_EXTRA_TOOL="true"
.venv\Scripts\python.exe -m app.mcp_servers.logistics
```

Add `query_delivery_eta` behind that environment switch in the logistics
server. Keep the customer-service process running and ask for a delivery ETA.
The Agent must see and call the new tool after the next turn refresh.

Record the before/after tool lists and process IDs in `dev-notes/ch08.md`.

- [ ] **Step 7: Verify database outcomes**

Run SQL against MySQL to prove:

- one confirmed ticket exists with the expected description and type;
- the cancelled ticket flow created no new row;
- `tool_audit_logs` has a cancelled `create_ticket` row with
  `status='permission_denied'`;
- the read timeout audit has `status='timeout'` and the expected
  `retry_count`;
- the write timeout audit has `status='timeout'` and `retry_count=0`.

Use `SELECT` statements only. Record the exact statements and outputs in
`dev-notes/ch08.md`.

- [ ] **Step 8: Run full backend verification**

Run:

```powershell
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.venv\Scripts\python.exe -m compileall -q app scripts tests/ch08
git diff --check
```

Expected: all tests pass; compile and whitespace checks pass.

- [ ] **Step 9: Run the frontend build and browser acceptance**

Run:

```powershell
npm run build
```

Workdir: `D:\std_selftest_frontend`

Then repeat the desktop/mobile confirm and cancel checks from Task 9 against
the final services.

- [ ] **Step 10: Update README and finish notes**

Add:

- MCP server start commands;
- customer-service start command;
- `scripts/eval_ch08.py` and `scripts/demo_ch08.py`;
- the new audit table and tool catalog behavior;
- the ticket confirmation SSE event contract.

Append final test counts, demo outputs, prompt metrics, database evidence,
frontend hashes, and remaining risks to `dev-notes/ch08.md`.

- [ ] **Step 11: Commit final documentation**

```powershell
git add eval_data/ch08_ticket_gate_eval_set.json scripts/eval_ch08.py scripts/demo_ch08.py app/mcp_servers/logistics.py README.md dev-notes/ch08.md
git commit -m "test: verify ch08 pluggable tool system"
```

---

## Plan Self-Review

### Spec Coverage

- Unified tool table and built-in discovery: Tasks 1 and 2.
- Per-turn MCP discovery and local server trust policy: Task 5.
- Dynamic Agent binding with no core edit for new tools: Tasks 2, 5, and 6.
- JSON Schema validation and model-facing rejection results: Task 4.
- Read/write permission and explicit ticket intent: Tasks 4, 7, and 8.
- Timeout, selective retry, result normalization, and error categories:
  Task 4.
- `tool_audit_logs` with no foreign key and non-blocking failures: Task 3,
  consumed in Task 4.
- Two independent Streamable HTTP MCP servers: Task 5.
- Ticket preview, confirm, cancel, and trusted UI compatibility: Tasks 8
  and 9.
- Prompt evaluation and acceptance demos: Task 10.

### Placeholder Scan

No `TBD`, `TODO`, deferred behavior, or unspecified acceptance command
remains. Task 5 allows one exact API adaptation only for server inspection
when the installed SDK names the introspection method differently; the
required tool-name output is fixed.

### Type Consistency

- `ToolDefinition`, `ToolExecutionContext`, and `ToolExecutionResult` are
  defined once in Task 1 and reused consistently.
- `ToolRegistry.register_mcp_tools` is not exposed to production; Task 5 adds
  `replace_mcp_server_tools` and `remove_mcp_server_tools` for atomic refresh.
- `TicketSlots` is defined in Task 1 and consumed by Tasks 4, 6, 7, and 8.
- `available_tool_names` is produced by Task 6 and consumed by Task 8.
- `ticket_request_allowed` and `ticket_slots` are produced by Task 7 and
  consumed by Tasks 6 and 8.
- `trusted_ui_context` is defined in Task 8 and used by all three trusted
  HTTP endpoints.
- `refresh_mcp_catalog(registry, client)` is defined in Task 5 and wired in
  Tasks 6 and 10.

### Execution Handoff

Plan complete and saved to
`docs/superpowers/plans/2026-09-25-ch08-pluggable-tool-system.md`.
