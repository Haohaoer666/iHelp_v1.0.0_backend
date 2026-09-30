# Ch08 Pluggable Tool System Design

## Summary

Upgrade the customer-service system from a fixed built-in tool dictionary to a
single pluggable tool registry. Built-in tools and MCP tools use the same
definition, validation, permission, execution, result-formatting, and audit
contracts. Add two independent business MCP servers, a hot-refreshed MCP tool
catalog, a guarded ticket confirmation flow, and a frontend confirmation card.

The design keeps the current LangGraph workflow and MySQL business data. SQLite
continues to store only LangGraph checkpoints. No skill mechanism, repository
system, or additional external business system is introduced.

## Goals

- Register built-in tools and MCP tools in one catalog with name, description,
  JSON Schema, origin, and local permission policy.
- Make newly discovered tools available to the main Agent without changing
  core code or restarting the customer-service service.
- Validate every tool call against JSON Schema before execution.
- Return validation and permission failures to the model as tool results.
- Enforce read/write policy locally, independent of MCP server annotations.
- Route every tool through one executor with timeout, selective retry,
  normalized errors, result projection, and audit logging.
- Persist every call attempt, including validation rejection and permission
  denial, in `tool_audit_logs` without a foreign key.
- Serve logistics and after-sales mock data from two independent MCP processes
  over Streamable HTTP.
- Require explicit user intent, complete ticket slots, and LangGraph interrupt
  confirmation before an Agent-initiated ticket is written.
- Preserve the existing complaint ticket button, refund form, and exchange form
  behavior.

## Non-Goals

- No Codex Skill loader or Skill runtime.
- No warehouse, CRM, payment, or other external system integration.
- No replacement of MySQL, SQLite checkpointing, FastAPI, LangGraph, OpenAI
  compatible chat models, or the existing frontend stack.
- No multi-worker coordination or distributed tool-catalog cache.
- No automatic retry for write operations.
- No trust in MCP `readOnlyHint`, `destructiveHint`, or equivalent server
  metadata.

## Context7 Findings

Documentation was checked through Context7 MCP `@upstash/context7-mcp` v4.1.1
on 2026-09-25. The following official sources shaped this design:

- MCP Python SDK, `/modelcontextprotocol/python-sdk`: the current installable
  1.x line uses `mcp.server.fastmcp.FastMCP`, decorator-based tools, and
  `run(transport="streamable-http")`; host, port, `/mcp` path, and JSON
  response behavior are provided when constructing `FastMCP`.
- LangChain MCP Adapters, `/langchain-ai/langchain-mcp-adapters`:
  `MultiServerMCPClient` accepts `transport="streamable_http"`, `url`,
  `timeout`, `sse_read_timeout`, and exposes `await client.get_tools()`.
- LangGraph 1.0.8, `/langchain-ai/langgraph/1.0.8`: `interrupt()` requires a
  checkpointer, resumes with `Command(resume=...)`, and re-executes the node
  from its beginning. No side effect may occur before the interrupt.
- SQLAlchemy 2.0 ORM, `/websites/sqlalchemy_en_20_orm`: declarative models use
  `Mapped` and `mapped_column`, including `JSON`, `Text`, `DateTime`, and
  `server_default`.
- FastAPI, `/websites/fastapi_tiangolo`: lifespan runs resource setup and
  teardown around `yield`; SSE/streaming uses an async iterator with an await
  point so cancellation works.

Implementation must re-check the installed package versions against these APIs
before changing dependency-sensitive code. If the resolved official SDK cannot
provide the required Streamable HTTP or dynamic discovery behavior, stop and
ask instead of replacing the specified stack.

## Tool Catalog

### Unified Definition

`ToolDefinition` is the only model consumed by the Agent and executor:

```python
@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Awaitable[Any]]
    origin: Literal["builtin", "mcp"]
    server_name: str | None
    side_effect: Literal["read", "write"]
    requires_ticket_confirmation: bool
```

`origin` and `server_name` are audit metadata. They do not grant permission.
Permission comes only from local registry policy.

### Built-In Discovery

Built-in tools live under `app/tools/builtin/`. Each module registers one tool
through a module-level `TOOL_DEFINITION` or a `register(registry)` function.
`discover_builtin_tools()` imports every module in that package and registers
what it exports. Adding a built-in tool therefore requires a new module and a
service restart, but no edit to the registry, graph, or Agent.

The built-in `query_logistics` implementation is removed from this package.
The MCP logistics server exports the single `query_logistics` tool.

### MCP Discovery

`MultiServerMCPClient` is configured for:

```python
{
    "logistics": {
        "transport": "streamable_http",
        "url": "http://127.0.0.1:8101/mcp",
        "timeout": 5,
        "sse_read_timeout": 30,
    },
    "after_sales": {
        "transport": "streamable_http",
        "url": "http://127.0.0.1:8102/mcp",
        "timeout": 5,
        "sse_read_timeout": 30,
    },
}
```

At the start of every user turn that can reach the Agent, the catalog calls
`refresh_mcp()`, which invokes `await client.get_tools()` and replaces the
current MCP definitions. Built-in definitions are never replaced. A server
failure records catalog status, omits that server's tools for the turn, and
does not block built-in tools or unrelated MCP servers.

Duplicate names are rejected. The MCP part of the catalog is replaced
atomically under an async lock so concurrent sessions never observe a
half-refreshed table. `tool_catalog_version` is a deterministic hash of the
sorted registered names, descriptions, schemas, origins, and policy flags.

### Local MCP Policy

MCP server metadata is informational only. The local policy table is
authoritative:

```python
MCP_SERVER_POLICIES = {
    "logistics": {"mode": "read_only"},
    "after_sales": {"mode": "read_only"},
}
```

Every discovered tool from a `read_only` server is registered as
`side_effect="read"`. Servers absent from this table are not registered for
Agent use. A future trusted write server must be explicitly configured with
its write tools and confirmation policy. Model output cannot change this
classification.

## Graph Integration

`ChatState` adds:

```python
ticket_request_allowed: bool
ticket_slots: dict[str, Any]
tool_catalog_version: str
tool_catalog_errors: list[dict[str, Any]]
```

After intent classification, routes that could contain an explicit ticket
request pass through a dedicated `ticket_request_gate` node:

- knowledge, core after-sales, and chitchat keep their existing routes;
- business, complaint, and other routes are checked by the gate;
- gate result `explicit_request=false` keeps the original route;
- gate result `explicit_request=true` routes to the Agent and bypasses the
  fixed complaint response.

The gate returns:

```json
{
  "explicit_request": true,
  "ticket_type": "投诉|咨询|售后|其他",
  "description": "用户原文中的问题描述或空字符串",
  "missing_fields": ["description"]
}
```

The gate may copy only facts present in the conversation. It may not invent a
ticket description or infer an unstated ticket type beyond the allowed
fallback. When required fields are missing, the Agent prompt instructs it to
ask the user and the `create_ticket` tool is withheld for that turn.

Before binding tools, the Agent receives the current catalog plus the
authoritative ticket slot state. The model can choose tool calls, but it cannot
create or alter the slot state.

## Execution Context

The executor accepts an execution context constructed only by server code:

```python
@dataclass(frozen=True)
class ToolExecutionContext:
    conversation_id: str
    tool_call_id: str
    caller: Literal["agent", "trusted_ui", "system"]
    trusted_ui_confirmation: bool = False
    ticket_request_allowed: bool = False
    ticket_slots: TicketSlots | None = None
    emit: Callable[[dict[str, Any]], None] | None = None
```

`execute_tool(name, args)` remains as a compatibility wrapper but constructs an
untrusted Agent context. It cannot set `trusted_ui_confirmation`. The explicit
ticket button, refund form, and exchange form use
`caller="trusted_ui"` and `trusted_ui_confirmation=True` because an HTTP user
action already represents confirmation.

## Unified Execution

`ToolExecutor.execute()` performs these steps in order:

1. Look up the current `ToolDefinition`.
2. Normalize ticket arguments from authoritative slots when the tool requires
   ticket confirmation.
3. Validate arguments against JSON Schema Draft 2020-12.
4. Apply local read/write and caller permission rules.
5. For `create_ticket`, require an explicit request, complete slots, and a
   tool-call batch containing only this call.
6. Emit the ticket preview and wait for LangGraph confirmation when required.
7. Execute with `asyncio.wait_for`.
8. Retry only when the tool is read-only and the error is transient.
9. Project the raw result into the model-facing result contract.
10. Attempt the audit write independently of the returned result.

Any failure after discovery returns a tool result. Exceptions are not allowed
to break the Agent stream merely because validation or permission failed.

### Validation Result

The validator collects all errors into a stable shape:

```json
{
  "ok": false,
  "status": "validation_rejected",
  "error": {
    "category": "invalid_arguments",
    "message": "参数校验失败",
    "issues": [
      {"path": "order_id", "message": "必填字段缺失"}
    ]
  }
}
```

The model receives this JSON as the tool result and can ask a follow-up
question or reissue a corrected call.

### Retry Policy

- Read timeout, connection failure, connection reset, and explicit MCP
  transport failures are retryable.
- Business empty results, `not_found`, validation errors, permission denials,
  and tool business exceptions are not retryable.
- Write tools never retry automatically.
- `retry_count` counts attempts after the first. One retry produces
  `retry_count=1`; no retry produces `retry_count=0`.
- The final timeout status is `timeout`; a successful retry returns
  `success` with the actual retry count.

### Result Contract

Every model-facing tool result is serialized JSON with
`ensure_ascii=False`:

```json
{
  "ok": true,
  "status": "success",
  "data": {"ticket_id": "TK...", "status": "待处理"}
}
```

The projection layer removes transport metadata, keeps answer-relevant fields,
and translates internal enum values. Built-in tools provide a projector where
domain fields are known. MCP tools use a generic recursive projection that
drops connection/session metadata and limits strings to the configured result
budget.

## Ticket Confirmation Flow

`create_ticket` is a local write tool with:

```python
requires_ticket_confirmation=True
```

The Agent may call it only after the gate has found an explicit user request
and all required slots are complete. The current batch must contain only
`create_ticket`; a mixed batch is denied with `permission_denied`. This avoids
side effects and duplicate read calls when LangGraph replays the node after an
interrupt.

The executor emits:

```json
{
  "type": "ticket_confirmation",
  "tool_call_id": "call-xxx",
  "ticket_type": "投诉",
  "description": "问题描述"
}
```

It then calls `interrupt()` with the same preview plus a confirmation token.
The node has not written `tickets` or audit state before this point.

Resume values are validated against the pending `tool_call_id`:

```json
{"type": "ticket_confirmed", "tool_call_id": "call-xxx"}
```

```json
{"type": "ticket_cancelled", "tool_call_id": "call-xxx"}
```

On confirmation, the executor writes one `Ticket`, returns the ticket number,
and audits `success`. On cancellation or an invalid resume value, it returns
`permission_denied`, writes no ticket, and audits `permission_denied`.

The frontend deduplicates preview cards by `tool_call_id` because the
interrupted node may replay. After a decision, both buttons lock and the
resulting assistant turn reports either the ticket number or cancellation.

## Audit Persistence

Add `ToolAuditLog` in `app/db/models.py`:

```text
tool_audit_logs
---------------
id
conversation_id
tool_call_id
tool_name
origin
server_name
arguments
result_summary
status
error
retry_count
duration_ms
created_at
```

`conversation_id` and `tool_call_id` are indexed but have no foreign key.
`arguments` is JSON and stores the original call arguments. `result_summary`,
`error`, `status`, `origin`, and `server_name` are text fields.
`created_at` uses a timezone-aware server default.

The audit writer catches and logs every persistence exception. An audit failure
never changes the tool result and never prevents ticket creation.

## MCP Servers

### Logistics Server

Process entry point: `app/mcp_servers/logistics.py`.

- Transport: Streamable HTTP.
- Default URL: `http://127.0.0.1:8101/mcp`.
- Tool: `query_logistics(order_id)`.
- Response: carrier, status, last update, and latest event generated randomly
  in process.
- No MySQL connection and no table creation.

### After-Sales Server

Process entry point: `app/mcp_servers/after_sales.py`.

- Transport: Streamable HTTP.
- Default URL: `http://127.0.0.1:8102/mcp`.
- Tools: `query_warranty(order_id)` and `query_return_progress(order_id)`.
- Responses are generated random mock data in process.
- No MySQL connection and no table creation.

Both servers are independent processes. Restarting one does not restart the
customer-service service.

## Frontend

The existing frontend remains at `D:\std_selftest_frontend`.

- Parse the new `ticket_confirmation` SSE event.
- Render a preview card containing ticket type and problem description.
- Render separate `确认提交` and `取消` buttons.
- Confirmation posts to `/api/chat` with `resume` and the exact
  `ticket_confirmed` payload.
- Cancellation posts to `/api/chat` with `resume` and the exact
  `ticket_cancelled` payload.
- Lock both buttons after the decision and keep the card visible with the
  result state.
- Keep the existing Ch05 `create_ticket` reply option and its direct
  `/api/tickets` behavior unchanged.

Frontend changes follow the Vibe Coding exception: direct implementation and
browser verification, without brainstorming or TDD for the visual layer.

## Compatibility

- Existing `query_order`, `query_product`, and `query_faq` behavior is
  preserved.
- Existing `query_logistics` calls route to the MCP definition with the same
  tool name.
- Existing ticket, refund, and exchange APIs are preserved and updated to
  construct trusted UI execution contexts.
- Existing message persistence and LangGraph checkpoints are unchanged.
- Existing old tests are migrated only where they assert the removed static
  `TOOLS_BY_NAME` implementation.
- `Base.metadata.create_all` creates the new audit table. No Alembic migration
  is introduced.

## Validation Strategy

Backend implementation follows TDD:

- registry discovery, duplicate rejection, JSON Schema validation, permission
  policy, result projection, retry policy, timeout status, and audit behavior;
- `TicketRequestGate` JSON parsing and state updates;
- LangGraph interrupt, confirm, cancel, replay safety, and trusted UI bypass;
- real MCP client discovery against two separately started server processes;
- dynamic MCP tool visibility after restarting only the changed server;
- read timeout retry and write timeout non-retry audit assertions.

The ticket gate is a prompt/data component. Its validation uses a labeled
evaluation set instead of unit-testing the model's language judgment. The set
must include explicit requests, complaints that are not requests to create a
ticket, missing descriptions, follow-up descriptions, negations, and
unrelated chat.

Frontend validation uses Vite build plus real browser interaction for confirm
and cancel paths, desktop and mobile viewport checks, and no-overlap checks.

## Acceptance Mapping

1. A new built-in module registers itself during startup; the Agent sees it
   without registry or Agent code edits.
2. The logistics and after-sales servers start as separate processes; the
   Agent answers logistics and after-sales questions through MCP tools.
3. A new tool added to an MCP server becomes visible after only that server is
   restarted; the customer-service process and code remain unchanged.
4. An explicit request with no description causes a follow-up. Complete slots
   produce a preview card; confirmation writes one ticket and the final reply
   contains its ticket number.
5. Cancellation writes no ticket and produces one `permission_denied` audit
   row for `create_ticket`.
6. A forced read timeout shows retries and terminal timeout audit data. A
   forced write timeout shows `retry_count=0`.

## Risks And Controls

- MCP catalog refresh depends on the adapter creating a fresh session after a
  server restart. The integration test restarts only the server and performs a
  second discovery from the already-running customer-service client.
- LangGraph replay can duplicate interrupt-side effects. The design forbids
  mixed ticket batches and performs no ticket or audit write before resume.
- External tool schemas may be malformed. A malformed definition is rejected at
  registration and reported as a catalog error rather than bound to the Agent.
- An unavailable MCP server may increase turn latency. Connection and SSE
  read timeouts are bounded by configuration, and built-in tools remain
  available.
