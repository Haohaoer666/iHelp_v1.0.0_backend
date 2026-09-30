"""Run deterministic Ch08 registry, MCP, ticket, and timeout demos."""

import argparse
import asyncio
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from app.config import settings
from app.db.base import Base
from app.db.models import Ticket
from app.tools import audit as audit_module
from app.tools.audit import AuditRecorder
from app.tools.builtin import create_ticket as ticket_module
from app.tools.builtin.create_ticket import TOOL_DEFINITION
from app.tools.executor import ToolExecutor, trusted_ui_context
from app.tools.mcp import build_mcp_client, refresh_mcp_catalog
from app.tools.models import (
    SideEffect,
    TicketSlots,
    ToolCaller,
    ToolDefinition,
    ToolExecutionContext,
    ToolOrigin,
)
from app.tools.registry import ToolRegistry, discover_builtin_tools


class MemoryAudit:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    async def record(self, **kwargs: Any) -> None:
        self.rows.append(kwargs)


def run_registry_demo() -> None:
    registry = ToolRegistry()
    discover_builtin_tools(registry)

    async def current_time() -> dict:
        return {"current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    registry.register(
        ToolDefinition(
            name="demo_current_time",
            description="返回当前时间，用于演示无核心代码改动的注册。",
            input_schema={"type": "object", "properties": {}},
            handler=current_time,
            origin=ToolOrigin.BUILTIN,
            server_name=None,
            side_effect=SideEffect.READ,
        )
    )
    print("registry_tools=" + ",".join(item.name for item in registry.list()))
    print("registered_without_registry_edit=demo_current_time")


def wait_for_port(port: int, timeout: float = 8.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"MCP server did not listen on port {port}")


def start_mcp_server(module: str, env: dict[str, str]) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", module],
        cwd=ROOT,
        env=env,
    )


async def run_mcp_demo(*, extra_tool: bool = False) -> None:
    env = os.environ.copy()
    env["MCP_DEMO_EXTRA_TOOL"] = "true" if extra_tool else "false"
    processes = [
        start_mcp_server("app.mcp_servers.logistics", env),
        start_mcp_server("app.mcp_servers.after_sales", env),
    ]
    try:
        wait_for_port(settings.mcp_logistics_port)
        wait_for_port(settings.mcp_after_sales_port)
        registry = ToolRegistry()
        errors = await refresh_mcp_catalog(registry, build_mcp_client())
        print("mcp_errors=" + repr(errors))
        print("mcp_tools=" + ",".join(item.name for item in registry.list()))
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


async def run_dynamic_mcp_demo() -> None:
    base_env = os.environ.copy()
    base_env["MCP_DEMO_EXTRA_TOOL"] = "false"
    logistics = start_mcp_server("app.mcp_servers.logistics", base_env)
    after_sales = start_mcp_server("app.mcp_servers.after_sales", base_env)
    processes = [logistics, after_sales]
    try:
        wait_for_port(settings.mcp_logistics_port)
        wait_for_port(settings.mcp_after_sales_port)
        registry = ToolRegistry()
        client = build_mcp_client()
        await refresh_mcp_catalog(registry, client)
        before = {item.name for item in registry.list()}
        print("mcp_before=" + ",".join(sorted(before)))

        logistics.terminate()
        logistics.wait(timeout=5)
        processes.remove(logistics)
        enabled_env = os.environ.copy()
        enabled_env["MCP_DEMO_EXTRA_TOOL"] = "true"
        logistics = start_mcp_server(
            "app.mcp_servers.logistics",
            enabled_env,
        )
        processes.append(logistics)
        wait_for_port(settings.mcp_logistics_port)

        await refresh_mcp_catalog(registry, client)
        after = {item.name for item in registry.list()}
        print("mcp_after=" + ",".join(sorted(after)))
        print(
            "new_tool_visible="
            + str("query_delivery_eta" in after).lower()
        )
        if "query_delivery_eta" not in after:
            raise RuntimeError("dynamic MCP tool was not discovered")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


async def run_ticket_demo(cancel: bool) -> None:
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
    ticket_module.AsyncSessionLocal = sessions
    audit = MemoryAudit()
    registry = ToolRegistry()
    registry.register(TOOL_DEFINITION)
    decision_type = "ticket_cancelled" if cancel else "ticket_confirmed"
    executor = ToolExecutor(
        registry=registry,
        audit=audit,
        interrupt_fn=lambda preview: {
            "type": decision_type,
            "tool_call_id": preview["tool_call_id"],
        },
    )
    result = await executor.execute(
        "create_ticket",
        {
            "conversation_id": "demo-ticket",
            "description": "快递一直不到",
            "ticket_type": "投诉",
        },
        ToolExecutionContext(
            conversation_id="demo-ticket",
            tool_call_id="demo-ticket-call",
            caller=ToolCaller.AGENT,
            ticket_request_allowed=True,
            ticket_slots=TicketSlots(
                ticket_type="投诉",
                description="快递一直不到",
            ),
        ),
    )
    async with sessions() as db:
        count = await db.scalar(select(func.count()).select_from(Ticket))
    print(f"ticket_status={result.status}")
    print(f"ticket_count={count}")
    print(f"audit_status={audit.rows[-1]['status']}")
    await engine.dispose()


async def run_timeout_demo(*, write: bool) -> None:
    async def slow(**kwargs: Any) -> dict:
        await asyncio.sleep(1)
        return {"ok": True}

    audit = MemoryAudit()
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="slow_write" if write else "slow_read",
            description="超时演示",
            input_schema={"type": "object", "properties": {}},
            handler=slow,
            origin=ToolOrigin.BUILTIN,
            server_name=None,
            side_effect=SideEffect.WRITE if write else SideEffect.READ,
        )
    )
    executor = ToolExecutor(
        registry=registry,
        audit=audit,
        interrupt_fn=lambda value: value,
    )
    original_timeout = settings.tool_timeout_seconds
    original_attempts = settings.tool_max_retries
    settings.tool_timeout_seconds = 0.01
    settings.tool_max_retries = 2
    try:
        context = (
            trusted_ui_context("demo-timeout", "demo-timeout-call")
            if write
            else ToolExecutionContext("demo-timeout", "demo-timeout-call")
        )
        result = await executor.execute(
            "slow_write" if write else "slow_read",
            {},
            context,
        )
    finally:
        settings.tool_timeout_seconds = original_timeout
        settings.tool_max_retries = original_attempts
    print(f"timeout_status={result.status}")
    print(f"timeout_retry_count={result.retry_count}")
    print(f"timeout_duration_ms={result.duration_ms}")
    print(f"audit_status={audit.rows[-1]['status']}")


async def run_all() -> None:
    run_registry_demo()
    await run_mcp_demo()
    await run_ticket_demo(cancel=False)
    await run_ticket_demo(cancel=True)
    await run_timeout_demo(write=False)
    await run_timeout_demo(write=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=[
            "registry",
            "mcp",
            "mcp-dynamic",
            "ticket-confirm",
            "ticket-cancel",
            "timeout-read",
            "timeout-write",
            "all",
        ],
        default="all",
    )
    parser.add_argument(
        "--extra-mcp-tool",
        action="store_true",
        help="Restart the logistics server with query_delivery_eta enabled.",
    )
    args = parser.parse_args()
    if args.scenario == "registry":
        run_registry_demo()
    elif args.scenario == "mcp":
        asyncio.run(run_mcp_demo(extra_tool=args.extra_mcp_tool))
    elif args.scenario == "mcp-dynamic":
        asyncio.run(run_dynamic_mcp_demo())
    elif args.scenario == "ticket-confirm":
        asyncio.run(run_ticket_demo(cancel=False))
    elif args.scenario == "ticket-cancel":
        asyncio.run(run_ticket_demo(cancel=True))
    elif args.scenario == "timeout-read":
        asyncio.run(run_timeout_demo(write=False))
    elif args.scenario == "timeout-write":
        asyncio.run(run_timeout_demo(write=True))
    else:
        asyncio.run(run_all())


if __name__ == "__main__":
    main()
