from contextlib import asynccontextmanager
import logging
from pathlib import Path

import aiosqlite
from fastapi import FastAPI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.api.chat import router as chat_router
from app.api.conversations import router as conversations_router
from app.api.exchanges import router as exchanges_router
from app.api.extract import router as extract_router
from app.api.knowledge import router as knowledge_router
from app.api.refunds import router as refunds_router
from app.api.tickets import router as tickets_router
from app.config import settings
from app.context.budget import ContextBudget
from app.context.manager import ContextManager, make_text_estimator
from app.context.tasks import SummaryTaskManager
from app.core.llm import get_chat_model
from app.db import repository
from app.db.session import dispose_db, init_db
from app.graph.builder import build_chat_graph
from app.tools.audit import audit_recorder
from app.tools.executor import ToolExecutor, configure_default_executor
from app.tools.mcp import build_mcp_client, refresh_mcp_catalog
from app.tools.registry import discover_builtin_tools, tool_registry


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def _configure_context_logging() -> None:
    """
        配置上下文日志文件处理器
        作用：将日志输出写入指定日志文件；增加防重复挂载逻辑，多次调用不会重复添加handler
    """
    log_path = Path(settings.context_log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    resolved = str(log_path.resolve())
    root = logging.getLogger()
    if any(
        isinstance(handler, logging.FileHandler)
        and handler.baseFilename == resolved
        for handler in root.handlers
    ):
        return
    file_handler = logging.FileHandler(resolved, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root.addHandler(file_handler)


def _build_context_resources():
    """
        构建上下文管理所需的全套资源对象（工厂函数，一次性组装依赖）
        返回：预算对象、上下文管理器、摘要任务管理器
    """
    budget = ContextBudget.from_settings()  #  负责：模型窗口计算、分层token额度分配、历史/瞬时峰值预算核算、超限判断
    estimator = make_text_estimator(settings.chinese_tokens_per_char)   # 建文本token估算器，传入中文每字符对应的token换算系数

    # 创建后台摘要任务管理器。
    # 当 Layer 2 经过缩短后仍然超过 layer2_budget 时，
    # ContextManager 会把待摘要的完整原始消息范围交给这里
    tasks = SummaryTaskManager(     # 层2原文超限后，执行对话压缩、摘要生成任务；持久化摘要到存储层
        model=get_chat_model(streaming=False),
        store=repository,
        estimator=estimator,
        min_chars=settings.summary_target_min_chars,
        max_chars=settings.summary_target_max_chars,
        max_tokens=settings.summary_segment_max_tokens,
        chinese_tokens_per_char=settings.chinese_tokens_per_char,
    )

    # 创建每轮对话使用的上下文管理器。
    # prepare_context 节点会调用 ContextManager.prepare()
    manager = ContextManager(   # 负责分层上下文维护：层1完整对话、层2摘要；判断裁剪阈值、拼装送入LLM的上下文消息
        store=repository,
        budget=budget,
        tasks=tasks,
        chinese_tokens_per_char=settings.chinese_tokens_per_char,
        assistant_layer2_chars=settings.assistant_layer2_chars, # 助手回复存入层2摘要的字符上限
        estimator=estimator,
    )
    return budget, manager, tasks


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_context_logging()
    await init_db()
    budget, context_manager, summary_tasks = _build_context_resources()
    if not budget.can_fit_one_turn:     # 【启动自检】校验上下文预算是否至少能放下一轮对话
        logging.getLogger(__name__).critical(
            "context_budget_insufficient history_budget=%d fixed_overhead=%d",
            budget.history_budget,
            budget.fixed_overhead,
        )
    checkpoint_path = Path(settings.langgraph_checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(str(checkpoint_path)) as connection:
        await connection.execute("PRAGMA journal_mode=WAL")
        await connection.execute("PRAGMA busy_timeout=5000")
        await connection.commit()

    try:
        async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
            discover_builtin_tools(tool_registry)   # 扫描 app/tools/builtin/ 下的内置工具模块，并注册到全局 tool_registry
            mcp_client = build_mcp_client()         # 创建一个 MultiServerMCPClient，配置 logistics 和 after_sales 两个 MCP Server,这里只是创建客户端配置，不会启动 MCP Server，也还没有真正发现 MCP 工具
            tool_executor = ToolExecutor(           # 创建统一工具执行器。所有工具调用都应该经过它，而不是直接调用工具函数。
                registry=tool_registry,
                audit=audit_recorder,
            )
            configure_default_executor(tool_executor)   # 启动时组装好的“正式工具执行器”放进一个全局位置，让那些没有通过 FastAPI 依赖注入拿 Executor 的接口也能使用它。
            app.state.mcp_client = mcp_client
            app.state.tool_registry = tool_registry
            app.state.tool_executor = tool_executor
            app.state.checkpointer = saver
            # 构建LangGraph对话图实例，传入持久化checkpointer，挂载到app.state全局状态
            app.state.chat_graph = build_chat_graph(
                understanding_model=get_chat_model(
                    streaming=False,
                    model=settings.query_understanding_model,
                ),
                ticket_gate_model=get_chat_model(
                    streaming=False,
                    model=settings.ticket_gate_model,
                ),
                context_manager=context_manager,
                context_budget=budget,
                checkpointer=saver,
                tool_registry=tool_registry,
                tool_refresh_service=lambda registry: refresh_mcp_catalog(
                    registry,
                    mcp_client,
                ),
                tool_executor=tool_executor,
            )
            yield
    finally:
        await summary_tasks.shutdown()
        await dispose_db()


app = FastAPI(title="智能客服", version="1.0.0", lifespan=lifespan)
app.include_router(chat_router)
app.include_router(conversations_router)
app.include_router(extract_router)
app.include_router(knowledge_router)
app.include_router(tickets_router)
app.include_router(refunds_router)
app.include_router(exchanges_router)
