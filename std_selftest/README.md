# iHelp 后端

iHelp 是一个基于 LangGraph 的电商智能客服后端，提供多轮对话、知识库
RAG、售后信息提取、工单确认、转人工、统一工具执行和 MCP 动态工具接入。

项目默认使用 mock 业务数据，适合本地开发、流程验证和评估，不直接连接真实
订单、物流、售后或客服系统。

## 目录

- [核心能力](#核心能力)
- [技术栈](#技术栈)
- [LangGraph 架构](#langgraph-架构)
- [快速开始](#快速开始)
- [接口概览](#接口概览)
- [工具系统与 MCP](#工具系统与-mcp)
- [知识库与 RAG](#知识库与-rag)
- [测试与评估](#测试与评估)
- [演示命令](#演示命令)
- [项目结构](#项目结构)
- [常见问题](#常见问题)

## 核心能力

- 多轮对话：SSE 流式响应，支持会话历史、摘要、指代消解和上下文预算控制。
- 意图分流：物流、订单、商品咨询、退款退货、售后、投诉、闲聊、转人工和其他。
- RAG 知识库：支持 Markdown、DOCX、DOC、文字型 PDF 和扫描型 PDF。
- 统一工具系统：内置工具和 MCP 工具使用同一套注册、Schema、权限和执行协议。
- 工具执行引擎：统一处理参数校验、权限、超时、重试、结果格式化、错误分诊和审计。
- 写操作确认：`create_ticket` 必须经过明确建单意图和前端确认；取消时审计为
  `permission_denied`。
- 转人工流程：复用上下文问题生成工单预览；没有问题描述时先追问；确认后回复
  “已介入客服”并总结问题。
- MCP 动态发现：物流和售后各运行一个独立 Streamable HTTP MCP Server；只重启
  Server 即可刷新工具目录，无需重启客服主服务。
- 结构化售后：售后信息提取、退款单和换货单接口。

## 技术栈

| 模块 | 技术 |
| --- | --- |
| Web API | FastAPI、Uvicorn、SSE |
| Agent / Workflow | LangChain、LangGraph、LangGraph Checkpoint SQLite |
| MCP | MCP Python SDK、`langchain-mcp-adapters` |
| 数据库 | MySQL、SQLAlchemy Async、`aiomysql` |
| 向量检索 | Milvus、混合检索、可选 Rerank |
| 文档解析 | PyMuPDF、pypdf、python-docx、RapidOCR |
| 测试与评估 | pytest、RAGAS、自定义离线评估脚本 |
| 环境管理 | Python 3.12、`uv` |

当前核心版本以 [pyproject.toml](pyproject.toml) 为准：

- `langchain==1.4.2`
- `langchain-core==1.6.5`
- `langchain-mcp-adapters==0.3.2`
- `langgraph==1.2.12`
- `langgraph-checkpoint-sqlite==3.1.1`
- `mcp==1.30.0`

## LangGraph 架构

### 主客服图

```text
START
  -> prepare_context
  -> understand_query
  -> classify_intent
  -> knowledge / business / core_after_sales / complaint / chitchat / human_transfer
  -> ticket_request_gate / order selector / retrieval / confidence gate
  -> ReAct Agent 或固定响应节点
  -> log_turn
  -> END
```

![主客服 LangGraph](graph.png)

主要 interrupt / resume 点：

- `select_order`：无订单号时展示订单选择器，前端回传 `order_selected`。
- `create_ticket`：工具执行引擎展示工单确认卡，前端回传
  `ticket_confirmed` 或 `ticket_cancelled`。
- `human_transfer_request`：复用上下文问题，或追问缺失的问题描述。

### ReAct Agent 子图

```text
START
  -> prepare_agent
  -> agent
  -> tools
  -> agent
  -> ensure_refund_option
  -> 返回主图
```

![Agent 工具调用 LangGraph](tool.png)

## 快速开始

### 1. 环境要求

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- MySQL 8
- Milvus 2.x，仅使用知识库检索时需要
- 可访问兼容 OpenAI API 的聊天、Embedding 和 Rerank 服务
- 可选：Microsoft Word 或 LibreOffice，用于旧版 `.doc` 转换

### 2. 安装依赖

```
  pyproject.toml
```



### 3. 配置 `.env`

至少需要确认以下配置：

| 配置 | 用途 |
| --- | --- |
| `CHAT_BASE_URL`、`CHAT_MODEL`、`CHAT_API_KEY` | 主聊天模型 |
| `EMBEDDING_*` | Embedding 服务 |
| `RERANK_*` | Rerank 服务 |
| `MYSQL_*` | MySQL 连接信息 |
| `MILVUS_*` | Milvus 地址和数据库 |
| `MCP_LOGISTICS_URL` | 物流 MCP 地址，默认 `http://127.0.0.1:8101/mcp` |
| `MCP_AFTER_SALES_URL` | 售后 MCP 地址，默认 `http://127.0.0.1:8102/mcp` |
| `TOOL_TIMEOUT_SECONDS`、`TOOL_MAX_RETRIES` | 工具超时和重试策略 |
| `CONTEXT_*`、`SUMMARY_*` | 上下文窗口和摘要预算 |

不要提交真实 `.env` 或 API Key。

### 4. 启动 MCP Server

物流和售后 MCP Server 需要分别启动。打开两个 PowerShell 终端：

```powershell
# 终端 A
cd D:\std_selftest
uv run python -m app.mcp_servers.logistics
```

```powershell
# 终端 B
cd D:\std_selftest
uv run python -m app.mcp_servers.after_sales
```

默认地址：

- 物流：`http://127.0.0.1:8101/mcp`
- 售后：`http://127.0.0.1:8102/mcp`

### 5. 启动 iHelp 后端

开发模式：

```powershell
cd D:\std_selftest
uv run uvicorn app.main:app --port 8000 --reload
```

单 worker 运行：

```powershell
uv run uvicorn app.main:app --port 8000 --workers 1
```

当前必须使用单 worker：LangGraph SQLite checkpointer、后台摘要任务和会话内状态
不跨进程协调。

启动后可访问：

- 后端：<http://127.0.0.1:8000>
- OpenAPI：<http://127.0.0.1:8000/docs>
- ReDoc：<http://127.0.0.1:8000/redoc>

### 6. 启动前端

前端位于独立目录，不在当前 Git 仓库内：

```powershell
cd D:\std_selftest_frontend
npm install
npm run dev
```

默认前端地址为 `http://127.0.0.1:5173/`。

## 接口概览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/api/chat` | 普通消息或 LangGraph resume，SSE 流式返回 |
| `POST` | `/api/extract` | 售后信息结构化提取 |
| `GET` | `/api/conversations?user_key=访客` | 会话列表 |
| `GET` | `/api/conversations/{id}/messages?user_key=访客` | 会话历史 |
| `DELETE` | `/api/conversations/{id}?user_key=访客` | 删除会话、消息和 checkpoint |
| `GET` | `/api/knowledge/stats` | 知识库统计 |
| `GET` | `/api/knowledge/chunks` | 知识块列表 |
| `POST` | `/api/knowledge/search` | 知识库检索 |
| `POST` | `/api/knowledge/build` | 构建或补齐知识库 |
| `POST` | `/api/knowledge/mine` | 从历史对话挖掘 QA |
| `POST` | `/api/tickets` | 前端已确认的建单入口 |
| `GET` | `/api/refunds/reasons` | 退款原因 |
| `POST` | `/api/refunds` | 提交退款单 |
| `GET` | `/api/exchanges/reasons` | 换货原因 |
| `POST` | `/api/exchanges` | 提交换货单 |

`/api/chat` 恢复中断时使用：

```json
{
  "session_id": "demo-session",
  "resume": {
    "type": "ticket_confirmed",
    "tool_call_id": "human-transfer-demo-session-42"
  }
}
```

## 工具系统与 MCP

### 统一工具协议

每个工具统一包含：

- 工具名
- 用途描述
- JSON Schema 参数定义
- 来源：`builtin` 或 `mcp`
- 只读/写副作用
- 是否要求工单确认

内置工具：

- `query_order`
- `query_product`
- `query_faq`
- `create_ticket`

MCP 工具：

- 物流 Server：`query_logistics`
- 售后 Server：`query_warranty`
- 售后 Server：`query_return_progress`

内置工具由 `app/tools/builtin/` 自动发现。MCP 工具由
`MultiServerMCPClient` 通过 Streamable HTTP 动态发现；MCP Server 重启后，服务端
下一次刷新工具目录即可看到新工具。

### 写操作和确认

`create_ticket` 是写操作，执行前必须满足：

1. 客户明确要求建工单，或进入明确的转人工流程。
2. 必填槽位完整。
3. 工具调用不与其他工具混批。
4. 前端确认 `ticket_confirmed`。

取消或未确认时，不执行 Handler，`tool_audit_logs.status` 记录为
`permission_denied`。

### 审计

`tool_audit_logs` 不挂外键，审计写入失败不会阻断业务执行。每次调用记录：

- 会话 ID 和 `tool_call_id`
- 工具名、来源和 MCP Server
- 调用参数和结果摘要
- 成功、失败、超时、校验拒绝或权限拒绝状态
- 错误说明、重试次数、耗时和时间

## 知识库与 RAG

构建知识库：

```powershell
uv run python scripts/build_knowledge_base.py
uv run python scripts/mine_qa.py
```

支持的文档：

- Markdown：`.md`
- Word：`.docx`
- 旧版 Word：`.doc`
- 文字型 PDF
- 扫描型 PDF，自动 OCR

默认 Milvus 位置：

- database：`Haohelper`
- collection：`knowledge`

主要 RAG 评估：

```powershell
uv run python RAG_test/scripts/eval_ch04.py `
  --strategies dense bm25 hybrid hybrid_rerank `
  --report-dir reports

uv run python RAG_test/scripts/eval_ch04.py `
  --strategies dense bm25 hybrid hybrid_rerank `
  --report-dir reports `
  --with-faithfulness
```

RAGAS：

```powershell
uv run python RAG_test/scripts/eval_ragas_ch04.py `
  --strategies hybrid_rerank `
  --ragas-top-k 5
```

报告位于 `reports/ch04_ragas_*` 和 `reports/ch04_eval_*`。

## 测试与评估

运行完整测试：

```powershell
uv run pytest -q
```

运行 Ch08 定向测试：

```powershell
uv run pytest -q tests/ch08
```

常用评估：

```powershell
uv run python scripts/eval_ch02.py
uv run python scripts/eval_ch03.py
uv run python scripts/eval_ch05.py
uv run python scripts/eval_ch06.py
uv run python scripts/eval_ch07.py
uv run python scripts/eval_ch08.py
```

需要调用真实模型时，先确认代理配置：

```powershell
Remove-Item Env:HTTP_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:HTTPS_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:ALL_PROXY -ErrorAction SilentlyContinue
```

## 演示命令

多轮聊天：

```powershell
bash scripts/demo_chat.sh
```

裸 Agent 与工作流：

```powershell
uv run python scripts/demo_bare_agent.py
uv run python scripts/demo_ch05.py "订单 1001 的物流到哪了" demo-1001
uv run python scripts/demo_ch06.py "订单 1001 的物流到哪了" demo-logistics
uv run python scripts/demo_ch06.py "这个能退吗" demo-refund --resume-order 1001
```

上下文管理：

```powershell
uv run python scripts/demo_ch07.py --profile cost_optimized --turns 20
uv run python scripts/demo_ch07.py --profile legacy_acceptance --turns 24 --ask-start-order
```

工具系统、MCP、工单和超时：

```powershell
uv run python scripts/demo_ch08.py --scenario registry
uv run python scripts/demo_ch08.py --scenario mcp
uv run python scripts/demo_ch08.py --scenario mcp-dynamic
uv run python scripts/demo_ch08.py --scenario ticket-confirm
uv run python scripts/demo_ch08.py --scenario ticket-cancel
uv run python scripts/demo_ch08.py --scenario timeout-read
uv run python scripts/demo_ch08.py --scenario timeout-write
uv run python scripts/demo_ch08.py --scenario all
```

## 项目结构

```text
app/
  api/                 FastAPI 路由
  context/             上下文预算、摘要、裁剪和日志
  core/                LLM、兼容工具入口
  db/                  SQLAlchemy 模型和会话
  graph/               LangGraph 主图、Agent 子图、节点和状态
  mcp_servers/         物流与售后 MCP Server
  services/            RAG、订单、理解和后台服务
  tools/
    builtin/           内置工具
    audit.py           审计
    executor.py        统一执行引擎
    mcp.py             MCP Client 和目录刷新
    models.py          工具共享协议
    registry.py        工具注册中心
scripts/               演示、建库和评估脚本
tests/                 单元、图和集成测试
RAG_test/              RAG 检索与 RAGAS 评估
docs/                  Superpowers 计划与设计
dev-notes/             分章节开发记录
```

## 常见问题

### MCP Server 未连接

确认两个 MCP 进程都在运行，并检查：

```powershell
Test-NetConnection 127.0.0.1 -Port 8101
Test-NetConnection 127.0.0.1 -Port 8102
```

同时检查 `.env` 中的 `MCP_LOGISTICS_URL` 和 `MCP_AFTER_SALES_URL`。

### 模型请求失败或超时

检查 `CHAT_API_KEY`、`CHAT_BASE_URL` 和网络。如果本机代理阻断
`api.deepseek.com` 或 `api.siliconflow.cn`，清空代理变量或加入 `NO_PROXY`。

### MySQL 连接失败

确认 MySQL 已启动、端口和密码正确。后端启动时会初始化数据库及表结构。

### Milvus 检索不可用

确认 Milvus 正在运行，数据库和 collection 已构建：

```powershell
uv run python scripts/build_knowledge_base.py
uv run python scripts/rebuild_hybrid_kb.py
```

### 工单确认卡不出现

检查：

1. `create_ticket` 是否进入统一执行器。
2. SSE 是否收到 `ticket_confirmation` 事件。
3. 前端是否回传相同的 `tool_call_id`。
4. 后端是否仍然使用单 worker 运行。

### 删除会话后工单仍保留

这是预期行为。删除会话会删除消息、摘要和 checkpoint，但 `tickets` 保留并从已删除
会话解绑。
