# iHelp Ch05 生产级客服编排设计

## 背景

当前 `/api/chat` 在单层函数中完成模型调用、工具选择、工具执行、RAG 引用提取和流式回复。它能够支撑前四章的功能演示，但确定性流程、Agent 循环、持久化和前端动作确认都耦合在一个接口中。

本章将客服系统升级为 Workflow 与 Agent 分层架构：

- Workflow 负责稳定、可审计的规则编排。
- 主力 Agent 作为一个可独立演进的节点，负责 ReAct 推理和工具调用。
- LangGraph State 贯穿完整请求。
- LangGraph checkpointer 持久化会话状态。
- 前端只接收结构化事件，并对“转人工”和“建工单”分别执行用户确认。

## 目标

1. 先提供一个不依赖 LangGraph 的裸 Agent ReAct 循环，说明其本质是“模型决策、工具执行、结果回填、再次决策”。
2. 使用 LangGraph 将同样的 Agent 循环重构为显式子图，并嵌入确定性的父工作流。
3. 父工作流固定经过指代消解、意图识别、按意图分流、知识检索、置信度闸、主力 Agent、日志记录。
4. 七类意图映射到知识类、业务数据类、投诉、闲聊四个出口。
5. 知识类在任何回答生成前完成检索与置信度判断。
6. 投诉和闲聊不调用模型。
7. Agent 支持多步 ReAct、缺少信息时追问、最大步数和 token 预算。
8. LangGraph State 与 checkpointer 支撑多轮会话恢复。
9. 前端分别模拟“转人工”，并通过工单接口真正创建工单。
10. 产出可重复的单元测试、意图评估集和功能演示命令。

## 非目标

- 不实现正式的指代消解。
- 不实现正式的意图识别提示词工程和模型调优。
- 不升级长期上下文压缩、摘要或记忆策略。
- 不接入 MCP。
- 不实现数据飞轮入库。
- 不新增业务工具。
- 不自动转人工。
- 不自动创建工单。

## 架构

### 父图

```text
START
  -> resolve_reference
  -> classify_intent
  -> route_by_intent
      -> knowledge:
           retrieve_knowledge
           -> confidence_gate
               -> agent
               -> fallback
      -> business:
           agent
      -> complaint:
           complaint_response
      -> chitchat:
           chitchat_response
  -> log_turn
  -> END
```

`resolve_reference` 本章原样透传。`classify_intent` 通过简单 prompt 输出 JSON。分流规则全部硬编码在路由函数中，不由模型选择出口。

### 四出口映射

| 意图 | 出口 | 行为 |
| --- | --- | --- |
| 商品咨询 | knowledge | 强制检索，过置信闸，再将证据和问题交给 Agent |
| 退款退货 | knowledge | 先检索政策，过置信闸，再将证据和问题交给 Agent；Agent 仍可查订单 |
| 物流 | business | 不预检索，直接进入 Agent |
| 订单 | business | 不预检索，直接进入 Agent |
| 售后 | business | 不预检索，直接进入 Agent |
| 投诉 | complaint | 不进入 Agent，返回固定安抚话术与可选动作 |
| 闲聊 | chitchat | 不进入 Agent，返回固定话术 |

### Agent 子图

```text
START
  -> agent
  -> tool_calls 非空且未超限 -> tools -> agent
  -> tool_calls 为空 -> END
  -> 达到步数或 token 预算 -> END
```

Agent 子图复用现有 Function Calling 工具：

- `query_order`
- `query_product`
- `query_logistics`

`query_faq` 继续作为知识检索能力存在，但只由父图的 `retrieve_knowledge` 节点调用，不绑定给 Agent，避免知识类请求绕过父图的强制检索与置信闸。`create_ticket` 也不绑定给 Agent。工单只允许由用户点击前端按钮后通过独立接口创建。

业务数据类请求允许 Agent 自主选择上述业务查询工具。

## State 设计

父图 State 使用 `TypedDict`，核心字段如下：

| 字段 | 用途 |
| --- | --- |
| `session_id` | 会话 ID，同时作为 checkpointer 的 `thread_id` |
| `messages` | LangGraph 持久化的对话消息 |
| `user_message` | 当前用户输入 |
| `resolved_query` | 指代消解后的查询，本章等于当前输入 |
| `intent` | 七类意图之一 |
| `route` | 四个出口之一 |
| `evidence` | 知识类检索证据 |
| `citations` | 前端可展示的引用元数据 |
| `sufficient` | 置信闸结果 |
| `reply` | 当前轮最终回复 |
| `reply_options` | 转人工、建工单等前端动作 |
| `agent_steps` | Agent ReAct 已执行步数 |
| `tool_trace` | 工具调用轨迹 |
| `route_reason` | 路由原因 |
| `error` | 可审计错误信息 |

`messages` 使用 LangGraph message reducer。每轮只向图输入新的用户消息，历史通过 checkpoint 恢复。现有 MySQL 消息表继续承担审计用途，不替代 checkpoint。

## 裸 Agent 热身

新增 `app/agents/bare_react.py`：

- `run_bare_agent(model, tools, messages, limits)`。
- 调用模型，读取 `AIMessage.tool_calls`。
- 有工具调用时逐个执行并构造 `ToolMessage`。
- 将工具结果回填后再次调用模型。
- 无工具调用时返回最终文本。
- 达到最大步数或 token 预算时停止。

新增 `scripts/demo_bare_agent.py`，打印每轮模型决策、工具调用与停止原因，作为“祛魅”演示。

生产链路不使用该 while 循环，而是使用等价的 LangGraph ReAct 子图。

## LangGraph 编排

新增以下模块：

- `app/graph/state.py`：State 定义和默认值。
- `app/graph/nodes.py`：确定性节点与路由。
- `app/graph/agent.py`：ReAct 子图。
- `app/graph/builder.py`：编译父图和子图。
- `app/graph/events.py`：节点事件、工具事件、文本事件和引用事件的统一结构。
- `app/services/turn_logger.py`：轮次轨迹和结构化日志。

图通过工厂函数构建，允许测试注入模型、工具执行器和 checkpointer。

## 意图识别

`classify_intent` 使用简单提示词，要求模型只输出：

```json
{"intent":"物流","reason":"用户询问订单物流进度"}
```

允许的意图固定为：

- 物流
- 订单
- 商品咨询
- 退款退货
- 售后
- 投诉
- 闲聊

解析失败时先运行确定性关键词兜底。仍无法判断时返回受控追问，不进入 Agent，并记录分类错误。

## 检索与置信度闸

知识类请求直接复用 `app.services.hybrid_retriever.retrieve` 和 `app.services.rag_service` 现有检索能力。

现有 `answer_knowledge_question` 同时完成召回和充足性判断。本章将其拆成两个可测试步骤：

- `retrieve_knowledge_evidence`：完成召回、章节路径补全和证据组装，供 `retrieve_knowledge` 节点调用。
- `assess_evidence`：只负责证据充足性判断和低置信问题落库，供 `confidence_gate` 节点调用。

这样日志中能明确看到两个节点分别完成。

置信闸位于检索之后、Agent 之前：

- 无召回：失败。
- rerank 后最高分或证据覆盖度不足：失败。
- 充足性模型判定为不足：失败。
- 失败时返回兜底话术，不进入 Agent。
- 失败问题写入 `low_confidence_questions`。
- 业务数据类没有检索证据，不经过该闸。

本章先使用现有 `RERANK_SCORE_THRESHOLD`、`KNOWLEDGE_SCORE_THRESHOLD` 和最小证据数做最简判断。正式置信度模型、校准和可观测性留给后续章节。

## Agent 运行控制

默认限制：

- `AGENT_MAX_STEPS=5`
- `AGENT_MAX_OUTPUT_TOKENS=800`
- `AGENT_TOKEN_BUDGET=4000`
- 历史消息继续使用 `TOKEN_BUDGET` 裁剪

每次 ReAct 迭代后更新 `agent_steps`、`tool_trace` 和近似 token 用量。达到限制时停止工具循环，返回当前已确认信息，并明确说明无法继续确认的部分。

简单业务问题应在一轮工具调用后结束。复杂问题允许多轮工具调用，例如先查订单，再根据订单工具结果查物流。

## 持久化

使用 LangGraph 官方 `AsyncSqliteSaver`：

- 默认路径：`data/langgraph_checkpoints.db`
- 配置项：`LANGGRAPH_CHECKPOINT_PATH`
- 线程键：`{"configurable": {"thread_id": session_id}}`
- SQLite 启用 WAL 与 busy timeout
- 本章按单实例、单 worker 运行
- 测试使用 `InMemorySaver`

FastAPI lifespan 初始化并持有 checkpointer，图编译一次，请求间复用。

## API 与 SSE

### `/api/chat`

请求格式保持不变：

```json
{"session_id":"s1","message":"订单 1001 的物流到哪了"}
```

响应继续使用 SSE，并支持以下事件：

```json
{"type":"node_status","node":"retrieve_knowledge","status":"running|success|skipped","detail":{}}
{"type":"tool_status","tool_name":"query_order","status":"running|success|error"}
{"type":"delta","text":"..."}
{"type":"citations","citations":[]}
{"type":"reply_options","options":[]}
```

结束帧保持：

```text
data: [DONE]
```

### `/api/tickets`

只在用户点击“建工单”时调用：

```json
{
  "session_id": "s1",
  "description": "用户投诉内容",
  "ticket_type": "投诉"
}
```

接口复用现有 `create_ticket` 工具写入 `tickets` 表。

### 转人工

“转人工”不调用后端。前端收到 `human_transfer` 选项后：

1. 用户点击按钮。
2. 按钮锁定为已完成状态。
3. 消息流追加“已转接人工客服”。
4. 显示“您好，我是客服小猫，请问有什么可以帮您的”。

未点击时，下一条消息正常进入图，不产生任何人工转接副作用。

## Agent 动作建议

投诉路径固定输出：

```json
[
  {"id":"human_transfer","label":"转人工"},
  {"id":"create_ticket","label":"建工单"}
]
```

Agent 认为需要人工介入时，在最终文本末尾使用内部标记：

```text
[[suggest:human_transfer|create_ticket]]
```

服务端过滤标记并单独发送 `reply_options`。该标记只表达建议，不触发后端动作。两个选项独立，可同时出现，也可只出现一个。

## 前端

继续修改 `D:\std_selftest_frontend` 中现有的原生 HTML、CSS 和 JavaScript 聊天页。

新增能力：

- 解析 `reply_options` SSE 事件。
- 在对应助手消息下渲染独立操作按钮。
- “转人工”执行前端模拟。
- “建工单”调用 `/api/tickets`。
- 建单过程中禁用按钮，成功后显示工单号并锁定。
- 两个按钮互不绑定。
- 用户不操作时继续聊天，状态正常流转。

## 错误处理

- 意图分类解析失败：关键词兜底，再失败则受控追问并记录错误。
- 检索异常：按证据不足处理，写低置信问题池。
- 工具异常：将结构化错误回填给 Agent。
- 模型异常：停止当前轮，发送 SSE error。
- checkpointer 异常：返回可审计错误，不假装成功。
- 达到 Agent 限制：返回当前可靠信息，不编造缺失结果。

## 测试与评估

### 裸循环测试

- 无工具直接收敛。
- 一次工具调用后收敛。
- 多步工具链。
- 工具错误回填。
- 缺参数追问。
- 最大步数停止。
- token 预算停止。

### LangGraph 节点测试

- 七类意图到四出口映射。
- 知识类必经检索和置信闸。
- 业务数据类跳过检索。
- 投诉零模型调用且不写工单。
- 闲聊零模型调用。
- Agent ReAct 至少两步工具链。
- checkpointer 多轮恢复。

### 意图评估

新增 `eval_data/ch05_intent_eval_set.json` 与 `scripts/eval_ch05.py`：

- 覆盖七类意图。
- 覆盖易混淆的退款政策与售后进度表达。
- 输出分类准确率。
- 输出四出口准确率。
- 输出混淆矩阵。

### 接口与前端测试

- `/api/chat` SSE 事件顺序和类型。
- `/api/tickets` 落库。
- 投诉选项独立渲染。
- 转人工无后端副作用。
- 建工单用户点击前无副作用。
- 前端构建通过。

## 验收标准

1. 政策问题日志出现 `retrieve_knowledge` 和 `confidence_gate`。
2. “订单 1001 的物流到哪了”由 Agent 自己调用 `query_logistics` 并回答。
3. “我要投诉”出现“转人工”“建工单”两个独立按钮；转人工只在前端模拟；建工单才写 `tickets` 表；不点击则后续消息正常。
4. 闲聊直接返回固定话术。
5. 先查订单再查物流的问题能看到至少两轮 Agent ReAct 工具步骤。

## 交付物

- 裸 Agent 循环与演示脚本。
- LangGraph 父图和 ReAct 子图。
- SQLite checkpointer 持久化。
- 意图评估集与评估脚本。
- 后端测试和 SSE 接口测试。
- 前端操作按钮。
- `dev-notes/ch05.md` 持续追加的过程记录。
- 功能演示命令、测试结果和 dev-notes 路径。
