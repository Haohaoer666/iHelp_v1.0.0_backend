# iHelp Ch05 Production Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the chat path into a deterministic LangGraph workflow with a persisted ReAct Agent node, independent human-transfer and ticket actions, and a runnable bare-loop comparison.

**Architecture:** Build a parent `StateGraph` for resolution, intent routing, knowledge retrieval, confidence gating, fixed branches, Agent execution, and turn logging. Build an explicit ReAct subgraph for the Agent. Compile both with the LangGraph `AsyncSqliteSaver` in production and `InMemorySaver` in tests.

**Tech Stack:** Python 3.12, FastAPI, LangGraph, LangChain Core, LangChain OpenAI, Async SQLite, SQLAlchemy, pytest, native HTML/CSS/JavaScript with Vite.

**Spec:** `docs/superpowers/specs/2026-09-18-ch05-production-workflow-design.md`

## Global Constraints

- Use LangGraph for graph orchestration, State, and the built-in checkpointer.
- Use `AsyncSqliteSaver` with `data/langgraph_checkpoints.db` by default and `session_id` as `thread_id`.
- Reuse existing `query_order`, `query_product`, `query_logistics`, `query_faq`, and `create_ticket`.
- Do not add new business tools.
- Do not bind `query_faq` or `create_ticket` to the Agent.
- Do not auto-transfer to a human.
- Do not auto-create a ticket.
- Knowledge intents must run retrieval and confidence gating before the Agent.
- Business-data intents skip retrieval and enter the Agent directly.
- Complaint and chitchat branches must not call the model.
- Keep existing MySQL message auditing.
- Run on a single process and single Uvicorn worker while using the SQLite checkpointer.
- Modify the existing frontend at `D:\std_selftest_frontend`.
- Use Context7 before changing library-specific setup or API calls.
- Preserve unrelated working-tree changes.

---

### Task 1: Bare ReAct Loop

**Files:**
- Create: `app/agents/__init__.py`
- Create: `app/agents/bare_react.py`
- Create: `scripts/demo_bare_agent.py`
- Test: `tests/test_bare_react.py`

**Interfaces:**
- Consumes: `BaseChatModel`, `BaseTool`, `BaseMessage`, `AIMessage`, `ToolMessage`, `execute_tool`, `ToolExecutionResult`.
- Produces: `AgentLimits`, `AgentRunResult`, `ToolRunner`, `EventCallback`, and `async def run_bare_agent(...) -> AgentRunResult`.

- [ ] **Step 1: Write failing tests for convergence, tool loops, and limits**

```python
import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from app.agents.bare_react import AgentLimits, run_bare_agent
from app.core.tool_runner import ToolExecutionResult


class ScriptedModel(BaseChatModel):
    responses: list[AIMessage]

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    async def ainvoke(self, messages, **kwargs):
        return self.responses.pop(0)


def fake_tool(name: str):
    async def call(tool_name, tool_args):
        return ToolExecutionResult(
            tool_name=tool_name,
            tool_args=tool_args,
            result=json.dumps({"tool": tool_name, "args": tool_args}),
            success=True,
        )
    return call


async def test_bare_agent_converges_without_tools():
    model = ScriptedModel(responses=[AIMessage(content="直接回答")])
    result = await run_bare_agent(
        model=model,
        tools=[],
        messages=[HumanMessage("你好")],
        limits=AgentLimits(),
    )

    assert result.reply == "直接回答"
    assert result.steps == 1
    assert result.stop_reason == "completed"


async def test_bare_agent_executes_multiple_tool_steps():
    model = ScriptedModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "query_logistics",
                        "args": {"order_id": "1001"},
                        "id": "c2",
                    }
                ],
            ),
            AIMessage(content="订单已发货，物流运输中。"),
        ]
    )
    result = await run_bare_agent(
        model=model,
        tools=[],
        messages=[HumanMessage("先查订单再查物流")],
        limits=AgentLimits(max_steps=5),
        tool_runner=fake_tool("unused"),
    )

    assert result.reply == "订单已发货，物流运输中。"
    assert [item["name"] for item in result.tool_trace] == [
        "query_order",
        "query_logistics",
    ]
    assert result.steps == 3


async def test_bare_agent_stops_at_max_steps():
    model = ScriptedModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}
                ],
            )
        ]
    )
    result = await run_bare_agent(
        model=model,
        tools=[],
        messages=[HumanMessage("查询订单")],
        limits=AgentLimits(max_steps=1),
        tool_runner=fake_tool("unused"),
    )

    assert result.stop_reason == "max_steps"
    assert result.steps == 1
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `uv run pytest tests/test_bare_react.py -v`

Expected: FAIL because `app.agents.bare_react` does not exist.

- [ ] **Step 3: Implement the minimal bare loop**

```python
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import BaseTool

from app.core.tool_runner import ToolExecutionResult, execute_tool


ToolRunner = Callable[[str, dict[str, Any]], Awaitable[ToolExecutionResult]]
EventCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class AgentLimits:
    max_steps: int = 5
    max_output_tokens: int = 800
    token_budget: int = 4000


@dataclass
class AgentRunResult:
    reply: str
    steps: int
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = "completed"
    token_usage: int = 0


def _text_content(message: AIMessage) -> str:
    return message.content if isinstance(message.content, str) else str(message.content)


async def run_bare_agent(
    model: BaseChatModel,
    tools: list[BaseTool],
    messages: list[BaseMessage],
    limits: AgentLimits,
    tool_runner: ToolRunner = execute_tool,
    on_event: EventCallback | None = None,
) -> AgentRunResult:
    bound_model = model.bind_tools(tools) if tools else model
    working = list(messages)
    tool_trace: list[dict[str, Any]] = []
    steps = 0

    while steps < limits.max_steps:
        if count_tokens_approximately(working) >= limits.token_budget:
            return AgentRunResult(
                reply="当前信息较多，暂时无法继续查询，请补充最关键的信息。",
                steps=steps,
                tool_trace=tool_trace,
                stop_reason="token_budget",
                token_usage=count_tokens_approximately(working),
            )

        ai = await bound_model.ainvoke(
            working,
            max_tokens=limits.max_output_tokens,
        )
        if not isinstance(ai, AIMessage):
            raise TypeError("Agent model must return AIMessage")
        working.append(ai)
        steps += 1

        if on_event:
            on_event({"type": "agent_step", "step": steps, "tool_calls": ai.tool_calls})

        if not ai.tool_calls:
            reply = _text_content(ai)
            return AgentRunResult(
                reply=reply,
                steps=steps,
                tool_trace=tool_trace,
                token_usage=count_tokens_approximately(working),
            )

        for call in ai.tool_calls:
            execution = await tool_runner(call["name"], call.get("args") or {})
            content = (
                execution.result
                if isinstance(execution.result, str)
                else json.dumps(execution.result, ensure_ascii=False)
            )
            if not execution.success:
                content = json.dumps({"error": execution.error}, ensure_ascii=False)
            tool_trace.append(
                {
                    "name": call["name"],
                    "args": call.get("args") or {},
                    "call_id": call.get("id") or f"call-{steps}",
                    "success": execution.success,
                    "content": content,
                }
            )
            working.append(
                ToolMessage(
                    content=content,
                    tool_call_id=call.get("id") or f"call-{steps}",
                    name=call["name"],
                )
            )

    return AgentRunResult(
        reply="我已经查询到部分信息，但还需要更多信息才能继续确认。请补充订单号或具体问题。",
        steps=steps,
        tool_trace=tool_trace,
        stop_reason="max_steps",
        token_usage=count_tokens_approximately(working),
    )
```

The implementation must import `json` at module top. Keep event payloads JSON-serializable.

- [ ] **Step 4: Run the tests and verify they pass**

Run: `uv run pytest tests/test_bare_react.py -v`

Expected: PASS.

- [ ] **Step 5: Add the runnable comparison script**

```python
import asyncio

from langchain_core.messages import HumanMessage

from app.agents.bare_react import AgentLimits, run_bare_agent
from app.core.llm import get_chat_model
from app.core.tools import TOOLS_BY_NAME


async def main() -> None:
    model = get_chat_model(streaming=False)
    result = await run_bare_agent(
        model=model,
        tools=[TOOLS_BY_NAME["query_order"], TOOLS_BY_NAME["query_logistics"]],
        messages=[HumanMessage("先查订单 1001 的状态，再查物流")],
        limits=AgentLimits(),
        on_event=print,
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 6: Commit**

```bash
git add app/agents scripts/demo_bare_agent.py tests/test_bare_react.py
git commit -m "feat: add bare react agent loop"
```

### Task 2: State, Intent Routing, and Evaluation

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `app/graph/__init__.py`
- Create: `app/graph/state.py`
- Create: `app/graph/intent.py`
- Create: `eval_data/ch05_intent_eval_set.json`
- Create: `scripts/eval_ch05.py`
- Test: `tests/test_graph_intent.py`

**Interfaces:**
- Consumes: `BaseChatModel`, `BaseMessage`, `HumanMessage`, `add_messages`.
- Produces: `ChatState`, `IntentDecision`, `INTENTS`, `INTENT_TO_ROUTE`, `parse_intent_response`, `fallback_intent`, `classify_intent_text`, `classify_intent_node`.

- [ ] **Step 1: Add LangGraph dependencies**

Run: `uv add langgraph langgraph-checkpoint-sqlite`

Expected: `pyproject.toml` and `uv.lock` include the new packages.

- [ ] **Step 2: Write failing state, parse, fallback, and routing tests**

```python
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from app.graph.intent import (
    INTENT_TO_ROUTE,
    classify_intent_text,
    fallback_intent,
    parse_intent_response,
)
from app.graph.state import ChatState


class StaticModel(BaseChatModel):
    content: str

    @property
    def _llm_type(self) -> str:
        return "static"

    async def ainvoke(self, messages, **kwargs):
        return AIMessage(content=self.content)


def test_seven_intents_map_to_four_routes():
    assert INTENT_TO_ROUTE == {
        "商品咨询": "knowledge",
        "退款退货": "knowledge",
        "物流": "business",
        "订单": "business",
        "售后": "business",
        "投诉": "complaint",
        "闲聊": "chitchat",
    }


def test_parse_intent_response_accepts_code_fence():
    decision = parse_intent_response(
        '```json\n{"intent":"物流","reason":"询问物流"}\n```'
    )
    assert decision.intent == "物流"
    assert decision.route == "business"


def test_fallback_intent_uses_keywords():
    assert fallback_intent("订单 1001 的物流到哪了").intent == "物流"
    assert fallback_intent("我要投诉快递员").intent == "投诉"


async def test_classify_intent_text_uses_model_json():
    decision = await classify_intent_text(
        "退货政策是什么",
        StaticModel(content='{"intent":"退款退货","reason":"政策咨询"}'),
    )
    assert decision.intent == "退款退货"
    assert decision.route == "knowledge"


def test_chat_state_has_required_fields():
    annotations = ChatState.__annotations__
    for field in [
        "messages",
        "user_message",
        "resolved_query",
        "intent",
        "route",
        "evidence",
        "citations",
        "sufficient",
        "reply",
        "reply_options",
        "agent_steps",
        "tool_trace",
        "route_reason",
        "error",
    ]:
        assert field in annotations
```

- [ ] **Step 3: Run the tests and verify they fail**

Run: `uv run pytest tests/test_graph_intent.py -v`

Expected: FAIL because the graph modules do not exist.

- [ ] **Step 4: Implement State and intent classification**

`app/graph/state.py` must define:

```python
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ChatState(TypedDict, total=False):
    session_id: str
    messages: Annotated[list[BaseMessage], add_messages]
    agent_messages: Annotated[list[BaseMessage], add_messages]
    user_message: str
    resolved_query: str
    intent: str
    route: str
    evidence: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    sufficient: bool
    refusal: str
    reply: str
    reply_options: list[dict[str, str]]
    agent_steps: int
    tool_trace: list[dict[str, Any]]
    route_reason: str
    stop_reason: str
    error: str
    turn_log: dict[str, Any]
```

`app/graph/intent.py` must define the fixed mapping, JSON parser, deterministic keyword fallback, and LLM classification. The prompt is exactly:

```text
你是电商客服意图分类器。只输出 JSON：
{"intent":"物流|订单|商品咨询|退款退货|售后|投诉|闲聊","reason":"简短原因"}

规则：
- 询问物流轨迹、快递进度归为“物流”。
- 查询订单状态、金额、创建信息归为“订单”。
- 询问商品参数、库存、价格归为“商品咨询”。
- 询问退款退货政策、到账规则、退货流程归为“退款退货”。
- 询问维修、换货、售后处理进度归为“售后”。
- 表达投诉、不满、要求追责归为“投诉”。
- 非购物业务闲聊归为“闲聊”。

用户：{message}
输出：
```

Invalid or unknown intent values must call `fallback_intent`. If the fallback cannot match a keyword, return `闲聊` only when the text contains a greeting such as `你好`; otherwise raise `IntentClassificationError`.

- [ ] **Step 5: Add the labeled intent evaluation set**

Create `eval_data/ch05_intent_eval_set.json`:

```json
[
  {"text": "订单 1001 的物流到哪了", "intent": "物流", "route": "business"},
  {"text": "快递什么时候能到", "intent": "物流", "route": "business"},
  {"text": "我的包裹现在在哪", "intent": "物流", "route": "business"},
  {"text": "帮我查下运单轨迹", "intent": "物流", "route": "business"},
  {"text": "订单 1001 是什么状态", "intent": "订单", "route": "business"},
  {"text": "我的订单为什么还没发货", "intent": "订单", "route": "business"},
  {"text": "查询订单金额", "intent": "订单", "route": "business"},
  {"text": "这个订单什么时候创建的", "intent": "订单", "route": "business"},
  {"text": "这款显示器刷新率多少", "intent": "商品咨询", "route": "knowledge"},
  {"text": "iHao V1 有货吗", "intent": "商品咨询", "route": "knowledge"},
  {"text": "这个商品多少钱", "intent": "商品咨询", "route": "knowledge"},
  {"text": "智慧屏支持投屏吗", "intent": "商品咨询", "route": "knowledge"},
  {"text": "退货政策是什么", "intent": "退款退货", "route": "knowledge"},
  {"text": "怎么申请退款", "intent": "退款退货", "route": "knowledge"},
  {"text": "七天无理由怎么算", "intent": "退款退货", "route": "knowledge"},
  {"text": "退货需要什么条件", "intent": "退款退货", "route": "knowledge"},
  {"text": "我的退款为什么还没到", "intent": "售后", "route": "business"},
  {"text": "换货审核多久", "intent": "售后", "route": "business"},
  {"text": "显示器坏了怎么维修", "intent": "售后", "route": "business"},
  {"text": "售后处理到哪一步了", "intent": "售后", "route": "business"},
  {"text": "我要投诉", "intent": "投诉", "route": "complaint"},
  {"text": "快递员态度太差了", "intent": "投诉", "route": "complaint"},
  {"text": "客服一直不处理我要投诉", "intent": "投诉", "route": "complaint"},
  {"text": "对这次售后非常不满意要追责", "intent": "投诉", "route": "complaint"},
  {"text": "你好", "intent": "闲聊", "route": "chitchat"},
  {"text": "今天天气怎么样", "intent": "闲聊", "route": "chitchat"},
  {"text": "你会讲笑话吗", "intent": "闲聊", "route": "chitchat"},
  {"text": "随便聊聊", "intent": "闲聊", "route": "chitchat"}
]
```

Create `scripts/eval_ch05.py`:

```python
import asyncio
import json
from collections import Counter
from pathlib import Path

from app.core.llm import get_chat_model
from app.graph.intent import classify_intent_text


async def main() -> int:
    rows = json.loads(
        Path("eval_data/ch05_intent_eval_set.json").read_text(encoding="utf-8")
    )
    model = get_chat_model(streaming=False)
    intent_correct = 0
    route_correct = 0
    confusion = Counter()

    for row in rows:
        decision = await classify_intent_text(row["text"], model)
        intent_correct += decision.intent == row["intent"]
        route_correct += decision.route == row["route"]
        confusion[(row["intent"], decision.intent)] += 1

    total = len(rows)
    print(f"intent_accuracy={intent_correct / total:.4f}")
    print(f"route_accuracy={route_correct / total:.4f}")
    for (expected, actual), count in sorted(confusion.items()):
        print(f"{expected}->{actual}: {count}")
    return 0 if min(intent_correct, route_correct) / total >= 0.85 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
```

- [ ] **Step 6: Run unit tests and evaluation**

Run: `uv run pytest tests/test_graph_intent.py -v`

Expected: PASS.

Run: `uv run python scripts/eval_ch05.py`

Expected: intent accuracy and route accuracy are each at least `0.85`.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock app/graph eval_data/ch05_intent_eval_set.json scripts/eval_ch05.py tests/test_graph_intent.py
git commit -m "feat: add workflow state and intent routing"
```

### Task 3: Split RAG Retrieval and Confidence Gate

**Files:**
- Modify: `app/db/repository.py`
- Modify: `app/services/rag_service.py`
- Modify: `tests/test_rag_service.py`
- Modify: `tests/test_tools.py`

**Interfaces:**
- Consumes: `retrieve`, `assemble_evidence`, `load_chunk_section_paths`, `insert_low_confidence_question`, `get_chat_model`.
- Produces: `retrieve_knowledge_evidence(...) -> dict`, `assess_evidence(...) -> dict`, and `answer_knowledge_question(...) -> dict`.

- [ ] **Step 1: Write failing tests for the split functions**

```python
async def test_retrieve_knowledge_evidence_returns_citations(monkeypatch):
    async def fake_retrieve(*args, **kwargs):
        return [{"chunk_id": "c1", "text": "支持 7 天退货", "score": 0.9, "category": "售后"}]

    async def fake_paths(chunk_ids):
        return {"c1": "售后政策 > 退货"}

    monkeypatch.setattr(rag_service, "retrieve", fake_retrieve)
    monkeypatch.setattr(rag_service, "load_chunk_section_paths", fake_paths)

    result = await rag_service.retrieve_knowledge_evidence("退货政策")

    assert result["matched"] is True
    assert result["evidence"][0]["chunk_id"] == "c1"
    assert result["evidence"][0]["section_path"] == "售后政策 > 退货"


async def test_assess_evidence_rejects_empty_and_records(monkeypatch):
    recorded = []

    async def fake_insert(question, reason, source_conversation_id, entry_point):
        recorded.append((question, reason, source_conversation_id, entry_point))

    monkeypatch.setattr(rag_service, "insert_low_confidence_question", fake_insert)

    result = await rag_service.assess_evidence(
        "未知政策",
        [],
        conversation_id="s1",
    )

    assert result["sufficient"] is False
    assert recorded == [
        ("未知政策", "未召回足够证据", "s1", "workflow_confidence_gate")
    ]


async def test_answer_knowledge_question_composes_split_functions(monkeypatch):
    async def fake_retrieve(*args, **kwargs):
        return {"matched": True, "evidence": [{"citation_id": 1, "text": "证据"}]}

    async def fake_assess(*args, **kwargs):
        return {"sufficient": True}

    monkeypatch.setattr(rag_service, "retrieve_knowledge_evidence", fake_retrieve)
    monkeypatch.setattr(rag_service, "assess_evidence", fake_assess)

    result = await rag_service.answer_knowledge_question("问题")

    assert result["sufficient"] is True
    assert result["context"] == "[1] 证据"
```

- [ ] **Step 2: Run the targeted tests and verify they fail**

Run: `uv run pytest tests/test_rag_service.py -v`

Expected: FAIL because the split functions do not exist.

- [ ] **Step 3: Implement the split service functions**

`retrieve_knowledge_evidence` performs normalization, retrieval, section path enrichment, and evidence assembly but never writes low-confidence records. Return:

```python
{
    "matched": bool(evidence),
    "evidence": evidence,
}
```

`assess_evidence` accepts the evidence list and performs threshold checks before the existing model sufficiency check. It writes low-confidence questions with `entry_point="workflow_confidence_gate"` and returns:

```python
{
    "sufficient": False,
    "reason": "未召回足够证据",
    "refusal": "抱歉，我暂时无法依据现有知识回答这个问题。",
}
```

when evidence is empty or below threshold, and:

```python
{
    "sufficient": True,
    "context": build_evidence_text(evidence),
}
```

when evidence is sufficient. Keep `answer_knowledge_question` as a compatibility wrapper that calls both functions.

- [ ] **Step 4: Update the low-confidence repository signature**

Change:

```python
async def insert_low_confidence_question(
    question: str,
    reason: str,
    source_conversation_id: str | None,
    entry_point: str = "query_faq_self_check",
) -> None:
```

Existing callers that omit `entry_point` keep the old behavior. Update tests that assert `query_faq_self_check` and add an assertion for `workflow_confidence_gate`.

- [ ] **Step 5: Run regression tests**

Run: `uv run pytest tests/test_rag_service.py tests/test_tools.py -v`

Expected: PASS. Existing `query_faq` behavior remains compatible.

- [ ] **Step 6: Commit**

```bash
git add app/db/repository.py app/services/rag_service.py tests/test_rag_service.py tests/test_tools.py
git commit -m "refactor: split rag retrieval from confidence gate"
```

### Task 4: Explicit LangGraph ReAct Subgraph

**Files:**
- Create: `app/graph/agent.py`
- Test: `tests/test_graph_agent.py`

**Interfaces:**
- Consumes: `ChatState`, `AgentLimits`, `TOOLS_BY_NAME`, `execute_tool`, `get_stream_writer`.
- Produces: `AGENT_TOOL_NAMES`, `build_react_agent(model, limits=..., tool_runner=...) -> CompiledStateGraph`.

- [ ] **Step 1: Write failing tests for graph convergence, tools, and limits**

```python
from langchain_core.messages import AIMessageChunk, HumanMessage
from app.graph.agent import build_react_agent
from app.agents.bare_react import AgentLimits


class StreamingScriptedModel:
    def __init__(self, responses):
        self.responses = responses

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = tools
        return self

    async def astream(self, messages, **kwargs):
        response = self.responses.pop(0)
        yield response


async def test_react_graph_converges_without_tools():
    graph = build_react_agent(
        model=StreamingScriptedModel([AIMessageChunk(content="你好")]),
        limits=AgentLimits(),
    )

    result = await graph.ainvoke(
        {
            "agent_messages": [HumanMessage("你好")],
            "agent_steps": 0,
            "tool_trace": [],
        },
        {"configurable": {"thread_id": "agent-1"}},
    )

    assert result["reply"] == "你好"
    assert result["stop_reason"] == "completed"


async def test_react_graph_executes_tools_before_answering():
    chunks = [
        AIMessageChunk(
            tool_calls=[
                {"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}
            ]
        ),
        AIMessageChunk(content="订单已发货。"),
    ]
    graph = build_react_agent(
        model=StreamingScriptedModel(chunks),
        limits=AgentLimits(),
    )

    result = await graph.ainvoke(
        {
            "agent_messages": [HumanMessage("查询订单 1001")],
            "agent_steps": 0,
            "tool_trace": [],
        },
        {"configurable": {"thread_id": "agent-2"}},
    )

    assert result["reply"] == "订单已发货。"
    assert result["tool_trace"][0]["name"] == "query_order"


async def test_react_graph_extracts_suggestion_marker():
    graph = build_react_agent(
        model=StreamingScriptedModel(
            [
                AIMessageChunk(
                    content="建议转人工处理。[[suggest:human_transfer|create_ticket]]"
                )
            ]
        ),
        limits=AgentLimits(),
    )

    result = await graph.ainvoke(
        {
            "agent_messages": [HumanMessage("这个问题一直解决不了")],
            "agent_steps": 0,
            "tool_trace": [],
        },
        {"configurable": {"thread_id": "agent-3"}},
    )

    assert result["reply"] == "建议转人工处理。"
    assert result["reply_options"] == [
        {"id": "human_transfer", "label": "转人工"},
        {"id": "create_ticket", "label": "建工单"},
    ]
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `uv run pytest tests/test_graph_agent.py -v`

Expected: FAIL because `app.graph.agent` does not exist.

- [ ] **Step 3: Implement the ReAct subgraph**

`AGENT_TOOL_NAMES` must be exactly:

```python
AGENT_TOOL_NAMES = ("query_order", "query_product", "query_logistics")
```

The graph nodes are:

- `prepare_agent`: build `agent_messages` from `state["messages"]` plus an evidence system message when `state["evidence"]` exists, then trim with `TOKEN_BUDGET`.
- `agent`: stream `model_with_tools.astream`, combine chunks, emit saved content deltas only when the final combined message has no tool calls, and increment `agent_steps`.
- `tools`: execute every tool call through the injected runner, emit `tool_status`, append `ToolMessage`, and extend `tool_trace`.
- `limit_response`: return the controlled "需要更多信息" reply when the step limit is reached with outstanding tool calls.

After the combined message has no tool calls, remove the internal suggestion marker before emitting user-visible text:

```python
SUGGESTION_PATTERN = re.compile(r"\[\[suggest:([^\]]+)\]\]")

def split_suggestions(text: str) -> tuple[str, list[dict[str, str]]]:
    match = SUGGESTION_PATTERN.search(text)
    if not match:
        return text, []
    ids = [value for value in match.group(1).split("|") if value]
    labels = {"human_transfer": "转人工", "create_ticket": "建工单"}
    options = [
        {"id": value, "label": labels[value]}
        for value in ids
        if value in labels
    ]
    return SUGGESTION_PATTERN.sub("", text).strip(), options
```

Conditional edge:

```python
def should_continue(state: ChatState) -> str:
    last = state["agent_messages"][-1]
    tool_calls = getattr(last, "tool_calls", [])
    if tool_calls and state.get("agent_steps", 0) < limits.max_steps:
        return "tools"
    if tool_calls:
        return "limit"
    return "end"
```

The factory compiles the subgraph before returning:

```python
return builder.compile()
```

Use `get_stream_writer()` from `langgraph.config` inside nodes and emit:

```python
{"type": "tool_status", "tool_name": name, "status": "running"}
{"type": "tool_status", "tool_name": name, "status": "success"}
{"type": "tool_status", "tool_name": name, "status": "error", "message": error}
{"type": "delta", "text": chunk}
{"type": "reply_options", "options": options}
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `uv run pytest tests/test_graph_agent.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/graph/agent.py tests/test_graph_agent.py
git commit -m "feat: add explicit langgraph react agent"
```

### Task 5: Parent Workflow and Checkpointer

**Files:**
- Create: `app/graph/nodes.py`
- Create: `app/graph/builder.py`
- Create: `app/services/turn_logger.py`
- Test: `tests/test_graph_workflow.py`

**Interfaces:**
- Consumes: `ChatState`, `classify_intent_text`, `retrieve_knowledge_evidence`, `assess_evidence`, `build_react_agent`, `TurnLogger`.
- Produces: `build_chat_graph(...)`, `route_after_classify(state)`, `route_after_gate(state)`, `COMPLAINT_REPLY`, `CHITCHAT_REPLY`, `FALLBACK_REPLY`, and `COMPLAINT_OPTIONS`.

Lock the factory signature:

```python
def build_chat_graph(
    *,
    intent_model=None,
    agent_model=None,
    checkpointer,
    intent_classifier=classify_intent_text,
    retrieval_service=retrieve_knowledge_evidence,
    evidence_assessor=assess_evidence,
    agent_builder=build_react_agent,
    turn_logger=None,
    limits=AgentLimits(),
):
    graph = StateGraph(ChatState)
    # Add the nodes and edges exactly as listed below.
    return graph.compile(checkpointer=checkpointer)
```

When `intent_model` or `agent_model` is `None`, the factory uses `get_chat_model(streaming=True)`. When `turn_logger` is `None`, it uses `TurnLogger()`.

- [ ] **Step 1: Write failing routing tests**

```python
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from app.graph.builder import build_chat_graph


async def test_complaint_does_not_call_model(monkeypatch):
    called = False

    class NoModelUse:
        async def ainvoke(self, *args, **kwargs):
            nonlocal called
            called = True
            raise AssertionError("complaint must not call model")


    class FakeTurnLogger:
        async def log(self, session_id, state):
            return {"logged": True}


    async def classify(text):
        return "投诉"

    graph = build_chat_graph(
        intent_model=NoModelUse(),
        agent_model=NoModelUse(),
        intent_classifier=classify,
        turn_logger=FakeTurnLogger(),
        checkpointer=InMemorySaver(),
    )
    result = await graph.ainvoke(
        {
            "session_id": "s1",
            "user_message": "我要投诉",
            "messages": [HumanMessage("我要投诉")],
        },
        {"configurable": {"thread_id": "s1"}},
    )

    assert called is False
    assert result["reply_options"] == [
        {"id": "human_transfer", "label": "转人工"},
        {"id": "create_ticket", "label": "建工单"},
    ]


async def test_knowledge_path_always_runs_retrieval_and_gate(monkeypatch):
    seen = []


    class FakeTurnLogger:
        async def log(self, session_id, state):
            return {"logged": True}


    async def classify(text):
        return "退款退货"

    async def retrieve(query):
        seen.append("retrieve")
        return {"matched": True, "evidence": [{"citation_id": 1, "text": "七天退货"}]}

    async def assess(query, evidence, conversation_id):
        seen.append("gate")
        return {"sufficient": True, "context": "[1] 七天退货"}

    graph = build_chat_graph(
        intent_model=None,
        agent_model=None,
        intent_classifier=classify,
        retrieval_service=retrieve,
        evidence_assessor=assess,
        agent_builder=lambda model, limits: RunnableLambda(
            lambda state: {"reply": "按政策处理", "agent_steps": 1}
        ),
        turn_logger=FakeTurnLogger(),
        checkpointer=InMemorySaver(),
    )

    result = await graph.ainvoke(
        {
            "session_id": "s1",
            "user_message": "退款政策是什么",
            "messages": [HumanMessage("退款政策是什么")],
        },
        {"configurable": {"thread_id": "s1"}},
    )
    assert seen == ["retrieve", "gate"]
    assert result["reply"] == "按政策处理"
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `uv run pytest tests/test_graph_workflow.py -v`

Expected: FAIL because the parent graph modules do not exist.

- [ ] **Step 3: Implement deterministic nodes and graph assembly**

The parent graph edge map is fixed:

```python
graph.add_edge(START, "resolve_reference")
graph.add_edge("resolve_reference", "classify_intent")
graph.add_conditional_edges(
    "classify_intent",
    route_after_classify,
    {
        "knowledge": "retrieve_knowledge",
        "business": "agent",
        "complaint": "complaint_response",
        "chitchat": "chitchat_response",
        "fallback": "fallback",
    },
)
graph.add_edge("retrieve_knowledge", "confidence_gate")
graph.add_conditional_edges(
    "confidence_gate",
    route_after_gate,
    {"agent": "agent", "fallback": "fallback"},
)
graph.add_edge("agent", "log_turn")
graph.add_edge("complaint_response", "log_turn")
graph.add_edge("chitchat_response", "log_turn")
graph.add_edge("fallback", "log_turn")
graph.add_edge("log_turn", END)
```

`resolve_reference` resets per-turn fields, sets `resolved_query=user_message`, and returns an empty `agent_messages` list. `classify_intent_node` writes `intent`, `route`, and `route_reason`.

Every node uses a shared helper that emits node observability:

```python
def emit_node(node: str, status: str, **detail) -> None:
    get_stream_writer()(
        {"type": "node_status", "node": node, "status": status, "detail": detail}
    )
```

`retrieve_knowledge` emits `running` before retrieval and `success` after retrieval. It also emits:

```python
{"type": "citations", "citations": state["citations"]}
```

`confidence_gate` emits `running`, then `success` with `{"sufficient": ...}`. `complaint_response` and `chitchat_response` emit `skipped` for `retrieve_knowledge` and `confidence_gate` so the log clearly shows the branch.

`complaint_response` returns:

```python
{
    "reply": "很抱歉给您带来不好的体验。您可以选择转人工继续反馈，或创建工单留痕等待处理。",
    "reply_options": COMPLAINT_OPTIONS,
}
```

`chitchat_response` returns:

```python
{"reply": "您好，我是 iHelp 智能客服，很高兴为您服务。"}
```

`fallback` returns:

```python
{"reply": state.get("refusal") or "抱歉，我暂时无法可靠回答这个问题。"}
```

The Agent node is the compiled ReAct subgraph and consumes `agent_messages`, `evidence`, `citations`, and `session_id`.

- [ ] **Step 4: Implement `TurnLogger`**

`TurnLogger.log(session_id, state)` must:

- call `ensure_conversation`
- append the user message
- append each tool call and tool result from `tool_trace`
- append the final assistant reply
- emit a structured `logger.info("turn_complete ...")` line containing `session_id`, `intent`, `route`, `agent_steps`, `retrieval`, `sufficient`, and `stop_reason`

The graph must call `turn_logger.log` in `log_turn`. Catch database exceptions inside the logger, log the exception, and still return `{"turn_log": ...}`.

`log_turn` also returns the assistant response to the persistent conversation state:

```python
return {
    "messages": [AIMessage(content=state["reply"])],
    "turn_log": {...},
}
```

- [ ] **Step 5: Implement checkpointer tests**

Add test helpers and tests:

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


class FakeTurnLogger:
    async def log(self, session_id, state):
        return {"logged": True}


async def classify_chitchat(text):
    return "闲聊"


def make_graph(checkpointer):
    return build_chat_graph(
        intent_model=None,
        agent_model=None,
        checkpointer=checkpointer,
        intent_classifier=classify_chitchat,
        agent_builder=lambda model, limits: RunnableLambda(
            lambda state: {"reply": "你好", "agent_steps": 1}
        ),
        turn_logger=FakeTurnLogger(),
    )


async def test_inmemory_checkpointer_restores_messages():
    graph = make_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "persist-1"}}
    await graph.ainvoke(
        {
            "session_id": "persist-1",
            "user_message": "你好",
            "messages": [HumanMessage("你好")],
        },
        config,
    )
    result = await graph.ainvoke(
        {
            "session_id": "persist-1",
            "user_message": "继续",
            "messages": [HumanMessage("继续")],
        },
        config,
    )
    assert len(result["messages"]) == 4


async def test_sqlite_checkpointer_restores_after_reopen(tmp_path):
    path = tmp_path / "checkpoints.db"
    config = {"configurable": {"thread_id": "persist-2"}}
    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        graph = make_graph(saver)
        await graph.ainvoke(
            {
                "session_id": "persist-2",
                "user_message": "你好",
                "messages": [HumanMessage("你好")],
            },
            config,
        )

    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        graph = make_graph(saver)
        snapshot = await graph.aget_state(config)
        assert len(snapshot.values["messages"]) == 2
```

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/test_graph_workflow.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add app/graph/nodes.py app/graph/builder.py app/services/turn_logger.py tests/test_graph_workflow.py
git commit -m "feat: assemble production workflow graph"
```

### Task 6: API Integration, SSE, and Ticket Endpoint

**Files:**
- Modify: `app/api/chat.py`
- Modify: `app/main.py`
- Create: `app/api/tickets.py`
- Modify: `app/schemas/chat.py`
- Modify: `app/config.py`
- Modify: `tests/test_chat_api.py`
- Create: `tests/test_tickets_api.py`

**Interfaces:**
- Consumes: compiled `build_chat_graph`, `AsyncSqliteSaver`, `execute_tool`, `ChatRequest`.
- Produces: `GET`-free `POST /api/chat` graph streaming and `POST /api/tickets`.

- [ ] **Step 1: Add configuration and schema fields**

Add:

```python
langgraph_checkpoint_path: str = "data/langgraph_checkpoints.db"
agent_max_steps: int = 5
agent_max_output_tokens: int = 800
agent_token_budget: int = 4000
```

Add `TicketRequest` with:

```python
class TicketRequest(BaseModel):
    session_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    ticket_type: str = Field(default="投诉", min_length=1)
```

- [ ] **Step 2: Write failing API tests**

```python
def test_chat_forwards_graph_custom_events():
    class FakeGraph:
        async def astream(self, inputs, config, stream_mode):
            assert inputs["user_message"] == "你好"
            assert config["configurable"]["thread_id"] == "s1"
            assert stream_mode == "custom"
            yield {"type": "delta", "text": "您好"}
            yield {"type": "reply_options", "options": []}

    app.dependency_overrides[chat_api.get_chat_graph] = lambda: FakeGraph()
    try:
        with TestClient(app) as client:
            with client.stream(
                "POST",
                "/api/chat",
                json={"session_id": "s1", "message": "你好"},
            ) as response:
                lines = list(response.iter_lines())
        assert any('"type": "delta"' in line for line in lines)
        assert any('"type": "reply_options"' in line for line in lines)
        assert lines[-1] == "data: [DONE]"
    finally:
        app.dependency_overrides.clear()
```

```python
def test_ticket_endpoint_calls_create_ticket(monkeypatch):
    async def fake_execute(tool_name, tool_args):
        assert tool_name == "create_ticket"
        assert tool_args["session_id"] == "s1"
        return ToolExecutionResult(
            tool_name=tool_name,
            tool_args=tool_args,
            result='{"ticket_id":"TK1","status":"open"}',
            success=True,
        )

    monkeypatch.setattr(tickets_api, "execute_tool", fake_execute)
    with TestClient(app) as client:
        response = client.post(
            "/api/tickets",
            json={
                "session_id": "s1",
                "description": "我要投诉",
                "ticket_type": "投诉",
            },
        )
    assert response.status_code == 200
    assert response.json()["ticket_id"] == "TK1"
```

- [ ] **Step 3: Run the API tests and verify they fail**

Run: `uv run pytest tests/test_chat_api.py tests/test_tickets_api.py -v`

Expected: FAIL because graph dependency and ticket route are absent.

- [ ] **Step 4: Wire the graph into FastAPI lifespan**

In `app/main.py`:

```python
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.config import settings
from app.graph.builder import build_chat_graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    path = Path(settings.langgraph_checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(path)) as checkpointer:
        app.state.chat_graph = build_chat_graph(checkpointer=checkpointer)
        yield
    await dispose_db()
```

`get_chat_graph` in `app/api/chat.py` reads `request.app.state.chat_graph`. Tests override it.

- [ ] **Step 5: Replace the old chat loop with graph event relay**

`chat()` must:

```python
async def event_stream():
    try:
        async for event in graph.astream(
            {
                "session_id": req.session_id,
                "user_message": req.message,
                "messages": [HumanMessage(req.message)],
            },
            {"configurable": {"thread_id": req.session_id}},
            stream_mode="custom",
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    except Exception:
        logger.exception("客服图执行失败 session_id=%s", req.session_id)
        yield "event: error\n"
        yield 'data: {"message":"上游服务暂时不可用，请稍后重试"}\n\n'
    yield "data: [DONE]\n\n"
```

Remove the direct `SessionStore`, tool execution, citation extraction, and streaming model calls from `chat.py`.

- [ ] **Step 6: Add the ticket endpoint**

`app/api/tickets.py`:

```python
@router.post("/api/tickets")
async def create_ticket_endpoint(req: TicketRequest):
    result = await execute_tool(
        "create_ticket",
        {
            "conversation_id": req.session_id,
            "description": req.description,
            "ticket_type": req.ticket_type,
        },
    )
    if not result.success:
        raise HTTPException(status_code=500, detail=result.error)
    return json.loads(result.result)
```

Register the router in `app.main`.

- [ ] **Step 7: Run the API and regression tests**

Run: `uv run pytest tests/test_chat_api.py tests/test_tickets_api.py tests/test_graph_workflow.py tests/test_graph_agent.py -v`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add app/api/chat.py app/api/tickets.py app/main.py app/schemas/chat.py app/config.py tests/test_chat_api.py tests/test_tickets_api.py
git commit -m "feat: expose workflow graph over chat sse"
```

### Task 7: Frontend Action Buttons and Confirmations

**Files:**
- Modify: `D:\std_selftest_frontend\src\main.js`
- Modify: `D:\std_selftest_frontend\src\styles.css`
- Modify: `D:\std_selftest_frontend\index.html` only if the existing chat markup needs an action container

**Interfaces:**
- Consumes: `reply_options` SSE events with `{id,label}` and `POST /api/tickets`.
- Produces: independent `human_transfer` and `create_ticket` actions under the assistant message.

- [ ] **Step 1: Extend the stream event parser**

In `streamChat`, add:

```javascript
let replyOptions = [];
```

In `processFrame`, add:

```javascript
} else if (payload.type === "reply_options") {
  replyOptions = Array.isArray(payload.options) ? payload.options : [];
}
```

After a successful stream:

```javascript
renderReplyOptions(
  bubble.closest(".message-stack"),
  bubble,
  replyOptions,
  message
);
```

- [ ] **Step 2: Implement independent frontend handlers**

```javascript
function renderReplyOptions(stack, bubble, options, userMessage) {
  if (!stack || !options.length) return;
  const group = document.createElement("div");
  group.className = "message-actions";

  for (const option of options) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "action-btn";
    button.dataset.actionId = option.id;
    button.textContent = option.label;

    if (option.id === "human_transfer") {
      button.addEventListener("click", () => {
        button.disabled = true;
        button.textContent = "已转人工";
        appendMessage("assistant", "已转接人工客服");
        appendMessage(
          "assistant",
          "您好，我是客服小猫，请问有什么可以帮您的"
        );
      });
    }

    if (option.id === "create_ticket") {
      button.addEventListener("click", async () => {
        button.disabled = true;
        const original = button.textContent;
        button.textContent = "建单中…";
        try {
          const response = await fetch("/api/tickets", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              session_id: elements.sessionId.value.trim() || state.sessionId,
              description: userMessage,
              ticket_type: "投诉",
            }),
          });
          const payload = await response.json();
          if (!response.ok) throw new Error(payload.detail || "建单失败");
          button.textContent = `已建单 ${payload.ticket_id}`;
        } catch (error) {
          button.disabled = false;
          button.textContent = original;
          showChatError(error.message || "建单失败");
        }
      });
    }

    group.appendChild(button);
  }
  stack.appendChild(group);
  scrollChatToEnd();
}
```

- [ ] **Step 3: Add stable action styling**

Add CSS classes:

```css
.message-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.action-btn {
  min-height: 34px;
  padding: 0 12px;
  border: 1px solid var(--border-strong);
  border-radius: 8px;
  background: var(--surface);
  color: var(--primary);
  font: inherit;
  font-size: 13px;
  font-weight: 700;
  cursor: pointer;
}

.action-btn:disabled {
  cursor: default;
  color: var(--muted);
  background: var(--surface-muted);
}
```

- [ ] **Step 4: Build and inspect the frontend**

Run: `npm run build`

Expected: Vite build succeeds.

Start the backend and frontend, then verify:

- Complaint shows two buttons.
- Clicking transfer adds the confirmation and greeting without a `/api/tickets` request.
- Clicking ticket sends one request and displays the ticket ID.
- Doing neither and sending another message does not create a ticket or alter the session.

- [ ] **Step 5: Record frontend changes**

Run:

```powershell
Get-FileHash src\main.js, src\styles.css, index.html -Algorithm SHA256
```

Append the modified paths, hashes, build result, and manual verification result to `dev-notes/ch05.md`. The frontend directory is not a Git repository, so no frontend commit is created.

### Task 8: Acceptance Run, Documentation, and Finish

**Files:**
- Modify: `README.md`
- Create: `scripts/demo_ch05.py`
- Modify: `dev-notes/ch05.md`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: reproducible acceptance commands and final process notes.

- [ ] **Step 1: Add a callable end-to-end graph demo**

Create `scripts/demo_ch05.py`:

```python
import asyncio
import json
import sys

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.graph.builder import build_chat_graph


async def main() -> None:
    question = sys.argv[1]
    session_id = sys.argv[2]
    graph = build_chat_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": session_id}}
    async for event in graph.astream(
        {
            "session_id": session_id,
            "user_message": question,
            "messages": [HumanMessage(question)],
        },
        config,
        stream_mode="custom",
    ):
        print(json.dumps(event, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
```

Run:

```bash
uv run python scripts/demo_ch05.py "订单 1001 的物流到哪了" demo-1001
```

Expected: tool status events for `query_logistics` and a final reply.

- [ ] **Step 2: Add README commands**

Add a Ch05 section containing:

```bash
uv run python scripts/demo_bare_agent.py
uv run python scripts/eval_ch05.py
uv run python scripts/demo_ch05.py "订单 1001 的物流到哪了" demo-1001
uv run pytest -v
```

Also document that the SQLite checkpoint database is `data/langgraph_checkpoints.db` and the server should run with one Uvicorn worker for this chapter.

- [ ] **Step 3: Run the complete test suite and evaluation**

Run:

```bash
uv run pytest -v
uv run python scripts/eval_ch05.py
```

Expected:

- all tests pass
- intent accuracy >= `0.85`
- route accuracy >= `0.85`

- [ ] **Step 4: Run the five acceptance scenarios**

1. Policy question: assert the event stream contains `retrieve_knowledge` and `confidence_gate`.
2. `订单 1001 的物流到哪了`: assert `query_logistics` tool status and a final answer.
3. `我要投诉`: assert both independent options; assert no ticket before click; assert one ticket row after clicking create; assert transfer is frontend-only.
4. Chitchat: assert fixed reply and no model-backed Agent event.
5. `先查订单 1001 的状态，再查物流`: assert `query_order` and `query_logistics` both appear in order.

- [ ] **Step 5: Update `dev-notes/ch05.md`**

Append the required per-stage sections with:

- the user's key original instruction
- the produced spec or plan path
- the review conclusion
- any rejected or corrected direction
- failures and rework

Do not postpone this to the end. Append after plan approval, each implementation task, code review, and finish.

- [ ] **Step 6: Run the final verification commands**

```bash
uv run pytest -v
uv run python scripts/eval_ch05.py
python -m compileall -q app scripts
git diff --check
```

For frontend:

```bash
npm run build
```

- [ ] **Step 7: Commit**

```bash
git add README.md scripts/demo_ch05.py dev-notes/ch05.md
git commit -m "docs: document ch05 workflow acceptance"
```

## Plan Self-Review

- Spec coverage: every architecture section maps to Tasks 1 through 8.
- Routing: all seven intents and four exits are covered by Task 2 tests.
- Knowledge safety: Task 3 and Task 5 separate retrieval and confidence gating; Task 4 excludes `query_faq` from Agent tools.
- Agent safety: Task 4 excludes `create_ticket` and Task 6 exposes it only from the user-clicked endpoint.
- Persistence: Task 5 tests `InMemorySaver`, and Task 6 uses the production `AsyncSqliteSaver` lifecycle.
- Frontend: Task 7 covers independent buttons and verifies the no-click path.
- Evaluation: Task 2 and Task 8 require labeled intent evaluation rather than treating prompt behavior as ordinary unit-testable code.
- No placeholders: all tasks identify exact files, interfaces, tests, commands, and commit messages.
