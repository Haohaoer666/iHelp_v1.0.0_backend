# iHelp Ch06 Query Understanding and Intent Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the placeholder diverter with context-aware query understanding, strict confidence-bearing intent classification, deterministic refund/after-sales order and policy collection, and resumable order-selection UI.

**Architecture:** Keep the Ch05 parent LangGraph. Add deterministic query, intent, core after-sales, interrupt, retrieval, and logging nodes around the existing ReAct Agent. Use the existing checkpointer for interrupt/resume and keep knowledge in one store.

**Tech Stack:** Python 3.12, FastAPI, LangChain, LangGraph, SQLAlchemy 2 async, MySQL, Milvus, pytest, native HTML/CSS/JavaScript, Vite.

**Spec:** `docs/superpowers/specs/2026-09-19-ch06-query-intent-routing-design.md`

## Global Constraints

- Use the existing chat upstream and LangGraph checkpointer.
- Do not add a model-training path, BERT classifier, cross-session memory, or a new orchestration component.
- Do not guess order numbers.
- Do not ask for a refund reason until the user submits the refund form.
- Keep query expansion retrieval-side only.
- Use Context7 documentation before changing dependency-specific APIs.
- The frontend follows the user's Vibe Coding exception and is verified visually instead of through TDD.
- Preserve current uncommitted Ch05 comments in the working tree.

---

### Task 1: Query Understanding

**Files:**
- Create: `app/services/conversation_understanding.py`
- Create: `tests/ch06/test_conversation_understanding.py`
- Modify: `app/core/llm.py`
- Modify: `app/config.py`
- Modify: `.env.example`

**Interfaces:**
- Consumes: `BaseChatModel`, `HumanMessage`, `AIMessage`, `get_chat_model(model=...)`.
- Produces: `QueryUnderstandingResult`, `understand_query(history, current_message, model) -> QueryUnderstandingResult`.

- [ ] **Step 1: Write failing tests for pass-through, pronoun resolution, invalid JSON fallback, and model selection.**

```python
class StaticModel:
    def __init__(self, content):
        self.content = content
        self.messages = []

    async def ainvoke(self, messages, **kwargs):
        self.messages = messages
        return AIMessage(content=self.content)


async def test_complete_question_passes_through():
    model = StaticModel('{"resolved_query":"退货政策是什么"}')
    result = await understand_query([], "退货政策是什么", model)
    assert result.resolved_query == "退货政策是什么"
    assert result.changed is False


async def test_pronoun_uses_conversation_history():
    history = [
        HumanMessage("订单 1001 的物流到哪了"),
        AIMessage("订单 1001 正在运输中。"),
    ]
    model = StaticModel('{"resolved_query":"订单 1001 是否可以退款"}')
    result = await understand_query(history, "它能退吗", model)
    assert result.resolved_query == "订单 1001 是否可以退款"
    assert result.changed is True


async def test_invalid_json_preserves_original_message():
    model = StaticModel("不是 JSON")
    result = await understand_query([], "这个能退吗", model)
    assert result.resolved_query == "这个能退吗"
    assert result.source == "fallback"
```

- [ ] **Step 2: Run the tests and verify they fail.**

Run: `uv run pytest tests/ch06/test_conversation_understanding.py -q`
Expected: FAIL because `conversation_understanding` does not exist.

- [ ] **Step 3: Implement the prompt, parser, and model factory selection.**

```python
@dataclass(frozen=True)
class QueryUnderstandingResult:
    resolved_query: str
    changed: bool
    source: str = "model"


async def understand_query(history, current_message, model) -> QueryUnderstandingResult:
    prompt = build_query_understanding_prompt(history, current_message)
    response = await model.ainvoke([HumanMessage(content=prompt)])
    try:
        payload = json.loads(strip_code_fence(_content_text(response.content)))
        resolved = str(payload["resolved_query"]).strip()
        if not resolved:
            raise ValueError("empty resolved_query")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return QueryUnderstandingResult(current_message, False, "fallback")
    return QueryUnderstandingResult(
        resolved_query=resolved,
        changed=resolved != current_message,
    )
```

Update `get_chat_model` so a per-call model override is possible. Empty
configuration values fall back to `settings.chat_model`.

- [ ] **Step 4: Run the tests and verify they pass.**

Run: `uv run pytest tests/ch06/test_conversation_understanding.py -q`
Expected: PASS.

- [ ] **Step 5: Append the task result to `dev-notes/ch06.md`.**

Include the exact user instruction, files produced, verification result, rejected
or corrected decisions, and rework.

- [ ] **Step 6: Commit.**

```bash
git add app/services/conversation_understanding.py app/core/llm.py app/config.py .env.example tests/ch06/test_conversation_understanding.py dev-notes/ch06.md
git commit -m "feat: add context-aware query understanding"
```

### Task 2: Strict Intent Classification and Model Escalation

**Files:**
- Modify: `app/graph/intent.py`
- Create: `tests/ch06/test_intent_classifier.py`
- Modify: `app/graph/state.py`

**Interfaces:**
- Consumes: `QueryUnderstandingResult.resolved_query`, primary model, optional low-cost model.
- Produces: `IntentDecision(intent, route, confidence, model_used)`, `classify_intent_text(message, primary_model, low_cost_model=None, use_low_cost_first=False) -> IntentDecision`.

- [ ] **Step 1: Write failing tests for the exact JSON shape, boundary examples, malformed output, and low-confidence escalation.**

```python
async def test_intent_returns_only_intent_and_confidence():
    model = StaticModel('{"intent":"退款退货","confidence":0.91}')
    result = await classify_intent_text("退货政策是什么", model)
    assert result.intent == "退款退货"
    assert result.confidence == 0.91
    assert result.route == "core_after_sales"


async def test_unknown_question_falls_back_to_other():
    model = StaticModel("not json")
    result = await classify_intent_text("帮我看下那个奇怪的东西", model)
    assert result.intent == "其他"
    assert result.confidence == 0.0


async def test_low_confidence_escalates_to_primary_model():
    low = StaticModel('{"intent":"订单","confidence":0.31}')
    primary = StaticModel('{"intent":"物流","confidence":0.92}')
    result = await classify_intent_text(
        "订单 1001 到哪了",
        primary,
        low_cost_model=low,
        use_low_cost_first=True,
        confidence_threshold=0.75,
    )
    assert result.intent == "物流"
    assert result.model_used == "upgraded"
```

- [ ] **Step 2: Run the tests and verify they fail.**

Run: `uv run pytest tests/ch06/test_intent_classifier.py -q`
Expected: FAIL because the new contract and `其他` route are missing.

- [ ] **Step 3: Implement the four-part prompt and strict parser.**

Use this route mapping:

```python
INTENT_TO_ROUTE = {
    "物流": "business",
    "订单": "business",
    "商品咨询": "knowledge",
    "退款退货": "core_after_sales",
    "售后": "core_after_sales",
    "投诉": "complaint",
    "闲聊": "chitchat",
    "其他": "other",
}
```

Reject extra JSON keys in tests. A malformed response returns
`IntentDecision("其他", "other", 0.0, "fallback")`.

- [ ] **Step 4: Run the tests and verify they pass.**

Run: `uv run pytest tests/ch06/test_intent_classifier.py -q`
Expected: PASS.

- [ ] **Step 5: Append the task result to `dev-notes/ch06.md`.**

- [ ] **Step 6: Commit.**

```bash
git add app/graph/intent.py app/graph/state.py tests/ch06/test_intent_classifier.py dev-notes/ch06.md
git commit -m "feat: add confidence-aware intent routing"
```

### Task 3: Query Expansion and Multi-Query Retrieval

**Files:**
- Create: `app/services/retrieval_queries.py`
- Modify: `app/services/rag_service.py`
- Create: `tests/ch06/test_retrieval_queries.py`

**Interfaces:**
- Consumes: `BaseChatModel`, hybrid `retrieve`, `assemble_evidence`.
- Produces: `QueryExpansion`, `expand_retrieval_queries(resolved_query, model)` and `retrieve_multi_query(queries, category=None)`.

- [ ] **Step 1: Write failing tests for strict `queries` JSON, deduplication, original-query retention, and merge behavior.**

```python
async def test_expansion_deduplicates_and_preserves_original_first():
    model = StaticModel(
        '{"queries":["七天无理由退货","签收后七天退货","七天无理由退货"]}'
    )
    result = await expand_retrieval_queries("这个能退吗", model)
    assert result.queries == ["这个能退吗", "七天无理由退货", "签收后七天退货"]


async def test_expansion_rejects_extra_fields():
    model = StaticModel('{"queries":["退款"],"reason":"x"}')
    result = await expand_retrieval_queries("退款", model)
    assert result.queries == ["退款"]
    assert result.source == "fallback"
```

- [ ] **Step 2: Run the tests and verify they fail.**

Run: `uv run pytest tests/ch06/test_retrieval_queries.py -q`
Expected: FAIL because the expansion service does not exist.

- [ ] **Step 3: Implement expansion and merge.**

Merge chunks with:

```python
by_chunk: dict[str, dict] = {}
for hit in merged:
    current = by_chunk.get(hit["chunk_id"])
    if current is None or float(hit.get("score", 0)) > float(current.get("score", 0)):
        by_chunk[hit["chunk_id"]] = hit
return sorted(by_chunk.values(), key=lambda item: float(item.get("score", 0)), reverse=True)
```

Every retrieval failure is recorded in `failed_queries`; successful queries
still return evidence.

- [ ] **Step 4: Run the tests and verify they pass.**

Run: `uv run pytest tests/ch06/test_retrieval_queries.py -q`
Expected: PASS.

- [ ] **Step 5: Append the task result to `dev-notes/ch06.md`.**

- [ ] **Step 6: Commit.**

```bash
git add app/services/retrieval_queries.py app/services/rag_service.py tests/ch06/test_retrieval_queries.py dev-notes/ch06.md
git commit -m "feat: add retrieval-side query expansion"
```

### Task 4: Deterministic Demo Orders

**Files:**
- Create: `app/services/order_catalog.py`
- Create: `tests/ch06/test_order_catalog.py`
- Modify: `app/core/tools.py`
- Modify: `tests/test_tools.py`

**Interfaces:**
- Consumes: order identifiers from the query or selector.
- Produces: `list_demo_orders()`, `get_demo_order(order_id)`, deterministic `query_order`.

- [ ] **Step 1: Write failing tests for stable order lookup and selector fields.**

```python
def test_demo_order_catalog_is_stable():
    first = get_demo_order("1001")
    second = get_demo_order("1001")
    assert first == second
    assert first["order_id"] == "1001"


def test_list_demo_orders_has_selector_fields():
    orders = list_demo_orders()
    assert {"order_id", "product_name", "status", "amount", "signed_at"} <= set(
        orders[0]
    )
```

- [ ] **Step 2: Run the tests and verify they fail.**

Run: `uv run pytest tests/ch06/test_order_catalog.py -q`
Expected: FAIL because the catalog does not exist.

- [ ] **Step 3: Implement the catalog and wire `query_order`.**

Use three deterministic orders:

```python
DEMO_ORDERS = {
    "1001": {
        "order_id": "1001",
        "product_name": "iHao 智能手表",
        "category": "智能穿戴",
        "amount": 899.0,
        "status": "已签收",
        "created_at": "2026-09-10",
        "signed_at": "2026-09-15",
        "condition": "未拆封",
    },
    "1002": {
        "order_id": "1002",
        "product_name": "iHao 蓝牙耳机",
        "category": "数码配件",
        "amount": 199.0,
        "status": "运输中",
        "created_at": "2026-09-17",
        "signed_at": None,
        "condition": "运输中",
    },
    "1003": {
        "order_id": "1003",
        "product_name": "定制手机壳",
        "category": "定制商品",
        "amount": 49.0,
        "status": "已完成",
        "created_at": "2026-08-12",
        "signed_at": "2026-08-20",
        "condition": "已拆封使用",
    },
}
```

`query_order` must return the catalog object as JSON.

- [ ] **Step 4: Run the tests and verify they pass.**

Run: `uv run pytest tests/ch06/test_order_catalog.py tests/test_tools.py -q`
Expected: PASS.

- [ ] **Step 5: Append the task result to `dev-notes/ch06.md`.**

- [ ] **Step 6: Commit.**

```bash
git add app/services/order_catalog.py app/core/tools.py tests/ch06/test_order_catalog.py tests/test_tools.py dev-notes/ch06.md
git commit -m "feat: add deterministic demo order catalog"
```

### Task 5: Graph State, Core Subflow, and Interrupt Resume

**Files:**
- Modify: `app/graph/state.py`
- Modify: `app/graph/nodes.py`
- Modify: `app/graph/builder.py`
- Modify: `app/graph/agent.py`
- Create: `tests/ch06/test_core_after_sales_graph.py`

**Interfaces:**
- Consumes: Task 1-4 services and `langgraph.types.interrupt`.
- Produces: parent graph nodes `understand_query`, `build_order_options`, `select_order`, `load_order`, `expand_queries`, `retrieve_policy`.

- [ ] **Step 1: Write failing graph tests for refund/after-sales flow, missing-order interrupt, resume, and other fallback.**

```python
async def test_missing_order_interrupts_and_resumes():
    graph = build_chat_graph(
        understanding_model=StaticModel('{"resolved_query":"这个能退吗"}'),
        intent_model=StaticModel('{"intent":"退款退货","confidence":0.95}'),
        agent_model=NoUseModel(),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "ch06-resume"}}
    first = await graph.ainvoke(
        {"session_id": "ch06-resume", "user_message": "这个能退吗", "messages": [HumanMessage("这个能退吗")]},
        config,
    )
    assert first["__interrupt__"]

    second = await graph.ainvoke(
        Command(resume={"type": "order_selected", "order_id": "1001"}),
        config,
    )
    assert second["order_id"] == "1001"
    assert second["evidence"]
```

- [ ] **Step 2: Run the tests and verify they fail.**

Run: `uv run pytest tests/ch06/test_core_after_sales_graph.py -q`
Expected: FAIL because the graph nodes and state fields are missing.

- [ ] **Step 3: Implement graph nodes and update the Agent prompt.**

The Agent receives a structured system instruction:

```python
if state.get("order_data"):
    lines.append("订单事实：")
    lines.append(json.dumps(state["order_data"], ensure_ascii=False))
    lines.append("只判断这一单是否符合本次检索到的退款退货政策。")
```

The Agent must not invent order facts or policy clauses.

- [ ] **Step 4: Run the tests and verify they pass.**

Run: `uv run pytest tests/ch06/test_core_after_sales_graph.py -q`
Expected: PASS.

- [ ] **Step 5: Append the task result to `dev-notes/ch06.md`.**

- [ ] **Step 6: Commit.**

```bash
git add app/graph/state.py app/graph/nodes.py app/graph/builder.py app/graph/agent.py tests/ch06/test_core_after_sales_graph.py dev-notes/ch06.md
git commit -m "feat: add deterministic core after-sales subflow"
```

### Task 6: Chat Resume API and Refund Form API

**Files:**
- Modify: `app/schemas/chat.py`
- Modify: `app/api/chat.py`
- Create: `app/api/refunds.py`
- Modify: `app/main.py`
- Modify: `tests/test_chat_api.py`
- Create: `tests/ch06/test_refunds_api.py`

**Interfaces:**
- Consumes: graph `astream`, `Command`, fixed refund reasons, `create_ticket`.
- Produces: message-or-resume chat requests, `/api/refunds`, `/api/refunds/reasons`.

- [ ] **Step 1: Write failing API tests for validation, resume forwarding, reason validation, and ticket creation.**

```python
def test_chat_forwards_resume_as_command():
    fake = FakeGraph()
    app.dependency_overrides[chat_api.get_chat_graph] = lambda: fake
    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/api/chat",
            json={"session_id": "s1", "resume": {"type": "order_selected", "order_id": "1001"}},
        ) as response:
            list(response.iter_lines())
    assert isinstance(fake.calls[0][0], Command)
```

- [ ] **Step 2: Run the tests and verify they fail.**

Run: `uv run pytest tests/test_chat_api.py tests/ch06/test_refunds_api.py -q`
Expected: FAIL because resume and refund routes are missing.

- [ ] **Step 3: Implement schemas, validation, and routes.**

Use a Pydantic model validator requiring exactly one of `message` and `resume`.
Use `REFUND_REASONS` from configuration and return 422 for an unsupported
reason.

- [ ] **Step 4: Run the tests and verify they pass.**

Run: `uv run pytest tests/test_chat_api.py tests/ch06/test_refunds_api.py -q`
Expected: PASS.

- [ ] **Step 5: Append the task result to `dev-notes/ch06.md`.**

- [ ] **Step 6: Commit.**

```bash
git add app/schemas/chat.py app/api/chat.py app/api/refunds.py app/main.py tests/test_chat_api.py tests/ch06/test_refunds_api.py dev-notes/ch06.md
git commit -m "feat: resume chat after order selection"
```

### Task 7: Order Selector and Refund Form Frontend

**Files:**
- Modify: `D:\std_selftest_frontend\src\main.js`
- Modify: `D:\std_selftest_frontend\src\styles.css`

**Interfaces:**
- Consumes: `order_selector`, `refund_form`, `/api/chat`, `/api/refunds/reasons`, `/api/refunds`.
- Produces: clickable order cards, resume continuation, refund reason form, success/error states.

- [ ] **Step 1: Refactor the chat request runner.**

Add a shared runner accepting:

```javascript
{
  sessionId,
  message,
  resume,
  userLabel,
  appendUser
}
```

Do not restart the graph with a synthetic order-number text message.

- [ ] **Step 2: Render order cards and resume on click.**

Use a `<button>` card with order number, product name, status, signed date, and
amount. On click, disable the clicked card, append a confirmation in the chat,
and call the shared runner with `resume={"type":"order_selected","order_id":...}`.

- [ ] **Step 3: Render the refund form.**

Fetch `/api/refunds/reasons`, render a `<select>` and submit button, then post
`{session_id, order_id, reason}` to `/api/refunds`. Lock the form after success.

- [ ] **Step 4: Build and visually verify.**

Run: `npm run build`

Verify desktop and mobile:

- selector cards are visible and do not overlap;
- clicking a card resumes without a duplicate user text message;
- refund form exposes only fixed reasons;
- successful submission shows the refund ticket number.

- [ ] **Step 5: Append hashes and manual verification to `dev-notes/ch06.md`.**

### Task 8: Evaluation, Demo, Documentation, and Final Verification

**Files:**
- Create: `eval_data/ch06_query_intent_eval_set.json`
- Create: `scripts/eval_ch06.py`
- Create: `scripts/demo_ch06.py`
- Modify: `README.md`
- Modify: `dev-notes/ch06.md`

**Interfaces:**
- Consumes: all previous tasks.
- Produces: reproducible evaluation and demo commands.

- [ ] **Step 1: Add the multi-turn evaluation set.**

Include at least 16 examples covering all eight decisions, multi-turn
logistics -> refund -> logistics, coreference, pass-through, and ambiguous
fallback.

- [ ] **Step 2: Add the evaluation script.**

Use both model calls and print:

```text
json_parse_rate=...
intent_accuracy=...
resolved_query_match_rate=...
other_fallback_rate=...
```

Return nonzero below the thresholds in the spec.

- [ ] **Step 3: Add the demo script.**

The script must support:

```bash
uv run python scripts/demo_ch06.py "物流 1001 到哪了" demo-flow
uv run python scripts/demo_ch06.py "这个能退吗" demo-flow --resume-order 1001
```

- [ ] **Step 4: Run final verification.**

Run:

```bash
uv run pytest -q
uv run python scripts/eval_ch06.py
python -m compileall -q app scripts
git diff --check
```

Frontend:

```bash
npm run build
```

- [ ] **Step 5: Append final results to `dev-notes/ch06.md`.**

- [ ] **Step 6: Commit.**

```bash
git add eval_data/ch06_query_intent_eval_set.json scripts/eval_ch06.py scripts/demo_ch06.py README.md dev-notes/ch06.md
git commit -m "docs: document ch06 query and intent routing"
```

## Plan Self-Review

- Query understanding, pass-through, and coreference are covered by Task 1 and
  Task 8.
- Strict intent JSON, `其他`, boundary examples, and confidence escalation are
  covered by Task 2.
- Retrieval-side expansion and multi-query deduplication are covered by Task 3.
- Order selection and deterministic order facts are covered by Task 4.
- Refund/after-sales order-first, policy-second, and Agent-last behavior is
  covered by Task 5.
- Resume transport and refund submission are covered by Task 6.
- Frontend order cards and refund form are covered by Task 7.
- Evaluation and final acceptance are covered by Task 8.
- No task changes the knowledge ingestion path, so expanded queries are not
  stored as multiple knowledge entries.
