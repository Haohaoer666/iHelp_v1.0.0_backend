# iHelp Ch07 Context Cost Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the 64K model window as a hard ceiling while making `24 x 800 = 19200` the effective history soft budget, enforcing actual token caps for volatile context, and removing the conflicting standalone Agent token gate.

**Architecture:** Extend the existing `app/context` package rather than introducing another graph loop. `ContextBudget` separates hard and soft history limits; token helpers count and truncate real request components; the trimmer excludes the current user message and selects bounded summaries; the assembler keeps one stable system prefix and one dynamic supplement; the Agent uses one full-request guard instead of `agent_token_budget`.

**Tech Stack:** Python 3.12, LangChain, LangGraph 1.0.x, SQLAlchemy 2 async, MySQL, pytest, FastAPI.

**Spec:** `docs/superpowers/specs/2026-09-23-ch07-context-cost-optimization-design.md`

## Global Constraints

- Keep `MODEL_CONTEXT_WINDOW=65536`, `DESIRED_RETAINED_TURNS=24`, and `STEADY_TURN_TOKEN_ESTIMATE=800` as the default operating profile.
- Keep `64K` as a hard ceiling, never as a normal target.
- Keep the three layers: raw recent layer 1, shortened middle layer 2, append-only summaries.
- Do not add semantic retrieval over history, cross-session memory, user profiles, or authentication.
- Do not replace LangChain, LangGraph, FastAPI, SQLAlchemy, or MySQL.
- Do not change the single-worker summary ownership model.
- Do not move summary or evidence into a separate system message.
- Do not write tool calls or tool results into `messages`.
- Preserve the complete `ChatState.messages` and checkpointer lifecycle.
- Preserve the legacy acceptance result `5650 / 3954 / 1695` through an explicit compatibility profile.
- Re-check LangChain, LangGraph, SQLAlchemy, and FastAPI interfaces through Context7 before implementation.

---

## File Structure

Modify:

- `app/config.py`: add cost-profile settings and deprecate the standalone Agent token gate.
- `.env.example`: document the new profile and token caps.
- `app/context/budget.py`: add hard/soft history budgets and component budgets.
- `app/context/tokens.py`: add actual prompt counting and token-bounded truncation.
- `app/context/models.py`: extend `ContextPack` with prompt usage and current-user data.
- `app/context/trimmer.py`: exclude the current user, cap summary injection, and expose budget metadata.
- `app/context/assembler.py`: enforce the fixed message order with a separate current-user message.
- `app/context/summary.py`: enforce the summary segment token cap.
- `app/context/tasks.py`: pass segment caps and log token-boundary information.
- `app/context/logging.py`: log complete prompt usage and summary cap data.
- `app/context/manager.py`: pass the current user id and effective limits into the trimmer.
- `app/graph/agent.py`: replace the 4000-token gate with the full-request guard and truncate tool payloads.
- `app/graph/builder.py`: wire new Agent limits and remove the old gate dependency.
- `app/main.py`: construct the new budget/resources and fail fast if the profile is invalid.
- `app/agents/bare_react.py`: keep the legacy runner compatible while accepting the new full-request limit.
- `tests/ch07/test_context_budget.py`
- `tests/ch07/test_token_estimator.py`
- `tests/ch07/test_context_trimmer.py`
- `tests/ch07/test_context_assembler.py`
- `tests/ch07/test_context_manager.py`
- `tests/ch07/test_summary_service.py`
- `tests/ch07/test_summary_tasks.py`
- `tests/ch07/test_context_logging.py`
- `tests/ch07/test_context_graph.py`
- `tests/ch07/test_context_acceptance.py`
- `tests/test_graph_agent.py`
- `tests/test_graph_workflow.py`
- `scripts/demo_ch07.py`
- `scripts/eval_ch07.py`
- `README.md`
- `dev-notes/ch07.md`

---

### Task 1: Context7 Gate, Configuration, and Budget Model

**Files:**

- Modify: `app/config.py`
- Modify: `.env.example`
- Modify: `app/context/budget.py`
- Modify: `tests/ch07/test_context_budget.py`
- Modify: `dev-notes/ch07.md`

**Interfaces:**

- Consumes: `app.config.settings`.
- Produces:
  - `ContextBudget.hard_history_budget`
  - `ContextBudget.soft_history_budget`
  - `ContextBudget.history_budget`
  - `ContextBudget.layer1_budget`
  - `ContextBudget.layer2_budget`
  - `ContextBudget.evidence_budget`
  - `ContextBudget.react_peak`

- [ ] **Step 1: Re-check the official interfaces through Context7**

Use the same Context7 MCP setup that was used for Ch07. Query and record the
actual source/version for:

```text
LangChain count_tokens_approximately
LangChain bind_tools
LangGraph StateGraph/add_messages/checkpointer
SQLAlchemy 2 async session transaction
```

Record the query result and version in `dev-notes/ch07.md`. Do not implement
the token or graph changes until this is complete.

If Context7 is not available in the current environment, stop and report the
missing tool instead of inventing signatures.

- [ ] **Step 2: Write the failing default-profile budget test**

Add to `tests/ch07/test_context_budget.py`:

```python
def test_default_cost_profile_uses_soft_history_budget():
    budget = ContextBudget(
        model_context_window=65536,
        max_output_tokens=2000,
        max_user_input_tokens=2000,
        max_agent_steps=3,
        tool_call_message_max_tokens=400,
        tool_result_max_tokens=1200,
        rerank_top_k=10,
        evidence_chunk_token_budget=400,
        evidence_total_max_tokens=3000,
        system_prompt_token_budget=519,
        summary_injection_token_budget=600,
        summary_segment_max_tokens=200,
        safety_margin_tokens=350,
        desired_retained_turns=24,
        steady_turn_token_estimate=800,
    )

    assert budget.evidence_budget == 3000
    assert budget.react_peak == 4824
    assert budget.hard_history_budget == 52243
    assert budget.soft_history_budget == 19200
    assert budget.history_budget == 19200
    assert budget.layer1_budget == 13439
    assert budget.layer2_budget == 5760
```

- [ ] **Step 3: Run the budget test and verify it fails**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/ch07/test_context_budget.py -q -p no:cacheprovider
```

Expected: FAIL because `ContextBudget` does not yet accept the new fields.

- [ ] **Step 4: Add the failing legacy compatibility test**

Add:

```python
def test_legacy_acceptance_profile_preserves_ch07_numbers():
    budget = ContextBudget(
        model_context_window=18000,
        max_output_tokens=2000,
        max_user_input_tokens=2000,
        max_agent_steps=3,
        tool_call_message_max_tokens=400,
        tool_result_max_tokens=1200,
        rerank_top_k=5,
        evidence_chunk_token_budget=400,
        evidence_total_max_tokens=None,
        system_prompt_token_budget=1800,
        summary_injection_token_budget=600,
        summary_segment_max_tokens=200,
        safety_margin_tokens=350,
        desired_retained_turns=24,
        steady_turn_token_estimate=800,
        legacy_acceptance=True,
    )

    assert budget.evidence_budget == 2000
    assert budget.react_peak == 3600
    assert budget.hard_history_budget == 5650
    assert budget.soft_history_budget == 19200
    assert budget.history_budget == 5650
```

- [ ] **Step 5: Implement configuration fields**

Add or update the fields in `app/config.py`:

```python
context_budget_profile: str = "cost_optimized"
history_soft_budget_enabled: bool = True
history_layer1_ratio: float = 0.70
history_layer2_ratio: float = 0.30

tool_call_message_max_tokens: int = 400
evidence_total_max_tokens: int = 3000
summary_segment_max_tokens: int = 200
```

Keep the existing values:

```python
model_context_window: int = 65536
max_user_input_tokens: int = 2000
tool_result_max_tokens: int = 1200
system_prompt_token_budget: int = 519
evidence_chunk_token_budget: int = 400
summary_injection_token_budget: int = 600
safety_margin_tokens: int = 350
desired_retained_turns: int = 24
steady_turn_token_estimate: int = 800
```

Do not delete `agent_token_budget` yet. Mark it deprecated in a comment and
remove its use from the new Agent path in Task 5.

- [ ] **Step 6: Implement the budget properties**

Extend `ContextBudget` with the new fields and properties:

```python
@property
def evidence_budget(self) -> int:
    if self.evidence_total_max_tokens is not None:
        return self.evidence_total_max_tokens
    return self.rerank_top_k * self.evidence_chunk_token_budget

@property
def react_peak(self) -> int:
    if self.legacy_acceptance:
        return self.max_agent_steps * self.tool_result_max_tokens
    return self.max_agent_steps * (
        self.tool_call_message_max_tokens
        + self.tool_result_max_tokens
        + 8
    )

@property
def fixed_overhead(self) -> int:
    return (
        self.system_prompt_token_budget
        + self.evidence_budget
        + self.summary_injection_token_budget
        + self.max_output_tokens
        + self.max_user_input_tokens
        + self.react_peak
        + self.safety_margin_tokens
    )

@property
def hard_history_budget(self) -> int:
    return max(0, self.model_context_window - self.fixed_overhead)

@property
def soft_history_budget(self) -> int:
    return self.desired_retained_turns * self.steady_turn_token_estimate

@property
def history_budget(self) -> int:
    if not self.history_soft_budget_enabled:
        return self.hard_history_budget
    return min(self.soft_history_budget, self.hard_history_budget)
```

Keep the existing layer split formula:

```python
layer1_budget = int((history_budget - 1) * 0.70)
layer2_budget = history_budget - 1 - layer1_budget
```

- [ ] **Step 7: Add profile construction in `from_settings()`**

`ContextBudget.from_settings()` must read:

```python
legacy_acceptance=settings.context_budget_profile == "legacy_acceptance"
history_soft_budget_enabled=settings.history_soft_budget_enabled
tool_call_message_max_tokens=settings.tool_call_message_max_tokens
evidence_total_max_tokens=settings.evidence_total_max_tokens
summary_segment_max_tokens=settings.summary_segment_max_tokens
```

The legacy profile must use the configured system budget, rerank-derived
evidence budget, and old ReAct formula so the Ch07 acceptance numbers remain
stable.

In the legacy profile, override the production `system_prompt_token_budget`
to `1800` and force `evidence_total_max_tokens=None` before constructing
`ContextBudget`. This keeps the historical `12350` fixed overhead and the
`5650 / 3954 / 1695` acceptance result.

- [ ] **Step 8: Run the budget tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/ch07/test_context_budget.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 9: Append Task 1 to `dev-notes/ch07.md` and commit**

Record the Context7 output, exact budget numbers, and any version mismatch.

```powershell
git add app/config.py .env.example app/context/budget.py tests/ch07/test_context_budget.py dev-notes/ch07.md
git commit -m "feat: add hard and soft context budget profiles"
```

---

### Task 2: Actual Token Counting and Bounded Truncation

**Files:**

- Modify: `app/context/tokens.py`
- Modify: `tests/ch07/test_token_estimator.py`
- Create: `tests/ch07/test_token_limits.py`

**Interfaces:**

- Consumes: `estimate_text_tokens`, `estimate_messages_tokens`.
- Produces:
  - `count_prompt_tokens(messages, tools=None) -> int`
  - `truncate_text_to_tokens(text, max_tokens, chinese_tokens_per_char) -> str`
  - `truncate_tool_content(content, max_tokens, chinese_tokens_per_char) -> str`

- [ ] **Step 1: Write the failing token-limit tests**

Create `tests/ch07/test_token_limits.py`:

```python
from langchain_core.messages import HumanMessage, SystemMessage

from app.context.tokens import (
    count_prompt_tokens,
    truncate_text_to_tokens,
    truncate_tool_content,
)


def test_full_prompt_count_includes_system_and_messages():
    prompt = [
        SystemMessage("固定客服人设"),
        HumanMessage("订单1001要退款"),
    ]
    assert count_prompt_tokens(prompt) > 0


def test_text_truncation_respects_budget():
    text = "订单1001要退款" * 100
    shortened = truncate_text_to_tokens(text, 20, 0.8)
    assert len(shortened) < len(text)
    assert shortened


def test_tool_content_truncation_respects_budget():
    content = '{"order_id":"1001","items":["' + "x" * 2000 + '"]}'
    shortened = truncate_tool_content(content, 40, 0.8)
    assert len(shortened) < len(content)
    assert shortened.endswith("...")
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/ch07/test_token_limits.py -q -p no:cacheprovider
```

Expected: FAIL because the new helpers do not exist.

- [ ] **Step 3: Implement actual prompt counting**

Use LangChain's current counter for the full request:

```python
from langchain_core.messages.utils import count_tokens_approximately


def count_prompt_tokens(
    messages: Sequence[BaseMessage],
    *,
    tools: Sequence[BaseTool | dict[str, Any]] | None = None,
) -> int:
    return count_tokens_approximately(list(messages), tools=tools)
```

Keep the CJK estimator for planning and deterministic truncation:

```python
def truncate_text_to_tokens(
    text: str,
    max_tokens: int,
    chinese_tokens_per_char: float,
) -> str:
    if max_tokens <= 0:
        return ""
    if estimate_text_tokens(text, chinese_tokens_per_char) <= max_tokens:
        return text

    suffix = "..."
    content_budget = max(
        0,
        max_tokens
        - estimate_text_tokens(suffix, chinese_tokens_per_char),
    )
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_text_tokens(
            text[:middle],
            chinese_tokens_per_char,
        ) <= content_budget:
            low = middle
        else:
            high = middle - 1
    return text[:low].rstrip() + suffix
```

`truncate_tool_content` is an alias with a clearer name for tool payloads:

```python
def truncate_tool_content(
    content: str,
    max_tokens: int,
    chinese_tokens_per_char: float,
) -> str:
    return truncate_text_to_tokens(
        content,
        max_tokens,
        chinese_tokens_per_char,
    )
```

- [ ] **Step 4: Run token and context tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/ch07/test_token_estimator.py tests/ch07/test_token_limits.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 5: Append Task 2 to `dev-notes/ch07.md` and commit**

```powershell
git add app/context/tokens.py tests/ch07/test_token_estimator.py tests/ch07/test_token_limits.py dev-notes/ch07.md
git commit -m "feat: add actual token counting and truncation"
```

---

### Task 3: Current User, Summary Caps, and Fixed Order Assembly

**Files:**

- Modify: `app/context/models.py`
- Modify: `app/context/trimmer.py`
- Modify: `app/context/assembler.py`
- Modify: `app/context/manager.py`
- Modify: `tests/ch07/test_context_trimmer.py`
- Modify: `tests/ch07/test_context_assembler.py`
- Modify: `tests/ch07/test_context_manager.py`

**Interfaces:**

- Consumes: `ContextBudget`, `ContextMessage`, `SummaryRow`.
- Produces:
  - `ContextPack.current_user_message`
  - `ContextPack.summary_injection_tokens`
  - `build_context_pack` gains `current_user_message_id: int | None`.
  - `build_model_messages` gains `current_user: BaseMessage | None`.

- [ ] **Step 1: Write failing current-user and summary-cap tests**

Add to `tests/ch07/test_context_trimmer.py`:

```python
def test_current_user_is_not_part_of_layer1():
    messages = [
        ContextMessage(1, "user", "第一轮问题"),
        ContextMessage(2, "assistant", "第一轮回答"),
        ContextMessage(3, "user", "当前问题"),
    ]
    pack = build_context_pack(
        messages=messages,
        summaries=[],
        layer1_from_msg_id=None,
        summary_upto_msg_id=0,
        current_user_message_id=3,
        budget=budget(100, 100),
        chinese_tokens_per_char=0.8,
        assistant_layer2_chars=80,
    )

    assert pack.current_user_message is not None
    assert pack.current_user_message.content == "当前问题"
    assert all(
        "当前问题" not in message.content
        for message in pack.layer1_messages
    )
```

Add a summary injection cap test:

```python
def test_summary_injection_keeps_latest_complete_segments():
    summaries = [
        SummaryRow(1, 2, "旧摘要" * 50),
        SummaryRow(3, 4, "新摘要"),
    ]
    pack = build_context_pack(
        messages=[
            ContextMessage(1, "user", "旧问题"),
            ContextMessage(2, "assistant", "旧回答"),
            ContextMessage(3, "user", "新问题"),
            ContextMessage(4, "assistant", "新回答"),
        ],
        summaries=summaries,
        layer1_from_msg_id=None,
        summary_upto_msg_id=4,
        current_user_message_id=None,
        budget=budget(100, 100),
        chinese_tokens_per_char=0.8,
        assistant_layer2_chars=80,
    )

    assert "新摘要" in pack.summaries_text
    assert pack.summary_injection_tokens <= 600
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/ch07/test_context_trimmer.py -q -p no:cacheprovider
```

Expected: FAIL because `build_context_pack` has no current-user or summary
selection behavior.

- [ ] **Step 3: Extend the context data contracts**

In `app/context/models.py`, add:

```python
@dataclass(frozen=True)
class PromptUsage:
    full_prompt_tokens: int
    system_tokens: int
    tool_definition_tokens: int
    summary_tokens: int
    evidence_tokens: int
    layer1_tokens: int
    layer2_tokens: int
    current_user_tokens: int
    react_peak_reserved: int
    output_reserved: int
    safety_margin_tokens: int
    history_soft_budget: int
    history_hard_budget: int
```

Add defaulted fields to `ContextPack`:

```python
current_user_message: BaseMessage | None = None
summary_injection_tokens: int = 0
prompt_usage: PromptUsage | None = None
```

- [ ] **Step 4: Exclude the current user from historical layers**

In `build_context_pack`, add `current_user_message_id: int | None`.
Find the matching `ContextMessage` before splitting layers:

```python
current_user = next(
    (
        item
        for item in ordered
        if item.message_id == current_user_message_id
    ),
    None,
)
historical = [
    item
    for item in ordered
    if item.message_id != current_user_message_id
]
```

Use `historical` for layer 1, layer 2, and summary-span calculations. Keep
the original ordered list only for summary snapshot selection.

- [ ] **Step 5: Cap summary injection**

Implement a helper:

```python
def _select_summaries(
    summaries: list[SummaryRow],
    max_tokens: int,
    chinese_tokens_per_char: float,
) -> tuple[str, int]:
    selected: list[str] = []
    used = 0
    for row in reversed(summaries):
        tokens = estimate_text_tokens(
            row.content,
            chinese_tokens_per_char,
        )
        if used + tokens > max_tokens:
            break
        selected.insert(0, row.content)
        used += tokens
    return "\n".join(selected), used
```

Use the selected text in `ContextPack.summaries_text` and in
`history_ctx`.

- [ ] **Step 6: Make assembly order explicit**

Change `build_model_messages` to accept:

```python
current_user: BaseMessage | None = None
```

Build:

```python
messages = [SystemMessage(fixed_system)]
messages.extend(pack.layer2_messages)
messages.extend(pack.layer1_messages)
if current_user is not None:
    messages.append(current_user)
supplement = _supplement_text(
    pack,
    evidence,
    order_data,
    after_sales_action,
)
if supplement:
    messages.append(HumanMessage(supplement))
return messages
```

No summary or evidence may be appended before the current user.

- [ ] **Step 7: Pass the current user through `ContextManager.prepare`**

In `ContextManager.prepare`, call:

```python
build_context_pack(
    messages=context_messages,
    summaries=summary_rows,
    layer1_from_msg_id=conversation.layer1_from_msg_id,
    summary_upto_msg_id=conversation.summary_upto_msg_id,
    current_user_message_id=current_user_message_id,
)
```

The returned pack is the only source for the current user in the Agent path.

- [ ] **Step 8: Run focused tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/ch07/test_context_trimmer.py tests/ch07/test_context_assembler.py tests/ch07/test_context_manager.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 9: Append Task 3 to `dev-notes/ch07.md` and commit**

```powershell
git add app/context/models.py app/context/trimmer.py app/context/assembler.py app/context/manager.py tests/ch07/test_context_trimmer.py tests/ch07/test_context_assembler.py tests/ch07/test_context_manager.py dev-notes/ch07.md
git commit -m "feat: cap summaries and separate current user context"
```

---

### Task 4: Summary Segment Limits and Structured Context Logging

**Files:**

- Modify: `app/context/summary.py`
- Modify: `app/context/tasks.py`
- Modify: `app/context/logging.py`
- Modify: `tests/ch07/test_summary_service.py`
- Modify: `tests/ch07/test_summary_tasks.py`
- Modify: `tests/ch07/test_context_logging.py`

**Interfaces:**

- Consumes: `truncate_text_to_tokens`, `PromptUsage`.
- Produces:
  - `summarize_span` gains required `max_tokens` and
    `chinese_tokens_per_char` arguments.
  - summary lifecycle logs with token-boundary data.
  - `log_model_ctx` gains keyword-only `usage: PromptUsage | None`.

- [ ] **Step 1: Write the failing summary token-limit test**

Add to `tests/ch07/test_summary_service.py`:

```python
@pytest.mark.asyncio
async def test_summary_body_is_token_bounded():
    class LongSummaryModel:
        async def ainvoke(self, messages):
            return AIMessage("订单1001要退款。" * 100)

    result = await summarize_span(
        [
            ContextMessage(1, "user", "订单1001要退款"),
            ContextMessage(2, "assistant", "已登记"),
        ],
        LongSummaryModel(),
        min_chars=20,
        max_chars=200,
        max_tokens=30,
        chinese_tokens_per_char=0.8,
    )

    assert estimate_text_tokens(result, 0.8) <= 30
    assert result
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/ch07/test_summary_service.py -q -p no:cacheprovider
```

Expected: FAIL because `summarize_span` does not accept token limits.

- [ ] **Step 3: Implement summary token truncation**

Extend `summarize_span`:

```python
async def summarize_span(
    messages: Sequence[ContextMessage],
    model: BaseChatModel,
    *,
    min_chars: int,
    max_chars: int,
    max_tokens: int,
    chinese_tokens_per_char: float,
) -> str:
    response = await model.ainvoke(
        [HumanMessage(content=build_summary_prompt(messages))]
    )
    body = _strip_code_fence(
        _content_text(getattr(response, "content", "") or "")
    )
    if not body:
        raise ValueError("summary model returned an empty body")
    if len(body) > max_chars:
        body = body[:max_chars].rstrip()
    body = truncate_text_to_tokens(
        body,
        max_tokens,
        chinese_tokens_per_char,
    )
    if len(body) < min_chars:
        raise ValueError("summary body is too short")
    return body
```

Keep the existing char limit as a prompt/shape guard.

- [ ] **Step 4: Pass the limit from `SummaryTaskManager`**

Add constructor arguments:

```python
max_tokens: int,
chinese_tokens_per_char: float,
```

Pass them to `summarize_span` in `_run` and include the applied token cap in
the `summary done` log.

- [ ] **Step 5: Add component-aware model logging**

Extend `log_model_ctx`:

```python
def log_model_ctx(
    session_id: str,
    messages: Sequence[BaseMessage],
    token_estimate: int,
    summary_text: str = "",
    *,
    usage: PromptUsage | None = None,
) -> None:
    _log_context(
        "model_ctx",
        session_id,
        messages,
        token_estimate,
        summary_text,
    )
    if usage is None:
        return
    logger.info(
        "model_ctx_usage session_id=%s full_prompt_tokens=%d "
        "system_tokens=%d tool_definition_tokens=%d summary_tokens=%d "
        "evidence_tokens=%d layer1_tokens=%d layer2_tokens=%d "
        "current_user_tokens=%d react_peak_reserved=%d output_reserved=%d "
        "safety_margin=%d history_soft_budget=%d history_hard_budget=%d",
        session_id,
        usage.full_prompt_tokens,
        usage.system_tokens,
        usage.tool_definition_tokens,
        usage.summary_tokens,
        usage.evidence_tokens,
        usage.layer1_tokens,
        usage.layer2_tokens,
        usage.current_user_tokens,
        usage.react_peak_reserved,
        usage.output_reserved,
        usage.safety_margin_tokens,
        usage.history_soft_budget,
        usage.history_hard_budget,
    )
```

When `usage` is present, append these named integer fields to the usage log:
`full_prompt_tokens`, `system_tokens`, `tool_definition_tokens`,
`summary_tokens`, `evidence_tokens`, `layer1_tokens`, `layer2_tokens`,
`current_user_tokens`, `react_peak_reserved`, `output_reserved`,
`safety_margin`, `history_soft_budget`, and `history_hard_budget`.

Keep `summary_json`, `messages_json`, `message_count`, and
`token_estimate` unchanged for existing grep compatibility.

- [ ] **Step 6: Run focused tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/ch07/test_summary_service.py tests/ch07/test_summary_tasks.py tests/ch07/test_context_logging.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 7: Append Task 4 to `dev-notes/ch07.md` and commit**

```powershell
git add app/context/summary.py app/context/tasks.py app/context/logging.py tests/ch07/test_summary_service.py tests/ch07/test_summary_tasks.py tests/ch07/test_context_logging.py dev-notes/ch07.md
git commit -m "feat: bound summaries and log complete context usage"
```

---

### Task 5: Replace the Agent Token Gate and Truncate ReAct Payloads

**Files:**

- Modify: `app/agents/bare_react.py`
- Modify: `app/graph/agent.py`
- Modify: `app/graph/builder.py`
- Modify: `app/main.py`
- Modify: `tests/test_graph_agent.py`
- Modify: `tests/test_graph_workflow.py`
- Modify: `tests/ch07/test_context_graph.py`

**Interfaces:**

- Consumes: `ContextBudget`, `count_prompt_tokens`, `truncate_tool_content`.
- Produces:
  - `AgentLimits.full_prompt_token_budget`
  - `AgentLimits.tool_call_message_max_tokens`
  - `AgentLimits.tool_result_max_tokens`
  - one full-request guard before each Agent model call.

- [ ] **Step 1: Write the failing full-request-gate tests**

Add to `tests/test_graph_agent.py`:

```python
def test_agent_uses_full_prompt_budget_not_legacy_token_budget(monkeypatch):
    captured = {}

    monkeypatch.setattr(settings, "agent_token_budget", 1)
    monkeypatch.setattr(settings, "model_context_window", 65536)
    monkeypatch.setattr(settings, "agent_max_output_tokens", 2000)

    def builder(model, limits):
        captured["limits"] = limits
        return RunnableLambda(lambda state: {"reply": "ok", "agent_steps": 1})

    build_chat_graph(
        intent_model=NoUseModel(),
        agent_model=NoUseModel(),
        intent_classifier=lambda text, model: None,
        agent_builder=builder,
        turn_logger=FakeTurnLogger(),
        checkpointer=InMemorySaver(),
    )

    assert captured["limits"].full_prompt_token_budget == 63186
```

Add a ReAct payload test:

```python
import json

from langchain_core.messages import AIMessage

from app.graph.agent import _truncate_tool_call_message


def test_tool_call_message_is_truncated_for_model_context():
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "query_order",
                "args": {"note": "x" * 10000},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )

    shortened = _truncate_tool_call_message(
        message,
        max_tokens=20,
        chinese_tokens_per_char=0.8,
    )

    assert len(
        json.dumps(
            shortened.tool_calls[0]["args"],
            ensure_ascii=False,
        )
    ) < 10000
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_graph_agent.py -q -p no:cacheprovider
```

Expected: FAIL because `AgentLimits` has no full-request fields and tools do
not truncate content.

- [ ] **Step 3: Extend `AgentLimits`**

Add fields while keeping compatibility defaults:

```python
@dataclass(frozen=True)
class AgentLimits:
    max_steps: int = 5
    max_output_tokens: int = 800
    token_budget: int = 4000  # deprecated compatibility field
    full_prompt_token_budget: int = 0
    tool_call_message_max_tokens: int = 400
    tool_result_max_tokens: int = 1200
```

When `full_prompt_token_budget` is zero, derive it from the legacy gate for
older callers.

- [ ] **Step 4: Implement the full-request guard in `call_agent`**

Replace:

```python
if count_tokens_approximately(state["agent_messages"]) >= limits.token_budget:
```

with:

```python
full_prompt_tokens = count_prompt_tokens(
    state["agent_messages"],
    tools=AGENT_TOOLS,
)
if full_prompt_tokens > limits.full_prompt_token_budget:
    writer({"type": "delta", "text": LIMIT_REPLY})
    return {
        "reply": LIMIT_REPLY,
        "stop_reason": "token_budget",
    }
```

`token_budget` remains only for old direct-call compatibility and must not
be the production guard.

- [ ] **Step 5: Truncate tool call arguments and results**

In `run_tools`, before building the `ToolMessage`:

```python
content = truncate_tool_content(
    _tool_content(execution),
    limits.tool_result_max_tokens,
    settings.chinese_tokens_per_char,
)
```

When recording `tool_trace`, store the truncated content or a length-limited
summary, not the unbounded original.

Tool arguments are still executed with their original values so a safety
limit cannot silently change the business operation. Before returning the
updated `agent_messages`, replace the last assistant message with a
model-visible copy whose serialized `tool_calls` arguments are truncated to
`tool_call_message_max_tokens`. This keeps the executed call correct while
preventing an oversized argument payload from being retained in the model
context.

```python
def _truncate_tool_call_message(
    message: AIMessage,
    max_tokens: int,
    chinese_tokens_per_char: float,
) -> AIMessage:
    truncated_calls = []
    for call in message.tool_calls:
        args_text = json.dumps(call.get("args") or {}, ensure_ascii=False)
        truncated_calls.append(
            {
                **call,
                "args": {
                    "_truncated": truncate_tool_content(
                        args_text,
                        max_tokens,
                        chinese_tokens_per_char,
                    )
                },
            }
        )
    return message.model_copy(update={"tool_calls": truncated_calls})
```

Return the truncated assistant copy together with the `ToolMessage` results
so the next model call sees bounded tool-call arguments.

- [ ] **Step 6: Build full-request limits in the graph builder**

In `build_chat_graph`:

```python
full_prompt_token_budget = (
    settings.model_context_window
    - settings.agent_max_output_tokens
    - settings.safety_margin_tokens
)
limits = limits or AgentLimits(
    max_steps=settings.agent_max_steps,
    max_output_tokens=settings.agent_max_output_tokens,
    full_prompt_token_budget=full_prompt_token_budget,
    tool_call_message_max_tokens=settings.tool_call_message_max_tokens,
    tool_result_max_tokens=settings.tool_result_max_tokens,
)
```

Keep `agent_token_budget` only as a deprecated input alias for callers that
still construct `AgentLimits` directly.

- [ ] **Step 7: Wire the budget resource in `main.py`**

Replace `agent_token_budget=settings.agent_token_budget` in the production
Agent limits with the full-request budget derived from `ContextBudget`.

Log the full request limit at startup:

```text
context_budget_profile=...
history_soft_budget=...
history_hard_budget=...
full_prompt_limit=...
```

- [ ] **Step 8: Run graph and Agent tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_graph_agent.py tests/test_graph_workflow.py tests/ch07/test_context_graph.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 9: Append Task 5 to `dev-notes/ch07.md` and commit**

```powershell
git add app/agents/bare_react.py app/graph/agent.py app/graph/builder.py app/main.py tests/test_graph_agent.py tests/test_graph_workflow.py tests/ch07/test_context_graph.py dev-notes/ch07.md
git commit -m "refactor: replace agent token gate with full request guard"
```

---

### Task 6: Compatibility Profile, Acceptance Demo, and Evaluation

**Files:**

- Modify: `app/config.py`
- Modify: `scripts/demo_ch07.py`
- Modify: `scripts/eval_ch07.py`
- Modify: `tests/ch07/test_context_acceptance.py`
- Modify: `tests/ch07/test_context_graph.py`
- Modify: `README.md`

**Interfaces:**

- Consumes: all budget, trimmer, logging, and Agent changes.
- Produces:
  - `CONTEXT_BUDGET_PROFILE=cost_optimized|legacy_acceptance`
  - deterministic default and acceptance demo profiles.
  - evaluation that includes the summary token cap.

- [ ] **Step 1: Write the failing default-profile behavior test**

Add to `tests/ch07/test_context_acceptance.py`:

```python
def test_default_profile_does_not_trigger_summary_for_short_session():
    budget = ContextBudget(
        model_context_window=65536,
        max_output_tokens=2000,
        max_user_input_tokens=2000,
        max_agent_steps=3,
        tool_call_message_max_tokens=400,
        tool_result_max_tokens=1200,
        rerank_top_k=10,
        evidence_chunk_token_budget=400,
        evidence_total_max_tokens=3000,
        system_prompt_token_budget=519,
        summary_injection_token_budget=600,
        summary_segment_max_tokens=200,
        safety_margin_tokens=350,
        desired_retained_turns=24,
        steady_turn_token_estimate=800,
    )

    assert budget.history_budget == 19200
    assert budget.layer1_budget + budget.layer2_budget < budget.history_budget
```

- [ ] **Step 2: Write the failing legacy-profile acceptance test**

Add:

```python
def test_legacy_profile_keeps_ch07_numbers():
    budget = ContextBudget(
        model_context_window=18000,
        max_output_tokens=2000,
        max_user_input_tokens=2000,
        max_agent_steps=3,
        tool_call_message_max_tokens=400,
        tool_result_max_tokens=1200,
        rerank_top_k=5,
        evidence_chunk_token_budget=400,
        evidence_total_max_tokens=None,
        system_prompt_token_budget=1800,
        summary_injection_token_budget=600,
        summary_segment_max_tokens=200,
        safety_margin_tokens=350,
        desired_retained_turns=24,
        steady_turn_token_estimate=800,
        legacy_acceptance=True,
    )

    assert budget.history_budget == 5650
    assert budget.layer1_budget == 3954
    assert budget.layer2_budget == 1695
```

- [ ] **Step 3: Run the acceptance tests and verify they fail**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/ch07/test_context_acceptance.py -q -p no:cacheprovider
```

Expected: FAIL until the profile and budget plumbing are complete.

- [ ] **Step 4: Update the demo profile switch**

`scripts/demo_ch07.py` must accept:

```text
--profile cost_optimized
--profile legacy_acceptance
```

The legacy profile exports the Ch07 values exactly. The cost profile uses the
default `64K / 24x800` values and prints:

```text
history_soft_budget=19200
history_hard_budget=...
full_prompt_limit=...
```

- [ ] **Step 5: Update the evaluation path**

`scripts/eval_ch07.py` must call `summarize_span` with:

```python
max_tokens=settings.summary_segment_max_tokens
chinese_tokens_per_char=settings.chinese_tokens_per_char
```

Keep the existing thresholds and failure report.

- [ ] **Step 6: Run the default and legacy demos**

Run:

```powershell
.venv\Scripts\python.exe scripts\demo_ch07.py --profile cost_optimized --turns 20 --session-id demo-cost-default
.venv\Scripts\python.exe scripts\demo_ch07.py --profile legacy_acceptance --turns 24 --session-id demo-cost-legacy --ask-start-order
```

Expected:

```text
cost_optimized:
  no summary trigger
  no layer1 downgrade
  full prompt bounded by hard_history_budget

legacy_acceptance:
  layer1 downgrade appears
  summary trigger appears
  summary done appears
  original order number and request are answered from summary
```

- [ ] **Step 7: Run summary evaluation**

Run:

```powershell
.venv\Scripts\python.exe scripts\eval_ch07.py
```

Expected:

```text
key_fact_recall >= 0.90
unsupported_fact_rate == 0
small_talk_leak_rate == 0
length_pass_rate >= 0.90
```

- [ ] **Step 8: Run focused verification**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/ch07 tests/test_graph_agent.py tests/test_graph_workflow.py tests/test_memory.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 9: Append Task 6 to `dev-notes/ch07.md` and commit**

Record both profile results, the exact prompt size observed, and any
compatibility correction.

```powershell
git add app/config.py scripts/demo_ch07.py scripts/eval_ch07.py tests/ch07/test_context_acceptance.py tests/ch07/test_context_graph.py README.md dev-notes/ch07.md
git commit -m "test: verify optimized and legacy context profiles"
```

---

### Task 7: Full Verification, Documentation, and Finish Notes

**Files:**

- Modify: `README.md` if commands changed.
- Modify: `dev-notes/ch07.md`
- No production code changes expected.

**Interfaces:**

- Consumes: all preceding tasks.
- Produces: final verification evidence and delivery notes.

- [ ] **Step 1: Run the complete backend suite**

Run:

```powershell
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Expected: all tests pass; record exact counts and warnings.

- [ ] **Step 2: Run compile and whitespace checks**

Run:

```powershell
.venv\Scripts\python.exe -m compileall -q app scripts tests/ch07
git diff --check
```

Expected: no errors.

- [ ] **Step 3: Run the context demos and evaluation**

Run:

```powershell
.venv\Scripts\python.exe scripts\eval_ch07.py
.venv\Scripts\python.exe scripts\demo_ch07.py --profile cost_optimized --turns 20 --session-id demo-final-cost
.venv\Scripts\python.exe scripts\demo_ch07.py --profile legacy_acceptance --turns 24 --session-id demo-final-legacy --ask-start-order
```

Expected: cost profile stays under the soft target; legacy profile emits
summary lifecycle logs and answers the original order question.

- [ ] **Step 4: Inspect the logs**

Run:

```powershell
Select-String -Path log\app.log -Pattern "model_ctx|history_ctx|summary trigger|summary done"
```

Confirm:

- every turn has `history_ctx`;
- every Agent call has `model_ctx`;
- `model_ctx` contains `full_prompt_tokens`;
- no summary is emitted as a system message;
- legacy acceptance logs show `layer1 downgrade`, `summary trigger`, and
  `summary done`.

- [ ] **Step 5: Append final verification to `dev-notes/ch07.md`**

Record:

- exact commands;
- pass/fail counts;
- summary metrics;
- default and legacy profile observations;
- any rejected or reworked approach;
- final commit hash.

- [ ] **Step 6: Commit final documentation**

```powershell
git add README.md dev-notes/ch07.md
git commit -m "docs: finish ch07 context cost optimization"
```

---

## Plan Self-Review

### Spec Coverage

- Hard/soft history budget: Task 1.
- Actual token counting and bounded truncation: Task 2.
- Current-user single count, summary injection cap, fixed assembly order:
  Task 3.
- Summary segment cap and complete logging: Task 4.
- Removal of the standalone Agent gate and ReAct payload limits:
  Task 5.
- Compatibility profile and acceptance demos: Task 6.
- Full verification and process notes: Task 7.

### Placeholder Scan

No `TBD`, `TODO`, deferred implementation, or unspecified test behavior is
present. Each task names exact files, interfaces, test intent, and a
verification command.

### Type Consistency

- `ContextBudget` uses the same hard/soft property names in Tasks 1, 3, 5,
  and 6.
- `PromptUsage` is defined in Task 3 and consumed in Tasks 4 and 5.
- `count_prompt_tokens` and `truncate_tool_content` are defined in Task 2
  and consumed in Task 5.
- `current_user_message`, `summary_injection_tokens`, and `prompt_usage` are
  defined in Task 3 and used consistently afterward.
- `legacy_acceptance` is explicitly preserved in Tasks 1, 4, and 6.

### Execution Handoff

Plan complete and saved to
`docs/superpowers/plans/2026-09-23-ch07-context-cost-optimization.md`.
