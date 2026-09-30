# iHelp Ch07 Context Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single fixed history trim with observable three-layer per-conversation context management, background append-only summaries, multi-conversation persistence, and a conversation sidebar.

**Architecture:** Keep the Ch06 LangGraph and checkpoint model. Add a focused `app/context` package for budget calculation, token estimation, trimming, assembly, summaries, and task ownership; integrate it through one `prepare_context` node and injected repositories. Persist only user/assistant turns and anchors in MySQL, leaving full checkpoint state independent from the reduced model input.

**Tech Stack:** Python 3.12, FastAPI, LangChain, LangGraph 1.0.x, SQLAlchemy 2 async, MySQL, pytest, native HTML/CSS/JavaScript, Vite.

**Spec:** `docs/superpowers/specs/2026-09-21-ch07-context-management-design.md`

## Global Constraints

- No semantic message retrieval, cross-session memory, user profile, or authentication system.
- Keep MySQL, SQLAlchemy, FastAPI, LangChain, LangGraph, and the existing chat model provider.
- Keep one LangGraph thread per `session_id`; both anchor columns reference `messages.id`.
- Preserve the complete `ChatState.messages`; the reduced context is derived and never assigned back.
- Query understanding and intent classification share `history_ctx`.
- The Agent receives the fixed-order `model_ctx`; summaries and evidence are never separate system messages.
- Tool calls and tool results are not written to `messages`.
- Summary tasks are process-local and non-blocking; deployment remains single-worker.
- Frontend work uses the repository's existing direct implementation and browser verification path.
- Context7 was checked at design time. Re-check dependency interfaces if the implementation changes dependency versions.

---

## File Structure

Create:

- `app/context/__init__.py`: public exports for the context package.
- `app/context/budget.py`: pure budget calculation and startup feasibility.
- `app/context/tokens.py`: calibrated CJK/ASCII/message token estimator.
- `app/context/models.py`: immutable context data contracts.
- `app/context/trimmer.py`: layer boundary selection and deterministic shortening.
- `app/context/assembler.py`: fixed-order model and history message construction.
- `app/context/manager.py`: repository-backed per-turn context preparation.
- `app/context/summary.py`: facts-only summary prompt and model call.
- `app/context/tasks.py`: non-blocking summary task ownership and lifecycle.
- `app/context/logging.py`: `model_ctx`, `history_ctx`, and summary log records.
- `app/api/conversations.py`: read-only conversation list and message APIs.
- `app/db/migrations.py`: additive Ch07 schema check.
- `tests/ch07/__init__.py`
- `tests/ch07/test_context_budget.py`
- `tests/ch07/test_token_estimator.py`
- `tests/ch07/test_conversation_repository.py`
- `tests/ch07/test_context_trimmer.py`
- `tests/ch07/test_context_assembler.py`
- `tests/ch07/test_summary_service.py`
- `tests/ch07/test_summary_tasks.py`
- `tests/ch07/test_context_graph.py`
- `tests/ch07/test_conversations_api.py`
- `tests/ch07/test_context_logging.py`
- `eval_data/ch07_context_summary_eval_set.json`
- `scripts/eval_ch07.py`
- `scripts/demo_ch07.py`

Modify:

- `app/config.py`
- `.env.example`
- `app/db/models.py`
- `app/db/repository.py`
- `app/db/session.py`
- `app/graph/state.py`
- `app/graph/builder.py`
- `app/graph/nodes.py`
- `app/graph/agent.py`
- `app/core/memory.py`
- `app/services/conversation_understanding.py`
- `app/graph/intent.py`
- `app/api/chat.py`
- `app/schemas/chat.py`
- `app/main.py`
- `app/services/turn_logger.py`
- `tests/test_chat_api.py`
- `tests/test_graph_workflow.py`
- `tests/test_graph_agent.py`
- `tests/test_graph_intent.py`
- `tests/ch06/test_conversation_understanding.py`
- `D:\std_selftest_frontend\index.html`
- `D:\std_selftest_frontend\src\main.js`
- `D:\std_selftest_frontend\src\styles.css`
- `README.md`
- `dev-notes/ch07.md`

---

### Task 1: Configuration, Budget Calibration, and Token Estimation

**Files:**
- Create: `app/context/__init__.py`
- Create: `app/context/budget.py`
- Create: `app/context/tokens.py`
- Create: `tests/ch07/__init__.py`
- Create: `tests/ch07/test_context_budget.py`
- Create: `tests/ch07/test_token_estimator.py`
- Modify: `app/config.py`
- Modify: `.env.example`

**Interfaces:**

- Consumes: `app.config.settings`.
- Produces: `ContextBudget.from_settings()`, `ContextBudget(...)`, `estimate_text_tokens(text, chinese_tokens_per_char)`, `estimate_message_tokens(message, chinese_tokens_per_char)`, `estimate_messages_tokens(messages, chinese_tokens_per_char)`.

- [ ] **Step 1: Write the failing budget calibration test**

```python
from app.context.budget import ContextBudget


def test_acceptance_budget_matches_required_numbers():
    budget = ContextBudget(
        model_context_window=18000,
        max_output_tokens=2000,
        max_user_input_tokens=2000,
        max_agent_steps=3,
        tool_result_max_tokens=1200,
        rerank_top_k=5,
        evidence_chunk_token_budget=400,
        system_prompt_token_budget=1800,
        summary_injection_token_budget=600,
        safety_margin_tokens=350,
        desired_retained_turns=24,
        steady_turn_token_estimate=800,
    )

    assert budget.fixed_overhead == 12350
    assert budget.history_budget == 5650
    assert budget.layer1_budget == 3954
    assert budget.layer2_budget == 1695


def test_only_shrinking_window_with_defaults_is_insufficient():
    budget = ContextBudget(
        model_context_window=18000,
        max_output_tokens=800,
        max_user_input_tokens=2000,
        max_agent_steps=5,
        tool_result_max_tokens=1200,
        rerank_top_k=20,
        evidence_chunk_token_budget=400,
        system_prompt_token_budget=1800,
        summary_injection_token_budget=600,
        safety_margin_tokens=350,
        desired_retained_turns=24,
        steady_turn_token_estimate=800,
    )

    assert budget.history_budget == 0
    assert budget.can_fit_one_turn is False
```

- [ ] **Step 2: Run the budget test and verify it fails**

Run: `uv run pytest tests/ch07/test_context_budget.py -q`

Expected: FAIL because `app.context.budget` does not exist.

- [ ] **Step 3: Write the failing estimator tests**

```python
from langchain_core.messages import AIMessage, HumanMessage

from app.context.tokens import estimate_messages_tokens, estimate_text_tokens


def test_cjk_is_not_folded_like_ascii():
    chinese = estimate_text_tokens("订单1001要退款", 0.80)
    ascii_text = estimate_text_tokens("order1001refund", 0.80)
    assert chinese > ascii_text


def test_message_overhead_is_included():
    one = estimate_messages_tokens([HumanMessage("你好")], 0.80)
    assert one > estimate_text_tokens("你好", 0.80)


def test_mixed_summary_estimate_is_stable():
    estimate = estimate_messages_tokens(
        [HumanMessage("订单1001"), AIMessage("已登记退款")],
        0.80,
    )
    assert 10 <= estimate <= 30
```

- [ ] **Step 4: Run estimator tests and verify they fail**

Run: `uv run pytest tests/ch07/test_token_estimator.py -q`

Expected: FAIL because `app.context.tokens` does not exist.

- [ ] **Step 5: Add configuration fields**

```python
from pydantic import AliasChoices, Field

model_context_window: int = 65536
max_user_input_tokens: int = 2000
tool_result_max_tokens: int = 1200
agent_max_steps: int = Field(
    default=5,
    validation_alias=AliasChoices("MAX_AGENT_STEPS", "AGENT_MAX_STEPS"),
)
agent_max_output_tokens: int = Field(
    default=800,
    validation_alias=AliasChoices(
        "MAX_OUTPUT_TOKENS",
        "AGENT_MAX_OUTPUT_TOKENS",
    ),
)
system_prompt_token_budget: int = 1800
evidence_chunk_token_budget: int = 400
summary_injection_token_budget: int = 600
safety_margin_tokens: int = 350
desired_retained_turns: int = 24
steady_turn_token_estimate: int = 800
chinese_tokens_per_char: float = 0.80
assistant_layer2_chars: int = 80
summary_target_min_chars: int = 20
summary_target_max_chars: int = 200
context_log_path: str = "log/app.log"
```

Set the default `rerank_top_k` to `20` in `Settings`. Keep the acceptance
override `RERANK_TOP_K=5` in the demonstration command.

- [ ] **Step 6: Implement the budget**

```python
@dataclass(frozen=True)
class ContextBudget:
    model_context_window: int
    max_output_tokens: int
    max_user_input_tokens: int
    max_agent_steps: int
    tool_result_max_tokens: int
    rerank_top_k: int
    evidence_chunk_token_budget: int
    system_prompt_token_budget: int
    summary_injection_token_budget: int
    safety_margin_tokens: int
    desired_retained_turns: int
    steady_turn_token_estimate: int

    @property
    def react_peak(self) -> int:
        return self.max_agent_steps * self.tool_result_max_tokens

    @property
    def fixed_overhead(self) -> int:
        return (
            self.system_prompt_token_budget
            + self.rerank_top_k * self.evidence_chunk_token_budget
            + self.summary_injection_token_budget
            + self.max_output_tokens
            + self.max_user_input_tokens
            + self.react_peak
            + self.safety_margin_tokens
        )

    @property
    def window_available(self) -> int:
        return max(0, self.model_context_window - self.fixed_overhead)

    @property
    def desired_history(self) -> int:
        return self.desired_retained_turns * self.steady_turn_token_estimate

    @property
    def history_budget(self) -> int:
        return min(self.desired_history, self.window_available)

    @property
    def layer1_budget(self) -> int:
        if self.history_budget <= 1:
            return 0
        return int((self.history_budget - 1) * 0.70)

    @property
    def layer2_budget(self) -> int:
        if self.history_budget <= 1:
            return 0
        return self.history_budget - 1 - self.layer1_budget

    @property
    def can_fit_one_turn(self) -> bool:
        return self.history_budget > 0
```

`from_settings()` passes `settings.agent_max_steps`,
`settings.agent_max_output_tokens`, and `settings.rerank_top_k` into the
dataclass.

- [ ] **Step 7: Implement the token estimator**

```python
CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
MESSAGE_OVERHEAD = 4


def estimate_text_tokens(text: str, chinese_tokens_per_char: float) -> int:
    if not text:
        return 0
    cjk_count = len(CJK_PATTERN.findall(text))
    other_count = len(text) - cjk_count
    estimate = cjk_count * chinese_tokens_per_char + other_count / 4
    return max(1, math.ceil(estimate))


def estimate_message_tokens(
    message: BaseMessage,
    chinese_tokens_per_char: float,
) -> int:
    content = message.content
    text = content if isinstance(content, str) else str(content)
    return MESSAGE_OVERHEAD + estimate_text_tokens(
        text,
        chinese_tokens_per_char,
    )
```

- [ ] **Step 8: Run focused tests**

Run: `uv run pytest tests/ch07/test_context_budget.py tests/ch07/test_token_estimator.py -q`

Expected: PASS.

- [ ] **Step 9: Append Task 1 to `dev-notes/ch07.md`**

Record the exact user requirement for budget calibration, created files,
test result, rejected hard-coded token constants, and any failure.

- [ ] **Step 10: Commit**

```bash
git add app/config.py .env.example app/context tests/ch07 dev-notes/ch07.md
git commit -m "feat: add calibrated context budget"
```

---

### Task 2: Conversation Anchors, Summary Table, and Repository

**Files:**
- Modify: `app/db/models.py`
- Modify: `app/db/repository.py`
- Modify: `app/db/session.py`
- Create: `app/db/migrations.py`
- Create: `tests/ch07/test_conversation_repository.py`

**Interfaces:**

- Consumes: `AsyncSessionLocal`, ORM models, `Message.id`.
- Produces:
  - `ensure_conversation(session_id, user_key=None) -> Conversation`
  - `get_conversation(session_id) -> Conversation | None`
  - `append_message(...) -> int`
  - `load_messages(session_id) -> list[Message]`
  - `load_summaries(session_id) -> list[ConversationSummary]`
  - `update_layer1_anchor(session_id, message_id) -> None`
  - `append_summary_and_advance(session_id, from_id, to_id, content, token_estimate) -> ConversationSummary`
  - `list_conversations(user_key) -> list[dict]`
  - `load_conversation_messages(conversation_id, user_key) -> list[dict] | None`
  - `ConversationOwnershipError`

- [ ] **Step 1: Write failing repository tests**

```python
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db import repository
from app.db.base import Base


@pytest.fixture
async def sqlite_repository(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    monkeypatch.setattr(repository, "AsyncSessionLocal", maker)
    yield repository
    await engine.dispose()


async def test_append_message_returns_database_id(sqlite_repository):
    await sqlite_repository.ensure_conversation("s1", "u1")
    message_id = await sqlite_repository.append_message("s1", "user", "你好")
    assert message_id > 0


async def test_summary_append_advances_anchor_once(sqlite_repository):
    await sqlite_repository.ensure_conversation("s1", "u1")
    first = await sqlite_repository.append_message("s1", "user", "订单 1001")
    second = await sqlite_repository.append_message("s1", "assistant", "已查询")
    await sqlite_repository.append_summary_and_advance(
        "s1", first, second, "用户询问订单 1001。", 8
    )
    await sqlite_repository.append_summary_and_advance(
        "s1", first, second, "用户询问订单 1001。", 8
    )
    summaries = await sqlite_repository.load_summaries("s1")
    assert len(summaries) == 1
```

- [ ] **Step 2: Run repository tests and verify they fail**

Run: `uv run pytest tests/ch07/test_conversation_repository.py -q`

Expected: FAIL because the summary model and repository interfaces do not exist.

- [ ] **Step 3: Extend the ORM models**

```python
class Conversation(Base):
    # existing fields remain
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        index=True,
    )
    summary_upto_msg_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    layer1_from_msg_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )


class ConversationSummary(Base):
    __tablename__ = "conversation_summaries"
    __table_args__ = (
        UniqueConstraint(
            "conversation_id",
            "covered_from_msg_id",
            "covered_to_msg_id",
            name="uq_conversation_summary_range",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("conversations.id"),
        index=True,
    )
    sequence_no: Mapped[int] = mapped_column(Integer)
    covered_from_msg_id: Mapped[int] = mapped_column(Integer)
    covered_to_msg_id: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    token_estimate: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        index=True,
    )
```

- [ ] **Step 4: Add the additive migration**

`ensure_ch07_schema()` inspects `conversations` and executes only missing
column DDL:

```python
ALTER TABLE conversations ADD COLUMN updated_at DATETIME NULL
ALTER TABLE conversations ADD COLUMN summary_upto_msg_id INTEGER NULL
ALTER TABLE conversations ADD COLUMN layer1_from_msg_id INTEGER NULL
```

Call it after `Base.metadata.create_all` in `init_db`. Do not drop columns or
rewrite existing tool rows.

- [ ] **Step 5: Implement repository writes and summary transaction**

`ensure_conversation` loads by primary key, inserts when missing, and raises
`ConversationOwnershipError` when an explicit `user_key` differs.

`append_summary_and_advance`:

```python
existing = await db.scalar(
    select(ConversationSummary).where(
        ConversationSummary.conversation_id == session_id,
        ConversationSummary.covered_from_msg_id == from_id,
        ConversationSummary.covered_to_msg_id == to_id,
    )
)
if existing:
    return existing

max_sequence = await db.scalar(
    select(func.coalesce(func.max(ConversationSummary.sequence_no), 0)).where(
        ConversationSummary.conversation_id == session_id
    )
)
row = ConversationSummary(
    conversation_id=session_id,
    sequence_no=int(max_sequence or 0) + 1,
    covered_from_msg_id=from_id,
    covered_to_msg_id=to_id,
    content=content,
    token_estimate=token_estimate,
)
conversation = await db.get(Conversation, session_id)
conversation.summary_upto_msg_id = to_id
db.add(row)
await db.commit()
await db.refresh(row)
return row
```

The actual implementation wraps the read, insert, and anchor update in one
`async with db.begin()` transaction and catches the unique-range conflict by
reloading and returning the existing row.

- [ ] **Step 6: Implement list and message reads**

Use a scalar subquery for the first user message id, a correlated `exists()`
for `summarized`, and `updated_at DESC`:

```python
first_user_id = (
    select(func.min(Message.id))
    .where(
        Message.conversation_id == Conversation.id,
        Message.role == "user",
    )
    .scalar_subquery()
)
summarized = (
    select(ConversationSummary.id)
    .where(ConversationSummary.conversation_id == Conversation.id)
    .exists()
)
```

`load_messages` returns all persisted rows in ascending id order for context
assembly. `load_conversation_messages` joins the conversation to verify
`user_name == user_key`, filters `role IN ("user", "assistant")`, and returns
serializable rows. If `updated_at` is null for a pre-Ch07 row, return
`created_at` as the fallback value.

- [ ] **Step 7: Run repository and existing DB tests**

Run: `uv run pytest tests/ch07/test_conversation_repository.py tests/test_memory.py -q`

Expected: PASS.

- [ ] **Step 8: Append Task 2 to `dev-notes/ch07.md`**

Record the anchor/table design, exact tests, rejected message migration, and
any conflict/rework.

- [ ] **Step 9: Commit**

```bash
git add app/db tests/ch07/test_conversation_repository.py dev-notes/ch07.md
git commit -m "feat: persist conversation anchors and summaries"
```

---

### Task 3: Three-Layer Trimmer and Fixed-Order Assembler

**Files:**
- Create: `app/context/models.py`
- Create: `app/context/trimmer.py`
- Create: `app/context/assembler.py`
- Create: `tests/ch07/test_context_trimmer.py`
- Create: `tests/ch07/test_context_assembler.py`
- Modify: `app/core/memory.py`

**Interfaces:**

- Consumes: `ContextBudget`, token estimator, persisted message rows.
- Produces:
  - `ContextMessage`
  - `SummaryRow`
  - `SummarySpan`
  - `ContextPack`
  - `build_context_pack(...) -> ContextPack`
  - `build_model_messages(pack, evidence, order_data, after_sales_action, fixed_system) -> list[BaseMessage]`
  - `build_history_messages(pack) -> list[BaseMessage]`

- [ ] **Step 1: Write failing layer tests**

```python
from langchain_core.messages import AIMessage

from app.context.budget import ContextBudget
from app.context.models import ContextMessage, SummarySpan
from app.context.trimmer import build_context_pack


def budget(layer1: int, layer2: int) -> ContextBudget:
    return ContextBudget(
        model_context_window=10000,
        max_output_tokens=0,
        max_user_input_tokens=0,
        max_agent_steps=0,
        tool_result_max_tokens=0,
        rerank_top_k=0,
        evidence_chunk_token_budget=0,
        system_prompt_token_budget=0,
        summary_injection_token_budget=0,
        safety_margin_tokens=0,
        desired_retained_turns=layer1 + layer2 + 1,
        steady_turn_token_estimate=1,
    )


def test_layer1_downgrade_only_moves_anchor():
    messages = [
        ContextMessage(1, "user", "问题一"),
        ContextMessage(2, "assistant", "回答一" * 20),
        ContextMessage(3, "user", "问题二"),
        ContextMessage(4, "assistant", "回答二"),
    ]
    pack = build_context_pack(
        messages=messages,
        summaries=[],
        layer1_from_msg_id=1,
        summary_upto_msg_id=0,
        budget=budget(layer1=18, layer2=7),
        chinese_tokens_per_char=1.0,
        assistant_layer2_chars=4,
    )
    assert pack.layer1_from_msg_id == 3
    assert [item.content for item in pack.layer1_messages] == ["问题二", "回答二"]
    assert pack.layer2_messages[0].content == "回答一" + "..."
    assert pack.layer2_messages[1].content == "问题一"


def test_layer2_overflow_marks_summary_span_without_waiting():
    messages = [
        ContextMessage(1, "user", "订单1001要退款"),
        ContextMessage(2, "assistant", "已经登记退款" * 20),
        ContextMessage(3, "user", "后来呢"),
        ContextMessage(4, "assistant", "处理中"),
    ]
    pack = build_context_pack(
        messages=messages,
        summaries=[],
        layer1_from_msg_id=3,
        summary_upto_msg_id=0,
        budget=budget(layer1=20, layer2=1),
        chinese_tokens_per_char=1.0,
        assistant_layer2_chars=4,
    )
    assert pack.summary_span == SummarySpan(1, 2)
    assert pack.layer1_messages[-1].content == "处理中"
```

- [ ] **Step 2: Run layer tests and verify they fail**

Run: `uv run pytest tests/ch07/test_context_trimmer.py -q`

Expected: FAIL because context models and trimmer do not exist.

- [ ] **Step 3: Write failing assembler tests**

```python
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.context.assembler import build_model_messages
from app.context.models import ContextPack


def test_model_context_order_is_stable():
    pack = ContextPack(
        summaries_text="用户问过订单1001。",
        layer2_messages=[HumanMessage("旧用户话")],
        layer1_messages=[HumanMessage("新用户话"), AIMessage("新回答")],
        history_ctx=[],
        layer1_from_msg_id=2,
        summary_span=None,
        layer2_tokens=4,
        layer1_tokens=4,
        token_estimate=8,
        trigger_summary=False,
        budget_error="",
    )
    messages = build_model_messages(
        pack=pack,
        evidence=[{"citation_id": 1, "text": "七天无理由"}],
        order_data={},
        after_sales_action="",
        fixed_system="固定人设和红线",
    )
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == "固定人设和红线"
    assert messages[1].content == "旧用户话"
    assert messages[2].content == "新用户话"
    assert messages[3].content == "新回答"
    assert "用户问过订单1001。" in messages[-1].content
    assert "[1] 七天无理由" in messages[-1].content
    assert not any(
        isinstance(item, SystemMessage) and "订单1001" in item.content
        for item in messages
    )
```

- [ ] **Step 4: Run assembler tests and verify they fail**

Run: `uv run pytest tests/ch07/test_context_assembler.py -q`

Expected: FAIL because the assembler does not exist.

- [ ] **Step 5: Implement immutable context models**

```python
@dataclass(frozen=True)
class ContextMessage:
    message_id: int
    role: str
    content: str
    tool_name: str | None = None

    def to_langchain(self) -> BaseMessage:
        if self.role == "user":
            return HumanMessage(self.content)
        if self.role == "assistant":
            return AIMessage(self.content)
        return ToolMessage(
            self.content,
            tool_call_id=f"persisted-{self.message_id}",
            name=self.tool_name or "tool",
        )

    @classmethod
    def from_record(cls, row) -> "ContextMessage":
        return cls(
            message_id=row.id,
            role=row.role,
            content=row.content or "",
            tool_name=row.tool_name,
        )


@dataclass(frozen=True)
class SummaryRow:
    covered_from_msg_id: int
    covered_to_msg_id: int
    content: str


@dataclass(frozen=True)
class SummarySpan:
    from_msg_id: int
    to_msg_id: int


@dataclass(frozen=True)
class ContextPack:
    summaries_text: str
    layer2_messages: list[BaseMessage]
    layer1_messages: list[BaseMessage]
    history_ctx: list[BaseMessage]
    layer1_from_msg_id: int | None
    summary_span: SummarySpan | None
    layer2_tokens: int
    layer1_tokens: int
    token_estimate: int
    trigger_summary: bool
    budget_error: str = ""
```

- [ ] **Step 6: Implement trimmer behavior**

The trimmer sorts by `message_id`, skips messages at or below
`summary_upto_msg_id`, and computes layer 1 from
`layer1_from_msg_id or first_message_id`.

Layer 1 fitting:

```python
while layer1 and (
    estimate_messages_tokens(layer1, chinese_tokens_per_char)
    > budget.layer1_budget
):
    next_human = next(
        (
            index
            for index, item in enumerate(layer1[1:], start=1)
            if item.role == "user"
        ),
        None,
    )
    if next_human is None:
        break
    layer1 = layer1[next_human:]
    layer1_from_msg_id = layer1[0].message_id
```

Layer 2 shortening:

```python
if item.role == "user":
    content = item.content
elif item.role == "assistant":
    content = (
        item.content
        if len(item.content) <= assistant_layer2_chars
        else item.content[:assistant_layer2_chars] + "..."
    )
elif item.role == "tool":
    status = "ok" if not item.content.startswith("error") else "error"
    content = f"[tool:{item.tool_name or 'tool'} {status} id={item.message_id}]"
```

If layer 2 exceeds `layer2_budget`, set
`summary_span=SummarySpan(first_layer2_id, last_layer2_id)` and keep only the
newest shortened layer 2 messages that fit. The oldest original messages stay
in MySQL untouched.

- [ ] **Step 7: Implement model assembly**

Always start with one fixed `SystemMessage`; append
`pack.layer2_messages`, then `pack.layer1_messages`, then one final
`HumanMessage` containing summaries, order facts, and evidence in that order.
Do not add a second system message. `build_history_messages` prepends one
human-labelled summary line to reduced layer 2 plus layer 1.

- [ ] **Step 8: Run trimmer, assembler, and memory tests**

Run: `uv run pytest tests/ch07/test_context_trimmer.py tests/ch07/test_context_assembler.py tests/test_memory.py -q`

Expected: PASS.

- [ ] **Step 9: Append Task 3 to `dev-notes/ch07.md`**

Record layer rules, test values, rejected data movement, and any boundary
rework.

- [ ] **Step 10: Commit**

```bash
git add app/context app/core/memory.py tests/ch07/test_context_trimmer.py tests/ch07/test_context_assembler.py dev-notes/ch07.md
git commit -m "feat: add three-layer context trimming"
```

---

### Task 4: Facts-Only Summary Service and Non-Blocking Task Manager

**Files:**
- Create: `app/context/summary.py`
- Create: `app/context/tasks.py`
- Create: `app/context/logging.py`
- Create: `tests/ch07/test_summary_service.py`
- Create: `tests/ch07/test_summary_tasks.py`
- Create: `tests/ch07/test_context_logging.py`

**Interfaces:**

- Consumes: `ContextMessage`, `SummarySpan`, `ConversationRepository`.
- Produces:
  - `build_summary_prompt(messages) -> str`
  - `summarize_span(messages, model, min_chars, max_chars) -> str`
  - `SummaryTaskManager.schedule(session_id, span, messages) -> str`
  - `SummaryTaskManager.wait_for(session_id) -> None`
  - `SummaryTaskManager.shutdown() -> None`
  - `log_history_ctx(...)`, `log_model_ctx(...)`

- [ ] **Step 1: Write failing prompt/service tests**

```python
from langchain_core.messages import AIMessage

from app.context.models import ContextMessage
from app.context.summary import build_summary_prompt, summarize_span


class StaticSummaryModel:
    async def ainvoke(self, messages):
        self.messages = messages
        return AIMessage("用户咨询订单1001退款，客服已登记，问题仍未解决。")


def test_summary_prompt_forbids_invention():
    prompt = build_summary_prompt(
        [
            ContextMessage(1, "user", "订单1001要退款"),
            ContextMessage(2, "assistant", "已登记"),
        ]
    )
    assert "不得编造" in prompt
    assert "订单1001" in prompt
    assert "寒暄" in prompt


async def test_summarize_span_returns_body_only():
    model = StaticSummaryModel()
    result = await summarize_span(
        [ContextMessage(1, "user", "订单1001要退款")],
        model,
        min_chars=5,
        max_chars=100,
    )
    assert result.startswith("用户咨询订单1001")
```

- [ ] **Step 2: Run summary tests and verify they fail**

Run: `uv run pytest tests/ch07/test_summary_service.py -q`

Expected: FAIL because the summary service does not exist.

- [ ] **Step 3: Implement the facts-only prompt and response cleanup**

The prompt contains only:

```text
你只提炼对话中明确出现的事实与诉求。
允许保留：商品、订单号、手机号、明确诉求、未解决问题。
禁止：编造、推测、补充对话中没有出现的订单、金额、时间、政策或承诺。
寒暄、客套、感谢和闲聊必须删除。
保留数字原样，不要改成示例号。
只输出摘要正文，不要 JSON、标题或 Markdown。
长度控制在 {min_chars} 到 {max_chars} 个中文字符。
```

`summarize_span` strips Markdown fences and surrounding whitespace, rejects an
empty body, and truncates only to the configured maximum.

- [ ] **Step 4: Write failing async task tests**

```python
import asyncio

import pytest
from langchain_core.messages import AIMessage

from app.context.models import ContextMessage, SummarySpan
from app.context.tasks import SummaryTaskManager


class SlowModel:
    async def ainvoke(self, messages):
        await asyncio.sleep(0.05)
        return AIMessage("用户询问订单1001。")


class Store:
    def __init__(self):
        self.calls = []

    async def append_summary_and_advance(self, **kwargs):
        self.calls.append(kwargs)


async def test_schedule_does_not_wait_for_model(monkeypatch):
    logs = []
    monkeypatch.setattr(
        "app.context.tasks.log_summary_event",
        lambda *args, **kwargs: logs.append((args, kwargs)),
    )
    store = Store()
    manager = SummaryTaskManager(
        model=SlowModel(),
        store=store,
        estimator=lambda text: len(text),
        min_chars=2,
        max_chars=100,
    )
    status = manager.schedule(
        "s1",
        SummarySpan(1, 2),
        [ContextMessage(1, "user", "订单1001")],
    )
    assert status == "running"
    assert store.calls == []
    await manager.wait_for("s1")
    assert store.calls[0]["from_id"] == 1
```

- [ ] **Step 5: Run task tests and verify they fail**

Run: `uv run pytest tests/ch07/test_summary_tasks.py -q`

Expected: FAIL because `SummaryTaskManager` does not exist.

- [ ] **Step 6: Implement task ownership and logging**

```python
class SummaryTaskManager:
    def __init__(self, model, store, estimator, min_chars, max_chars):
        self._model = model
        self._store = store
        self._estimator = estimator
        self._min_chars = min_chars
        self._max_chars = max_chars
        self._tasks: dict[str, asyncio.Task] = {}

    def schedule(self, session_id, span, messages):
        current = self._tasks.get(session_id)
        if current and not current.done():
            log_summary_event("skip", session_id, reason="already_running")
            return "running"
        log_summary_event("trigger", session_id, span=span)
        task = asyncio.create_task(self._run(session_id, span, messages))
        self._tasks[session_id] = task
        task.add_done_callback(lambda _: self._tasks.pop(session_id, None))
        return "running"
```

`_run` emits `start`, calls `summarize_span`, stores
`append_summary_and_advance`, emits `done`, and catches exceptions with
`fail`. `wait_for` awaits the registered task when present. `shutdown` waits
up to two seconds, then cancels unfinished tasks.

- [ ] **Step 7: Implement structured logging**

Use `logging.getLogger("app.context")`. Build one-line escaped JSON payloads:

```python
logger.info(
    "model_ctx session_id=%s summary_json=%s messages_json=%s "
    "message_count=%d token_estimate=%d",
    session_id,
    json.dumps(summary_text, ensure_ascii=False),
    json.dumps([serialize_message(item) for item in messages], ensure_ascii=False),
    len(messages),
    token_estimate,
)
```

`log_history_ctx` uses the same shape with the `history_ctx` prefix.
`log_summary_event` emits `summary trigger|start|done|skip|fail` plus
`covered_from`, `covered_to`, `elapsed_ms`, or `error` when applicable.

- [ ] **Step 8: Run summary, task, and logging tests**

Run: `uv run pytest tests/ch07/test_summary_service.py tests/ch07/test_summary_tasks.py tests/ch07/test_context_logging.py -q`

Expected: PASS.

- [ ] **Step 9: Append Task 4 to `dev-notes/ch07.md`**

Record prompt rules, non-blocking test, task ownership, and any prompt
evaluation failure.

- [ ] **Step 10: Commit**

```bash
git add app/context tests/ch07/test_summary_service.py tests/ch07/test_summary_tasks.py tests/ch07/test_context_logging.py dev-notes/ch07.md
git commit -m "feat: add background context summaries"
```

---

### Task 5: Context Manager and LangGraph Integration

**Files:**
- Create: `app/context/manager.py`
- Create: `tests/ch07/test_context_graph.py`
- Modify: `app/graph/state.py`
- Modify: `app/graph/builder.py`
- Modify: `app/graph/nodes.py`
- Modify: `app/graph/agent.py`
- Modify: `app/services/conversation_understanding.py`
- Modify: `app/graph/intent.py`
- Modify: `app/services/turn_logger.py`
- Modify: `tests/test_graph_workflow.py`
- Modify: `tests/test_graph_agent.py`
- Modify: `tests/test_graph_intent.py`
- Modify: `tests/ch06/test_conversation_understanding.py`

**Interfaces:**

- Consumes: repository functions, `ContextBudget`, trimmer, assembler, summary task manager.
- Produces:
  - `ContextManager.prepare(session_id, current_user_message_id) -> ContextPack`
  - `make_prepare_context_node(manager)`
  - `route_after_prepare_context(state) -> "understand" | "fallback"`
  - State field `context_pack: ContextPack`

- [ ] **Step 1: Write a failing graph test for budget failure**

```python
from langgraph.checkpoint.memory import InMemorySaver

from app.graph.builder import build_chat_graph


class NoModel:
    async def ainvoke(self, *args, **kwargs):
        raise AssertionError("model must not be called")

    async def astream(self, *args, **kwargs):
        raise AssertionError("model must not be called")
        yield

    def bind_tools(self, tools, **kwargs):
        return self


class FailingContextManager:
    async def prepare(self, session_id, current_user_message_id):
        from app.context.models import ContextPack

        return ContextPack(
            summaries_text="",
            layer2_messages=[],
            layer1_messages=[],
            history_ctx=[],
            layer1_from_msg_id=None,
            summary_span=None,
            layer2_tokens=0,
            layer1_tokens=0,
            token_estimate=0,
            trigger_summary=False,
            budget_error="上下文预算不足",
        )


async def test_budget_failure_returns_before_any_model_call():
    graph = build_chat_graph(
        intent_model=NoModel(),
        agent_model=NoModel(),
        context_manager=FailingContextManager(),
        checkpointer=InMemorySaver(),
    )
    result = await graph.ainvoke(
        {
            "session_id": "s-budget",
            "current_user_message_id": 1,
            "user_message": "你好",
            "messages": [HumanMessage("你好")],
        },
        {"configurable": {"thread_id": "s-budget"}},
    )
    assert result["reply"] == "上下文预算不足"
```

- [ ] **Step 2: Run the graph test and verify it fails**

Run: `uv run pytest tests/ch07/test_context_graph.py::test_budget_failure_returns_before_any_model_call -q`

Expected: FAIL because `context_manager` and `prepare_context` do not exist.

- [ ] **Step 3: Add context state**

```python
from app.context.models import ContextPack

context_pack: ContextPack
current_user_message_id: int
```

`_reset_turn_update()` must not clear `context_pack` or
`current_user_message_id`.

- [ ] **Step 4: Implement `ContextManager.prepare`**

```python
async def prepare(
    self,
    session_id: str,
    current_user_message_id: int,
) -> ContextPack:
    conversation = await self._store.get_conversation(session_id)
    rows = await self._store.load_messages(session_id)
    summaries = await self._store.load_summaries(session_id)
    pack = build_context_pack(
        messages=[ContextMessage.from_record(row) for row in rows],
        summaries=[
            SummaryRow(
                row.covered_from_msg_id,
                row.covered_to_msg_id,
                row.content,
            )
            for row in summaries
        ],
        layer1_from_msg_id=conversation.layer1_from_msg_id,
        summary_upto_msg_id=conversation.summary_upto_msg_id,
        budget=self._budget,
        chinese_tokens_per_char=self._chinese_tokens_per_char,
        assistant_layer2_chars=self._assistant_layer2_chars,
    )
    if pack.layer1_from_msg_id != conversation.layer1_from_msg_id:
        await self._store.update_layer1_anchor(
            session_id,
            pack.layer1_from_msg_id,
        )
    if pack.trigger_summary and pack.summary_span:
        span_messages = [
            item
            for item in messages
            if pack.summary_span.from_msg_id
            <= item.message_id
            <= pack.summary_span.to_msg_id
        ]
        self._tasks.schedule(session_id, pack.summary_span, span_messages)
    log_history_ctx(session_id, pack)
    return pack
```

`prepare` never awaits `SummaryTaskManager.schedule`.

- [ ] **Step 5: Add `prepare_context` to the graph**

```python
graph.add_node("prepare_context", make_prepare_context_node(context_manager))
graph.add_conditional_edges(
    "prepare_context",
    route_after_prepare_context,
    {"understand": "understand_query", "fallback": "fallback"},
)
```

The legacy `START -> understand_query` edge is removed. If
`context_manager is None`, the node constructs a non-persistent pack directly
from `state["messages"]`; production passes the repository-backed manager.

- [ ] **Step 6: Share `history_ctx` with query understanding and intent**

In `make_understanding_node`, pass `pack.history_ctx` as history and keep the
last stored human message as the current user message.

Extend intent input:

```python
async def classify_intent_text(
    message: str,
    model: BaseChatModel,
    *,
    history: Sequence[BaseMessage] | None = None,
    ...
) -> IntentDecision:
```

The classification prompt includes a compact `最近对话` block from the same
`history_ctx`. Existing callers without history continue to work.

- [ ] **Step 7: Replace Agent history with fixed-order `model_ctx`**

In `prepare_agent`:

```python
pack = state.get("context_pack")
if pack:
    agent_messages = build_model_messages(
        pack=pack,
        evidence=state.get("evidence") or [],
        order_data=state.get("order_data") or {},
        after_sales_action=state.get("after_sales_action", ""),
        fixed_system=CUSTOMER_SERVICE_SYSTEM,
    )
else:
    agent_messages = [SystemMessage(CUSTOMER_SERVICE_SYSTEM), *trimmed]
log_model_ctx(state["session_id"], agent_messages, pack)
```

Remove dynamic order/evidence content from the system message. Put it in the
final human supplement after the current user turn.

- [ ] **Step 8: Change persistence ownership**

API persistence writes the user message before graph execution.
`TurnLogger.log` appends only the assistant reply, updates
`Conversation.updated_at`, and continues to log `turn_complete`. The current
user message is never appended a second time on resume.

- [ ] **Step 9: Run focused graph suites**

Run:

```bash
uv run pytest tests/ch07/test_context_graph.py tests/test_graph_workflow.py tests/test_graph_agent.py tests/test_graph_intent.py -q
```

Expected: PASS. Existing tests receive `context_manager=None` unless they
specifically test Ch07 behavior.

- [ ] **Step 10: Append Task 5 to `dev-notes/ch07.md`**

Record fixed context order, checkpoint preservation, user-message persistence
ownership, and any test regressions.

- [ ] **Step 11: Commit**

```bash
git add app/context/manager.py app/graph app/services/conversation_understanding.py app/services/turn_logger.py tests/ch07/test_context_graph.py tests/test_graph_workflow.py tests/test_graph_agent.py tests/test_graph_intent.py tests/ch06/test_conversation_understanding.py dev-notes/ch07.md
git commit -m "feat: integrate managed context with LangGraph"
```

---

### Task 6: Chat User Key, Read APIs, Startup Budget Check, and File Logging

**Files:**
- Create: `app/api/conversations.py`
- Create: `tests/ch07/test_conversations_api.py`
- Modify: `app/schemas/chat.py`
- Modify: `app/api/chat.py`
- Modify: `app/main.py`
- Modify: `tests/test_chat_api.py`

**Interfaces:**

- Consumes: repository list/read functions, `ContextBudget.from_settings()`, `ContextManager`, `SummaryTaskManager`.
- Produces:
  - `ChatRequest.user_key`
  - `GET /api/conversations`
  - `GET /api/conversations/{conversation_id}/messages`
  - startup `context_budget_insufficient` log when needed.

- [ ] **Step 1: Write failing API tests**

```python
from fastapi.testclient import TestClient

from app.main import app


def test_conversation_list_requires_user_key():
    with TestClient(app) as client:
        response = client.get("/api/conversations")
    assert response.status_code == 422


def test_message_api_filters_tool_rows(monkeypatch):
    async def fake_load_conversation_messages(conversation_id, user_key):
        return [
            {
                "id": 1,
                "role": "user",
                "content": "订单1001",
                "created_at": "2026-09-21T10:00:00",
            },
            {
                "id": 2,
                "role": "assistant",
                "content": "已查询",
                "created_at": "2026-09-21T10:00:01",
            },
        ]

    monkeypatch.setattr(
        "app.api.conversations.load_conversation_messages",
        fake_load_conversation_messages,
    )
    with TestClient(app) as client:
        response = client.get(
            "/api/conversations/s1/messages?user_key=u1"
        )
    assert response.status_code == 200
    assert [item["role"] for item in response.json()] == ["user", "assistant"]
```

- [ ] **Step 2: Run API tests and verify they fail**

Run: `uv run pytest tests/ch07/test_conversations_api.py -q`

Expected: FAIL because the routes do not exist.

- [ ] **Step 3: Add request and response schemas**

```python
class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1)
    user_key: str = Field(default="demo-user", min_length=1, max_length=64)
    message: str | None = None
    resume: dict[str, Any] | None = None


class ConversationListItem(BaseModel):
    id: str
    preview: str
    created_at: datetime
    updated_at: datetime
    summarized: bool


class ConversationMessageItem(BaseModel):
    id: int
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime
```

- [ ] **Step 4: Persist the new user message before graph execution**

In `/api/chat`, only for a non-resume request:

```python
await ensure_conversation(req.session_id, req.user_key)
user_message_id = await append_message(
    req.session_id,
    "user",
    req.message,
)
graph_input = {
    "session_id": req.session_id,
    "user_key": req.user_key,
    "current_user_message_id": user_message_id,
    "user_message": req.message,
    "messages": [HumanMessage(req.message)],
}
```

On `ConversationOwnershipError`, return HTTP 409. Resume input remains
`Command(resume=req.resume)` and does not append another user row.

- [ ] **Step 5: Add read-only routes**

```python
@router.get("/api/conversations", response_model=list[ConversationListItem])
async def conversations(user_key: str = Query(min_length=1)):
    return await list_conversations(user_key)


@router.get(
    "/api/conversations/{conversation_id}/messages",
    response_model=list[ConversationMessageItem],
)
async def conversation_messages(
    conversation_id: str,
    user_key: str = Query(min_length=1),
):
    rows = await load_conversation_messages(conversation_id, user_key)
    if rows is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return rows
```

Repository reads filter `role IN ("user", "assistant")`.

- [ ] **Step 6: Wire startup resources and logging**

In `lifespan`:

1. Build `ContextBudget.from_settings()`.
2. If `not budget.can_fit_one_turn`, log critical
   `context_budget_insufficient` with the fixed-overhead breakdown; keep the
   app serving so requests return `上下文预算不足`.
3. Build `SummaryTaskManager`.
4. Build `ContextManager`.
5. Pass the manager to `build_chat_graph`.
6. Register the manager's `shutdown` in the lifespan `finally`.
7. Create the parent directory of `settings.context_log_path` and add a
   `FileHandler` with the existing formatter.

Include `app.api.conversations.router` in `app.include_router`.

- [ ] **Step 7: Update existing chat API tests**

Assertions change to include `user_key` and `current_user_message_id`.
Fake repositories return deterministic ids. Resume tests assert that the fake
append function is not called again.

- [ ] **Step 8: Run API, startup, and logging tests**

Run:

```bash
uv run pytest tests/ch07/test_conversations_api.py tests/test_chat_api.py tests/ch07/test_context_logging.py -q
```

Expected: PASS.

- [ ] **Step 9: Append Task 6 to `dev-notes/ch07.md`**

Record API contracts, startup alarm, file logging path, and rejected
unscoped conversation reads.

- [ ] **Step 10: Commit**

```bash
git add app/api app/schemas/chat.py app/main.py tests/ch07/test_conversations_api.py tests/test_chat_api.py dev-notes/ch07.md
git commit -m "feat: add conversation history APIs"
```

---

### Task 7: Multi-Conversation Frontend Sidebar

**Files:**
- Modify: `D:\std_selftest_frontend\index.html`
- Modify: `D:\std_selftest_frontend\src\main.js`
- Modify: `D:\std_selftest_frontend\src\styles.css`

**Interfaces:**

- Consumes: `GET /api/conversations?user_key=...`, `GET /api/conversations/{id}/messages?user_key=...`, `/api/chat`.
- Produces: persistent demo `user_key`, conversation sidebar, click-to-reload, new-conversation flow.

- [ ] **Step 1: Add the sidebar markup**

Inside `#chat-view`, wrap the existing header/frame/composer in a
two-column chat layout and add:

```html
<aside id="conversation-sidebar" class="conversation-sidebar">
  <div class="conversation-sidebar-head">
    <h2>会话</h2>
    <button id="new-conversation" class="btn-icon" type="button">+</button>
  </div>
  <div id="conversation-list" class="conversation-list"></div>
</aside>
```

The existing `#new-chat` button remains available; both controls call the same
new-session function.

- [ ] **Step 2: Add frontend user and conversation state**

```javascript
const userKey =
  localStorage.getItem("ihelp-user-key") ||
  `demo-${crypto.randomUUID()}`;
localStorage.setItem("ihelp-user-key", userKey);

state.conversations = [];
state.loadingConversationId = "";
```

Every `/api/chat` body includes `user_key: userKey`.

- [ ] **Step 3: Load and render the sidebar**

```javascript
async function loadConversations() {
  try {
    const response = await fetch(
      `/api/conversations?user_key=${encodeURIComponent(userKey)}`
    );
    if (!response.ok) throw new Error("conversation list unavailable");
    state.conversations = await response.json();
    renderConversationList();
  } catch {
    elements.conversationSidebar.hidden = true;
  }
}
```

Each row is a `<button>` with preview, relative time, and a summarized badge.
Clicking the active conversation is a no-op. Clicking another row fetches its
messages, clears the visible log, renders user/assistant bubbles, updates
`state.sessionId`, then shows the empty composer.

- [ ] **Step 4: Implement new conversation behavior**

```javascript
function startNewConversation() {
  if (state.streaming) return;
  state.sessionId = makeSessionId();
  elements.sessionId.value = state.sessionId;
  localStorage.setItem("ihelp-session", state.sessionId);
  clearChatLog();
  renderConversationList();
  loadConversations();
}
```

Do not delete or mutate previous conversation rows.

- [ ] **Step 5: Add responsive styling**

Desktop uses a fixed 230-260px sidebar next to the chat panel. At `max-width:
760px`, place the sidebar above the chat frame with horizontal, non-overlapping
rows. Reuse existing colors and 8px card radius or less. Long previews use
two-line clamping and do not resize the selected row.

- [ ] **Step 6: Build and browser-check**

Run:

```bash
cd D:\std_selftest_frontend
npm run build
npm run dev -- --host 127.0.0.1
```

Verify:

- create two conversations;
- send at least one message in each;
- switch between them and confirm original user/assistant history reloads;
- continue chatting in the older conversation;
- sidebar failure does not block the composer;
- desktop and 390px mobile layouts have no overlap.

- [ ] **Step 7: Append Task 7 to `dev-notes/ch07.md`**

Record the interaction contract, build result, screenshot dimensions, and any
layout rework.

- [ ] **Step 8: Commit**

```bash
git add D:\std_selftest_frontend\index.html D:\std_selftest_frontend\src\main.js D:\std_selftest_frontend\src\styles.css dev-notes/ch07.md
git commit -m "feat: add conversation sidebar"
```

---

### Task 8: Summary Evaluation, Demonstrations, Documentation, and Final Acceptance

**Files:**
- Create: `eval_data/ch07_context_summary_eval_set.json`
- Create: `scripts/eval_ch07.py`
- Create: `scripts/demo_ch07.py`
- Modify: `README.md`
- Modify: `dev-notes/ch07.md`

**Interfaces:**

- Consumes: all Ch07 services and APIs.
- Produces: summary quality metrics, deterministic log demonstration, default-window acceptance, final documentation.

- [ ] **Step 1: Add the labeled summary set**

Use this shape:

```json
[
  {
    "id": "order-refund",
    "span": [
      {"role": "user", "content": "订单1001的手机壳裂了，我要退款"},
      {"role": "assistant", "content": "已为您登记退款。"}
    ],
    "must_include": ["订单1001", "退款"],
    "must_not_include": ["订单9999", "已退款到账"],
    "small_talk": ["你好", "谢谢"]
  }
]
```

Include at least ten cases covering product, order number, phone number,
unresolved issue, multiple numbers, small talk, and an unsupported fact.

- [ ] **Step 2: Write the evaluation script**

`scripts/eval_ch07.py` loads the model with `get_chat_model(streaming=False)`,
calls `summarize_span`, and prints:

```text
key_fact_recall=...
unsupported_fact_rate=...
small_talk_leak_rate=...
length_pass_rate=...
```

Return exit code `1` when any threshold fails:

```text
key_fact_recall < 0.90
unsupported_fact_rate > 0
small_talk_leak_rate > 0
length_pass_rate < 0.90
```

Write failed cases to `reports/ch07_summary_failures.json`.

- [ ] **Step 3: Write the demonstration script**

`scripts/demo_ch07.py` accepts:

```text
--profile default|acceptance
--turns 20
--session-id demo-ch07
--user-key demo-user
--ask-start-order
```

For the acceptance profile, apply exactly:

```text
MODEL_CONTEXT_WINDOW=18000
MAX_OUTPUT_TOKENS=2000
MAX_USER_INPUT_TOKENS=2000
MAX_AGENT_STEPS=3
TOOL_RESULT_MAX_TOKENS=1200
RERANK_TOP_K=5
```

The script uses the real configured chat model, creates or reuses the
conversation, sends deterministic turns, waits for summary tasks at the end,
and prints the summary rows from MySQL. It also sends
`最开始那个订单后来怎么说`.

- [ ] **Step 4: Add a deterministic test for the acceptance sequence**

```python
async def test_acceptance_sequence_closes_layers(monkeypatch):
    # Use a fake 18000-token budget and deterministic long turns.
    # Assert the captured logs contain:
    assert "layer1 downgrade" in captured
    assert "summary trigger" in captured
    assert "summary done" in captured
```

The test must not require a live model or MySQL. Inject fake trimmer,
repository, and summary task runner.

- [ ] **Step 5: Run focused tests and summary evaluation**

Run:

```bash
uv run pytest tests/ch07 -q
uv run python scripts/eval_ch07.py
```

Expected: all Ch07 tests pass; summary metrics meet the thresholds.

- [ ] **Step 6: Run default-window demonstration**

Run:

```bash
uv run python scripts/demo_ch07.py --profile default --turns 20 --session-id demo-default
```

Expected:

- no `layer1 downgrade`;
- no `summary trigger`;
- no summary rows;
- 20 turns complete without a token-budget fallback.

- [ ] **Step 7: Run acceptance-window demonstration**

Run:

```bash
uv run python scripts/demo_ch07.py --profile acceptance --turns 24 --session-id demo-budget --ask-start-order
```

Expected logs:

```text
layer1 downgrade ...
summary trigger ... layer2 ... > ...
summary done ... 第1段 ...
```

The final answer must mention the original order number and the request stored
in the summary.

- [ ] **Step 8: Run the full verification set**

Run:

```bash
uv run pytest -q -p no:cacheprovider
uv run python -m compileall -q app scripts
git diff --check
cd D:\std_selftest_frontend
npm run build
```

Expected: all commands pass. Record exact pass counts and warnings.

- [ ] **Step 9: Update README**

Add:

```text
uv run python scripts/eval_ch07.py
uv run python scripts/demo_ch07.py --profile default --turns 20
uv run python scripts/demo_ch07.py --profile acceptance --turns 24 --ask-start-order
```

Document the sidebar endpoints and the single-worker requirement.

- [ ] **Step 10: Append final task results to `dev-notes/ch07.md`**

Record exact user instructions, files/commands, accepted and rejected
decisions, failures, rework, test counts, evaluation metrics, and browser
verification.

- [ ] **Step 11: Commit**

```bash
git add eval_data/ch07_context_summary_eval_set.json scripts/eval_ch07.py scripts/demo_ch07.py README.md dev-notes/ch07.md
git commit -m "feat: verify ch07 context management"
```

---

## Plan Self-Review

### Spec Coverage

- Budget formula, exact `5650 / 3954 / 1695`, Chinese calibration, and startup
  alarm: Task 1 and Task 6.
- Conversation anchors, append-only summary table, additive schema check, and
  user-key ownership: Task 2.
- Raw layer 1, shortened layer 2, append-only layer 3, anchor-only movement,
  and tool-result one-line shortening: Task 3.
- Facts-only summary prompt, append-only ranges, async non-blocking ownership,
  skip/fail behavior, and summary logs: Task 4.
- LangGraph `add_messages`, checkpoint preservation, full-state/reduced-input
  separation, fixed context order, no summary system message, and shared
  `history_ctx`: Task 5.
- `/api/chat` user key, `/api/conversations`,
  `/api/conversations/{id}/messages`, startup resource wiring, and
  `log/app.log`: Task 6.
- Multi-conversation sidebar, reload, continue, new conversation, silent
  sidebar degradation, and responsive checks: Task 7.
- Summary labeled evaluation, default-window and acceptance-window demos,
  README, final commands, and dev-notes: Task 8.
- No semantic retrieval, cross-session memory, user profile, or technology
  replacement appears in any task.

### Placeholder Scan

The plan contains no `TBD`, `TODO`, deferred implementation, or unspecified
test behavior. Every code-producing task includes exact paths, interfaces,
test intent, and a runnable verification command.

### Type Consistency

- `ContextBudget` field names are identical in Tasks 1, 3, 5, and 6.
- `ContextMessage`, `SummaryRow`, `SummarySpan`, and `ContextPack` are defined
  in Task 3 and consumed with the same names in Tasks 4 and 5.
- Repository functions defined in Task 2 are used with the same signatures in
  Tasks 4, 5, and 6.
- `ContextManager.prepare` returns `ContextPack` consistently.
- `model_ctx` and `history_ctx` log names match the spec and tests.
- `user_key` is the only ownership identifier in the API and frontend.
