# iHelp Ch07 Context Management Design

## Background

Ch06 keeps the full conversation in `ChatState.messages` with the
`add_messages` reducer and a LangGraph checkpointer. The Agent still trims the
model input with one fixed `TOKEN_BUDGET`, while recent context required by
query understanding and intent classification is passed separately.

That is not sufficient for a long customer-service conversation:

- the budget is disconnected from the model context window;
- old raw messages are dropped without a durable summary;
- tool results are written to the audit `messages` table even though they are
  not stable conversation turns;
- there are no explicit boundaries between recent, shortened, and summarized
  history;
- there is no multi-conversation sidebar or read API.

Ch07 replaces the placeholder trimming with per-conversation context
management. It does not add cross-conversation memory or a user profile.

## Goal

Build a three-layer, observable, per-conversation context manager that:

1. Keeps recent turns as raw messages.
2. Deterministically shortens the middle layer.
3. Extends an append-only summary history for the oldest layer without
   blocking the current response.
4. Derives token budgets from the model window and current Agent limits.
5. Keeps the complete LangGraph state separate from the reduced model input.
6. Logs the exact summary and sliding-window context used by every model call.
7. Exposes read-only conversation APIs and a multi-conversation chat sidebar.

## Non-Goals

- No semantic retrieval over old conversation messages.
- No topic importance or key-fact retention layer beyond the summary.
- No cross-session memory.
- No user profile, personalization, or authentication system.
- No replacement of LangGraph, LangChain, SQLAlchemy, FastAPI, or MySQL.
- No multi-worker background-task coordination.

## Context7 Findings

Documentation was checked before implementation through the official Context7
MCP server v4.1.1:

- LangGraph `/langchain-ai/langgraph/1.0.8`: `StateGraph` state fields can use
  `Annotated[..., add_messages]`; checkpointer-backed execution uses a
  per-thread config; full state is durable and should not be replaced by the
  trimmed input.
- LangChain `/websites/reference_langchain`: `trim_messages` accepts
  `max_tokens`, a callable `token_counter`, `strategy`, `allow_partial`,
  `start_on`, `end_on`, `include_system`, and `text_splitter`. The project will
  use a calibrated callable token counter and `start_on="human"`.
- SQLAlchemy `/websites/sqlalchemy_en_20_orm`: async work uses
  `AsyncSession`; an explicit `session.begin()` or `commit()` boundary is
  required. ORM joins and aggregate expressions use `select()`,
  `func.max()`, and correlated `exists()`.
- FastAPI `/websites/fastapi_tiangolo`: application resources belong in the
  async `lifespan`; GET routes use typed query parameters and response models;
  SSE continues to use an async generator with an await point.

The Context7 guidance is reflected in the interfaces below. No dependency
version replacement is required by this design.

## Architecture

Add an independent context-management layer around model calls rather than
turning summarization into another graph loop:

```text
START
  -> prepare_context
       -> fallback                 (budget cannot fit one turn)
       -> understand_query
          -> classify_intent
          -> existing Ch06 routes
             -> Agent / knowledge / fixed response
  -> log_turn
  -> END
```

`prepare_context` only derives an immutable `ContextPack` from the full
LangGraph state and persisted anchors. It never mutates
`ChatState.messages`. Query understanding and intent classification consume the
same `history_ctx`; the Agent consumes `model_ctx`.

The context layer is split into focused units:

- `ContextBudget`: maps configuration to token budgets and validates startup
  feasibility.
- `TokenEstimator`: calibrated CJK/ASCII/message-overhead estimation.
- `ContextTrimmer`: builds layer 1 and layer 2 and advances the layer boundary.
- `ContextAssembler`: produces the fixed model-message order.
- `SummaryService`: calls the summary model with a strict facts-only prompt.
- `SummaryTaskManager`: owns non-blocking per-conversation summary tasks.
- `ConversationRepository`: anchors, summaries, list, and message read APIs.

## Data Model

### `conversations`

Modify:

| Column | Meaning |
| --- | --- |
| `user_name` | Existing column reused as the opaque demo `user_key` |
| `updated_at` | Last turn timestamp used by the sidebar |
| `summary_upto_msg_id` | Last `messages.id` covered by an appended summary |
| `layer1_from_msg_id` | First `messages.id` in the raw recent layer |

Both anchor columns are nullable. Null means the conversation has not yet
degraded. Moving an anchor never moves or rewrites message rows.

### `messages`

Keep the existing table for compatibility. New turns write only `user` and
`assistant` roles. Tool-call and tool-result rows are no longer written.
Historical tool rows remain readable only for database compatibility and are
excluded from the conversation message API.

### `conversation_summaries`

Create an append-only table:

| Column | Meaning |
| --- | --- |
| `id` | Auto-increment primary key |
| `conversation_id` | Foreign key to `conversations.id` |
| `sequence_no` | Monotonic segment number per conversation |
| `covered_from_msg_id` | Inclusive first message id |
| `covered_to_msg_id` | Inclusive last message id |
| `content` | Summary text |
| `token_estimate` | Estimated summary tokens |
| `created_at` | Creation timestamp |

Add a unique constraint on
`(conversation_id, covered_from_msg_id, covered_to_msg_id)`. A summary insert
and the corresponding `summary_upto_msg_id` update commit in one transaction.

`Base.metadata.create_all` creates the new table. Existing `conversations`
tables require a small idempotent startup schema check for the new columns;
the application does not add Alembic.

## State

Extend `ChatState` with derived fields:

| Field | Meaning |
| --- | --- |
| `context_summaries` | Summary rows included in the current turn |
| `context_history` | Reduced layer 1 and layer 2 messages |
| `context_history_count` | Number of messages in the reduced history |
| `context_token_estimate` | Estimated tokens in `model_ctx` |
| `context_budget_error` | Empty when the turn fits |
| `summary_task_status` | `not_needed`, `running`, `done`, `skipped`, or `failed` |

The full `messages` field remains the source of truth with the
`add_messages` reducer. Each node returns only new messages. The reduced
context is never assigned back to `messages`.

## Token Budget

All values are configuration-backed:

```text
react_peak =
    max_agent_steps * tool_result_max_tokens

fixed_overhead =
    system_prompt_token_budget
    + rerank_top_k * evidence_chunk_token_budget
    + summary_injection_token_budget
    + max_output_tokens
    + max_user_input_tokens
    + react_peak
    + safety_margin_tokens

window_available =
    max(0, model_context_window - fixed_overhead)

desired_history =
    desired_retained_turns * steady_turn_token_estimate

history_budget =
    min(desired_history, window_available)
```

Defaults:

```text
MODEL_CONTEXT_WINDOW=65536
MAX_OUTPUT_TOKENS=800
MAX_USER_INPUT_TOKENS=2000
MAX_AGENT_STEPS=5
TOOL_RESULT_MAX_TOKENS=1200
RERANK_TOP_K=20
SYSTEM_PROMPT_TOKEN_BUDGET=1800
EVIDENCE_CHUNK_TOKEN_BUDGET=400
SUMMARY_INJECTION_TOKEN_BUDGET=600
SAFETY_MARGIN_TOKENS=350
DESIRED_RETAINED_TURNS=24
STEADY_TURN_TOKEN_ESTIMATE=800
```

With these defaults, changing only `MODEL_CONTEXT_WINDOW=18000` produces a
zero history budget and the startup insufficiency alarm. The full acceptance
override below supplies the smaller evidence and ReAct peaks needed to yield
the calibrated `5650`.

The layer split reserves one token for the boundary separator:

```text
layer1_budget = floor((history_budget - 1) * 0.70)
layer2_budget = history_budget - 1 - layer1_budget
```

The acceptance configuration:

```text
MODEL_CONTEXT_WINDOW=18000
MAX_OUTPUT_TOKENS=2000
MAX_USER_INPUT_TOKENS=2000
MAX_AGENT_STEPS=3
TOOL_RESULT_MAX_TOKENS=1200
RERANK_TOP_K=5
```

must produce exactly:

```text
history_budget=5650
layer1_budget=3954
layer2_budget=1695
```

These values are asserted by a dedicated calibration test.

Startup validation logs a critical `context_budget_insufficient` event when
`history_budget` cannot fit one minimum user/assistant turn. A request in that
state routes directly to the existing fallback response without calling a
model, returning `上下文预算不足`.

## Token Estimation

`TokenEstimator` replaces the unconditional use of
`count_tokens_approximately`:

- CJK code points are counted at the configured per-character ratio.
- Non-CJK text is counted at approximately four characters per token.
- Each message adds a fixed structural overhead.
- Tool results add a small wrapper overhead.

Defaults are conservative and configuration-backed. A labeled calibration set
verifies the formula against representative Chinese customer-service phrases,
tool JSON, and mixed text. Budget tests and estimator tests are changed
together so changing only one side cannot silently invert the net effect.

## Three-Layer Trimming

Messages are ordered by `messages.id`.

### Layer 1: Raw Recent History

- Starts at `layer1_from_msg_id`.
- Must start on a user message.
- Keeps complete user and assistant content unchanged.
- If the raw layer exceeds `layer1_budget`, move `layer1_from_msg_id` toward
  newer messages until the oldest complete turns fit.
- Log `layer1 downgrade X->Y` with the previous and new message ids.

### Layer 2: Deterministic Middle Layer

- Spans from `summary_upto_msg_id` (exclusive) to
  `layer1_from_msg_id` (exclusive).
- Keeps user content unchanged.
- Keeps only the first 80 characters of assistant content and appends `...`
  when shortened.
- Replaces persisted tool messages with one line such as
  `[tool:query_order ok result=...]`.
- If layer 2 exceeds `layer2_budget`, trigger the background summary task.
- The current response never waits for that task. If the deterministic layer
  still cannot fit, omit its oldest shortened items from the current model
  input only.

### Layer 3: Append-Only Summaries

- Each summary row covers a non-overlapping older range.
- Existing summary rows are loaded as read-only background.
- New summaries cover only the uncovered layer 2 range at task start.
- Old summaries are never rewritten or merged into a new summary.

## Background Summary

The trigger decision is based on estimated usage, not message count.

Task lifecycle:

1. `summary trigger`: layer 2 estimate exceeds its budget.
2. `summary start`: a per-conversation task acquires ownership.
3. The task snapshots the uncovered message ids, including the current layer 1
   start as the exclusive upper boundary.
4. The summary model receives the facts-only prompt and the snapshot.
5. A successful result is inserted into `conversation_summaries`.
6. In the same transaction, `summary_upto_msg_id` advances to the inclusive
   snapshot end.
7. `summary done` records sequence number, coverage, and elapsed time.

`skip` is emitted when another task already covers the range or no new span
exists. `fail` records the exception without advancing an anchor, so the same
span can be retried safely after restart.

Task ownership is process-local and keyed by `session_id`; deployment remains
single-worker. Application shutdown waits briefly for running tasks, cancels
remaining tasks, and leaves their anchors unchanged.

## Summary Prompt Contract

The summary prompt extracts only:

- products mentioned;
- order numbers or phone numbers explicitly stated;
- explicit customer requests;
- unresolved problems.

Rules:

- Never invent a fact not present in the span.
- Remove greetings and small talk.
- Preserve numbers exactly.
- Prefer short factual statements.
- Output only the summary body, without JSON, Markdown, headings, or preamble.
- Target 20 to 200 Chinese characters, matching the requirement's
  "tens to a couple hundred characters" range.

The model does not receive old summaries for rewriting. Existing summaries are
provided only later, as background during answer generation.

## Context Assembly

`model_ctx` has a fixed order:

```text
1. SystemMessage: system persona, red lines, fixed suffix
2. Layer 2: deterministically shortened history
3. Layer 1: raw recent history
4. HumanMessage: current user message
5. HumanMessage: early summaries + current retrieval evidence
```

Tool definitions remain bound with `bind_tools` in a fixed order. They are part
of the stable request prefix and are serialized into `model_ctx` for
observation. Tools are not appended after variable history.

Summaries and evidence are never added as a separate `SystemMessage`. The
upstream template may hoist all system messages ahead of variable content,
which would invalidate the stable prefix and push tool definitions behind it.

`history_ctx` contains one summary line per loaded summary plus the same
reduced sliding window. It is produced every turn, including turns that later
use the fixed chitchat or fallback response.

## Observability

Add a file handler for:

```text
log/app.log
```

The console handler remains active. Each model-context record is one physical
log line with escaped JSON so `grep model_ctx` and `grep history_ctx` return
usable records:

```text
model_ctx session_id=... summary_json=... messages_json=... message_count=... token_estimate=...
history_ctx session_id=... summary_json=... messages_json=... message_count=... token_estimate=...
```

Summary records include:

```text
summary trigger session_id=... layer2_tokens=... budget=...
summary start session_id=... covered_from=... covered_to=...
summary done session_id=... sequence_no=... covered_from=... covered_to=... elapsed_ms=...
summary skip session_id=... reason=...
summary fail session_id=... error=... elapsed_ms=...
```

Log payloads contain conversation content but no API keys or credentials.

## API

### `/api/chat`

Add `user_key` to the request body. It is an opaque demo identifier, not a
profile.

```json
{
  "session_id": "s-abc",
  "user_key": "demo-user",
  "message": "最开始那个订单后来怎么说"
}
```

On first use, `ensure_conversation` stores `user_key` in
`conversations.user_name`. A later request for the same session with a
different key is rejected with `409`.

### `GET /api/conversations`

Query parameter:

```text
user_key
```

Response:

```json
[
  {
    "id": "s-abc",
    "preview": "最开始那个订单后来怎么说",
    "created_at": "2026-09-21T10:00:00",
    "updated_at": "2026-09-21T10:20:00",
    "summarized": true
  }
]
```

Ordering is `updated_at DESC`. The preview is the first user message,
truncated for display.

### `GET /api/conversations/{conversation_id}/messages`

Query parameter:

```text
user_key
```

Returns only user and assistant rows in ascending id order:

```json
[
  {
    "id": 1,
    "role": "user",
    "content": "订单 1001 什么时候到",
    "created_at": "2026-09-21T10:00:00"
  }
]
```

A missing conversation or a user-key mismatch returns `404`.

## Frontend

The existing native HTML/CSS/JavaScript and Vite frontend remains the target.

The chat page adds a conversation sidebar:

- lists the current user's conversations newest-first;
- shows the first-question preview and a summarized marker;
- highlights the active conversation;
- loads messages and replaces the visible chat log when a conversation is
  selected;
- creates only a new empty session for `新对话`, without deleting old rows;
- keeps sending, stopping, and reconnecting on the selected session.

Sidebar loading is best-effort. A failed list request hides the list and does
not block chat. A failed message reload preserves the current chat area and
shows the normal request error state.

## Error Handling

- Invalid or over-budget current input: return the existing fallback flow with
  a clear `上下文预算不足` reason before model invocation when a hard limit is
  exceeded.
- Summary model failure: log and leave the anchor unchanged.
- Summary insert conflict: treat the range as already covered and skip.
- History reload failure: keep the LangGraph execution result; persistence
  errors continue to be audit-logged.
- Conversation list failure: frontend silently degrades.
- Database schema mismatch: startup adds only the known additive Ch07 columns
  and table; it does not drop or rewrite existing data.

## Testing

### Unit and Integration

- Budget calibration produces `5650 / 3954 / 1695` for the acceptance config.
- Startup validation reports an insufficient budget before a model is called.
- Token estimation calibration covers CJK, ASCII, mixed text, and tool JSON.
- Layer 1 raw content is unchanged and starts on a user message.
- Layer 1 overflow moves only `layer1_from_msg_id`.
- Layer 2 shortens assistant content and tool messages while preserving user
  content.
- Summary ranges are append-only, non-overlapping, and idempotent.
- A delayed summary task does not block the current response.
- A failed summary task does not advance `summary_upto_msg_id`.
- Assembly order and stable tool-definition order are exact.
- No summary is emitted as a system message.
- `messages` persistence writes only user and assistant rows.
- Checkpoint recovery still restores the full `messages` state.
- Conversation list filtering, first-question preview, summary marker, and
  message reload are covered.

### Summary Evaluation

Add `eval_data/ch07_context_summary_eval_set.json` and
`scripts/eval_ch07.py`.

The evaluated cases cover:

- product, order number, phone number, explicit request, and unresolved issue;
- multiple numbers with exact preservation;
- small talk removal;
- no invented entities;
- mixed Chinese and tool-result text;
- summary length.

Thresholds:

```text
key_fact_recall >= 0.90
unsupported_fact_rate == 0
small_talk_leak_rate == 0
length_pass_rate >= 0.90
```

### Acceptance Demonstration

Default window:

```text
chat 20+ turns
no layer downgrade
no summary trigger
no summary rows
no token overflow
```

Acceptance window:

```text
MODEL_CONTEXT_WINDOW=18000
MAX_OUTPUT_TOKENS=2000
MAX_USER_INPUT_TOKENS=2000
MAX_AGENT_STEPS=3
TOOL_RESULT_MAX_TOKENS=1200
RERANK_TOP_K=5
```

The logs must show:

```text
layer1 downgrade ...
summary trigger ... layer2 ... > ...
summary done ... 第N段 ...
```

After summary completion, asking `最开始那个订单后来怎么说` must answer with
the order number and request retained in the summary.

Frontend acceptance:

- create or open multiple conversations;
- select between them;
- reload the complete user/assistant history;
- continue chatting in the selected conversation;
- verify no overlap at desktop and mobile widths.

## Configuration

Add:

```text
MODEL_CONTEXT_WINDOW=65536
MAX_OUTPUT_TOKENS=800
MAX_USER_INPUT_TOKENS=2000
MAX_AGENT_STEPS=5
TOOL_RESULT_MAX_TOKENS=1200
RERANK_TOP_K=20
SYSTEM_PROMPT_TOKEN_BUDGET=1800
EVIDENCE_CHUNK_TOKEN_BUDGET=400
SUMMARY_INJECTION_TOKEN_BUDGET=600
SAFETY_MARGIN_TOKENS=350
DESIRED_RETAINED_TURNS=24
STEADY_TURN_TOKEN_ESTIMATE=800
CHINESE_TOKENS_PER_CHAR=0.80
ASSISTANT_LAYER2_CHARS=80
SUMMARY_TARGET_MIN_CHARS=20
SUMMARY_TARGET_MAX_CHARS=200
CONTEXT_LOG_PATH=log/app.log
```

Existing `agent_max_output_tokens` is the source for `MAX_OUTPUT_TOKENS`;
the environment alias is kept explicit and does not duplicate the value.

## Deliverables

- Budget, estimator, trimmer, assembler, summary service, and task manager.
- Conversation anchors and append-only summary table.
- Graph context preparation with full-state preservation.
- Context logging in `log/app.log`.
- User-key-scoped read APIs.
- Multi-conversation frontend sidebar.
- Summary evaluation set and script.
- Default-window and acceptance-window demonstrations.
- `dev-notes/ch07.md` updated after each completed phase.
