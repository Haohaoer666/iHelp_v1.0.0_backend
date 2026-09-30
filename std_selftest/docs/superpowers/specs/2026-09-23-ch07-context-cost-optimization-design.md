# iHelp Ch07 Context Cost Optimization Design

## Background

Ch07 already introduced an append-only summary table, two conversation
anchors, a `ContextManager`, and a per-turn `ContextPack`. The current budget
code also already contains the core idea of a retained-history target:

```text
desired_history =
    desired_retained_turns * steady_turn_token_estimate

history_budget =
    min(desired_history, window_available)
```

That is the right direction, but the effective request is still not fully
bounded:

- `agent_token_budget=4000` is still a separate pre-request gate and can stop a
  request far below the configured window;
- `summary_injection_token_budget` is reserved but the actual injected summary
  text is not hard-capped;
- evidence is budgeted from `RERANK_TOP_K * evidence_chunk_token_budget`, while
  retrieval can return a different number of chunks and the content is not
  actually trimmed to a token limit;
- the ReAct peak counts tool results but not tool-call messages, tool
  arguments, or real per-step payload truncation;
- the current user message is reserved through
  `MAX_USER_INPUT_TOKENS` and can still be counted inside layer 1;
- `model_ctx.token_estimate` does not describe the complete request, so the
  log can understate system, tool, evidence, summary, and ReAct cost;
- the legacy acceptance numbers are useful regression data but are not
  explicitly separated from the production cost profile.

This design keeps the existing three-layer context model and makes the cost
boundary explicit without changing the required technology stack.

## Goal

Make the context request predictable and cheaper to operate by using:

```text
64K model window as a hard ceiling
24 x 800 = 19.2K as the normal history soft budget
actual-token caps for volatile request components
```

The change must:

1. Preserve the current three-layer semantics.
2. Keep `64K` as a safety ceiling, not as a target to fill.
3. Keep `24 x 800 = 19.2K` as the default retained-history target.
4. Remove the conflicting standalone `agent_token_budget` path.
5. Count the current user message exactly once.
6. Truncate summaries, evidence, and tool payloads by actual token limits.
7. Count and log the complete request before a model call.
8. Preserve the legacy acceptance calculation behind an explicit
   compatibility profile.

## Non-Goals

- No cross-session memory.
- No user profile or personalization.
- No semantic retrieval over old conversation messages.
- No replacement of LangGraph, LangChain, SQLAlchemy, FastAPI, or MySQL.
- No multi-worker summary coordination.
- No per-intent history budget in the first implementation.

Per-intent budgets are deferred because `prepare_context` executes before
intent classification. Implementing them would require an additional
classification pass or a second context assembly pass, which is not justified
until production cost data shows a need.

## Context7 Gate

Before implementation, re-check the current official documentation through
Context7 for:

- LangChain `count_tokens_approximately` behavior and message accounting;
- LangChain `bind_tools` request serialization and tool order;
- LangGraph `StateGraph`, `add_messages`, and checkpointer state recovery;
- SQLAlchemy 2 async transaction ownership and summary insertion;
- FastAPI lifespan resource setup if an existing interface changes.

No implementation should rely on remembered signatures when the installed
version differs from the documented example.

## Budget Model

### Planning and actual counts

There are two distinct token concepts:

1. **Planning estimate**
   `estimate_text_tokens` remains the conservative CJK/ASCII estimator used
   before assembly and for budget decisions.
2. **Actual prompt count**
   A full-request counter measures the assembled `SystemMessage`, layered
   messages, current user message, dynamic supplement, tool-call messages,
   and tool results using the available LangChain token counter or model
   tokenizer.

The estimator may be conservative, but the final guard must use the best
available actual prompt count.

### Hard and soft history budgets

```text
W = MODEL_CONTEXT_WINDOW

O = MAX_OUTPUT_TOKENS
S = SAFETY_MARGIN_TOKENS
P = actual system and tool-definition tokens
C = actual current-user tokens
E = min(EVIDENCE_TOTAL_MAX_TOKENS, actual evidence tokens)
M = min(SUMMARY_INJECTION_MAX_TOKENS, selected summary tokens)
R = MAX_AGENT_STEPS x
    (TOOL_CALL_MESSAGE_MAX_TOKENS + TOOL_RESULT_MAX_TOKENS + overhead)

hard_history_budget =
    max(0, W - O - S - P - C - E - M - R)

soft_history_budget =
    DESIRED_RETAINED_TURNS x STEADY_TURN_TOKEN_ESTIMATE

history_budget =
    min(hard_history_budget, soft_history_budget)
```

The production default is:

```text
W = 65536
O = 2000
S = 350
P = approximately 519 for the current system prompt and three tools
C = 2000
E = 3000
M = 600
R = 3 x (400 + 1200 + 8) = 4824

hard_history_budget = 52243
soft_history_budget = 19200
history_budget = 19200
```

The layer split is:

```text
layer1_budget = floor((history_budget - 1) x 0.70)
layer2_budget =
    history_budget - 1 - layer1_budget
```

For the default profile this is approximately `13439 / 5760`, or
`13.4K / 5.76K`.

### Request size under the default profile

```text
initial prompt maximum
= P + C + E + M + history_budget
= approximately 25319 tokens

ReAct peak input
= initial prompt maximum + R
= approximately 30143 tokens

total window reservation including output and safety
= approximately 32493 tokens
```

Typical sessions will be lower because evidence, summaries, and tool calls are
not present on every turn. `64K` is reserved for unexpected long inputs and
tool-heavy turns, not used as a normal operating level.

## Configuration

The production profile introduces or formalizes:

```text
MODEL_CONTEXT_WINDOW=65536
HISTORY_SOFT_BUDGET_ENABLED=true
DESIRED_RETAINED_TURNS=24
STEADY_TURN_TOKEN_ESTIMATE=800
HISTORY_LAYER1_RATIO=0.70
HISTORY_LAYER2_RATIO=0.30

MAX_OUTPUT_TOKENS=2000
MAX_USER_INPUT_TOKENS=2000
MAX_AGENT_STEPS=3
TOOL_CALL_MESSAGE_MAX_TOKENS=400
TOOL_RESULT_MAX_TOKENS=1200

EVIDENCE_TOTAL_MAX_TOKENS=3000
SUMMARY_SEGMENT_MAX_TOKENS=200
SUMMARY_INJECTION_MAX_TOKENS=600
SAFETY_MARGIN_TOKENS=350
```

`EVIDENCE_TOTAL_MAX_TOKENS` is the authoritative cap. When it is unset, the
legacy `RERANK_TOP_K * EVIDENCE_CHUNK_TOKEN_BUDGET` value may be derived for
backward compatibility. Actual evidence content is still truncated to the
effective cap.

`agent_token_budget` is deprecated. It may remain as a settings alias for one
release cycle, but the production Agent path must not use it as a separate
request gate. The only final guard is:

```text
actual_full_prompt_tokens + MAX_OUTPUT_TOKENS + SAFETY_MARGIN_TOKENS
<= MODEL_CONTEXT_WINDOW
```

## Entry Budget Profile

### Default: cost_optimized

- Uses actual system/tool/evidence/summary/tool payload counts.
- Uses the `19.2K` history soft budget.
- Includes tool-call messages and tool results in the ReAct peak.
- Enforces hard token limits at the point where content enters the request.
- Logs complete component usage.

### Compatibility: legacy_acceptance

The existing Ch07 demonstration contract must remain available:

```text
MODEL_CONTEXT_WINDOW=18000
MAX_OUTPUT_TOKENS=2000
MAX_USER_INPUT_TOKENS=2000
MAX_AGENT_STEPS=3
TOOL_RESULT_MAX_TOKENS=1200
RERANK_TOP_K=5
SYSTEM_PROMPT_TOKEN_BUDGET=1800
EVIDENCE_CHUNK_TOKEN_BUDGET=400
SUMMARY_INJECTION_TOKEN_BUDGET=600
SAFETY_MARGIN_TOKENS=350
DESIRED_RETAINED_TURNS=24
STEADY_TURN_TOKEN_ESTIMATE=800
```

This profile preserves:

```text
history_budget=5650
layer1_budget=3954
layer2_budget=1695
```

For this compatibility profile only:

```text
P = SYSTEM_PROMPT_TOKEN_BUDGET
E = RERANK_TOP_K x EVIDENCE_CHUNK_TOKEN_BUDGET
R = MAX_AGENT_STEPS x TOOL_RESULT_MAX_TOKENS
```

The legacy profile does not add `TOOL_CALL_MESSAGE_MAX_TOKENS` to the ReAct
peak. New profiles do.

The compatibility profile exists for regression, evaluation, and the existing
demo command. It is not the production default.

## Context Assembly Contract

The complete request order is fixed:

```text
1. SystemMessage: persona, red lines, stable tool definition prefix
2. layer2: deterministically shortened older messages
3. layer1: recent raw messages
4. current user message
5. one dynamic HumanMessage:
   early summaries + retrieval evidence + current order facts
```

Rules:

- Only the first message may be a system message in the assembled context.
- Summaries and evidence must never be separate system messages.
- The current user message is not also included in layer1.
- Tool definitions remain in a stable order through `bind_tools`.
- Dynamic order facts and evidence must not be hoisted into the stable system
  prefix.

`history_ctx` is generated on every turn and contains the selected summary
text plus the reduced sliding window. It is used by reference resolution and
intent classification, including fixed chitchat/fallback paths.

## Three-Layer Behavior

### Layer 1: raw recent history

- Keeps complete user and assistant messages unchanged.
- Excludes `current_user_message_id` from the historical layer.
- Moves only `layer1_from_msg_id` when raw content exceeds `layer1_budget`.
- Does not modify rows in `messages`.

### Layer 2: deterministic shortened history

- Keeps user content unchanged.
- Keeps at most 80 characters of assistant content and appends `...` when
  shortened.
- Replaces large tool results with a one-line identifier.
- Is measured with the planning estimator.
- Triggers background summarization only when its actual usage exceeds
  `layer2_budget`.

### Layer 3: append-only summaries

- Each database row covers one non-overlapping message range.
- A segment has a hard generation limit of `SUMMARY_SEGMENT_MAX_TOKENS`.
- Injection selects the newest complete segments whose total estimated tokens
  do not exceed `SUMMARY_INJECTION_MAX_TOKENS`.
- A segment is not partially injected; if the newest segment alone exceeds
  the cap, it is trimmed only at generation/persistence time and a warning is
  logged.
- Existing summaries are never rewritten or merged.

## ReAct and Tool Context

Within a single Agent execution:

- an `AIMessage` containing `tool_calls` is appended to `agent_messages`;
- each `ToolMessage` result is appended to `agent_messages`;
- tool args and tool results are each truncated to their configured token
  budget before append;
- the ReAct loop checks the next-step worst case before making another model
  call;
- when the next step cannot fit, the graph returns the existing limit response
  instead of overflowing the model window.

Tool calls and tool results:

- are transient within the current Agent execution;
- are not written to `messages`;
- are not included in layer1 or layer2 for later turns;
- remain visible in `model_ctx` and tool trace logs.

If a tool fact must survive into later turns, the final assistant answer must
state it or the summary input must explicitly include the relevant turn.
Semantic retrieval of old tool results is out of scope.

## Summary Task Contract

The existing background task ownership model remains:

```text
trigger -> start -> done
                   -> fail
                   -> skip
```

Every event logs:

- `session_id`;
- covered message range;
- trigger tokens versus budget;
- elapsed time for start/done/fail;
- sequence number for completed summaries;
- skip reason or error text.

`ContextManager.prepare` must never await a summary task. A failed task does
not advance `summary_upto_msg_id` and the same range can be retried safely.

The summary prompt remains facts-only. It may preserve products, order
numbers, phone numbers, explicit requests, and unresolved issues. It must not
invent any fact not present in the source span, and it removes greetings and
small talk.

## Persistence and State

`conversations` and `messages` remain the authoritative source for historical
context assembly. The LangGraph checkpoint remains the authoritative graph
state store.

The two paths do not compete:

- `messages` is read to build the reduced `ContextPack`;
- `ChatState.messages` continues to use the `add_messages` reducer and is
  durable through the checkpointer;
- `prepare_context` never replaces `ChatState.messages`;
- an interrupt/resume does not append the current user message twice;
- tool rows are not inserted into `messages`.

This preserves the original Ch07 requirement that full history and reduced
model input have separate lifecycles.

## Observability

`model_ctx` must log the complete request and a component breakdown:

```text
model_ctx
  session_id=...
  full_prompt_tokens=...
  prompt_limit=...
  model_context_window=...
  system_tokens=...
  tool_definition_tokens=...
  summary_tokens=...
  evidence_tokens=...
  layer1_tokens=...
  layer2_tokens=...
  current_user_tokens=...
  react_peak_reserved=...
  output_reserved=...
  safety_margin=...
  history_soft_budget=...
  history_hard_budget=...
  summary_json=...
  messages_json=...
```

`history_ctx` must be logged every turn with:

- selected summary text;
- sliding-window messages;
- message count;
- token estimate.

The log must remain one physical line per event so `grep model_ctx` and
`grep history_ctx` return complete records.

## Error Handling

The degradation order is fixed:

1. Trim optional evidence to `EVIDENCE_TOTAL_MAX_TOKENS`.
2. Trim summary injection to `SUMMARY_INJECTION_MAX_TOKENS`.
3. Trim layer2 to `layer2_budget`.
4. Move `layer1_from_msg_id` toward newer messages.
5. If mandatory fixed content, current input, output reserve, and safety
   margin still do not fit, return `上下文预算不足` without calling the model.

The current user's message is validated during `prepare_context` before any
model invocation in the request. The API persists the user message before
entering the graph, then passes `current_user_message_id` to the context
manager. A current message exceeding `MAX_USER_INPUT_TOKENS` produces the
existing fallback path and a structured log record.

No dynamic component may silently push the complete request beyond the model
window.

## Compatibility and Migration

- No new table is required; existing `conversation_summaries` remains the
  storage unit.
- `agent_token_budget` is deprecated, not deleted abruptly.
- Existing conversation rows and historical tool rows are left untouched.
- The compatibility profile preserves the exact Ch07 acceptance numbers.
- The default profile changes only request budgeting and truncation, not the
  existing UI or conversation ownership behavior.

## Testing Strategy

### Budget and estimator tests

- Default profile computes `hard=52243`, `soft=19200`, `history=19200`.
- Layer split is exactly `13439 / 5760` under the existing `-1` guard.
- Legacy profile computes `5650 / 3954 / 1695`.
- CJK and ASCII estimation remains calibrated together with the budget.
- Actual full prompt token counting includes every message component.

### Context behavior tests

- Current user appears exactly once.
- Layer1 contains no current user message.
- Layer2 preserves user content and shortens assistant/tool content.
- Only one system message appears.
- Summary and evidence appear after the current user message.
- Summary injection never exceeds `SUMMARY_INJECTION_MAX_TOKENS`.
- Evidence is truncated to `EVIDENCE_TOTAL_MAX_TOKENS`.
- Tool calls and tool results are truncated before append.
- Summary trigger is based on token usage, not message count.
- A slow summary task does not delay the current response.

### Acceptance tests

Default profile with 20+ turns:

- no `layer1 downgrade`;
- no `summary trigger`;
- no summary rows;
- no budget fallback;
- full prompt stays within the configured hard ceiling.

Legacy acceptance profile:

- logs contain `layer1 downgrade`;
- logs contain `summary trigger` with layer2 tokens and budget;
- logs contain `summary done` with sequence number and elapsed time;
- asking `最开始那个订单后来怎么说` answers from the retained summary facts.

### Summary evaluation

Run the existing labeled summary evaluation set with the same thresholds:

```text
key_fact_recall >= 0.90
unsupported_fact_rate == 0
small_talk_leak_rate == 0
length_pass_rate >= 0.90
```

## Deliverables

- Updated design and implementation plan documents.
- Soft/hard budget model and legacy compatibility profile.
- Actual token counting and truncation for evidence, summaries, and tools.
- Full-request Agent guard replacing the standalone `agent_token_budget`.
- Complete `model_ctx` and `history_ctx` observability.
- Regression, acceptance, and summary evaluation results.
- `dev-notes/ch07.md` updated at each implementation and review stage.
