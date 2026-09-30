# iHelp Ch06 Query Understanding and Intent Routing Design

## Background

Ch05 established a deterministic LangGraph workflow and a ReAct Agent node. Its
query resolution and intent classification are intentionally minimal:

- `resolve_reference` passes the current message through unchanged.
- Intent classification classifies the current message without conversation
  history.
- Intent output contains `intent` and `reason`, not a confidence score.
- Refund and after-sales requests do not have a deterministic order and policy
  collection subflow.

Ch06 turns the placeholder diverter into the production routing node for the
customer-service workflow. Query resolution, query rewriting, query expansion,
intent classification, order selection, policy retrieval, and refund form
handling become explicit parts of the graph and API contract.

## Goal

Build a context-aware and confidence-aware routing layer that:

1. Resolves pronouns and colloquial expressions with the conversation history.
2. Classifies one of seven business intents or the fallback `Other` intent with
   strict JSON output.
3. Forces refund/return and after-sales requests through a deterministic order
   and policy collection subflow.
4. Interrupts safely when an order number is missing, obtains the order through
   a chat order selector, and resumes the same graph run.
5. Expands only high-risk refund/after-sales retrieval queries, merges and
   deduplicates evidence, and leaves the final eligibility judgment to the main
   Agent.
6. Supports a high-accuracy model by default and an optional low-cost-first
   confidence upgrade path.

## Non-Goals

- No fine-tuning or BERT intent classifier.
- No cross-session memory.
- No duplicate storage of expanded queries in the knowledge base.
- No new search service, queue, or orchestration component.
- No replacement of the existing chat upstream or model provider.

## Context7 Findings

The implementation follows the current official documentation checked through
Context7:

- LangGraph Python 1.x uses checkpointer-backed `interrupt()` for human input.
  Resume input is supplied with `Command(resume=...)`. A node containing
  `interrupt()` is re-executed from its beginning after resume, so side effects
  must not precede the interrupt.
- FastAPI `StreamingResponse` supports async generators; generators must reach
  an await point so client disconnect cancellation can propagate.
- SQLAlchemy 2.0 async sessions continue to use `AsyncSession` and explicit
  transaction boundaries. Ch06 does not add a database abstraction.

## Architecture

The parent graph remains the deterministic workflow. Ch06 changes the first
half of the graph:

```text
START
  -> understand_query
  -> classify_intent
  -> route_after_classify
       -> core_after_sales
       |    -> build_order_options
       |    -> select_order (interrupt when required)
       |    -> load_order
       |    -> expand_queries
       |    -> retrieve_policy
       |    -> confidence_gate
       |    -> agent
       |
       -> knowledge
       |    -> retrieve_knowledge
       |    -> confidence_gate
       |    -> agent
       |
       -> business -> agent
       -> complaint -> complaint_response
       -> chitchat -> chitchat_response
       -> other -> fallback
  -> log_turn
  -> END
```

`退款退货` and `售后` use `core_after_sales`. `商品咨询` uses the existing
knowledge path. `物流`, `订单`, and `商品咨询` continue to use the existing
Agent behavior. `投诉` and `闲聊` keep their fixed branches. `其他` uses a
controlled fallback and never forces the request into a business intent.

## Query Understanding

### Contracts

`understand_query` combines reference resolution and query rewriting. It uses
the current user message plus recent conversation history. It returns:

```json
{"resolved_query":"用户主动申请退款，想确认订单 1001 是否可以退"}
```

The model is instructed to:

- Resolve pronouns such as `它`, `这个`, and `那单` from the latest unambiguous
  conversation context.
- Normalize colloquial wording into a standard customer-service question.
- Return the original question unchanged when it is already complete and
  unambiguous.
- Never invent an order number, product model, amount, or policy condition.

No history is persisted beyond the LangGraph session checkpoint and the
existing MySQL audit trail.

### State

Add these fields to `ChatState`:

| Field | Meaning |
| --- | --- |
| `query_history` | User/assistant messages supplied to the understanding prompt |
| `query_changed` | Whether the resolved query differs from the current message |
| `query_understanding_source` | `model` or `fallback` |
| `intent_confidence` | Confidence returned by the intent classifier |
| `intent_model_used` | `primary`, `low_cost`, or `upgraded` |
| `order_id` | Selected or extracted order number |
| `order_options` | Cards shown by the order selector |
| `order_data` | Deterministic order facts used by the Agent |
| `expanded_queries` | Deduplicated retrieval queries |
| `retrieval_query_count` | Number of retrieval queries executed |
| `pending_action` | `order_selection`, `refund_form`, or empty |

## Intent Classification

### Output Contract

Intent classification uses one strict JSON shape:

```json
{"intent":"物流","confidence":0.94}
```

The only valid business choices are:

- 物流
- 订单
- 商品咨询
- 退款退货
- 售后
- 投诉
- 闲聊

`其他` is the eighth fallback choice. The parser rejects unknown intent names,
missing confidence, non-numeric confidence, and values outside `[0, 1]`.

### Prompt

The prompt is built from four required parts:

1. Enumerate the seven business intents plus `其他` as a multiple-choice
   classification.
2. Require JSON with only `intent` and `confidence`.
3. Include boundary examples for refund policy versus after-sales progress,
   order status versus logistics, product consultation versus after-sales, and
   ambiguous text.
4. State that `其他` is the safety fallback and that uncertain requests must
   not be forced into a business intent.

### Model Routing

The default path uses the configured primary chat model once. This favors
accuracy while the intent evaluation is being established.

An optional cost-saving path uses `INTENT_LOW_COST_MODEL` first. If its
confidence is below `INTENT_CONFIDENCE_THRESHOLD`, the primary model re-judges
once. The final decision records whether it came from the low-cost model or the
upgraded primary model.

## Query Expansion

Query expansion runs only for `退款退货` and `售后`.

Input:

```json
{"resolved_query":"用户想确认订单 1001 是否支持七天无理由退货"}
```

Output:

```json
{"queries":["七天无理由退货条件","签收后七天退货时效","退款退货政策"]}
```

The output must contain exactly one top-level field named `queries` whose value
is an array of non-empty strings. The service trims, removes duplicates while
preserving order, limits the list to `QUERY_EXPANSION_MAX_QUERIES`, and
guarantees that the original resolved query is the first retrieval query.

The expansion is retrieval-side only. Knowledge chunks remain stored once.

## Core After-Sales Subflow

### Order Collection

`build_order_options` reads a deterministic demo order catalog and emits:

```json
{
  "type": "order_selector",
  "orders": [
    {
      "order_id": "1001",
      "product_name": "iHao 智能手表",
      "status": "已签收",
      "amount": 899.0,
      "signed_at": "2026-09-15"
    }
  ]
}
```

If the resolved query already contains an order number, the graph skips the
selector and loads that order directly.

If an order number is missing, `select_order` calls `interrupt(...)`. The
interrupt value contains the selector payload:

```json
{"type":"order_selection","orders":[...]}
```

The resume value must be:

```json
{"type":"order_selected","order_id":"1001"}
```

The node validates the selected order against the offered list. The node does
not call `interrupt()` after a side effect.

The order catalog is deterministic and shared by the selector and
`query_order`. This keeps the card, the order tool result, and the policy
decision consistent during demos and tests. The adapter boundary is isolated
so a real order source can replace it later without changing the graph
contract.

### Policy Retrieval

`expand_queries` calls the query-expansion model. `retrieve_policy` runs all
expanded queries through the existing hybrid retriever, merges results by
`chunk_id`, keeps the highest score, sorts the merged results, and passes the
result through the existing evidence assembly and confidence gate.

The Agent receives:

- the resolved query;
- the selected order facts;
- the deduplicated policy evidence and citations;
- an instruction to decide only whether that order can be refunded under the
  retrieved policy.

No policy-only heuristic inside retrieval decides eligibility. The main Agent
performs the final reasoning.

## Slot Handling

- Missing order number: interrupt and show the order selector. The model never
  guesses an order number.
- Refund reason: never ask a follow-up question during intent classification or
  order selection. It is selected from a fixed list only when the user submits
  the refund form.
- Requirement clarification: remains in the main Agent.
- Intent classification only answers what the user wants to do, not whether all
  execution slots are already available.

Fixed refund reasons:

- 商品质量问题
- 七天无理由
- 尺寸或型号不合适
- 收到商品破损
- 少件或错件
- 其他

## API and SSE Contracts

### `/api/chat`

The request model accepts either a new message or an interrupt resume payload:

```json
{"session_id":"s1","message":"这个能退吗"}
```

```json
{
  "session_id":"s1",
  "resume":{"type":"order_selected","order_id":"1001"}
}
```

Exactly one of `message` and `resume` must be present. A new message invokes the
graph normally. A resume invokes the graph with `Command(resume=...)` and the
same `thread_id`.

The SSE stream continues to use custom events and ends with:

```text
data: [DONE]
```

New event types:

```json
{"type":"query_understood","original":"这个能退吗","resolved_query":"用户想确认订单 1001 是否可以退款","changed":true}
{"type":"intent","intent":"退款退货","confidence":0.94,"model_used":"primary"}
{"type":"order_selector","orders":[...]}
{"type":"order_selected","order_id":"1001","order":{...}}
{"type":"expanded_queries","queries":["...","..."],"count":3}
{"type":"refund_form","order_id":"1001","reasons":["商品质量问题","七天无理由"]}
```

### `/api/refunds`

`POST /api/refunds` validates the order number and fixed reason, then reuses the
existing `create_ticket` tool with `ticket_type="退款"`. The description stores
the order number and selected reason. No new persistence component is added.

`GET /api/refunds/reasons` returns the fixed reason list used by the frontend.

## Frontend

The existing native HTML/CSS/JavaScript frontend remains the target.

The chat stream handles:

- `query_understood` and `intent` events for status display only;
- `order_selector` by rendering clickable order cards in the chat stream;
- resume by posting `{"session_id":...,"resume":{...}}` and continuing the same
  assistant message flow;
- `refund_form` by rendering a compact form with a fixed-reason select;
- refund submission through `/api/refunds`.

The frontend work follows the user's Vibe Coding exception: it is implemented
directly from the requested interaction, then visually verified in a browser.
It does not use TDD or code review steps.

## Errors and Observability

- Query understanding JSON failure: use the original message and mark
  `query_understanding_source=fallback`.
- Intent JSON failure: return `其他` with confidence `0.0` and log the parse
  failure.
- Missing or invalid order selection: re-emit the selector without guessing.
- Expansion failure: retrieve the original resolved query only.
- Multi-query retrieval failure for one query: record the failed query and
  continue with the successful queries.
- Refund form validation failure: return a 422 response from FastAPI/Pydantic.

Turn logs add:

- resolved query;
- intent confidence;
- intent model route;
- selected order number;
- expanded query count;
- policy retrieval count;
- pending action.

## Testing and Evaluation

### Unit and Integration

- Query understanding returns a standalone query and preserves an already
  complete query.
- Intent parser accepts the exact JSON shape and rejects invalid values.
- Intent fallback produces `其他` when output is malformed or uncertain.
- Boundary few-shot cases classify refund policy separate from after-sales
  progress.
- Query expansion produces only the `queries` field, deduplicates, and enforces
  the maximum count.
- Multi-query retrieval merges duplicate chunks and keeps the highest score.
- Refund and after-sales routes always collect order data and policy evidence
  before Agent execution.
- Missing order number interrupts and resume continues the same graph run.
- Invalid resume value re-prompts instead of inventing an order.
- `/api/chat` accepts either message or resume, but rejects both or neither.
- `/api/refunds` validates fixed reasons and reuses `create_ticket`.
- Existing Ch05 workflow tests remain green after adapting the namespaced test
  package.

### Labeled Evaluation

Add `eval_data/ch06_query_intent_eval_set.json` with multi-turn conversations:

- logistics -> refund -> logistics;
- pronoun resolution to the most recent order;
- complete question unchanged;
- refund policy versus after-sales progress boundary cases;
- ambiguous and out-of-domain questions expected to become `其他`.

Add `scripts/eval_ch06.py` reporting:

- JSON parse rate;
- intent accuracy;
- resolved-query match rate;
- fallback-to-Other rate for ambiguous examples;
- confusion matrix.

The evaluation command fails below:

- JSON parse rate `0.98`;
- intent accuracy `0.90`;
- resolved-query match rate `0.80`.

### Browser Acceptance

1. A multi-turn logistics -> refund -> logistics conversation resolves each
   pronoun and classifies each turn correctly.
2. A complete question passes through unchanged.
3. `这个能退吗` resolves the order reference and shows order and policy events.
4. Asking for a refund without an order number shows clickable order cards.
5. Selecting a card resumes the same flow and reaches the Agent decision.
6. The refund form lists only fixed reasons and submits a refund ticket.

## Rollout and Configuration

Add:

```text
QUERY_UNDERSTANDING_MODEL=
QUERY_EXPANSION_MODEL=
INTENT_LOW_COST_MODEL=
INTENT_CONFIDENCE_THRESHOLD=0.75
INTENT_USE_LOW_COST_FIRST=false
QUERY_EXPANSION_MAX_QUERIES=4
REFUND_REASONS=商品质量问题,七天无理由,尺寸或型号不合适,收到商品破损,少件或错件,其他
```

Empty model values mean "use the existing primary chat model".

## Deliverables

- Query understanding service and prompt.
- Intent classification contract and evaluation.
- Query expansion and multi-query evidence merge.
- Deterministic demo order catalog and order selector interrupt flow.
- Refund reason API and refund form API.
- Chat SSE resume support.
- Frontend order cards and refund form.
- `dev-notes/ch06.md` appended after every completed phase.
- Demo commands, evaluation results, and final verification output.
