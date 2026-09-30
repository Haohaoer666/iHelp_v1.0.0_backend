from types import SimpleNamespace

from app.context.budget import ContextBudget
from app.context.manager import ContextManager


class Store:
    async def get_conversation(self, session_id):
        return SimpleNamespace(
            layer1_from_msg_id=1,
            summary_upto_msg_id=None,
        )

    async def load_messages(self, session_id):
        return [
            SimpleNamespace(
                id=1,
                role="user",
                content="太长" * 100,
                tool_name=None,
            )
        ]

    async def load_summaries(self, session_id):
        return []

    async def update_layer1_anchor(self, session_id, message_id):
        return None


class Tasks:
    def schedule(self, *args, **kwargs):
        raise AssertionError("summary must not start for oversized input")


async def test_oversized_current_user_is_rejected_before_model():
    manager = ContextManager(
        store=Store(),
        budget=ContextBudget(
            model_context_window=10000,
            max_output_tokens=0,
            max_user_input_tokens=10,
            max_agent_steps=0,
            tool_result_max_tokens=0,
            rerank_top_k=0,
            evidence_chunk_token_budget=0,
            system_prompt_token_budget=0,
            summary_injection_token_budget=0,
            safety_margin_tokens=0,
            desired_retained_turns=100,
            steady_turn_token_estimate=1,
        ),
        tasks=Tasks(),
        chinese_tokens_per_char=1.0,
        assistant_layer2_chars=80,
        estimator=lambda text: len(text),
    )

    pack = await manager.prepare("s1", 1)

    assert pack.budget_error == "上下文预算不足"

