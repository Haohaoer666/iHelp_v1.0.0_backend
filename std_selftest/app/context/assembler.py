"""Assemble stable model input and the shared history context."""

import json

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.context.models import ContextPack


def _evidence_text(evidence: list[dict]) -> str:
    return "\n".join(
        f"[{item.get('citation_id', index)}] {item.get('text', '')}"
        for index, item in enumerate(evidence, start=1)
    )


def _supplement_text(
    pack: ContextPack,
    evidence: list[dict],
    order_data: dict,
    after_sales_action: str,
) -> str:
    """
        拼接【给LLM的附加参考文本块】
        把分层上下文摘要、订单数据、售后流程标记、RAG检索到的政策证据组装成一段文本，
        拼进prompt，作为回答用户问题的外部事实依据。
        返回的字符串会追加到LLM提示词里，作为参考材料，不是对话消息历史。
    """
    sections: list[str] = []
    if pack.summaries_text:
        sections.append(f"【早期会话梗概】\n{pack.summaries_text}")
    if order_data:
        sections.append(
            "【本次订单事实】\n"
            + json.dumps(order_data, ensure_ascii=False)
        )
        action_label = {
            "exchange": "换货",
            "refund": "退款退货",
            "repair": "维修售后",
        }.get(after_sales_action, "售后")
        sections.append(
            f"当前属于{action_label}核心子流程。"
            "只依据订单事实和政策证据判断，不补造订单或政策内容；"
            "订单已确认，不得再次索要订单号。"
        )
    if evidence:
        sections.append(f"【本轮检索证据】\n{_evidence_text(evidence)}")
    return "\n\n".join(sections)


def build_model_messages(
    *,
    pack: ContextPack,
    evidence: list[dict],
    order_data: dict,
    after_sales_action: str,
    fixed_system: str,
    current_user: BaseMessage | None = None,
) -> list[BaseMessage]:
    """
        组装送给LLM的完整消息列表（LangChain BaseMessage数组）
        分层上下文构造：系统提示词 + layer2(高层摘要) + layer1(最近原始对话) + 补充业务材料
    """
    messages: list[BaseMessage] = [SystemMessage(fixed_system)]
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


def build_history_messages(pack: ContextPack) -> list[BaseMessage]:
    """
        从ContextPack组装送入LLM的对话消息列表（分层上下文）
        组装顺序：历史摘要 → layer2消息 → layer1消息，拼成最终消息数组，用于指代消解节点
    """
    messages: list[BaseMessage] = []
    if pack.summaries_text:
        messages.append(HumanMessage(f"【历史摘要】\n{pack.summaries_text}"))
    messages.extend(pack.layer2_messages)
    messages.extend(pack.layer1_messages)
    return messages
