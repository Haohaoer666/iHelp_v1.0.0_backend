"""Intent classification and deterministic workflow routing."""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage

from app.graph.state import ChatState
from app.tools.ticket_gate import deterministic_ticket_request
from app.tools.transfer_gate import (
    is_human_transfer_request,
    is_standalone_transfer_cancel,
)


INTENTS = (
    "物流",
    "订单",
    "商品咨询",
    "退款退货",
    "售后",
    "投诉",
    "闲聊",
    "转人工",
)
OTHER_INTENT = "其他"

INTENT_TO_ROUTE = {
    "物流": "business",
    "订单": "business",
    "商品咨询": "knowledge",
    "退款退货": "core_after_sales",
    "售后": "core_after_sales",
    "投诉": "complaint",
    "闲聊": "chitchat",
    "转人工": "human_transfer",
    OTHER_INTENT: "other",
}

AFTER_SALES_MCP_KEYWORDS = (
    "保修",
    "在保",
    "质保",
    "退货进度",
    "退货到哪",
    "退货处理到哪",
    "退款进度",
)


class IntentClassificationError(ValueError):
    """Raised when a structured intent response violates the contract."""


@dataclass(frozen=True)
class IntentDecision:
    intent: str
    route: str
    confidence: float
    model_used: str

    @property
    def reason(self) -> str:
        return f"{self.model_used} confidence={self.confidence:.2f}"


def other_intent(model_used: str = "fallback") -> IntentDecision:
    return IntentDecision(
        intent=OTHER_INTENT,
        route=INTENT_TO_ROUTE[OTHER_INTENT],
        confidence=0.0,
        model_used=model_used,
    )


def is_after_sales_mcp_query(text: str) -> bool:
    normalized = text.strip().lower()
    return any(
        keyword in normalized
        for keyword in AFTER_SALES_MCP_KEYWORDS
    )


def route_after_sales_mcp_query(
    decision: IntentDecision,
    message: str,
) -> IntentDecision:
    if decision.intent != "售后" or not is_after_sales_mcp_query(message):
        return decision
    return IntentDecision(
        intent=decision.intent,
        route="business",
        confidence=decision.confidence,
        model_used=f"{decision.model_used}+mcp",
    )


def _clean_json(value: str) -> str:
    value = value.strip()
    if not value.startswith("```"):
        return value
    value = re.sub(r"^```(?:json)?\s*", "", value)
    value = re.sub(r"\s*```$", "", value)
    return value.strip()


def parse_intent_response(
    value: str,
    *,
    model_used: str = "primary",
) -> IntentDecision:
    payload: Any = json.loads(_clean_json(value))
    if not isinstance(payload, dict) or set(payload) != {"intent", "confidence"}:
        raise IntentClassificationError("意图 JSON 必须只包含 intent 和 confidence")

    intent = payload.get("intent")
    confidence = payload.get("confidence")
    if intent not in INTENT_TO_ROUTE:
        raise IntentClassificationError(f"未知意图: {intent}")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= float(confidence) <= 1
    ):
        raise IntentClassificationError("confidence 必须是 0 到 1 之间的数值")

    return IntentDecision(
        intent=intent,
        route=INTENT_TO_ROUTE[intent],
        confidence=float(confidence),
        model_used=model_used,
    )


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def fallback_intent(text: str) -> IntentDecision:
    """Deterministic keyword fallback retained for diagnostic callers."""
    normalized = text.strip().lower()

    if is_human_transfer_request(normalized):
        intent = "转人工"
    elif _contains_any(normalized, ("投诉", "态度差", "追责", "非常不满意")):
        intent = "投诉"
    elif _contains_any(normalized, ("物流", "快递", "包裹", "运单", "到哪了", "什么时候到")):
        intent = "物流"
    elif _contains_any(
        normalized,
        (
            "退款为什么",
            "退款还没到",
            "换货审核",
            "维修",
            "售后处理",
            "保修",
            "在保",
            "质保",
            "退货进度",
            "退货到哪",
            "退款进度",
        ),
    ):
        intent = "售后"
    elif _contains_any(
        normalized,
        ("退货政策", "申请退款", "七天无理由", "退货条件", "怎么退"),
    ):
        intent = "退款退货"
    elif _contains_any(normalized, ("订单", "发货", "金额", "什么时候创建")):
        intent = "订单"
    elif _contains_any(
        normalized,
        ("刷新率", "有货", "多少钱", "支持", "参数", "价格", "库存"),
    ):
        intent = "商品咨询"
    elif _contains_any(normalized, ("你好", "天气", "笑话", "随便聊")):
        intent = "闲聊"
    else:
        return other_intent()

    return route_after_sales_mcp_query(
        IntentDecision(
            intent=intent,
            route=INTENT_TO_ROUTE[intent],
            confidence=0.0,
            model_used="keyword_fallback",
        ),
        text,
    )


def _history_text(history: Sequence[BaseMessage] | None) -> str:
    if not history:
        return "无"
    lines = []
    for item in history:
        role = "用户" if isinstance(item, HumanMessage) else "客服"
        content = item.content if isinstance(item.content, str) else str(item.content)
        if content.strip():
            lines.append(f"{role}：{content.strip()}")
    return "\n".join(lines) or "无"


def _new_classification_prompt(
    message: str,
    history: Sequence[BaseMessage] | None = None,
) -> str:
    return f"""你是电商客服意图分类器。请从以下九类中选择最符合用户“要做什么”的一项：

1. 物流
2. 订单
3. 商品咨询
4. 退款退货
5. 售后
6. 投诉
7. 闲聊
8. 转人工
9. 其他

严格只输出 JSON，不要解释，不要 Markdown：
{{"intent":"物流|订单|商品咨询|退款退货|售后|投诉|闲聊|转人工|其他","confidence":0.0}}

分类边界：
- 问“退货政策、七天无理由怎么算”= 退款退货；问“我的退款/换货处理到哪一步”= 售后。
- 问“订单状态、金额、创建时间”= 订单；问“包裹、快递、运单轨迹”= 物流。
- 问“商品参数、库存、价格、是否支持某功能”= 商品咨询；商品损坏后的维修/换货安排 = 售后。
- 明确表达不满、要求追责 = 投诉；与购物无关的寒暄 = 闲聊。
- 明确要求转人工、找人工客服、联系客服人员 = 转人工。
- 信息不足或无法可靠归入以上八类时 = 其他。拿不准时必须选其他，不要硬塞。

示例：
用户：七天无理由怎么算？
输出：{{"intent":"退款退货","confidence":0.96}}

用户：我的退款为什么还没到？
输出：{{"intent":"售后","confidence":0.91}}

用户：订单 1001 是什么状态？
输出：{{"intent":"订单","confidence":0.95}}

用户：包裹现在到哪了？
输出：{{"intent":"物流","confidence":0.97}}

用户：帮我看看这个奇怪的东西是不是那个意思？
输出：{{"intent":"其他","confidence":0.83}}

最近对话：
{_history_text(history)}

用户：{message}
输出："""


async def _classify_once(
    message: str,
    model: BaseChatModel,   # 本次要使用的模型实例（低成本小模型 / 主大模型）
    model_used: str,
    history: Sequence[BaseMessage] | None = None,
) -> IntentDecision:
    """
        单次意图分类执行函数
    """
    response = await model.ainvoke(
        [
            HumanMessage(
                content=_new_classification_prompt(message, history)
            )
        ]
    )
    content = getattr(response, "content", "") or ""
    if not isinstance(content, str):
        content = str(content)
    return route_after_sales_mcp_query(
        parse_intent_response(content, model_used=model_used),
        message,
    )


async def classify_intent_text(
    message: str,
    model: BaseChatModel,   #主模型（高精度大模型）
    *,
    history: Sequence[BaseMessage] | None = None,
    low_cost_model: BaseChatModel | None = None,    # 低成本小模型
    use_low_cost_first: bool = False,
    confidence_threshold: float = 0.75,
) -> IntentDecision:
    if is_standalone_transfer_cancel(message):
        return IntentDecision(
            intent="转人工",
            route="human_transfer",
            confidence=1.0,
            model_used="human_transfer_cancel_gate",
        )
    if is_human_transfer_request(message):
        return IntentDecision(
            intent="转人工",
            route="human_transfer",
            confidence=1.0,
            model_used="human_transfer_gate",
        )
    ticket_intent = deterministic_ticket_request(message)
    if ticket_intent is not None and ticket_intent.explicit_request:
        return IntentDecision(
            intent=OTHER_INTENT,
            route=INTENT_TO_ROUTE[OTHER_INTENT],
            confidence=1.0,
            model_used="ticket_gate",
        )
    if use_low_cost_first and low_cost_model is not None:
        try:
            low_decision = await _classify_once(
                message,
                low_cost_model,
                "low_cost",
                history,
            )    # 调用低成本小模型做意图分类，标记来源 low_cost
        except (
            json.JSONDecodeError,
            IntentClassificationError,
            TypeError,
            AttributeError,
            ValueError,
        ):
            low_decision = other_intent("low_cost")     # 小模型调用/解析报错，生成兜底意图，标记来源 low_cost
        if low_decision.confidence >= confidence_threshold:     # 判断：小模型置信度达标，直接返回结果，不调用昂贵大模型，节省token成本
            return low_decision

        try:
            return await _classify_once(
                message,
                model,
                "upgraded",
                history,
            ) # 小模型置信不够，升级：调用主大模型重新分类，标记来源 upgraded
        except (
            json.JSONDecodeError,
            IntentClassificationError,
            TypeError,
            AttributeError,
            ValueError,
        ):
            return other_intent("fallback")     # 升级到大模型也失败，返回兜底意图，标记 fallback

    # ===== 分支2：不开启小模型优先模式，直接使用主模型进行意图识别
    try:
        return await _classify_once(message, model, "primary", history)
    except (
        json.JSONDecodeError,
        IntentClassificationError,
        TypeError,
        AttributeError,
        ValueError,
    ):
        return other_intent()       # 主模型调用异常，兜底返回默认other意图


async def classify_intent_node(
    state: ChatState,
    model: BaseChatModel,
) -> dict[str, Any]:
    decision = await classify_intent_text(state["resolved_query"], model)
    return {
        "intent": decision.intent,
        "route": decision.route,
        "route_reason": decision.reason,
        "intent_confidence": decision.confidence,
        "intent_model_used": decision.model_used,
    }
