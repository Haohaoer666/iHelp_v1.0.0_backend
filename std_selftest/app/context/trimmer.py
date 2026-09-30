"""Build raw, shortened, and summarized context layers."""

from langchain_core.messages import HumanMessage

from app.context.budget import ContextBudget
from app.context.models import (
    ContextMessage,
    ContextPack,
    SummaryRow,
    SummarySpan,
)
from app.context.tokens import (
    estimate_messages_tokens,
    estimate_text_tokens,
)


def _shorten_message(
    message: ContextMessage,
    assistant_chars: int,
) -> str:
    """
        精简消息文本，用于layer2旧消息压缩：
        user消息原样保留；assistant超长则截断加省略号；tool消息压缩成简短标记文本
    """
    if message.role == "user":
        return message.content
    if message.role == "assistant":
        if len(message.content) <= assistant_chars:     #80
            return message.content
        return message.content[:assistant_chars] + "..."
    status = "error" if message.content.startswith("error") else "ok"
    return f"[tool:{message.tool_name or 'tool'} {status} id={message.message_id}]"


def _fit_newest(
    messages: list[ContextMessage],
    max_tokens: int,
    chinese_tokens_per_char: float,
) -> list[ContextMessage]:
    """
        从消息列表里保留最新的若干消息，整体总token不超过max_tokens；
        倒序遍历，从最新消息往前收集，凑到token上限就停止，保证保留的都是最新对话
    """
    if not messages:
        return []
    kept: list[ContextMessage] = []
    used = 0
    for message in reversed(messages):
        tokens = estimate_messages_tokens(
            [message.to_langchain()],
            chinese_tokens_per_char,
        )
        if not kept and tokens > max_tokens:
            return []
        if kept and used + tokens > max_tokens:
            break
        kept.insert(0, message)
        used += tokens
    return kept


def _select_summaries(
    summaries: list[SummaryRow],
    max_tokens: int,
    chinese_tokens_per_char: float,
) -> tuple[str, int]:
    """在摘要token上限内，选取**最新的连续摘要片段**。"""
    if max_tokens <= 0:
        return "", 0
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


def build_context_pack(
    *,
    messages: list[ContextMessage],
    summaries: list[SummaryRow],
    layer1_from_msg_id: int | None,
    summary_upto_msg_id: int | None,
    budget: ContextBudget,
    chinese_tokens_per_char: float,
    assistant_layer2_chars: int,
    current_user_message_id: int | None = None,
) -> ContextPack:
    """
        分层上下文核心裁剪函数：
        对会话消息做分层切分：已摘要的历史摘要文本 + layer2（旧消息精简版） + layer1（最近完整原始对话）；
        不断裁剪layer1保证其token不超layer1预算；如果layer2超限,则后台生成摘要，并精简layer2；
        组装最终送入LLM的完整上下文 history_ctx，返回不可变ContextPack
    """
    ordered = sorted(messages, key=lambda item: item.message_id)    # 消息按message_id从小到大排序
    if not ordered:
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
        )

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
    summary_floor = summary_upto_msg_id or 0    # 已经归档摘要的消息上限，这部分消息不再重复参与layer1/layer2，只用摘要文本
    eligible = [item for item in historical if item.message_id > summary_floor]    # eligible：还未归档成摘要，需要参与本次分层裁剪的消息

    # eligible为空：所有历史消息都已经归档到摘要表里。只加载摘要，没有layer1/layer2原始消息
    if not eligible:
        summaries_text, summary_injection_tokens = _select_summaries(   # 从历史摘要列表挑选摘要文本，不超过摘要注入token预算
            summaries,
            budget.summary_injection_token_budget,
            chinese_tokens_per_char,
        )
        history_ctx = (     # 把摘要包装成一条【历史摘要】消息，放到上下文头部
            [HumanMessage(f"【历史摘要】\n{summaries_text}")]
            if summaries_text
            else []
        )
        return ContextPack(
            summaries_text=summaries_text,
            layer2_messages=[],
            layer1_messages=[],
            history_ctx=history_ctx,
            layer1_from_msg_id=layer1_from_msg_id,
            summary_span=None,
            layer2_tokens=0,
            layer1_tokens=0,
            token_estimate=0,
            trigger_summary=False,
            budget_error="上下文预算不足" if budget.history_budget <= 0 else "",
            current_user_message=(
                current_user.to_langchain()
                if current_user is not None
                else None
            ),
            summary_injection_tokens=summary_injection_tokens,
        )

    start_index = 0         # 定位layer1起始位置：找到layer1_from_msg_id对应的下标
    if layer1_from_msg_id is not None:
        for index, item in enumerate(eligible):
            if item.message_id >= layer1_from_msg_id:
                start_index = index
                break
    while start_index < len(eligible) and eligible[start_index].role != "user": # 保证layer1第一条消息必须是用户消息；如果锚点落在助手消息，向后移动直到找到user
        start_index += 1
    if start_index >= len(eligible):    # 兜底：全部都是助手消息，取最后一条
        start_index = max(0, len(eligible) - 1)

    layer1 = eligible[start_index:]
    layer1_from = layer1[0].message_id
    while (
        len(layer1) > 1
        and estimate_messages_tokens(
            [item.to_langchain() for item in layer1],
            chinese_tokens_per_char,
        )
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
        layer1_from = layer1[0].message_id



    layer2_original = [     # layer2：在eligible里面，ID小于layer1_from的旧消息（待压缩区域）,即是 上一轮摘要处理完之后新增、还没有被归档成 SummaryRow 的消息
        item
        for item in eligible
        if item.message_id < layer1_from
    ]
    layer2_shortened = [
        ContextMessage(
            message_id=item.message_id,
            role=item.role,
            content=_shorten_message(item, assistant_layer2_chars), # 截断助手长回复，最多保留assistant_layer2_chars个字符。
            tool_name=item.tool_name,
        )
        for item in layer2_original
    ]
    layer2_tokens = estimate_messages_tokens(   # 计算精简后layer2的token
        [item.to_langchain() for item in layer2_shortened],
        chinese_tokens_per_char,
    )
    layer2_trigger_tokens = layer2_tokens
    trigger_summary = bool(     # 判断：layer2超限，标记需要触发后台摘要任务
        layer2_shortened and layer2_tokens > budget.layer2_budget
    )
    summary_span = None
    if trigger_summary:
        summary_span = SummarySpan(     # 标记摘要区间：layer2原始消息的起止ID，交给SummaryTaskManager后台LLM生成摘要
            layer2_original[0].message_id,
            layer2_original[-1].message_id,
        )
        layer2_shortened = _fit_newest(  # 再次精简layer2，调用_fit_newest，只保留最新消息，压缩到layer2预算内（本轮临时方案）
            layer2_shortened,
            budget.layer2_budget,
            chinese_tokens_per_char,
        )
        layer2_tokens = estimate_messages_tokens(
            [item.to_langchain() for item in layer2_shortened],
            chinese_tokens_per_char,
        )

    # 转为Langchain消息对象，用于拼装prompt
    layer1_messages = [item.to_langchain() for item in layer1]
    layer2_messages = [item.to_langchain() for item in layer2_shortened]
    summaries_text, summary_injection_tokens = _select_summaries(
        summaries,
        budget.summary_injection_token_budget,
        chinese_tokens_per_char,
    )

    history_ctx: list = []
    if summaries_text:
        history_ctx.append(HumanMessage(f"【历史摘要】\n{summaries_text}"))
    history_ctx.extend(layer2_messages)
    history_ctx.extend(layer1_messages)
    layer1_tokens = estimate_messages_tokens(
        layer1_messages,
        chinese_tokens_per_char,
    )

    return ContextPack(
        summaries_text=summaries_text,      # 当前会话全部历史摘要
        layer2_messages=layer2_messages,
        layer1_messages=layer1_messages,
        history_ctx=history_ctx,            # 组装后的历史上下文：历史摘要消息 + layer2_messages + layer1_messages，主要用于日志和长度统计
        layer1_from_msg_id=layer1_from,     # 本轮 layer1 的第一条消息 ID，会保存为会话锚点，下一轮从这里附近继续选择
        summary_span=summary_span,          # 需要后台摘要的消息 ID 范围
        layer2_tokens=layer2_tokens,        # 实际放入本轮的 layer2 token 数
        layer1_tokens=layer1_tokens,        # 最终实际保留的 layer1 原文消息 token 数
        token_estimate=layer1_tokens + layer2_tokens,  # 本轮历史上下文的估算 token 总数
        trigger_summary=trigger_summary,  # 是否触发后台摘要：layer2 缩短后仍超过 layer2_budget 时为 True
        budget_error="上下文预算不足" if budget.history_budget <= 0 else "",  # 当历史预算本身为 0 或负数时写入错误；图中据此走 fallback
        layer2_trigger_tokens=layer2_trigger_tokens,  # 触发摘要判断时的 layer2 token 数，即 _fit_newest 裁剪前的缩短后 token，用于日志和摘要任务统计
        current_user_message=(
            current_user.to_langchain()
            if current_user is not None
            else None
        ),
        summary_injection_tokens=summary_injection_tokens,
    )


#layer2如果大于预算的话---->裁剪到预算范围内、此外标记整个layer2的起始，是否压缩：True
