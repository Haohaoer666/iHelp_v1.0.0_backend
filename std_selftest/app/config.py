from urllib.parse import quote_plus

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    chat_model: str
    chat_base_url: str
    chat_api_key: str
    token_budget: int = 2000
    query_understanding_model: str | None = None
    query_expansion_model: str | None = None
    intent_low_cost_model: str | None = None
    intent_confidence_threshold: float = 0.75
    intent_use_low_cost_first: bool = False
    query_expansion_max_queries: int = 4
    refund_reasons: str = (
        "商品质量问题,七天无理由,尺寸或型号不合适,"
        "收到商品破损,少件或错件,其他"
    )
    exchange_reasons: str = (
        "尺码不合适,颜色或款式不喜欢,商品破损,"
        "少件或错件,质量问题,其他"
    )

    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = "123456"
    mysql_database: str = "std_selfproject_db2"

    tool_timeout_seconds: float = 15.0
    tool_max_retries: int = 2
    ticket_gate_model: str | None = None
    ticket_gate_confidence_threshold: float = 0.75
    mcp_transport_timeout_seconds: float = 5.0
    mcp_sse_read_timeout_seconds: float = 30.0
    mcp_logistics_url: str = "http://127.0.0.1:8101/mcp"
    mcp_after_sales_url: str = "http://127.0.0.1:8102/mcp"
    mcp_logistics_port: int = 8101
    mcp_after_sales_port: int = 8102
    audit_result_max_chars: int = 1000

    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_database: str = "Haohelper"
    milvus_collection: str = "knowledge"

    embedding_base_url: str = "https://api.siliconflow.cn/v1"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_api_key: str = ""
    embedding_dim: int = 1024

    rerank_base_url: str = "https://api.siliconflow.cn/v1"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_api_key: str = ""
    rerank_top_k: int = 20
    rerank_score_threshold: float = 0.35

    knowledge_top_k: int = 3
    knowledge_score_threshold: float = 0.35

    langgraph_checkpoint_path: str = "data/langgraph_checkpoints.db"
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
    # Deprecated compatibility field. The production Agent path uses the
    # full-request ContextBudget guard instead.
    agent_token_budget: int = 4000

    context_budget_profile: str = "cost_optimized"
    history_soft_budget_enabled: bool = True
    history_layer1_ratio: float = 0.70
    history_layer2_ratio: float = 0.30
    model_context_window: int = 65536            # 模型总上下文窗口
    max_user_input_tokens: int = 2000            # 给当前用户输入预留的 token
    tool_call_message_max_tokens: int = 400      # 单次工具调用消息的参数预算
    tool_result_max_tokens: int = 1200           #单次工具结果的预算峰值。ReAct 瞬时峰值按 MAX_AGENT_STEPS × tool_result_max_tokens 预留。
    system_prompt_token_budget: int = 519       #生产档按实际 system + 工具定义估算；legacy 档显式覆盖为 1800
    evidence_chunk_token_budget: int = 400      #每个检索证据块的规划 token。证据预算为 RERANK_TOP_K × evidence_chunk_token_budget。
    evidence_total_max_tokens: int | None = 3000 #证据总预算；为空时兼容旧的 RERANK_TOP_K × 单块预算
    summary_injection_token_budget: int = 600   #给早期梗概注入预留的固定 token
    summary_segment_max_tokens: int = 200       #单段摘要硬上限
    safety_margin_tokens: int = 350             #防止估算误差、消息结构变化和模型 tokenizer 差异的安全余量。
    desired_retained_turns: int = 24            #希望尽量保留的历史轮数，用于计算 desired_history
    steady_turn_token_estimate: int = 800       #每轮正常对话的规划 token 估算，乘以保留轮数得到期望历史预算。
    chinese_tokens_per_char: float = 0.80       #中文 token 估算系数
    assistant_layer2_chars: int = 80            #层 2 中客服答复最多保留的字符数
    summary_target_min_chars: int = 20          #摘要 prompt 的目标下限，也是摘要评估的长度下限。当前按“几十到一两百字”校准为 20。
    summary_target_max_chars: int = 200         #摘要 prompt 的目标上限，summarize_span 超长时会截到 200 字符。
    context_log_path: str = "log/app.log"

    ocr_enabled: bool = True
    ocr_min_text_chars: int = 20
    ocr_render_zoom: float = 2.0

    @property       #把这个方法变成 只读属性
    def database_url(self) -> str:
        user = quote_plus(self.mysql_user)
        password = quote_plus(self.mysql_password)
        return (
            f"mysql+aiomysql://{user}:{password}@{self.mysql_host}:{self.mysql_port}"
            f"/{self.mysql_database}?charset=utf8mb4"
        )

    @property
    def async_database_url(self) -> str:
        return self.database_url

    @property   # 取退货理由
    def refund_reason_list(self) -> list[str]:
        return [
            item.strip()
            for item in self.refund_reasons.split(",")
            if item.strip()
        ]

    @property   # 取换货理由
    def exchange_reason_list(self) -> list[str]:
        return [
            item.strip()
            for item in self.exchange_reasons.split(",")
            if item.strip()
        ]


settings = Settings()
