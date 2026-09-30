"""
定义了Query改写/扩写两个函数
"""

import json
import re

from langchain_core.messages import HumanMessage

from app.core.llm import get_chat_model


SYNONYMS = {
    "到账": ["到款", "打款", "到账时间"],
    "退款": ["退钱", "退货退款"],
    "发货": ["发出", "寄出"],
}


#专门用来 清洗大模型返回的内容 。很多 LLM 输出 JSON 时，会自动包一层 markdown 代码块
def _strip_code_fence(value: str) -> str:
    value = value.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    return value.strip()


# Query改写
async def normalize_query(query: str) -> dict:
    # 构造提示词：让大模型把用户口语化问句，转换成标准化检索JSON，附带示例
    prompt = f"""把用户口语改写成标准检索问法，只输出 JSON。
示例：
用户：钱什么时候打回来
输出：{{"normalized_query":"退款到账时间","category":null,"is_knowledge_question":true}}

用户：{query}
输出："""
    model = get_chat_model(streaming=False)
    response = await model.ainvoke([HumanMessage(content=prompt)])
    content = getattr(response, "content", "") or ""
    try:
        result = json.loads(_strip_code_fence(content))
    except json.JSONDecodeError:
        # JSON解析失败时降级兜底：直接使用原始query，不做改写
        return {"normalized_query": query, "category": None, "is_knowledge_question": True}
    return {
        "normalized_query": result.get("normalized_query") or query,
        "category": result.get("category"),
        "is_knowledge_question": bool(result.get("is_knowledge_question", True)),
    }

# Query扩写
def expand_synonyms(normalized_query: str) -> str:
    terms = [normalized_query]
    for key, synonyms in SYNONYMS.items():
        if key in normalized_query:
            terms.extend(synonyms)
    return " ".join(dict.fromkeys(terms))   #利用字典key自动去重，保留首次出现顺序,去重后的所有词项用空格拼接成一个字符串返回


#调用Query扩写函数
def build_retrieval_query(normalized: dict) -> str:
    return expand_synonyms(normalized.get("normalized_query") or "")
