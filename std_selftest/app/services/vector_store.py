"""
基于 Milvus 异步客户端封装 VectorStore ->
自动创建数据库与集合 ->
存储带分类、文本、稠密向量、问题列表的文档块，Milvus 内置 BM25 生成稀疏向量 ->
提供 upsert 写入、稠密向量检索、BM25 关键词检索、RRF 混合检索接口，支持 category 分类过滤 ->
统一标准化返回检索结果，附带集合删除、数据 flush 能力。
"""

import json

from pymilvus import (
    AnnSearchRequest,
    AsyncMilvusClient,
    DataType,
    Function,
    FunctionType,
    RRFRanker,
)

from app.config import settings

#生成 Milvus 的 filter 过滤表达式，按`category`分类过滤,只检索指定分类的文档块
def _category_expr(category: str | None) -> str:
    if not category:
        return ""
    safe = category.replace('"', '\\"')
    return f'category == "{safe}"'


#Milvus 检索返回结果标准化，统一输出格式。
def _normalize_hit(hit: dict) -> dict:
    entity = hit["entity"]
    raw_questions = entity.get("questions", "")
    try:
        questions = json.loads(raw_questions or "[]")
    except (TypeError, json.JSONDecodeError):
        questions = []
    return {
        "chunk_id": entity["chunk_id"],
        "category": entity["category"],
        "question": questions[0] if questions else entity["category"],
        "questions": questions,
        "text": entity["text"],
        "score": float(hit["distance"]),
    }


class VectorStore:
    def __init__(self) -> None:
        self.root_client = AsyncMilvusClient(uri=settings.milvus_uri)       #根客户端，用于**创建数据库**（创建库的时候不能指定 db_name）
        self.client = AsyncMilvusClient(                                    #业务客户端，绑定指定 database，后续集合操作都用这个
            uri=settings.milvus_uri,
            db_name=settings.milvus_database,
        )
        self.collection = settings.milvus_collection
        self._database_ready = False

    async def _ensure_database(self) -> None:   #  保证目标数据库一定存在**，不存在就新建。
        if self._database_ready:
            return
        databases = await self.root_client.list_databases()
        if settings.milvus_database not in databases:
            await self.root_client.create_database(settings.milvus_database)
        self._database_ready = True

    async def ensure_collection(self) -> None:  #  保证目标数据库的collection一定存在，不存在就新建。
        await self._ensure_database()
        if await self.client.has_collection(self.collection):
            return

        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("chunk_id", DataType.VARCHAR, max_length=128)
        schema.add_field("category", DataType.VARCHAR, max_length=255)
        schema.add_field("questions", DataType.VARCHAR, max_length=4096)
        schema.add_field(
            "text",
            DataType.VARCHAR,
            max_length=65535,
            enable_analyzer=True,
            analyzer_params={"type": "chinese"},
        )
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=settings.embedding_dim)   #dense 向量字段，用于语义检索
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)                        #稀疏向量字段，专门存放BM25关键词向量

        #注册 BM25 function,写入一条文档`text`， Milvus 内部自动对 text 中文分词，算出 BM25 稀疏向量，自动写入 sparse 字段 。
        schema.add_function(
            Function(
                name="bm25",
                function_type=FunctionType.BM25,
                input_field_names=["text"],
                output_field_names=["sparse"],
            )
        )
        #先在本地写好索引建造清单（稠密向量索引、稀疏关键词索引），然后创建 Milvus 集合并一次性建好这两个索引，最后把集合 + 索引加载进内存
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        index_params.add_index(
            field_name="sparse",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )
        await self.client.create_collection(
            collection_name=self.collection,
            schema=schema,
            index_params=index_params,
        )
        await self.client.load_collection(self.collection)

    async def upsert(   #写入一个知识块到 Milvus
        self,
        chunk_id: str,
        category: str,
        text: str,
        vector: list[float],
        questions: list[str] | None = None,
    ) -> int:
        await self.ensure_collection()
        questions_json = json.dumps(questions or [], ensure_ascii=False)[:4096]
        data = [
            {
                "chunk_id": chunk_id[:128],
                "category": category[:80],
                "questions": questions_json,
                "text": text[:20000],
                "vector": vector,
            }
        ]
        result = await self.client.insert(self.collection, data)
        return int(result["ids"][0])

    async def search(   #dense 向量检索
        self,
        vector: list[float],
        top_k: int | None = None,
        category: str | None = None,
    ) -> list[dict]:
        await self.ensure_collection()
        limit = top_k or settings.knowledge_top_k
        result = await self.client.search(
            collection_name=self.collection,
            data=[vector],
            anns_field="vector",
            limit=limit,
            search_params={"metric_type": "COSINE"},
            filter=_category_expr(category),
            output_fields=["chunk_id", "category", "questions", "text"],
        )
        hits = result[0]
        normalized = []
        for hit in hits:
            entity = hit["entity"]
            raw_questions = entity.get("questions", "")
            try:
                question_list = json.loads(raw_questions or "[]")
            except (TypeError, json.JSONDecodeError):
                question_list = []
            normalized.append(
                {
                    "chunk_id": entity["chunk_id"],
                    "category": entity["category"],
                    "question": question_list[0] if question_list else entity["category"],
                    "questions": question_list,
                    "text": entity["text"],
                    "distance": float(hit["distance"]),
                }
            )
        return normalized

    async def search_bm25(      #BM25 全文检索
        self,
        query: str,
        top_k: int = 10,
        category: str | None = None,
    ) -> list[dict]:
        await self.ensure_collection()
        result = await self.client.search(
            collection_name=self.collection,
            data=[query],
            anns_field="sparse",
            search_params={"metric_type": "BM25"},
            limit=top_k,
            filter=_category_expr(category),
            output_fields=["chunk_id", "category", "questions", "text"],
        )
        return [_normalize_hit(hit) for hit in result[0]]

    async def hybrid_search(        #dense + BM25 混合检索(索引和检索都在Milvus中用Chinese_analyzer分析器完成)
        self,
        vector: list[float],
        query: str,
        top_k: int = 10,
        category: str | None = None,
    ) -> list[dict]:
        await self.ensure_collection()
        expr = _category_expr(category)
        dense = AnnSearchRequest(
            data=[vector],
            anns_field="vector",
            param={"metric_type": "COSINE"},
            limit=top_k,
            filter=expr,
        )
        sparse = AnnSearchRequest(
            data=[query],
            anns_field="sparse",
            param={"metric_type": "BM25"},
            limit=top_k,
            filter=expr,
        )
        result = await self.client.hybrid_search(
            collection_name=self.collection,
            reqs=[dense, sparse],
            ranker=RRFRanker(k=60),
            limit=top_k,
            output_fields=["chunk_id", "category", "questions", "text"],
        )
        return [_normalize_hit(hit) for hit in result[0]]

    #删除 collection
    async def reset(self) -> None:
        await self._ensure_database()
        if await self.client.has_collection(self.collection):
            await self.client.drop_collection(self.collection)

    #把数据刷入可查询状态
    async def flush(self) -> None:
        await self.client.flush(self.collection)

    #关闭 Milvus 连接
    async def close(self) -> None:
        await self.client.close()
        await self.root_client.close()
