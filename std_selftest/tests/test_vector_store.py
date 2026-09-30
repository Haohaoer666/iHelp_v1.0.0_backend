import json

from pymilvus import DataType, FunctionType, RRFRanker

from app.services import vector_store


class FakeMilvusClient:
    def __init__(self):
        self.inserted_data = None
        self.schema_fields = []
        self.schema_functions = []
        self.indexes = []
        self.hybrid_calls = []
        self.search_calls = []

    async def has_collection(self, collection_name):
        return True

    async def insert(self, collection_name, data):
        self.inserted_data = data
        return {"ids": [7]}

    async def search(
        self,
        collection_name,
        data,
        anns_field=None,
        search_params=None,
        limit=10,
        filter="",
        output_fields=None,
    ):
        self.search_calls.append((data, anns_field, search_params, limit, filter))
        return [
            [
                {
                    "entity": {
                        "chunk_id": "chunk-bm25",
                        "category": "商品FAQ",
                        "questions": "[]",
                        "text": "断电后拆下集便仓。",
                    },
                    "distance": 0.8,
                }
            ]
        ]

    async def hybrid_search(self, collection_name, reqs, ranker, limit, output_fields):
        self.hybrid_calls.append((reqs, ranker, limit))
        return [
            [
                {
                    "entity": {
                        "chunk_id": "chunk-hybrid",
                        "category": "商品FAQ",
                        "questions": "[]",
                        "text": "智能猫砂盆断电后清理。",
                    },
                    "distance": 0.032,
                }
            ]
        ]

    async def close(self):
        return None


class RecordingSchema:
    def __init__(self):
        self.fields = []
        self.functions = []

    def add_field(self, name, dtype, **kwargs):
        self.fields.append((name, dtype, kwargs))

    def add_function(self, function):
        self.functions.append(function)


def make_store():
    store = vector_store.VectorStore()
    store.client = FakeMilvusClient()
    store.root_client = FakeMilvusClient()
    store._database_ready = True
    return store


def make_store_with_recorder():
    store = vector_store.VectorStore()
    fake = FakeMilvusClient()
    store.client = fake
    store.root_client = FakeMilvusClient()
    store._database_ready = True
    return store, fake


async def test_upsert_writes_questions_json():
    store = make_store()

    milvus_id = await store.upsert(
        "chunk-1",
        "商品FAQ",
        "断电后拆下集便仓，用清水冲洗晾干。",
        [0.1, 0.2],
        questions=["智能猫砂盆怎么清理"],
    )

    assert milvus_id == 7
    assert store.client.inserted_data[0]["questions"] == json.dumps(
        ["智能猫砂盆怎么清理"],
        ensure_ascii=False,
    )
    await store.close()


async def test_search_returns_first_question():
    store, fake = make_store_with_recorder()

    hits = await store.search([0.1, 0.2], top_k=1)

    assert fake.search_calls[0][1] == "vector"
    assert hits[0]["question"] == "商品FAQ"
    assert hits[0]["chunk_id"] == "chunk-bm25"
    assert hits[0]["distance"] == 0.8
    await store.close()


async def test_dense_search_applies_category_filter():
    store, fake = make_store_with_recorder()

    await store.search([0.1, 0.2], top_k=1, category="商品FAQ")

    assert fake.search_calls[0][4] == 'category == "商品FAQ"'
    await store.close()


async def test_search_bm25_uses_sparse_field():
    store, fake = make_store_with_recorder()

    hits = await store.search_bm25(
        "智能猫砂盆",
        top_k=10,
        category="商品FAQ",
    )

    assert hits[0]["chunk_id"] == "chunk-bm25"
    assert fake.search_calls == [
        (
            ["智能猫砂盆"],
            "sparse",
            {"metric_type": "BM25"},
            10,
            'category == "商品FAQ"',
        )
    ]
    await store.close()


async def test_hybrid_search_builds_dense_and_bm25_requests():
    store, fake = make_store_with_recorder()

    hits = await store.hybrid_search(
        [0.1, 0.2],
        "智能猫砂盆",
        top_k=10,
        category="商品FAQ",
    )

    assert hits[0]["chunk_id"] == "chunk-hybrid"
    assert len(fake.hybrid_calls) == 1
    reqs, ranker, limit = fake.hybrid_calls[0]
    assert limit == 10
    assert isinstance(ranker, RRFRanker)
    assert reqs[0].anns_field == "vector"
    assert reqs[1].anns_field == "sparse"
    assert reqs[0].data == [[0.1, 0.2]]
    assert reqs[1].data == ["智能猫砂盆"]
    assert reqs[0].expr == 'category == "商品FAQ"'
    await store.close()


async def test_ensure_collection_registers_bm25_function():
    class CreateFakeClient(FakeMilvusClient):
        def __init__(self):
            super().__init__()
            self.has_collection_value = False
            self.schema = RecordingSchema()
            self.created = False

        async def has_collection(self, collection_name):
            return self.has_collection_value

        def create_schema(self, **kwargs):
            return self.schema

        def prepare_index_params(self):
            return self

        def add_index(self, field_name, index_type, metric_type):
            self.indexes.append((field_name, index_type, metric_type))

        async def create_collection(self, collection_name, schema, index_params):
            self.created = True

        async def load_collection(self, collection_name):
            return None

    store = vector_store.VectorStore()
    fake = CreateFakeClient()
    store.client = fake
    store.root_client = FakeMilvusClient()
    store._database_ready = True

    await store.ensure_collection()

    assert fake.created is True
    assert any(
        name == "sparse" and dtype == DataType.SPARSE_FLOAT_VECTOR
        for name, dtype, _ in fake.schema.fields
    )
    text_field = next(field for field in fake.schema.fields if field[0] == "text")
    assert text_field[2]["enable_analyzer"] is True
    assert text_field[2]["analyzer_params"] == {"type": "chinese"}
    assert fake.schema.functions[0].name == "bm25"
    assert fake.schema.functions[0].type == FunctionType.BM25
    assert any(field == "sparse" for field, _, _ in fake.indexes)
    await store.close()
