from RAG_test.scripts import eval_ragas_ch04


def test_is_refusal_response():
    assert eval_ragas_ch04.is_refusal_response("抱歉，现有知识不足以回答。")
    assert not eval_ragas_ch04.is_refusal_response("iHao V1 支持 200Hz。")


def test_build_ragas_rows_skips_refusal_and_empty_contexts():
    rows = [
        {
            "query": "q1",
            "contexts": ["c1"],
            "response": "a1",
            "ground_truth_answer": "ref1",
            "expect_refusal": False,
        },
        {
            "query": "q2",
            "contexts": [],
            "response": "a2",
            "ground_truth_answer": "ref2",
            "expect_refusal": False,
        },
        {
            "query": "q3",
            "contexts": ["c3"],
            "response": "a3",
            "ground_truth_answer": "",
            "expect_refusal": True,
        },
    ]

    samples = eval_ragas_ch04.build_ragas_rows(rows)

    assert len(samples) == 1
    assert samples[0]["user_input"] == "q1"


def test_refusal_accuracy():
    rows = [
        {"expect_refusal": True, "response": "抱歉，无法回答。"},
        {"expect_refusal": True, "response": "可以，200Hz。"},
        {"expect_refusal": False, "response": "普通回答"},
    ]

    assert eval_ragas_ch04.refusal_accuracy(rows) == 0.5
