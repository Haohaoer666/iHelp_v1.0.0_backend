from RAG_test.scripts import eval_ch04


def test_recall_at_k():
    assert eval_ch04.recall_at_k(["a", "b"], ["a"], k=3) == 1.0
    assert eval_ch04.recall_at_k(["x", "y"], ["a"], k=3) == 0.0
    assert eval_ch04.recall_at_k([], [], k=3) == 1.0


def test_mrr():
    assert eval_ch04.mrr(["a", "b"], ["a"]) == 1.0
    assert eval_ch04.mrr(["a", "b"], ["b"]) == 0.5
    assert eval_ch04.mrr(["a", "b"], ["c"]) == 0.0
