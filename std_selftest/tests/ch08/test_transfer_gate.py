import pytest

from app.graph.intent import fallback_intent
from app.tools.transfer_gate import (
    extract_transfer_issue,
    is_human_transfer_cancel,
    is_human_transfer_request,
    is_standalone_transfer_cancel,
)


@pytest.mark.parametrize(
    "text",
    [
        "不要转人工",
        "不需要人工客服",
        "别转人工",
        "不转人工",
        "不找人工",
        "不接人工",
        "不联系人工",
        "取消转人工",
        "人工客服态度差",
    ],
)
def test_transfer_gate_rejects_negation_and_complaint(text):
    assert is_human_transfer_request(text) is False


@pytest.mark.parametrize(
    "text",
    [
        "转人工",
        "我要转人工",
        "帮我找人工客服",
        "需要人工客服",
    ],
)
def test_transfer_gate_accepts_explicit_requests(text):
    assert is_human_transfer_request(text) is True


def test_complaint_about_manual_service_keeps_complaint_intent():
    assert fallback_intent("人工客服态度差").intent == "投诉"
    assert extract_transfer_issue("不要转人工") == ""


@pytest.mark.parametrize(
    "text",
    [
        "算了",
        "不用了",
        "取消转人工",
        "我不转了",
        "不找人工",
        "不接人工",
        "不联系人工",
    ],
)
def test_transfer_cancel_phrases(text):
    assert is_human_transfer_cancel(text) is True


def test_normal_issue_is_not_transfer_cancel():
    assert is_human_transfer_cancel("退款一直没到") is False
    assert is_human_transfer_cancel("要不要转人工") is False


def test_manual_service_hours_is_not_transfer_request():
    assert is_human_transfer_request("我要找人工客服几点下班") is False


def test_standalone_transfer_cancel_command():
    assert is_standalone_transfer_cancel("取消转人工") is True
    assert is_standalone_transfer_cancel("不要转人工，帮我查物流") is False
