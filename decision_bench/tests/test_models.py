"""Unit tests for Decision Model adapters."""

import pytest
from decision_bench.models import create_model, MockDecisionModel


def test_mock_model_primitives():
    model = MockDecisionModel("test-mock", simulated_latency_ms=0.0)

    # Choice
    out_choice = model.predict_choice("科技数码新品", ["科技数码", "美食探店"])
    assert out_choice.is_success
    assert out_choice.value in ["科技数码", "美食探店"]
    assert out_choice.confidence is not None

    # Noul
    out_noul = model.predict_noul("震惊！惊天大秘密！")
    assert out_noul.is_success
    assert isinstance(out_noul.value, bool)

    # Score
    out_score = model.predict_score("内容评测", ["1", "2", "3", "4", "5"])
    assert out_score.is_success
    assert 1.0 <= out_score.value <= 5.0

    # Rank
    out_rank = model.rank("手机", ["苹果手机评测", "红烧肉", "大衣"])
    assert out_rank.is_success
    assert isinstance(out_rank.value, list)
    assert len(out_rank.value) == 3


def test_create_model_factory():
    m1 = create_model("mock")
    assert isinstance(m1, MockDecisionModel)

    m2 = create_model("http://127.0.0.1:8700")
    assert m2.base_url == "http://127.0.0.1:8700"
