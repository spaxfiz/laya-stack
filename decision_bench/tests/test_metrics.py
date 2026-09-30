"""Unit tests for DecisionBench metrics suite."""

import pytest
from decision_bench.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    spearman_rank_correlation,
    tolerance_accuracy,
    ndcg_at_k,
    mean_reciprocal_rank,
    hit_at_k,
    expected_calibration_error,
    brier_score,
    latency_summary
)


def test_classification_metrics():
    y_true = ["A", "B", "A", "C"]
    y_pred = ["A", "B", "C", "C"]

    acc = accuracy_score(y_true, y_pred)
    assert acc == 0.75

    prf = precision_recall_fscore_support(y_true, y_pred, labels=["A", "B", "C"])
    assert "precision" in prf
    assert "recall" in prf
    assert "f1" in prf
    assert prf["per_class"]["A"]["recall"] == 0.5

    conf = confusion_matrix(y_true, y_pred, labels=["A", "B", "C"])
    assert conf["labels"] == ["A", "B", "C"]
    assert conf["matrix"][0][0] == 1  # True A predicted A


def test_regression_metrics():
    y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_pred = [1.0, 2.2, 2.9, 4.1, 4.8]

    mae = mean_absolute_error(y_true, y_pred)
    assert mae < 0.2

    mse = mean_squared_error(y_true, y_pred)
    assert mse < 0.1

    rho = spearman_rank_correlation(y_true, y_pred)
    assert rho == 1.0  # Perfect monotonic rank order

    tol_acc = tolerance_accuracy(y_true, y_pred, tolerance=0.5)
    assert tol_acc == 1.0


def test_ranking_metrics():
    # True item is at index 0, predicted rank is [A, B, C] -> reciprocal rank = 1
    mrr = mean_reciprocal_rank([["A", "B", "C"], ["B", "A", "C"]], ["A", "A"])
    assert mrr == (1.0 + 0.5) / 2.0  # 0.75

    hit1 = hit_at_k([["A", "B"], ["B", "A"]], ["A", "A"], k=1)
    assert hit1 == 0.5

    ndcg = ndcg_at_k([1.0, 0.0, 0.0], k=3)
    assert ndcg == 1.0


def test_calibration_and_latency():
    ece = expected_calibration_error([1, 0, 1, 1], [0.9, 0.1, 0.8, 0.85])
    assert 0.0 <= ece <= 1.0

    bs = brier_score([1, 0], [0.9, 0.1])
    assert bs == 0.01

    lat = latency_summary([5.0, 10.0, 15.0, 20.0])
    assert lat["count"] == 4
    assert lat["min_ms"] == 5.0
    assert lat["max_ms"] == 20.0
    assert lat["p50_ms"] == 12.5
