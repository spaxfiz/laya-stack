"""Regression and ordinal scoring metrics for DecisionBench."""

import math
from typing import Any, Dict, List


def mean_absolute_error(y_true: List[float], y_pred: List[float]) -> float:
    """Calculate Mean Absolute Error (MAE)."""
    if not y_true or len(y_true) != len(y_pred):
        return 0.0
    mae = sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)
    return round(mae, 4)


def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    """Calculate Mean Squared Error (MSE)."""
    if not y_true or len(y_true) != len(y_pred):
        return 0.0
    mse = sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true)
    return round(mse, 4)


def root_mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    """Calculate Root Mean Squared Error (RMSE)."""
    return round(math.sqrt(mean_squared_error(y_true, y_pred)), 4)


def pearson_correlation(y_true: List[float], y_pred: List[float]) -> float:
    """Calculate Pearson correlation coefficient r."""
    n = len(y_true)
    if n < 2 or n != len(y_pred):
        return 0.0

    mean_t = sum(y_true) / n
    mean_p = sum(y_pred) / n

    cov = sum((t - mean_t) * (p - mean_p) for t, p in zip(y_true, y_pred))
    var_t = sum((t - mean_t) ** 2 for t in y_true)
    var_p = sum((p - mean_p) ** 2 for p in y_pred)

    denominator = math.sqrt(var_t * var_p)
    if denominator == 0.0:
        return 0.0
    return round(cov / denominator, 4)


def _rank_data(arr: List[float]) -> List[float]:
    """Assign average ranks to tied values in a list."""
    n = len(arr)
    sorted_indices = sorted(range(n), key=lambda i: arr[i])
    ranks = [0.0] * n

    i = 0
    while i < n:
        j = i
        while j + 1 < n and arr[sorted_indices[j + 1]] == arr[sorted_indices[i]]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # 1-based rank average
        for k in range(i, j + 1):
            ranks[sorted_indices[k]] = avg_rank
        i = j + 1

    return ranks


def spearman_rank_correlation(y_true: List[float], y_pred: List[float]) -> float:
    """Calculate Spearman rank-order correlation coefficient rho."""
    if len(y_true) < 2 or len(y_true) != len(y_pred):
        return 0.0

    rank_t = _rank_data(y_true)
    rank_p = _rank_data(y_pred)
    return pearson_correlation(rank_t, rank_p)


def tolerance_accuracy(y_true: List[float], y_pred: List[float], tolerance: float = 1.0) -> float:
    """Percentage of predictions within ±tolerance of true value."""
    if not y_true or len(y_true) != len(y_pred):
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if abs(t - p) <= tolerance)
    return round(correct / len(y_true), 4)


def regression_summary(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    """Produce comprehensive summary of regression and scoring metrics."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "pearson_r": pearson_correlation(y_true, y_pred),
        "spearman_rho": spearman_rank_correlation(y_true, y_pred),
        "exact_match": tolerance_accuracy(y_true, y_pred, tolerance=0.5),
        "within_1_level": tolerance_accuracy(y_true, y_pred, tolerance=1.0)
    }
