"""Metrics suite for DecisionBench."""

from .classification import accuracy_score, precision_recall_fscore_support, confusion_matrix
from .regression import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    spearman_rank_correlation,
    pearson_correlation,
    tolerance_accuracy,
    regression_summary
)
from .ranking import ndcg_at_k, mean_reciprocal_rank, hit_at_k
from .calibration import expected_calibration_error, brier_score
from .latency import latency_summary, percentile

__all__ = [
    "accuracy_score",
    "precision_recall_fscore_support",
    "confusion_matrix",
    "mean_absolute_error",
    "mean_squared_error",
    "root_mean_squared_error",
    "spearman_rank_correlation",
    "pearson_correlation",
    "tolerance_accuracy",
    "regression_summary",
    "ndcg_at_k",
    "mean_reciprocal_rank",
    "hit_at_k",
    "expected_calibration_error",
    "brier_score",
    "latency_summary",
    "percentile"
]
