"""Probability calibration metrics for Decision Models."""

from typing import List, Tuple


def brier_score(y_true_binary: List[int], y_prob: List[float]) -> float:
    """Calculate Brier score (mean squared error of probabilistic predictions)."""
    if not y_true_binary or len(y_true_binary) != len(y_prob):
        return 0.0
    bs = sum((p - t) ** 2 for t, p in zip(y_true_binary, y_prob)) / len(y_true_binary)
    return round(bs, 4)


def expected_calibration_error(
    y_true_binary: List[int],
    y_prob: List[float],
    n_bins: int = 10
) -> float:
    """
    Calculate Expected Calibration Error (ECE).
    Measures the absolute difference between predicted confidence and observed empirical accuracy.
    """
    n = len(y_true_binary)
    if n == 0 or n != len(y_prob):
        return 0.0

    bin_boundaries = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        # Include right edge on the last bin
        if i == n_bins - 1:
            bin_indices = [idx for idx, p in enumerate(y_prob) if low <= p <= high]
        else:
            bin_indices = [idx for idx, p in enumerate(y_prob) if low <= p < high]

        bin_size = len(bin_indices)
        if bin_size > 0:
            avg_conf = sum(y_prob[idx] for idx in bin_indices) / bin_size
            avg_acc = sum(y_true_binary[idx] for idx in bin_indices) / bin_size
            ece += (bin_size / n) * abs(avg_acc - avg_conf)

    return round(ece, 4)
