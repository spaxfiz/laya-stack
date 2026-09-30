"""Classification metrics for DecisionBench."""

from typing import Any, Dict, List, Optional, Set, Tuple
from collections import Counter


def accuracy_score(y_true: List[Any], y_pred: List[Any]) -> float:
    """Compute overall accuracy score between true and predicted labels."""
    if not y_true or len(y_true) != len(y_pred):
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return round(correct / len(y_true), 4)


def precision_recall_fscore_support(
    y_true: List[Any],
    y_pred: List[Any],
    labels: Optional[List[Any]] = None,
    average: str = "macro"
) -> Dict[str, Any]:
    """Compute precision, recall, and F1 score per label and aggregated."""
    if not y_true or len(y_true) != len(y_pred):
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "per_class": {}}

    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)), key=lambda x: str(x))

    per_class: Dict[Any, Dict[str, float]] = {}
    supports: Dict[Any, int] = {}

    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        support = sum(1 for t in y_true if t == label)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class[label] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support
        }
        supports[label] = support

    total_support = sum(supports.values())

    if average == "macro":
        valid_labels = [l for l in labels if supports[l] > 0 or any(p == l for p in y_pred)]
        if not valid_labels:
            avg_p, avg_r, avg_f1 = 0.0, 0.0, 0.0
        else:
            avg_p = sum(per_class[l]["precision"] for l in valid_labels) / len(valid_labels)
            avg_r = sum(per_class[l]["recall"] for l in valid_labels) / len(valid_labels)
            avg_f1 = sum(per_class[l]["f1"] for l in valid_labels) / len(valid_labels)
    elif average == "weighted":
        if total_support == 0:
            avg_p, avg_r, avg_f1 = 0.0, 0.0, 0.0
        else:
            avg_p = sum(per_class[l]["precision"] * supports[l] for l in labels) / total_support
            avg_r = sum(per_class[l]["recall"] * supports[l] for l in labels) / total_support
            avg_f1 = sum(per_class[l]["f1"] * supports[l] for l in labels) / total_support
    else:  # micro
        total_tp = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        avg_p = total_tp / len(y_pred) if y_pred else 0.0
        avg_r = avg_p
        avg_f1 = avg_p

    return {
        "precision": round(avg_p, 4),
        "recall": round(avg_r, 4),
        "f1": round(avg_f1, 4),
        "per_class": per_class
    }


def confusion_matrix(
    y_true: List[Any],
    y_pred: List[Any],
    labels: Optional[List[Any]] = None
) -> Dict[str, Any]:
    """Calculate confusion matrix representation."""
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)), key=lambda x: str(x))

    label_to_idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    matrix = [[0 for _ in range(n)] for _ in range(n)]

    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            matrix[label_to_idx[t]][label_to_idx[p]] += 1

    return {
        "labels": labels,
        "matrix": matrix
    }
