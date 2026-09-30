from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple
import numpy as np

def compute_classification_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Any]:
    total = len(y_true)
    if total == 0:
        return {}

    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = correct / total

    # Per-class TP, FP, FN
    tp = Counter()
    fp = Counter()
    fn = Counter()

    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            tp[yt] += 1
        else:
            fn[yt] += 1
            fp[yp] += 1

    per_class = {}
    precisions = []
    recalls = []
    f1s = []

    for l in labels:
        p = tp[l] / max(tp[l] + fp[l], 1)
        r = tp[l] / max(tp[l] + fn[l], 1)
        f1 = (2 * p * r) / max(p + r, 1e-6)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        per_class[l] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "support": tp[l] + fn[l]
        }

    macro_p = float(np.mean(precisions))
    macro_r = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    return {
        "total_samples": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class
    }
