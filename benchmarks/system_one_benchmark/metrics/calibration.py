from typing import Any, Dict, List
import numpy as np

def compute_ece(confidences: List[float], y_true: List[str], y_pred: List[str], n_bins: int = 10) -> Dict[str, Any]:
    """Compute Expected Calibration Error (ECE) and bin reliability statistics."""
    confidences = np.array(confidences)
    accuracies = np.array([1.0 if yt == yp else 0.0 for yt, yp in zip(y_true, y_pred)])
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bins_data = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if np.sum(in_bin) > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            gap = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += gap * prop_in_bin
            mce = max(mce, gap)
            
            bins_data.append({
                "bin_range": f"({bin_lower:.1f}, {bin_upper:.1f}]",
                "count": int(np.sum(in_bin)),
                "avg_confidence": round(float(avg_confidence_in_bin), 4),
                "accuracy": round(float(accuracy_in_bin), 4),
                "gap": round(float(gap), 4)
            })

    return {
        "ece": round(float(ece), 4),
        "mce": round(float(mce), 4),
        "bins": bins_data
    }
