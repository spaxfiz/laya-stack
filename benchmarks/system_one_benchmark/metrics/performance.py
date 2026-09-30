import statistics
from typing import Any, Dict, List
import numpy as np

def compute_performance_stats(latencies_ms: List[float], total_time_sec: float) -> Dict[str, Any]:
    if not latencies_ms:
        return {}

    total_requests = len(latencies_ms)
    rps = total_requests / max(total_time_sec, 1e-4)

    return {
        "total_requests": total_requests,
        "total_time_sec": round(total_time_sec, 3),
        "rps": round(rps, 2),
        "mean_ms": round(statistics.mean(latencies_ms), 2),
        "median_ms": round(statistics.median(latencies_ms), 2),
        "p90_ms": round(float(np.percentile(latencies_ms, 90)), 2),
        "p95_ms": round(float(np.percentile(latencies_ms, 95)), 2),
        "p99_ms": round(float(np.percentile(latencies_ms, 99)), 2),
        "min_ms": round(min(latencies_ms), 2),
        "max_ms": round(max(latencies_ms), 2),
        "std_ms": round(statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0, 2)
    }
