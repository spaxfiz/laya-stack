"""Latency and inference performance metrics for Decision Models."""

import math
from typing import Dict, List


def percentile(data: List[float], p: float) -> float:
    """Calculate the p-th percentile of a sorted list of numbers."""
    if not data:
        return 0.0
    k = (len(data) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return data[int(k)]
    d0 = data[int(f)] * (c - k)
    d1 = data[int(c)] * (k - f)
    return d0 + d1


def latency_summary(latencies_ms: List[float], total_duration_seconds: float = 0.0) -> Dict[str, float]:
    """
    Produce a full statistical summary of inference latency (in milliseconds)
    and throughput (QPS).
    """
    if not latencies_ms:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "std_ms": 0.0,
            "min_ms": 0.0,
            "p50_ms": 0.0,
            "p90_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
            "qps": 0.0
        }

    sorted_latencies = sorted(latencies_ms)
    n = len(sorted_latencies)
    mean = sum(sorted_latencies) / n
    variance = sum((x - mean) ** 2 for x in sorted_latencies) / n
    std = math.sqrt(variance)

    if total_duration_seconds > 0:
        qps = n / total_duration_seconds
    else:
        # Sum of latencies in seconds if executed serially
        serial_duration_sec = sum(sorted_latencies) / 1000.0
        qps = n / serial_duration_sec if serial_duration_sec > 0 else 0.0

    return {
        "count": n,
        "mean_ms": round(mean, 2),
        "std_ms": round(std, 2),
        "min_ms": round(sorted_latencies[0], 2),
        "p50_ms": round(percentile(sorted_latencies, 50), 2),
        "p90_ms": round(percentile(sorted_latencies, 90), 2),
        "p95_ms": round(percentile(sorted_latencies, 95), 2),
        "p99_ms": round(percentile(sorted_latencies, 99), 2),
        "max_ms": round(sorted_latencies[-1], 2),
        "qps": round(qps, 2)
    }
