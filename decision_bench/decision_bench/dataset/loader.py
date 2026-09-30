"""Dataset loader and partitioning for DecisionBench."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from .schema import BenchmarkSample, DifficultyLevel


DEFAULT_DATASET_PATH = Path(__file__).parent / "data" / "decision_benchmark_1024.json"


def load_dataset(
    path: Optional[Path] = None,
    difficulty: Optional[DifficultyLevel] = None,
    category: Optional[str] = None,
    limit: Optional[int] = None
) -> List[BenchmarkSample]:
    """Load benchmark samples with optional filtering."""
    file_path = path or DEFAULT_DATASET_PATH
    if not file_path.exists():
        raise FileNotFoundError(f"Benchmark dataset not found at {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    samples = [BenchmarkSample(**item) for item in raw_data]

    if difficulty:
        samples = [s for s in samples if s.difficulty == difficulty]

    if category:
        samples = [s for s in samples if s.ground_truth_track == category]

    if limit and limit > 0:
        samples = samples[:limit]

    return samples


def get_dataset_statistics(samples: List[BenchmarkSample]) -> Dict[str, Any]:
    """Calculate summary statistics of the loaded dataset."""
    from collections import Counter
    total = len(samples)
    if total == 0:
        return {}

    track_dist = Counter(s.ground_truth_track for s in samples)
    diff_dist = Counter(s.difficulty.value for s in samples)
    cb_count = sum(1 for s in samples if s.ground_truth_is_clickbait)
    ad_count = sum(1 for s in samples if s.ground_truth_is_ad)

    return {
        "total_samples": total,
        "track_distribution": dict(track_dist.most_common()),
        "difficulty_distribution": dict(diff_dist),
        "clickbait_count": cb_count,
        "clickbait_ratio": round(cb_count / total, 4),
        "commercial_ad_count": ad_count,
        "commercial_ad_ratio": round(ad_count / total, 4)
    }
