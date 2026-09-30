"""Unit tests for dataset loading and schema validation."""

import pytest
from decision_bench.dataset.loader import load_dataset, get_dataset_statistics
from decision_bench.dataset.schema import BenchmarkSample, DifficultyLevel


def test_dataset_loading_full():
    samples = load_dataset()
    assert len(samples) == 1024
    assert isinstance(samples[0], BenchmarkSample)
    assert samples[0].title
    assert samples[0].ground_truth_track
    assert 1 <= samples[0].ground_truth_viral_score <= 5


def test_dataset_filtering():
    limit_samples = load_dataset(limit=50)
    assert len(limit_samples) == 50

    tech_samples = load_dataset(category="科技数码")
    assert len(tech_samples) > 0
    assert all(s.ground_truth_track == "科技数码" for s in tech_samples)

    hard_samples = load_dataset(difficulty=DifficultyLevel.HARD)
    assert len(hard_samples) > 0
    assert all(s.difficulty == DifficultyLevel.HARD for s in hard_samples)


def test_dataset_statistics():
    samples = load_dataset()
    stats = get_dataset_statistics(samples)
    assert stats["total_samples"] == 1024
    assert len(stats["track_distribution"]) == 12
    assert stats["clickbait_count"] > 0
    assert stats["commercial_ad_count"] > 0
