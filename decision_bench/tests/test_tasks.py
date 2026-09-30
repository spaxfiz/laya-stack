"""Unit tests for task runners."""

import pytest
from decision_bench.dataset.loader import load_dataset
from decision_bench.models.mock_model import MockDecisionModel
from decision_bench.tasks import get_task, TASK_REGISTRY


@pytest.fixture
def mock_model():
    return MockDecisionModel("test-mock", simulated_latency_ms=0.0)


@pytest.fixture
def sample_data():
    return load_dataset(limit=15)


def test_all_tasks_execution(mock_model, sample_data):
    for task_name in TASK_REGISTRY:
        task = get_task(task_name)
        result = task.run(mock_model, sample_data)
        assert result.num_samples == len(sample_data)
        assert result.execution_time_seconds >= 0.0
        assert len(result.predictions) == len(sample_data)
        assert isinstance(result.metrics, dict)
        assert "p50_ms" in result.latency
