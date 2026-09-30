"""End-to-end evaluation pipeline and reporting tests."""

import json
import pytest
from decision_bench.dataset.loader import load_dataset
from decision_bench.models.mock_model import MockDecisionModel
from decision_bench.evaluator import DecisionBenchEvaluator
from decision_bench.reporting import generate_markdown_report, export_json_report


def test_full_pipeline_run(tmp_path):
    evaluator = DecisionBenchEvaluator()
    model = MockDecisionModel("e2e-mock-model", simulated_latency_ms=0.0)

    report = evaluator.run(model, limit=10)

    # Check overall report fields
    assert report.model_name == "e2e-mock-model"
    assert 0.0 <= report.overall_quality_score <= 100.0
    assert 0.0 <= report.overall_speed_score <= 100.0
    assert 0.0 <= report.decision_index <= 100.0
    assert len(report.task_results) == 6

    # Test Markdown generation
    md_content = generate_markdown_report(report)
    assert "# 🚀 DecisionBench" in md_content
    assert "e2e-mock-model" in md_content

    # Test JSON export
    json_path = tmp_path / "report.json"
    export_json_report(report, str(json_path), include_samples=False)
    assert json_path.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["model_name"] == "e2e-mock-model"
    assert "overall_quality_score" in data
