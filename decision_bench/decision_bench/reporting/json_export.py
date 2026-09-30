"""JSON report exporter for DecisionBench."""

import json
from pathlib import Path
from decision_bench.evaluator import BenchmarkReport


def export_json_report(
    report: BenchmarkReport,
    output_path: str,
    include_samples: bool = False
) -> None:
    """Export benchmark report to a JSON file."""
    data = report.model_dump()
    if not include_samples:
        for t_data in data.get("task_results", {}).values():
            t_data.pop("predictions", None)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
