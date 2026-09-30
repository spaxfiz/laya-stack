"""Core benchmark evaluator orchestrator for DecisionBench."""

import platform
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

from decision_bench.dataset.loader import load_dataset
from decision_bench.dataset.schema import BenchmarkSample, DifficultyLevel
from decision_bench.models.base import BaseDecisionModel
from decision_bench.tasks.base import TaskResult
from decision_bench.tasks import TASK_REGISTRY, get_task


class BenchmarkReport(BaseModel):
    """Unified benchmark evaluation report."""
    model_name: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    task_results: Dict[str, TaskResult]
    overall_quality_score: float = Field(..., description="Overall accuracy/quality composite score (0-100)")
    overall_speed_score: float = Field(..., description="Overall latency efficiency score (0-100)")
    decision_index: float = Field(..., description="Composite Decision Index (D-Index 0-100)")
    environment: Dict[str, Any] = Field(default_factory=dict)
    summary_table: Dict[str, Any] = Field(default_factory=dict)


def compute_decision_index(task_results: Dict[str, TaskResult]) -> Dict[str, float]:
    """
    Compute Decision Quality, Speed Score, and composite Decision Index (D-Index).
    Decision Index = 0.7 * Quality + 0.3 * Speed
    """
    quality_components = []
    p50_latencies = []

    for name, res in task_results.items():
        p50 = res.latency.get("p50_ms", 50.0)
        p50_latencies.append(p50)

        m = res.metrics
        if "accuracy" in m:
            quality_components.append(m["accuracy"] * 100.0)
        elif "top1_hit_rate" in m:
            quality_components.append(m["top1_hit_rate"] * 100.0)
        elif "within_1_level" in m:
            quality_components.append(m["within_1_level"] * 100.0)
        elif "robustness_accuracy" in m:
            quality_components.append(m["robustness_accuracy"] * 100.0)

    quality_score = sum(quality_components) / len(quality_components) if quality_components else 0.0

    # Speed Score: target is <= 15ms -> 100; 50ms -> 75; 100ms -> 50; >= 500ms -> 10
    avg_p50 = sum(p50_latencies) / len(p50_latencies) if p50_latencies else 100.0
    if avg_p50 <= 15.0:
        speed_score = 100.0
    elif avg_p50 <= 50.0:
        speed_score = 100.0 - ((avg_p50 - 15.0) / 35.0) * 25.0
    elif avg_p50 <= 200.0:
        speed_score = 75.0 - ((avg_p50 - 50.0) / 150.0) * 35.0
    else:
        speed_score = max(5.0, 40.0 - (avg_p50 / 50.0))

    d_index = 0.70 * quality_score + 0.30 * speed_score

    return {
        "quality_score": round(quality_score, 2),
        "speed_score": round(speed_score, 2),
        "decision_index": round(d_index, 2),
        "average_p50_ms": round(avg_p50, 2)
    }


class DecisionBenchEvaluator:
    """
    Evaluator engine for running Decision Model benchmarks.
    """

    def __init__(
        self,
        samples: Optional[List[BenchmarkSample]] = None,
        warmup_runs: int = 2
    ):
        self.samples = samples or load_dataset()
        self.warmup_runs = warmup_runs

    def run(
        self,
        model: BaseDecisionModel,
        tasks: Optional[List[str]] = None,
        limit: Optional[int] = None,
        difficulty: Optional[DifficultyLevel] = None,
        progress_callback: Optional[Callable[[str, int, int, Any], None]] = None
    ) -> BenchmarkReport:
        """
        Run the benchmark battery against the target model.
        """
        # Warmup model
        if self.warmup_runs > 0:
            model.warmup(self.warmup_runs)

        # Filter dataset
        eval_samples = self.samples
        if difficulty:
            eval_samples = [s for s in eval_samples if s.difficulty == difficulty]
        if limit and limit > 0:
            eval_samples = eval_samples[:limit]

        # Determine tasks to run
        selected_tasks = tasks or list(TASK_REGISTRY.keys())
        if "all" in selected_tasks:
            selected_tasks = list(TASK_REGISTRY.keys())

        task_results: Dict[str, TaskResult] = {}

        for task_key in selected_tasks:
            task_instance = get_task(task_key)

            # Special filtering for adversarial task if requested specifically
            if task_key == "adversarial":
                adv_samples = [
                    s for s in eval_samples
                    if s.difficulty in (DifficultyLevel.ADVERSARIAL, DifficultyLevel.HARD)
                ]
                task_samples = adv_samples if len(adv_samples) >= 5 else eval_samples
            else:
                task_samples = eval_samples

            def on_progress(current: int, total: int, pred: Any):
                if progress_callback:
                    progress_callback(task_key, current, total, pred)

            result = task_instance.run(
                model=model,
                samples=task_samples,
                progress_callback=on_progress
            )
            task_results[task_key] = result

        # Compute index & scores
        index_scores = compute_decision_index(task_results)

        env_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor() or "Apple Silicon",
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Build summary table dictionary
        summary = {}
        for k, v in task_results.items():
            primary_metric = "N/A"
            m = v.metrics
            if "accuracy" in m:
                primary_metric = f"Acc: {m['accuracy'] * 100:.1f}%"
            elif "top1_hit_rate" in m:
                primary_metric = f"Top-1: {m['top1_hit_rate'] * 100:.1f}%"
            elif "within_1_level" in m:
                primary_metric = f"Within-1: {m['within_1_level'] * 100:.1f}%"
            elif "robustness_accuracy" in m:
                primary_metric = f"RobAcc: {m['robustness_accuracy'] * 100:.1f}%"

            summary[k] = {
                "task_name": v.task_name,
                "type": v.task_type.value,
                "samples": v.num_samples,
                "primary_metric": primary_metric,
                "p50_ms": v.latency.get("p50_ms", 0.0),
                "p95_ms": v.latency.get("p95_ms", 0.0),
                "qps": v.latency.get("qps", 0.0)
            }

        return BenchmarkReport(
            model_name=model.name,
            task_results=task_results,
            overall_quality_score=index_scores["quality_score"],
            overall_speed_score=index_scores["speed_score"],
            decision_index=index_scores["decision_index"],
            environment=env_info,
            summary_table=summary
        )
