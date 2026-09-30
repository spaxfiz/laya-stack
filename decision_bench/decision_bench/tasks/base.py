"""Base task definition and result container for DecisionBench."""

import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

from decision_bench.dataset.schema import BenchmarkSample, TaskType
from decision_bench.models.base import BaseDecisionModel


class SamplePrediction(BaseModel):
    """Detailed record of a single sample's evaluation."""
    sample_id: str
    context: str
    ground_truth: Any
    prediction: Any
    confidence: Optional[float] = None
    is_correct: bool = False
    latency_ms: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskResult(BaseModel):
    """Aggregated evaluation results of a task executed against a model."""
    task_name: str
    task_type: TaskType
    description: str
    model_name: str
    num_samples: int
    metrics: Dict[str, Any]
    latency: Dict[str, float]
    execution_time_seconds: float
    error_count: int = 0
    predictions: List[SamplePrediction] = Field(default_factory=list)


class BaseTask(ABC):
    """Abstract base class for benchmark tasks."""

    def __init__(self, name: str, task_type: TaskType, description: str):
        self.name = name
        self.task_type = task_type
        self.description = description

    @abstractmethod
    def evaluate_sample(
        self,
        model: BaseDecisionModel,
        sample: BenchmarkSample
    ) -> SamplePrediction:
        """Run single sample evaluation against the model."""
        pass

    @abstractmethod
    def compute_metrics(
        self,
        predictions: List[SamplePrediction]
    ) -> Dict[str, Any]:
        """Aggregate prediction records into statistical metrics."""
        pass

    def run(
        self,
        model: BaseDecisionModel,
        samples: List[BenchmarkSample],
        progress_callback: Optional[Callable[[int, int, SamplePrediction], None]] = None
    ) -> TaskResult:
        """Execute task over all samples and generate TaskResult."""
        from decision_bench.metrics.latency import latency_summary

        predictions: List[SamplePrediction] = []
        latencies_ms: List[float] = []
        errors = 0

        start_time = time.perf_counter()
        total_samples = len(samples)

        for idx, sample in enumerate(samples):
            pred = self.evaluate_sample(model, sample)
            predictions.append(pred)
            latencies_ms.append(pred.latency_ms)
            if pred.error:
                errors += 1

            if progress_callback:
                progress_callback(idx + 1, total_samples, pred)

        total_duration = time.perf_counter() - start_time
        metrics = self.compute_metrics(predictions)
        lat_summary = latency_summary(latencies_ms, total_duration_seconds=total_duration)

        return TaskResult(
            task_name=self.name,
            task_type=self.task_type,
            description=self.description,
            model_name=model.name,
            num_samples=len(samples),
            metrics=metrics,
            latency=lat_summary,
            execution_time_seconds=round(total_duration, 2),
            error_count=errors,
            predictions=predictions
        )
