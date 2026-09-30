"""DecisionBench: Universal Benchmark Suite for System One and Decision Models."""

from .dataset.schema import BenchmarkSample, TaskType, DifficultyLevel
from .dataset.loader import load_dataset, get_dataset_statistics
from .models import (
    BaseDecisionModel,
    DecisionOutput,
    MockDecisionModel,
    SystemOneHttpModel,
    LayaMlxModel,
    OpenAICompatibleModel,
    create_model
)
from .tasks import (
    BaseTask,
    TaskResult,
    SamplePrediction,
    TrackChoiceTask,
    ClickbaitNoulTask,
    AdDetectionNoulTask,
    ViralScoreTask,
    CandidateRankingTask,
    AdversarialStressTask,
    TASK_REGISTRY,
    get_task,
    list_tasks
)
from .evaluator import DecisionBenchEvaluator, BenchmarkReport
from .reporting import (
    generate_markdown_report,
    save_markdown_report,
    print_report_to_console,
    export_json_report
)

__version__ = "1.0.0"

__all__ = [
    "BenchmarkSample",
    "TaskType",
    "DifficultyLevel",
    "load_dataset",
    "get_dataset_statistics",
    "BaseDecisionModel",
    "DecisionOutput",
    "MockDecisionModel",
    "SystemOneHttpModel",
    "LayaMlxModel",
    "OpenAICompatibleModel",
    "create_model",
    "BaseTask",
    "TaskResult",
    "SamplePrediction",
    "TrackChoiceTask",
    "ClickbaitNoulTask",
    "AdDetectionNoulTask",
    "ViralScoreTask",
    "CandidateRankingTask",
    "AdversarialStressTask",
    "TASK_REGISTRY",
    "get_task",
    "list_tasks",
    "DecisionBenchEvaluator",
    "BenchmarkReport",
    "generate_markdown_report",
    "save_markdown_report",
    "print_report_to_console",
    "export_json_report"
]
