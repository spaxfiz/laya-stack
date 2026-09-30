"""Tasks registry and execution module for DecisionBench."""

from typing import Dict, List, Type
from .base import BaseTask, TaskResult, SamplePrediction
from .track_choice import TrackChoiceTask, ALL_TRACKS
from .clickbait_noul import ClickbaitNoulTask
from .ad_detection_noul import AdDetectionNoulTask
from .viral_score import ViralScoreTask
from .ranking_task import CandidateRankingTask
from .adversarial_stress import AdversarialStressTask


TASK_REGISTRY: Dict[str, Type[BaseTask]] = {
    "choice": TrackChoiceTask,
    "clickbait": ClickbaitNoulTask,
    "ad": AdDetectionNoulTask,
    "score": ViralScoreTask,
    "rank": CandidateRankingTask,
    "adversarial": AdversarialStressTask
}


def get_task(name: str, **kwargs) -> BaseTask:
    """Instantiate a task by shorthand identifier."""
    name_clean = name.lower().strip()
    if name_clean not in TASK_REGISTRY:
        available = ", ".join(TASK_REGISTRY.keys())
        raise ValueError(f"Unknown task '{name}'. Available tasks: {available}")
    return TASK_REGISTRY[name_clean](**kwargs)


def list_tasks() -> List[Dict[str, str]]:
    """Return list of available tasks with description."""
    res = []
    for key, task_cls in TASK_REGISTRY.items():
        inst = task_cls()
        res.append({
            "key": key,
            "name": inst.name,
            "task_type": inst.task_type.value,
            "description": inst.description
        })
    return res


__all__ = [
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
    "ALL_TRACKS"
]
