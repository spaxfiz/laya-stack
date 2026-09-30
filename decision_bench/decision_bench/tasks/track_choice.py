"""12-Way Vertical Track Choice Task."""

from typing import Any, Dict, List
from decision_bench.dataset.schema import BenchmarkSample, TaskType
from decision_bench.models.base import BaseDecisionModel
from decision_bench.metrics.classification import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
from decision_bench.metrics.calibration import expected_calibration_error
from .base import BaseTask, SamplePrediction


ALL_TRACKS = [
    "科技数码",
    "萌宠动物",
    "美妆护肤",
    "穿搭时尚",
    "美食探店",
    "职场成长",
    "家居生活",
    "运动健康",
    "母婴育儿",
    "旅游出行",
    "情感心理",
    "摄影剪辑"
]


class TrackChoiceTask(BaseTask):
    """
    Evaluates 12-way discrete track classification performance.
    Measures accuracy, macro-F1, per-vertical F1, confusion matrix, and ECE.
    """

    def __init__(self, options: List[str] = ALL_TRACKS):
        super().__init__(
            name="track_choice",
            task_type=TaskType.CHOICE,
            description="12-Way Xiaohongshu Vertical Domain Track Classification"
        )
        self.options = options

    def evaluate_sample(
        self,
        model: BaseDecisionModel,
        sample: BenchmarkSample
    ) -> SamplePrediction:
        out = model.predict_choice(
            prompt=sample.full_context,
            options=self.options,
            instruction="请根据笔记标题和内容，准确选择最符合的唯一内容分类赛道。"
        )

        chosen = str(out.value).strip() if out.value else ""
        is_correct = (chosen == sample.ground_truth_track)

        return SamplePrediction(
            sample_id=sample.id,
            context=sample.title,
            ground_truth=sample.ground_truth_track,
            prediction=chosen,
            confidence=out.confidence,
            is_correct=is_correct,
            latency_ms=out.latency_ms,
            error=out.error,
            metadata=out.metadata
        )

    def compute_metrics(
        self,
        predictions: List[SamplePrediction]
    ) -> Dict[str, Any]:
        y_true = [p.ground_truth for p in predictions]
        y_pred = [p.prediction for p in predictions]

        acc = accuracy_score(y_true, y_pred)
        prf = precision_recall_fscore_support(y_true, y_pred, labels=self.options, average="macro")
        conf_mat = confusion_matrix(y_true, y_pred, labels=self.options)

        # Compute calibration for top-choice confidence
        valid_conf_preds = [p for p in predictions if p.confidence is not None]
        if valid_conf_preds:
            y_binary = [1 if p.is_correct else 0 for p in valid_conf_preds]
            y_conf = [p.confidence for p in valid_conf_preds] # type: ignore
            ece = expected_calibration_error(y_binary, y_conf)
        else:
            ece = 0.0

        return {
            "accuracy": acc,
            "macro_precision": prf["precision"],
            "macro_recall": prf["recall"],
            "macro_f1": prf["f1"],
            "ece": ece,
            "per_class": prf["per_class"],
            "confusion_matrix": conf_mat
        }
