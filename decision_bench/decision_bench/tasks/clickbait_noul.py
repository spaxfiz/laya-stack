"""Clickbait Boolean Verification Task (Noul)."""

from typing import Any, Dict, List
from decision_bench.dataset.schema import BenchmarkSample, TaskType
from decision_bench.models.base import BaseDecisionModel
from decision_bench.metrics.classification import accuracy_score, precision_recall_fscore_support
from decision_bench.metrics.calibration import expected_calibration_error, brier_score
from .base import BaseTask, SamplePrediction


class ClickbaitNoulTask(BaseTask):
    """
    Evaluates binary boolean verification on clickbait / sensationalism detection.
    Tests model's ability to act as a content moderation & quality filter.
    """

    def __init__(self):
        super().__init__(
            name="clickbait_noul",
            task_type=TaskType.NOUL,
            description="Binary Clickbait and Sensationalism Identification"
        )

    def evaluate_sample(
        self,
        model: BaseDecisionModel,
        sample: BenchmarkSample
    ) -> SamplePrediction:
        out = model.predict_noul(
            prompt=sample.full_context,
            instruction="判断该笔记标题或内容是否具有标题党、浮夸博眼球、情绪煽动或制造焦虑特征？"
        )

        pred_bool = bool(out.value)
        is_correct = (pred_bool == sample.ground_truth_is_clickbait)

        return SamplePrediction(
            sample_id=sample.id,
            context=sample.title,
            ground_truth=sample.ground_truth_is_clickbait,
            prediction=pred_bool,
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
        prf = precision_recall_fscore_support(y_true, y_pred, labels=[True, False], average="macro")

        # Positive class (Clickbait = True) metrics
        pos_metrics = prf["per_class"].get(True, {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0})

        # Calibration
        valid_conf_preds = [p for p in predictions if p.confidence is not None]
        if valid_conf_preds:
            y_binary = [1 if p.is_correct else 0 for p in valid_conf_preds]
            y_conf = [p.confidence for p in valid_conf_preds] # type: ignore
            ece = expected_calibration_error(y_binary, y_conf)
            brier = brier_score(y_binary, y_conf)
        else:
            ece = 0.0
            brier = 0.0

        return {
            "accuracy": acc,
            "macro_f1": prf["f1"],
            "clickbait_precision": pos_metrics["precision"],
            "clickbait_recall": pos_metrics["recall"],
            "clickbait_f1": pos_metrics["f1"],
            "clickbait_support": pos_metrics["support"],
            "ece": ece,
            "brier_score": brier
        }
