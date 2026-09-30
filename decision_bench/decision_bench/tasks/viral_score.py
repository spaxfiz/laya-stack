"""Viral Potential Rating Task (Score Primitive)."""

from typing import Any, Dict, List
from decision_bench.dataset.schema import BenchmarkSample, TaskType
from decision_bench.models.base import BaseDecisionModel
from decision_bench.metrics.regression import regression_summary
from .base import BaseTask, SamplePrediction


SCORE_LEVELS = [
    "1-极低潜力: 平淡日常无反差",
    "2-较低潜力: 常规经验分享",
    "3-中等潜力: 具备实用价值与清晰痛点",
    "4-较高潜力: 情绪共鸣强烈或高新知爆点",
    "5-极高爆款: 现象级反差或全网热搜话题"
]


class ViralScoreTask(BaseTask):
    """
    Evaluates continuous / ordinal scoring primitive on viral potential rating (1-5).
    Tests the model's calibration on regression and rank-order consistency.
    """

    def __init__(self):
        super().__init__(
            name="viral_score",
            task_type=TaskType.SCORE,
            description="Viral Engagement Potential Score Rating (1-5 Scale)"
        )

    def evaluate_sample(
        self,
        model: BaseDecisionModel,
        sample: BenchmarkSample
    ) -> SamplePrediction:
        out = model.predict_score(
            prompt=sample.full_context,
            criteria=SCORE_LEVELS,
            instruction="评估该内容的爆款潜力和情绪调动指数，评分范围为1到5分。"
        )

        try:
            pred_score = float(out.value)
        except (ValueError, TypeError):
            pred_score = 2.5

        is_exact = abs(pred_score - sample.ground_truth_viral_score) < 0.5

        return SamplePrediction(
            sample_id=sample.id,
            context=sample.title,
            ground_truth=float(sample.ground_truth_viral_score),
            prediction=round(pred_score, 2),
            confidence=out.confidence,
            is_correct=is_exact,
            latency_ms=out.latency_ms,
            error=out.error,
            metadata=out.metadata
        )

    def compute_metrics(
        self,
        predictions: List[SamplePrediction]
    ) -> Dict[str, Any]:
        y_true = [float(p.ground_truth) for p in predictions]
        y_pred = [float(p.prediction) for p in predictions]

        reg_summary = regression_summary(y_true, y_pred)
        return reg_summary
