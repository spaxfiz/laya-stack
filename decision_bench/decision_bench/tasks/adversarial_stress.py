"""Adversarial and Polysemy Disambiguation Stress Test Task."""

from typing import Any, Dict, List
from decision_bench.dataset.schema import BenchmarkSample, DifficultyLevel, TaskType
from decision_bench.models.base import BaseDecisionModel
from decision_bench.metrics.classification import accuracy_score, precision_recall_fscore_support
from .base import BaseTask, SamplePrediction
from .track_choice import ALL_TRACKS


class AdversarialStressTask(BaseTask):
    """
    Stress-tests model robustness against polysemy, entity confusion,
    and cross-domain distractor terms (e.g. 苹果/Apple, 牦牛/Yak, 咖啡/Cat cafe).
    """

    def __init__(self, options: List[str] = ALL_TRACKS):
        super().__init__(
            name="adversarial_stress",
            task_type=TaskType.ADVERSARIAL,
            description="Adversarial Polysemy and Cross-Domain Entity Disambiguation"
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
            instruction="【高难度抗干扰测试】请精准识别笔记的真实主核心赛道，排除双关语或交叉领域的误导干扰。"
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
            metadata={
                "difficulty": sample.difficulty.value,
                "notes": sample.notes or ""
            }
        )

    def compute_metrics(
        self,
        predictions: List[SamplePrediction]
    ) -> Dict[str, Any]:
        y_true = [p.ground_truth for p in predictions]
        y_pred = [p.prediction for p in predictions]

        acc = accuracy_score(y_true, y_pred)
        prf = precision_recall_fscore_support(y_true, y_pred, labels=self.options, average="macro")

        # Breakdown by difficulty if both hard and adversarial exist
        hard_preds = [p for p in predictions if p.metadata.get("difficulty") == "hard"]
        adv_preds = [p for p in predictions if p.metadata.get("difficulty") == "adversarial"]

        hard_acc = accuracy_score([p.ground_truth for p in hard_preds], [p.prediction for p in hard_preds]) if hard_preds else None
        adv_acc = accuracy_score([p.ground_truth for p in adv_preds], [p.prediction for p in adv_preds]) if adv_preds else None

        # Collect failure cases
        failures = [
            {
                "id": p.sample_id,
                "title": p.context,
                "expected": p.ground_truth,
                "predicted": p.prediction,
                "notes": p.metadata.get("notes", "")
            }
            for p in predictions if not p.is_correct
        ]

        return {
            "robustness_accuracy": acc,
            "macro_f1": prf["f1"],
            "hard_subset_accuracy": hard_acc,
            "adversarial_subset_accuracy": adv_acc,
            "total_adversarial_tested": len(predictions),
            "failure_case_count": len(failures),
            "top_failures": failures[:10]
        }
