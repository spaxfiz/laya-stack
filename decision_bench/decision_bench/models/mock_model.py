"""Mock Decision Model for fast testing, validation, and benchmarking baseline."""

import random
import time
from typing import Any, Dict, List, Optional
from .base import BaseDecisionModel, DecisionOutput


class MockDecisionModel(BaseDecisionModel):
    """
    Mock decision model for unit testing and CI benchmarks.
    Can be configured with deterministic seed, target accuracy, or simulated latency.
    """

    def __init__(
        self,
        name: str = "mock-decision-model",
        simulated_latency_ms: float = 2.0,
        accuracy_bias: float = 0.85,
        seed: int = 42
    ):
        super().__init__(name=name)
        self.simulated_latency_ms = simulated_latency_ms
        self.accuracy_bias = accuracy_bias
        self.rng = random.Random(seed)

    def _simulate_delay(self) -> float:
        start = time.perf_counter()
        if self.simulated_latency_ms > 0:
            # Jitter latency slightly
            jitter = self.rng.uniform(0.8, 1.2)
            time.sleep((self.simulated_latency_ms * jitter) / 1000.0)
        return round((time.perf_counter() - start) * 1000.0, 3)

    def predict_choice(
        self,
        prompt: str,
        options: List[str],
        instruction: str = "Select the best matching category"
    ) -> DecisionOutput:
        lat = self._simulate_delay()
        if not options:
            return DecisionOutput(value="", latency_ms=lat, error="No options provided")

        # Pick option based on substring match in prompt, else random
        matched = [opt for opt in options if opt in prompt]
        if matched and self.rng.random() < self.accuracy_bias:
            chosen = matched[0]
        else:
            chosen = self.rng.choice(options)

        # Build synthetic probabilities
        probs = {}
        for opt in options:
            probs[opt] = 0.8 if opt == chosen else (0.2 / max(1, len(options) - 1))

        return DecisionOutput(
            value=chosen,
            confidence=probs[chosen],
            probabilities=probs,
            latency_ms=lat,
            metadata={"simulated": True}
        )

    def predict_noul(
        self,
        prompt: str,
        instruction: str = "Evaluate whether the proposition is true or false"
    ) -> DecisionOutput:
        lat = self._simulate_delay()
        # Heuristic keywords for simulation
        clickbait_signals = ["震惊", "居然", "绝了", "千万别", "秘密", "闭眼入", "血亏", "太绝"]
        has_signal = any(sig in prompt for sig in clickbait_signals)

        if has_signal:
            is_true = True if self.rng.random() < self.accuracy_bias else False
        else:
            is_true = False if self.rng.random() < self.accuracy_bias else True

        prob = 0.88 if is_true else 0.12
        return DecisionOutput(
            value=is_true,
            confidence=prob if is_true else 1.0 - prob,
            latency_ms=lat,
            metadata={"simulated": True}
        )

    def predict_score(
        self,
        prompt: str,
        criteria: Any,
        instruction: str = "Rate the sample according to the criteria"
    ) -> DecisionOutput:
        lat = self._simulate_delay()
        # Criteria can be list of levels or dict
        if isinstance(criteria, list):
            n_levels = len(criteria)
        else:
            n_levels = 5

        # Heuristic score between 1 and n_levels
        score = self.rng.randint(1, n_levels)
        return DecisionOutput(
            value=float(score),
            confidence=0.8,
            latency_ms=lat,
            metadata={"simulated": True}
        )

    def rank(
        self,
        prompt: str,
        candidates: List[str],
        instruction: Optional[str] = None
    ) -> DecisionOutput:
        lat = self._simulate_delay()
        sorted_candidates = sorted(
            candidates,
            key=lambda c: sum(1 for word in prompt.split() if word in c),
            reverse=True
        )
        return DecisionOutput(
            value=sorted_candidates,
            confidence=0.85,
            latency_ms=lat,
            metadata={"simulated": True}
        )
