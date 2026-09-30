"""In-process Python MLX model adapter using laya_mlx."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from .base import BaseDecisionModel, DecisionOutput


class LayaMlxModel(BaseDecisionModel):
    """
    Direct in-process MLX Decision Model adapter.
    Runs on Apple Silicon GPU via MLX without HTTP overhead.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        name: str = "laya-mlx-native"
    ):
        super().__init__(name=name)
        try:
            import laya_mlx
        except ImportError:
            raise ImportError("laya_mlx is not installed. Please install laya-mlx to use LayaMlxModel.")

        if not model_path:
            # Default to local cache
            default_path = (
                Path.home() /
                ".cache/huggingface/hub/models--aac6fef--laya-multilingual-mlx/snapshots/f2b4faf51023039425946074e2cf1361d2db11d5"
            )
            model_path = str(default_path)

        self.agent = laya_mlx.Agent(model_path)

    def warmup(self, n: int = 1) -> None:
        for _ in range(n):
            try:
                self.predict_choice("Warmup text", ["A", "B"])
            except Exception:
                pass

    def predict_choice(
        self,
        prompt: str,
        options: List[str],
        instruction: str = "Select the best matching category"
    ) -> DecisionOutput:
        criteria = {opt: opt for opt in options}
        questions = {
            "decision": {
                "type": "choice",
                "instructions": instruction,
                "criteria": criteria
            }
        }

        start = time.perf_counter()
        try:
            res = self.agent.system_one(prompt, questions)
            lat_ms = (time.perf_counter() - start) * 1000.0

            ans = res.get("answers", {}).get("decision", {})
            chosen = ans.get("choice", "")
            confidence = ans.get("confidence", 0.0)
            probs = ans.get("probabilities", {})

            return DecisionOutput(
                value=chosen,
                confidence=confidence,
                probabilities=probs,
                latency_ms=round(lat_ms, 2),
                metadata=res.get("usage", {})
            )
        except Exception as e:
            lat_ms = (time.perf_counter() - start) * 1000.0
            return DecisionOutput(
                value=options[0] if options else "",
                latency_ms=lat_ms,
                error=str(e)
            )

    def predict_noul(
        self,
        prompt: str,
        instruction: str = "Evaluate whether the proposition is true or false"
    ) -> DecisionOutput:
        questions = {
            "decision": {
                "type": "noul",
                "instructions": instruction
            }
        }

        start = time.perf_counter()
        try:
            res = self.agent.system_one(prompt, questions)
            lat_ms = (time.perf_counter() - start) * 1000.0

            ans = res.get("answers", {}).get("decision", {})
            noul_val = ans.get("noul", 0.0)
            confidence = ans.get("confidence", 0.0)
            is_true = noul_val >= 0.5

            return DecisionOutput(
                value=is_true,
                confidence=confidence,
                probabilities={"True": noul_val, "False": 1.0 - noul_val},
                latency_ms=round(lat_ms, 2),
                metadata=res.get("usage", {})
            )
        except Exception as e:
            lat_ms = (time.perf_counter() - start) * 1000.0
            return DecisionOutput(
                value=False,
                latency_ms=lat_ms,
                error=str(e)
            )

    def predict_score(
        self,
        prompt: str,
        criteria: Any,
        instruction: str = "Rate the sample according to the criteria"
    ) -> DecisionOutput:
        if isinstance(criteria, list):
            crit_list = [str(c) for c in criteria]
        elif isinstance(criteria, dict):
            crit_list = [str(v) for v in criteria.values()]
        else:
            crit_list = ["1-极低", "2-较低", "3-一般", "4-较高", "5-极高"]

        questions = {
            "decision": {
                "type": "score",
                "instructions": instruction,
                "criteria": crit_list
            }
        }

        start = time.perf_counter()
        try:
            res = self.agent.system_one(prompt, questions)
            lat_ms = (time.perf_counter() - start) * 1000.0

            ans = res.get("answers", {}).get("decision", {})
            raw_score = ans.get("score", 0.0)
            confidence = ans.get("confidence", 0.0)
            normalized_score = raw_score + 1.0

            return DecisionOutput(
                value=round(normalized_score, 2),
                confidence=confidence,
                probabilities=ans.get("probabilities", {}),
                latency_ms=round(lat_ms, 2),
                metadata=res.get("usage", {})
            )
        except Exception as e:
            lat_ms = (time.perf_counter() - start) * 1000.0
            return DecisionOutput(
                value=2.0,
                latency_ms=lat_ms,
                error=str(e)
            )

    def rank(
        self,
        prompt: str,
        candidates: List[str],
        instruction: Optional[str] = None
    ) -> DecisionOutput:
        # Rank by querying candidate probabilities
        inst = instruction or "Select the best match"
        criteria = {c: c for c in candidates}
        questions = {
            "rank": {
                "type": "choice",
                "instructions": inst,
                "criteria": criteria
            }
        }

        start = time.perf_counter()
        try:
            res = self.agent.system_one(prompt, questions)
            lat_ms = (time.perf_counter() - start) * 1000.0

            ans = res.get("answers", {}).get("rank", {})
            probs = ans.get("probabilities", {})
            sorted_candidates = sorted(
                candidates,
                key=lambda c: probs.get(c, 0.0),
                reverse=True
            )

            return DecisionOutput(
                value=sorted_candidates,
                confidence=probs.get(sorted_candidates[0], 0.0) if sorted_candidates else 0.0,
                probabilities=probs,
                latency_ms=round(lat_ms, 2)
            )
        except Exception as e:
            lat_ms = (time.perf_counter() - start) * 1000.0
            return DecisionOutput(
                value=candidates,
                latency_ms=lat_ms,
                error=str(e)
            )
