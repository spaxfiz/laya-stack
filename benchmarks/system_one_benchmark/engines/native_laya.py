import time
from typing import Any, Dict, List, Optional
from pathlib import Path
import laya_mlx
from .base import BaseDecisionEngine

DEFAULT_HF_SNAPSHOT = str(
    Path.home() / ".cache/huggingface/hub/models--aac6fef--laya-multilingual-mlx/snapshots/f2b4faf51023039425946074e2cf1361d2db11d5"
)

class NativeLayaEngine(BaseDecisionEngine):
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or DEFAULT_HF_SNAPSHOT
        self.agent = laya_mlx.Agent(self.model_path)

    @property
    def name(self) -> str:
        return "Native-Laya-MLX (Python SDK)"

    @property
    def description(self) -> str:
        return "In-process MLX DecisionModel with 2-layer Cross-Attention Head & Scorer"

    def predict(self, state: str, questions: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        res = self.agent.predict(state, questions)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        res["latency_ms"] = round(elapsed_ms, 2)
        return res

    def rank(self, context: str, candidates: List[str], question: Optional[str] = None) -> List[Dict[str, Any]]:
        q = {
            "rank": {
                "type": "choice",
                "instructions": question or "Choose the best candidate.",
                "criteria": {str(i): c for i, c in enumerate(candidates)}
            }
        }
        res = self.predict(context, q)
        probs = res["answers"]["rank"]["probabilities"]
        sorted_indices = sorted(probs.items(), key=lambda x: -x[1])
        return [
            {"rank": idx + 1, "candidate": candidates[int(k)], "prob": float(p)}
            for idx, (k, p) in enumerate(sorted_indices)
        ]
