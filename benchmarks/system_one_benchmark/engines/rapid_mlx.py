import time
import json
import urllib.request
from typing import Any, Dict, List, Optional
from .base import BaseDecisionEngine

class RapidMlxEngine(BaseDecisionEngine):
    def __init__(self, endpoint_url: str = "http://127.0.0.1:8700"):
        self.endpoint_url = endpoint_url.rstrip("/")

    @property
    def name(self) -> str:
        return "Rapid-MLX System-One (HTTP API)"

    @property
    def description(self) -> str:
        return f"Rapid-MLX Standalone Daemon on {self.endpoint_url} (First-class LayaBackend)"

    def predict(self, state: str, questions: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        req = urllib.request.Request(
            f"{self.endpoint_url}/v1/systemone",
            data=json.dumps({"state": state, "questions": questions}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        elapsed_ms = (time.perf_counter() - t0) * 1000
        data["latency_ms"] = round(elapsed_ms, 2)
        return data

    def rank(self, context: str, candidates: List[str], question: Optional[str] = None) -> List[Dict[str, Any]]:
        req = urllib.request.Request(
            f"{self.endpoint_url}/v1/rank",
            data=json.dumps({"context": context, "answers": candidates, "question": question}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["ranked"]
