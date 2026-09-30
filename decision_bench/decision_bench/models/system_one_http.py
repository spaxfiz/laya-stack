"""HTTP Client adapter for System One / Rapid-MLX REST APIs."""

import time
from typing import Any, Dict, List, Optional
import httpx
from .base import BaseDecisionModel, DecisionOutput


class SystemOneHttpModel(BaseDecisionModel):
    """
    Adapter for System One HTTP servers (such as Rapid-MLX, custom inference proxies, etc.).
    Communicates via `/v1/systemone` and `/v1/rank`.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8700",
        name: str = "system-one-http",
        timeout: float = 30.0,
        headers: Optional[Dict[str, str]] = None
    ):
        super().__init__(name=name)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            headers=headers or {}
        )

    def close(self):
        self.client.close()

    def warmup(self, n: int = 1) -> None:
        """Ping the model to warm up GPU/cache."""
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
        # Rapid-MLX expects criteria as a dictionary {key: description}
        criteria = {opt: opt for opt in options}
        payload = {
            "state": prompt,
            "questions": {
                "decision": {
                    "type": "choice",
                    "instructions": instruction,
                    "criteria": criteria
                }
            }
        }

        start = time.perf_counter()
        try:
            resp = self.client.post("/v1/systemone", json=payload)
            lat_ms = (time.perf_counter() - start) * 1000.0

            if resp.status_code != 200:
                return DecisionOutput(
                    value=options[0] if options else "",
                    latency_ms=lat_ms,
                    error=f"HTTP {resp.status_code}: {resp.text}"
                )

            data = resp.json()
            ans = data.get("answers", {}).get("decision", {})
            chosen = ans.get("choice", "")
            confidence = ans.get("confidence", 0.0)
            probs = ans.get("probabilities", {})

            # Use server reported latency if provided, or client measured
            server_lat = data.get("latency_ms", lat_ms)

            return DecisionOutput(
                value=chosen,
                confidence=confidence,
                probabilities=probs,
                latency_ms=round(server_lat, 2),
                metadata=data.get("usage", {})
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
        payload = {
            "state": prompt,
            "questions": {
                "decision": {
                    "type": "noul",
                    "instructions": instruction
                }
            }
        }

        start = time.perf_counter()
        try:
            resp = self.client.post("/v1/systemone", json=payload)
            lat_ms = (time.perf_counter() - start) * 1000.0

            if resp.status_code != 200:
                return DecisionOutput(
                    value=False,
                    latency_ms=lat_ms,
                    error=f"HTTP {resp.status_code}: {resp.text}"
                )

            data = resp.json()
            ans = data.get("answers", {}).get("decision", {})
            noul_val = ans.get("noul", 0.0)  # probability of True
            confidence = ans.get("confidence", 0.0)
            is_true = noul_val >= 0.5

            server_lat = data.get("latency_ms", lat_ms)

            return DecisionOutput(
                value=is_true,
                confidence=confidence,
                probabilities={"True": noul_val, "False": 1.0 - noul_val},
                latency_ms=round(server_lat, 2),
                metadata=data.get("usage", {})
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
        # Standardize criteria into ordered list of levels
        if isinstance(criteria, list):
            crit_list = [str(c) for c in criteria]
        elif isinstance(criteria, dict):
            crit_list = [str(v) for v in criteria.values()]
        else:
            crit_list = ["1-极低", "2-较低", "3-一般", "4-较高", "5-极高"]

        payload = {
            "state": prompt,
            "questions": {
                "decision": {
                    "type": "score",
                    "instructions": instruction,
                    "criteria": crit_list
                }
            }
        }

        start = time.perf_counter()
        try:
            resp = self.client.post("/v1/systemone", json=payload)
            lat_ms = (time.perf_counter() - start) * 1000.0

            if resp.status_code != 200:
                return DecisionOutput(
                    value=2.0,
                    latency_ms=lat_ms,
                    error=f"HTTP {resp.status_code}: {resp.text}"
                )

            data = resp.json()
            ans = data.get("answers", {}).get("decision", {})
            raw_score = ans.get("score", 0.0)
            confidence = ans.get("confidence", 0.0)
            # Rapid-MLX score is 0-indexed float across levels (e.g. 0 to len-1).
            # We normalize to 1-based scale (1 to 5)
            normalized_score = raw_score + 1.0

            server_lat = data.get("latency_ms", lat_ms)

            return DecisionOutput(
                value=round(normalized_score, 2),
                confidence=confidence,
                probabilities=ans.get("probabilities", {}),
                latency_ms=round(server_lat, 2),
                metadata=data.get("usage", {})
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
        payload = {
            "context": prompt,
            "answers": candidates
        }
        if instruction:
            payload["question"] = instruction

        start = time.perf_counter()
        try:
            resp = self.client.post("/v1/rank", json=payload)
            lat_ms = (time.perf_counter() - start) * 1000.0

            if resp.status_code != 200:
                return DecisionOutput(
                    value=candidates,
                    latency_ms=lat_ms,
                    error=f"HTTP {resp.status_code}: {resp.text}"
                )

            data = resp.json()
            ranked_items = data.get("ranked", [])
            sorted_candidates = [item["candidate"] for item in ranked_items]
            server_lat = data.get("latency_ms", lat_ms)

            return DecisionOutput(
                value=sorted_candidates,
                confidence=ranked_items[0].get("prob", 0.0) if ranked_items else 0.0,
                latency_ms=round(server_lat, 2),
                metadata={"ranked": ranked_items}
            )
        except Exception as e:
            lat_ms = (time.perf_counter() - start) * 1000.0
            return DecisionOutput(
                value=candidates,
                latency_ms=lat_ms,
                error=str(e)
            )
