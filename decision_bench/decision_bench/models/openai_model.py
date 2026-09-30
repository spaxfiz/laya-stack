"""OpenAI-compatible LLM adapter for DecisionBench."""

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import httpx
from .base import BaseDecisionModel, DecisionOutput


class OpenAICompatibleModel(BaseDecisionModel):
    """
    Adapter for any OpenAI-compatible API (vLLM, Ollama, Groq, FastChat, OpenAI).
    Enables benchmarking large models / LLMs on the exact same decision primitives.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "EMPTY",
        model: str = "default",
        name: Optional[str] = None,
        timeout: float = 30.0
    ):
        super().__init__(name=name or f"openai-{model}")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=self.timeout
        )

    def close(self):
        self.client.close()

    def _chat_completion(self, system_prompt: str, user_prompt: str) -> Tuple[str, float]:
        start = time.perf_counter()
        resp = self.client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 128
            }
        )
        lat_ms = (time.perf_counter() - start) * 1000.0
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        return content, lat_ms

    def predict_choice(
        self,
        prompt: str,
        options: List[str],
        instruction: str = "Select the best matching category"
    ) -> DecisionOutput:
        sys = (
            f"You are a fast decision engine. Task: {instruction}. "
            f"Options: {json.dumps(options, ensure_ascii=False)}. "
            f"Respond ONLY with the exact chosen option name, nothing else."
        )
        try:
            content, lat_ms = self._chat_completion(sys, prompt)
            # Find closest option match
            chosen = options[0] if options else ""
            for opt in options:
                if opt in content:
                    chosen = opt
                    break

            return DecisionOutput(
                value=chosen,
                confidence=0.9,
                latency_ms=round(lat_ms, 2)
            )
        except Exception as e:
            return DecisionOutput(
                value=options[0] if options else "",
                latency_ms=0.0,
                error=str(e)
            )

    def predict_noul(
        self,
        prompt: str,
        instruction: str = "Evaluate whether the proposition is true or false"
    ) -> DecisionOutput:
        sys = (
            f"You are a fast binary verifier. Question: {instruction}. "
            f"Answer with ONLY 'TRUE' or 'FALSE'."
        )
        try:
            content, lat_ms = self._chat_completion(sys, prompt)
            is_true = "TRUE" in content.upper()
            return DecisionOutput(
                value=is_true,
                confidence=0.85,
                latency_ms=round(lat_ms, 2)
            )
        except Exception as e:
            return DecisionOutput(
                value=False,
                latency_ms=0.0,
                error=str(e)
            )

    def predict_score(
        self,
        prompt: str,
        criteria: Any,
        instruction: str = "Rate the sample according to the criteria"
    ) -> DecisionOutput:
        sys = (
            f"You are an evaluator. Task: {instruction}. Rating scale: 1 to 5. "
            f"Respond ONLY with a single numeric number from 1 to 5."
        )
        try:
            content, lat_ms = self._chat_completion(sys, prompt)
            match = re.search(r"([1-5](?:\.\d+)?)", content)
            val = float(match.group(1)) if match else 2.5
            return DecisionOutput(
                value=val,
                confidence=0.8,
                latency_ms=round(lat_ms, 2)
            )
        except Exception as e:
            return DecisionOutput(
                value=2.5,
                latency_ms=0.0,
                error=str(e)
            )

    def rank(
        self,
        prompt: str,
        candidates: List[str],
        instruction: Optional[str] = None
    ) -> DecisionOutput:
        sys = (
            f"Rank candidates by relevance to the query. Candidates: {json.dumps(candidates, ensure_ascii=False)}. "
            f"Respond with a JSON list of the candidate strings ordered from highest to lowest relevance."
        )
        try:
            content, lat_ms = self._chat_completion(sys, prompt)
            match = re.search(r"\[.*\]", content, re.DOTALL)
            if match:
                ranked = json.loads(match.group(0))
            else:
                ranked = candidates
            return DecisionOutput(
                value=ranked,
                confidence=0.85,
                latency_ms=round(lat_ms, 2)
            )
        except Exception as e:
            return DecisionOutput(
                value=candidates,
                latency_ms=0.0,
                error=str(e)
            )
