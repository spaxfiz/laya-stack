import time
import json
import urllib.request
import numpy as np
from typing import Any, Dict, List, Optional
from .base import BaseDecisionEngine

class OmlxEmbedEngine(BaseDecisionEngine):
    def __init__(self, endpoint_url: str = "http://localhost:8090/v1/embeddings", api_key: str = "123456"):
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self._cache = {}

    @property
    def name(self) -> str:
        return "oMLX Bi-Encoder Embeddings (HTTP API)"

    @property
    def description(self) -> str:
        return "oMLX ModernBERT Sentence Embeddings with Cosine Similarity Matching"

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        try:
            req = urllib.request.Request(
                self.endpoint_url,
                data=json.dumps({"model": "laya-multilingual-mlx", "input": texts}).encode("utf-8"),
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return np.array([x["embedding"] for x in data["data"]])
        except Exception:
            # Fallback zero vector when offline
            return np.zeros((len(texts), 768), dtype=np.float32)

    def predict(self, state: str, questions: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        answers = {}
        try:
            state_emb = self._get_embeddings([state])[0]
        except Exception:
            state_emb = np.zeros(768, dtype=np.float32)

        for qid, qdef in questions.items():
            qtype = qdef.get("type", "choice")
            criteria = qdef.get("criteria", {})
            labels = list(criteria.keys()) if isinstance(criteria, dict) else list(criteria)
            
            # Cache label embeddings
            cache_key = tuple(labels)
            if cache_key not in self._cache:
                self._cache[cache_key] = self._get_embeddings(labels)
            label_embs = self._cache[cache_key]

            sims = np.dot(label_embs, state_emb)
            # Temperature scaling for ModernBERT
            logits = sims * 12.0
            probs = np.exp(logits - np.max(logits))
            probs /= np.sum(probs)

            best_idx = int(np.argmax(probs))
            conf = float(np.max(probs))
            prob_dict = {label: round(float(probs[i]), 4) for i, label in enumerate(labels)}

            answers[qid] = {
                "type": qtype,
                "choice": labels[best_idx],
                "confidence": round(conf, 4),
                "probabilities": prob_dict
            }

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "answers": answers,
            "latency_ms": round(elapsed_ms, 2)
        }

    def rank(self, context: str, candidates: List[str], question: Optional[str] = None) -> List[Dict[str, Any]]:
        c_embs = self._get_embeddings([context])[0]
        cand_embs = self._get_embeddings(candidates)
        sims = np.dot(cand_embs, c_embs)
        probs = np.exp(sims * 10.0 - np.max(sims * 10.0))
        probs /= np.sum(probs)
        sorted_pairs = sorted(zip(candidates, probs), key=lambda x: -x[1])
        return [
            {"rank": idx + 1, "candidate": cand, "prob": round(float(p), 4)}
            for idx, (cand, p) in enumerate(sorted_pairs)
        ]
