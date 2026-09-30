from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class BaseDecisionEngine(ABC):
    """Abstract base class for all System One / DecisionModel benchmark candidates."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the engine."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of the architecture and deployment."""
        pass

    @abstractmethod
    def predict(self, state: str, questions: Dict[str, Any]) -> Dict[str, Any]:
        """Execute decision query on state with specified typed questions.
        
        Returns dict matching System One standard schema:
        {
            "answers": {
                "<question_id>": {
                    "type": "choice" | "score" | "noul",
                    "choice": str,
                    "confidence": float,
                    "probabilities": Dict[str, float]
                }
            },
            "latency_ms": float
        }
        """
        pass

    @abstractmethod
    def rank(self, context: str, candidates: List[str], question: Optional[str] = None) -> List[Dict[str, Any]]:
        """Rank text candidates relative to context.
        
        Returns list:
        [
            {"rank": 1, "candidate": str, "prob": float},
            ...
        ]
        """
        pass
