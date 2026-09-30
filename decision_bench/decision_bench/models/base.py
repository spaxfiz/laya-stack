"""Base interfaces and data structures for Decision Models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DecisionOutput(BaseModel):
    """Normalized output from any Decision Model / System One inference."""
    value: Any = Field(..., description="Decision result (e.g. chosen label, boolean, numeric score, ranked list)")
    confidence: Optional[float] = Field(None, description="Confidence score in [0.0, 1.0]")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Probability distribution over candidates if available")
    latency_ms: float = Field(0.0, description="End-to-end inference latency in milliseconds")
    error: Optional[str] = Field(None, description="Error message if inference failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional model/engine telemetry")

    @property
    def is_success(self) -> bool:
        return self.error is None


class BaseDecisionModel(ABC):
    """
    Abstract Base Class for any Decision Model / System One engine.
    Supports in-process MLX models, REST APIs, or LLM-based decision systems.
    """

    def __init__(self, name: str = "base_model"):
        self.name = name

    @abstractmethod
    def predict_choice(
        self,
        prompt: str,
        options: List[str],
        instruction: str = "Select the best matching category"
    ) -> DecisionOutput:
        """
        N-way discrete choice primitive.
        Returns the chosen option name in `value`, and optionally candidate probabilities.
        """
        pass

    @abstractmethod
    def predict_noul(
        self,
        prompt: str,
        instruction: str = "Evaluate whether the proposition is true or false"
    ) -> DecisionOutput:
        """
        Binary boolean verification primitive (noul).
        Returns boolean True/False in `value`, and positive probability in `confidence`.
        """
        pass

    @abstractmethod
    def predict_score(
        self,
        prompt: str,
        criteria: Any,
        instruction: str = "Rate the sample according to the criteria"
    ) -> DecisionOutput:
        """
        Ordinal or continuous rating primitive.
        Returns numeric float or integer rating in `value`.
        """
        pass

    @abstractmethod
    def rank(
        self,
        prompt: str,
        candidates: List[str],
        instruction: Optional[str] = None
    ) -> DecisionOutput:
        """
        Candidate ranking primitive.
        Returns list of candidate strings sorted by predicted relevance in `value`.
        """
        pass

    def warmup(self, n: int = 1) -> None:
        """Optional warmup routine to stabilize GPU/JIT compilation."""
        pass
