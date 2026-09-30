"""Models module for DecisionBench."""

from typing import Any, Optional
from .base import BaseDecisionModel, DecisionOutput
from .mock_model import MockDecisionModel
from .system_one_http import SystemOneHttpModel
from .laya_mlx_model import LayaMlxModel
from .openai_model import OpenAICompatibleModel


def create_model(target: str, **kwargs: Any) -> BaseDecisionModel:
    """
    Factory function to initialize a decision model adapter.

    Examples:
        create_model("http://127.0.0.1:8700") -> SystemOneHttpModel
        create_model("mock", accuracy_bias=0.9) -> MockDecisionModel
        create_model("laya") -> LayaMlxModel
        create_model("openai", base_url="http://localhost:11434/v1", model="llama3") -> OpenAICompatibleModel
    """
    target_lower = target.lower().strip()

    if target_lower in ("mock", "dummy", "fake"):
        return MockDecisionModel(**kwargs)

    if target_lower.startswith("http://") or target_lower.startswith("https://"):
        # Default to SystemOneHttpModel for HTTP targets unless openai requested
        if kwargs.get("provider") == "openai" or "openai" in target_lower:
            return OpenAICompatibleModel(base_url=target, **kwargs)
        return SystemOneHttpModel(base_url=target, **kwargs)

    if target_lower in ("laya", "mlx", "local"):
        return LayaMlxModel(**kwargs)

    if target_lower.startswith("openai"):
        return OpenAICompatibleModel(**kwargs)

    # If it's a file path or huggingface path
    return LayaMlxModel(model_path=target, **kwargs)


__all__ = [
    "BaseDecisionModel",
    "DecisionOutput",
    "MockDecisionModel",
    "SystemOneHttpModel",
    "LayaMlxModel",
    "OpenAICompatibleModel",
    "create_model"
]
