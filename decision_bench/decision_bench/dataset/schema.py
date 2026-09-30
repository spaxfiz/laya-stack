"""Schema definitions for DecisionBench dataset items and tasks."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TaskType(str, Enum):
    CHOICE = "choice"          # Discrete N-way classification
    NOUL = "noul"              # Binary boolean verification
    SCORE = "score"            # Continuous / Ordinal scoring (1-5)
    RANK = "rank"              # Candidate preference ranking
    ADVERSARIAL = "adversarial"# Adversarial entity / semantic stress testing


class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    ADVERSARIAL = "adversarial"


class BenchmarkSample(BaseModel):
    """Unified benchmark sample schema representing a real-world decision task."""
    id: str = Field(..., description="Unique sample identifier")
    title: str = Field(..., description="Note title or query context")
    snippet: Optional[str] = Field(None, description="Optional body excerpt or detail context")
    ground_truth_track: str = Field(..., description="Target category track (1 of 12 verticals)")
    ground_truth_is_clickbait: bool = Field(False, description="Whether title is misleading/clickbait")
    ground_truth_is_ad: bool = Field(False, description="Whether note is a commercial sponsorship")
    ground_truth_viral_score: int = Field(2, ge=1, le=5, description="Viral/emotional intensity rating (1-5)")
    difficulty: DifficultyLevel = Field(DifficultyLevel.MEDIUM, description="Sample difficulty level")
    author: Optional[str] = Field(None, description="Author identifier if available")
    notes: Optional[str] = Field(None, description="Annotation rationale or edge case notes")

    @property
    def full_context(self) -> str:
        """Formatted contextual representation for Decision Models."""
        if self.snippet and self.snippet.strip():
            return f"小红书笔记标题: {self.title.strip()}\n正文片段: {self.snippet.strip()}"
        return f"小红书笔记标题: {self.title.strip()}"
