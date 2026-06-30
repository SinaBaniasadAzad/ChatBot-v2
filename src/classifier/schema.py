"""
Pydantic data models for the model output (the data "shape" only).

The logic that converts/repairs/validates the raw LLM output lives in
output_parser.py.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Candidate(BaseModel):
    label: str
    evidence: list[str] = Field(default_factory=list)


class LayerOutput(BaseModel):
    candidates: list[Candidate] = Field(default_factory=list)
    needs_clarification: bool = False

    @property
    def top(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None


class ClassificationOutput(BaseModel):
    layers: dict[str, LayerOutput] = Field(default_factory=dict)
    clarifying_question: str | None = None
    suggested_summary: str = ""
    reasoning: str = ""
