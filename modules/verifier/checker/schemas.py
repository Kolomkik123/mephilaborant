from typing import Literal
from pydantic import BaseModel, Field


class ClaimAssessment(BaseModel):
    statement: str
    verdict: Literal["confirmed", "unsupported", "contradicted"]
    source_ids: list[str] = Field(default_factory=list)
    comment: str = ""


class SourceJudgeResult(BaseModel):
    claims: list[ClaimAssessment] = Field(default_factory=list)
    overall_status: Literal["pass", "revise", "needs_review"]
    issues: list[str] = Field(default_factory=list)


class ExaminerJudgeResult(BaseModel):
    question_valid: bool
    expected_answer_valid: bool
    scientifically_correct: bool
    within_experiment_scope: bool
    ambiguity_detected: bool = False
    issues: list[str] = Field(default_factory=list)
    overall_status: Literal["pass", "revise", "needs_review"]


class NumericIssue(BaseModel):
    quantity_key: str
    expected_value: float
    expected_unit: str | None = None
    observed_values: list[float] = Field(default_factory=list)
    message: str


class NumericValidationResult(BaseModel):
    status: Literal["pass", "revise", "needs_review"]
    checked_quantities: list[str] = Field(default_factory=list)
    issues: list[NumericIssue] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SecurityResult(BaseModel):
    suspicious: bool
    matched_patterns: list[str] = Field(default_factory=list)
    action: Literal["allow", "isolate", "refuse"]
    explanation: str = ""


class CheckerDecision(BaseModel):
    verdict: Literal["pass", "revise", "refuse", "needs_review"]
    feedback: list[str] = Field(default_factory=list)
    refusal_reason: str | None = None
