from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


UserRole = Literal["student", "worker"]
BugType = Literal[
    "syntax_error",
    "indentation_error",
    "logic_error",
    "variable_misuse",
    "no_bug",
]
SkillLevel = Literal["novice", "professional"]


class AnalyzeRequest(BaseModel):
    code: str = Field(..., min_length=1, description="Python code submitted by the user.")
    role: UserRole = Field(default="student", description="Declared user role for feedback style.")
    language: str = Field(default="python", description="Programming language of the submission.")


class BugClassification(BaseModel):
    bug_type: BugType
    confidence: float = Field(..., ge=0, le=1)
    line_number: int | None = None
    description: str = ""


class SkillPrediction(BaseModel):
    skill_level: SkillLevel
    confidence: float = Field(..., ge=0, le=1)
    source: str = "model"
    description: str = ""


class FeedbackItem(BaseModel):
    title: str
    message: str


class FeedbackResponse(BaseModel):
    summary: str
    explanation: str
    next_steps: list[str]
    tone: str


class LessonItem(BaseModel):
    lesson_id: str
    title: str
    description: str
    difficulty: str
    url: str


class AnalyzeResponse(BaseModel):
    bug: BugClassification
    skill: SkillPrediction
    feedback: FeedbackResponse
    lessons: list[LessonItem]


class HealthResponse(BaseModel):
    status: str
    service: str
