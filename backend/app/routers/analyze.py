from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BugClassification,
    SkillPrediction,
)
from app.services import bug_classifier, feedback_generator, lesson_recommender, skill_detector


router = APIRouter(prefix="/analyze", tags=["analyze"])


@router.post("", response_model=AnalyzeResponse)
def analyze_code(payload: AnalyzeRequest) -> AnalyzeResponse:
    if payload.language.lower() != "python":
        raise HTTPException(status_code=400, detail="Only Python analysis is supported in this phase.")

    bug_result = bug_classifier.classify(payload.code)
    skill_result = skill_detector.detect_skill(payload.code)
    feedback = feedback_generator.generate_feedback(
        bug_type=bug_result.bug_type,
        role=payload.role,
        skill_level=skill_result.skill_level,
        confidence=bug_result.confidence,
        line_number=bug_result.line_number,
    )
    lessons = lesson_recommender.recommend(bug_result.bug_type, payload.role)

    return AnalyzeResponse(
        bug=BugClassification(
            bug_type=bug_result.bug_type,
            confidence=bug_result.confidence,
            line_number=bug_result.line_number,
            description=bug_result.description,
        ),
        skill=SkillPrediction(
            skill_level=skill_result.skill_level,
            confidence=skill_result.confidence,
            source=skill_result.source,
            description=skill_result.description,
        ),
        feedback=feedback,
        lessons=lessons,
    )
