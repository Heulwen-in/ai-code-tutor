from __future__ import annotations

import logging
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.models.database import get_db
from app.models.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BugClassification,
    SkillPrediction,
)
from app.services import bug_classifier, feedback_generator, lesson_recommender, skill_detector
from app.services import learning_loop

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["analyze"])

DB = Annotated[Session, Depends(get_db)]

# Optional bearer — auto_error=False means unauthenticated requests still work
_optional_bearer = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)
OptionalToken = Annotated[Optional[str], Depends(_optional_bearer)]


def _user_id_from_token(token: Optional[str]) -> Optional[int]:
    """Extract user_id from JWT without raising — anonymous if invalid/absent."""
    if not token:
        return None
    try:
        from jose import JWTError, jwt
        from app.routers.auth import ALGORITHM, SECRET_KEY
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        sub = payload.get("sub")
        return int(sub) if sub else None
    except Exception:
        return None


@router.post("", response_model=AnalyzeResponse)
def analyze_code(
    payload: AnalyzeRequest,
    db: DB,
    token: OptionalToken = None,
) -> AnalyzeResponse:
    if payload.language.lower() != "python":
        raise HTTPException(status_code=400, detail="Only Python analysis is supported in this phase.")

    bug_result   = bug_classifier.classify(payload.code)
    skill_result = skill_detector.detect_skill(payload.code)
    feedback     = feedback_generator.generate_feedback(
        bug_type=bug_result.bug_type,
        role=payload.role,
        skill_level=skill_result.skill_level,
        confidence=bug_result.confidence,
        line_number=bug_result.line_number,
        bug_subtype=bug_result.bug_subtype,
        code=payload.code,
    )
    lessons = lesson_recommender.recommend(bug_result.bug_type, payload.role)

    response = AnalyzeResponse(
        bug=BugClassification(
            bug_type=bug_result.bug_type,
            confidence=bug_result.confidence,
            line_number=bug_result.line_number,
            description=bug_result.description,
            bug_subtype=bug_result.bug_subtype,
            subtype_confidence=bug_result.subtype_confidence,
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

    # Persist — logged-in users get user_id attached, anonymous users get None
    try:
        user_id = _user_id_from_token(token)
        learning_loop.save_analysis(
            db=db,
            user_id=user_id,
            code=payload.code,
            bug_result=bug_result,
            skill_result=skill_result,
            language=payload.language,
        )
    except Exception:
        logger.exception("Failed to save analysis session to DB")

    return response
