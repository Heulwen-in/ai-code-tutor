from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.models.database import AnalysisHistory, User, get_db
from app.routers.auth import _hash_password, _verify_password, get_current_user
from app.services import learning_loop

router = APIRouter(prefix="/users", tags=["users"])

CurrentUser = Annotated[User, Depends(get_current_user)]
DB          = Annotated[Session, Depends(get_db)]


# ── Schemas ───────────────────────────────────────────────────────────────────

class UserProfile(BaseModel):
    id: int
    email: str
    role: str
    display_name: str | None = None

    model_config = {"from_attributes": True}


class HistoryItem(BaseModel):
    id: int
    created_at: str
    bug_type: str
    bug_confidence: float
    skill_level: str
    language: str

    model_config = {"from_attributes": True}


class RolePatch(BaseModel):
    role: str


class ProfilePatch(BaseModel):
    display_name: str | None = None


class PasswordChange(BaseModel):
    current_password: str
    new_password: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/me", response_model=UserProfile)
def get_me(current_user: CurrentUser) -> User:
    return current_user


@router.patch("/me", response_model=UserProfile)
def update_profile(payload: ProfilePatch, current_user: CurrentUser, db: DB) -> User:
    """Update editable profile fields (currently the display name)."""
    if payload.display_name is not None:
        name = payload.display_name.strip()
        current_user.display_name = name or None
    db.commit()
    db.refresh(current_user)
    return current_user


@router.post("/me/password", status_code=204)
def change_password(payload: PasswordChange, current_user: CurrentUser, db: DB) -> None:
    """Change the account password after verifying the current one."""
    if not _verify_password(payload.current_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect.")
    if len(payload.new_password) < 6:
        raise HTTPException(status_code=422, detail="New password must be at least 6 characters.")
    current_user.hashed_password = _hash_password(payload.new_password)
    db.commit()


@router.patch("/me/role", response_model=UserProfile)
def update_role(
    payload: RolePatch,
    current_user: CurrentUser,
    db: DB,
) -> User:
    if payload.role not in ("student", "worker"):
        raise HTTPException(status_code=422, detail="role must be 'student' or 'worker'.")
    current_user.role = payload.role
    db.commit()
    db.refresh(current_user)
    return current_user


@router.get("/me/progress")
def get_progress(current_user: CurrentUser, db: DB) -> dict:
    return learning_loop.get_user_progress(db, current_user.id)


@router.get("/me/stats")
def get_stats(current_user: CurrentUser, db: DB) -> dict:
    return learning_loop.get_stats(db, current_user.id)


@router.get("/me/achievements")
def get_achievements(current_user: CurrentUser, db: DB) -> list[dict]:
    return learning_loop.compute_achievements(db, current_user.id)


@router.get("/me/reviews")
def get_reviews(current_user: CurrentUser, db: DB) -> list[dict]:
    """Spaced-repetition review queue (due items first)."""
    return learning_loop.get_review_queue(db, current_user.id, current_user.role)


@router.get("/me/history", response_model=list[HistoryItem])
def get_history(
    current_user: CurrentUser,
    db: DB,
    limit: int = 20,
) -> list[AnalysisHistory]:
    rows = (
        db.query(AnalysisHistory)
        .filter_by(user_id=current_user.id)
        .order_by(AnalysisHistory.created_at.desc())
        .limit(max(1, min(limit, 100)))
        .all()
    )
    # Convert datetime to ISO string for JSON serialisation
    for row in rows:
        row.created_at = row.created_at.isoformat()
    return rows
