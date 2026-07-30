from __future__ import annotations

import os
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker

# ── Engine ────────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./tutor_ai.db",   # relative path — saved next to the backend process
)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ── Base ──────────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


# ── Models ────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id         = Column(Integer, primary_key=True, index=True)
    email      = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    display_name = Column(String(120), nullable=True)   # optional editable name
    role       = Column(String(16), default="student")   # "student" | "worker"
    is_active  = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    history  = relationship("AnalysisHistory", back_populates="user", cascade="all, delete-orphan")
    progress = relationship("LessonProgress",  back_populates="user", cascade="all, delete-orphan")
    reviews  = relationship("ReviewSchedule",  back_populates="user", cascade="all, delete-orphan")


class AnalysisHistory(Base):
    __tablename__ = "analysis_history"

    id          = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at  = Column(DateTime, default=datetime.utcnow, index=True)

    # Submitted code (truncated for storage)
    code_snippet  = Column(Text, nullable=False)
    language      = Column(String(32), default="python")

    # Bug classifier output
    bug_type      = Column(String(32), nullable=False)
    bug_confidence = Column(Float, nullable=False)
    line_number   = Column(Integer, nullable=True)

    # Skill detector output
    skill_level   = Column(String(16), nullable=False)
    skill_source  = Column(String(32), default="model")

    user = relationship("User", back_populates="history")


class LessonProgress(Base):
    __tablename__ = "lesson_progress"

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    lesson_id  = Column(String(64), nullable=False, index=True)
    status     = Column(String(16), default="started")   # "started" | "completed"
    started_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="progress")


class ReviewSchedule(Base):
    """Spaced-repetition review item — one row per (user, bug type).

    Leitner-style: a detected bug resets the item to the first interval; each
    successful review (matching lesson completed, or clean code submitted while
    the item is due) advances it through growing intervals until mastered.
    """
    __tablename__ = "review_schedule"

    id            = Column(Integer, primary_key=True, index=True)
    user_id       = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    bug_type      = Column(String(32), nullable=False)
    interval_days = Column(Integer, default=1)           # 1 → 3 → 7 → 14
    next_due      = Column(DateTime, nullable=False)
    status        = Column(String(16), default="active")  # "active" | "mastered"
    updated_at    = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="reviews")


# ── Helpers ───────────────────────────────────────────────────────────────────

def create_tables() -> None:
    """Create all tables, then add any newly introduced columns in place."""
    Base.metadata.create_all(bind=engine)
    _ensure_columns()


def _ensure_columns() -> None:
    """Idempotently add columns that post-date the existing dev database.

    SQLite's create_all never ALTERs an existing table, so a column added to a
    model after the DB file was first created would otherwise be missing.
    """
    inspector = inspect(engine)
    existing = {col["name"] for col in inspector.get_columns("users")}
    if "display_name" not in existing:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE users ADD COLUMN display_name VARCHAR(120)"))


def get_db():
    """FastAPI dependency that yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
