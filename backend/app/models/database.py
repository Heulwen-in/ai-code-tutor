"""
database.py
-----------
SQLAlchemy ORM models and database engine setup.

Uses SQLite for development / demonstration (zero-config).
Switch to PostgreSQL for production by changing DATABASE_URL in .env.

Tables:
  users            — registered users (email, hashed password, role)
  analysis_history — each /analyze call result stored per user
  lesson_progress  — which lessons a user has started / completed
"""

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
    role       = Column(String(16), default="student")   # "student" | "worker"
    is_active  = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    history  = relationship("AnalysisHistory", back_populates="user", cascade="all, delete-orphan")
    progress = relationship("LessonProgress",  back_populates="user", cascade="all, delete-orphan")


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


# ── Helpers ───────────────────────────────────────────────────────────────────

def create_tables() -> None:
    """Create all tables. Call once at app startup."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency that yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
