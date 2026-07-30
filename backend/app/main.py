from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the backend/ directory so BUG_CLASSIFIER_PATH and other
# env vars are available before any service module reads os.getenv().
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.models.database import create_tables
from app.models.schemas import HealthResponse
from app.routers import analyze, auth, lessons, users
from app.services import bug_classifier, skill_detector


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialise database tables (no-op if already exist)
    create_tables()
    # Load ML models once at startup
    bug_classifier.load_model()
    skill_detector.load_model()
    yield


app = FastAPI(
    title="AI Programming Tutor API",
    description=(
        "Backend API for bug classification, skill detection, feedback, "
        "and lesson recommendation."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
_origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
ALLOWED_ORIGINS = [o.strip() for o in _origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(analyze.router)
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(lessons.router)


# ── Health endpoints ──────────────────────────────────────────────────────────
@app.get("/", response_model=HealthResponse)
def root() -> HealthResponse:
    return HealthResponse(status="ok", service="AI Programming Tutor API")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="backend")
