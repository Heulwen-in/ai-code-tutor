from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.models.schemas import HealthResponse
from app.routers import analyze
from app.services import bug_classifier, skill_detector


@asynccontextmanager
async def lifespan(app: FastAPI):
    bug_classifier.load_model()
    skill_detector.load_model()
    yield


app = FastAPI(
    title="AI Programming Tutor API",
    description="Backend API for bug classification, skill detection, feedback, and lesson recommendation.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router)


@app.get("/", response_model=HealthResponse)
def root() -> HealthResponse:
    return HealthResponse(status="ok", service="AI Programming Tutor API")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="backend")
