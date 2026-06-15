# ── Build stage ───────────────────────────────────────────────────────────────
# Use a slim Python 3.11 image; CPU-only inference is sufficient for the demo.
FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies needed by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Dependencies ──────────────────────────────────────────────────────────────
COPY requirements.txt .
# CPU-only PyTorch wheel (~200 MB instead of ~2 GB GPU variant)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY app/ ./app/

# ── ML models ─────────────────────────────────────────────────────────────────
# Models are gitignored (too large for GitHub).
# Either:
#   a) Build the image on a machine that has already trained the models, OR
#   b) Mount the models directory at runtime:
#        docker run -v /host/path/ml_models:/app/app/ml_models ...
COPY app/ml_models/ ./app/ml_models/

# ── Runtime ───────────────────────────────────────────────────────────────────
ENV USE_MOCK_BUG_CLASSIFIER=false \
    BUG_CLASSIFIER_PATH=/app/app/ml_models/codebert_model \
    SKILL_MODEL_DIR=/app/app/ml_models \
    DATABASE_URL=sqlite:///./tutor_ai.db \
    ACCESS_TOKEN_EXPIRE_MINUTES=60

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
