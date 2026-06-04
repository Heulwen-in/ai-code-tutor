# AI-Based Personalised Programming Tutor

> A web-based AI tutoring system that analyses Python code, detects bug types, estimates the learner's skill level, and returns role-adaptive feedback — all through a single REST endpoint.

**BSc Computer Science Final Year Project**  
**Author:** Nguyen Ngoc Gia Han
**Stack:** FastAPI · CodeBERT · Next.js · SQLAlchemy · Docker

---

## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [What Has Been Done](#what-has-been-done)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Running the Backend](#running-the-backend)
- [Running the ML Pipeline](#running-the-ml-pipeline)
- [API Reference](#api-reference)
- [Model Performance Summary](#model-performance-summary)
- [Configuration](#configuration)
- [Future Work](#future-work)

---

## Project Overview

The system accepts Python code submitted by a user (either a **student** or a **professional worker**), classifies the error type using a fine-tuned CodeBERT model, estimates the user's experience level from code style features, and returns:

- The detected **bug type** with confidence score and line number
- The inferred **skill level** (novice / professional)
- **Role-appropriate feedback** — beginner-friendly explanations for students, concise best-practice notes for workers
- **Recommended lessons** matched to the bug type and role

### Error Classes

| Label | Description | Detection method |
|---|---|---|
| `syntax_error` | Missing colon, bracket, invalid expression | Rule-based (`ast.parse`) + CodeBERT |
| `indentation_error` | Wrong block indentation, mixed tabs/spaces | Rule-based only (`ast.parse`) |
| `logic_error` | Off-by-one, wrong condition, infinite loop, early return | CodeBERT |
| `variable_misuse` | Wrong variable, name typo, uninitialised, wrong scope | CodeBERT |
| `no_bug` | No common issue detected | Confidence threshold gate *(planned)* |

---

## System Architecture

```
User (browser)
     │
     ▼
┌──────────────────────────────────────┐
│  Next.js Frontend                    │
│  Code Editor (Monaco) · Role select  │
│  Feedback panel · Lesson cards       │
└──────────────────┬───────────────────┘
                   │ POST /analyze
                   ▼
┌──────────────────────────────────────┐
│  FastAPI Backend                     │
│  ┌──────────────────────────────┐    │
│  │  bug_classifier.py           │    │
│  │  1. Rule-based parse check   │    │
│  │  2. CodeBERT inference       │    │
│  └──────────────┬───────────────┘    │
│  ┌──────────────▼───────────────┐    │
│  │  skill_detector.py           │    │
│  │  Behaviour features → LR     │    │
│  └──────────────┬───────────────┘    │
│  ┌──────────────▼───────────────┐    │
│  │  feedback_generator.py       │    │
│  │  Role-adaptive templates     │    │
│  └──────────────┬───────────────┘    │
│  ┌──────────────▼───────────────┐    │
│  │  lesson_recommender.py       │    │
│  │  Bug type + role → lessons   │    │
│  └──────────────────────────────┘    │
└──────────────────────────────────────┘
         │                │
    codebert_model/   skill_model.pkl
    (HuggingFace)     (scikit-learn LR)
```

---

## What Has Been Done

### Phase 1 — Foundation ✅

| Task | Status | Key output |
|---|---|---|
| Literature review (6 sections + similar systems comparison table) | ✅ Done | `docs/literature_review_v2.docx` |
| EDA — BuggedPythonLeetCode (14,113 samples, 15 bug types) | ✅ Done | `ml_pipeline/eda_output/` |
| EDA — LeetCodeDataset (2,869 correct Python solutions) | ✅ Done | `ml_pipeline/eda_output/` |
| EDA — PyBugHive (1,255 real open-source bugs, 36 repos) | ✅ Done | `ml_pipeline/eda_output/` |
| Dataset strategy confirmed (3 datasets, 3 roles) | ✅ Done | See Section 2 of progress report |

### Phase 2 — ML Pipeline ✅

| Task | Status | Key metric |
|---|---|---|
| Data preparation — 15 bug types → 4 classes + stratified split | ✅ Done | Train 9,877 / Val 2,117 / Test 2,117 |
| Feature extraction — 40 features (AST + token + behaviour) | ✅ Done | `ml_pipeline/data/processed/` |
| Feature histograms — all 40 features + top 12 + behaviour | ✅ Done | `ml_pipeline/data/processed/*.png` |
| Baseline classifiers v2 — 10 models, 2 pipelines, GridSearchCV tuning | ✅ Done | Best: SVM TF-IDF macro F1=0.9278 |
| Transformer fine-tuning — BERT + RoBERTa + CodeBERT | ✅ Done | Best: CodeBERT macro F1=0.9855 |
| Indentation error rule-based detector | ✅ Done | `bug_classifier.py` detect_parse_error() |
| PyBugHive generalisation evaluation | ✅ Done | Macro F1=0.2833 (domain shift documented) |
| Skill detector — novice vs professional (LR proxy classifier) | ✅ Done | Macro F1=0.8542 |
| Backend integration — FastAPI /analyze endpoint running | ✅ Done | All 4 services wired, smoke tests passing |

### Phase 3 — Frontend & Integration 🔄 In Progress

| Task | Status |
|---|---|
| Next.js analysis page (code editor, feedback panel, lesson cards) | 🔄 Planned |
| Connect frontend to /analyze endpoint | 🔄 Planned |
| Clean-code gate (binary buggy/clean detector before multi-class) | 🔄 Planned |
| Progress tracking / learning loop (resubmission history) | 🔄 Planned |
| Final thesis write-up | 🔄 Planned |
| Demo preparation (6-8 scripted test cases) | 🔄 Planned |

---

## Repository Structure

```
code-analyzer-ai/
│
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI entry point, CORS, model preload
│   │   ├── routers/
│   │   │   ├── analyze.py             # POST /analyze — core endpoint
│   │   │   ├── lessons.py             # GET /lessons/recommend
│   │   │   ├── users.py               # GET /user/progress, PATCH /user/score
│   │   │   └── auth.py                # POST /auth/login, /auth/register
│   │   ├── services/
│   │   │   ├── bug_classifier.py      # Rule-based detector + CodeBERT inference
│   │   │   ├── skill_detector.py      # Behaviour features → novice/professional
│   │   │   ├── feedback_generator.py  # Role-adaptive feedback templates
│   │   │   └── lesson_recommender.py  # Bug type + role → lesson catalogue
│   │   ├── models/
│   │   │   ├── schemas.py             # Pydantic: AnalyzeRequest, FeedbackResponse...
│   │   │   └── database.py            # SQLAlchemy ORM: User, CodeSession, BugDetection
│   │   └── ml_models/
│   │       ├── codebert_model/        # Fine-tuned CodeBERT (HuggingFace format)
│   │       ├── bert_model/            # Fine-tuned BERT (comparison only)
│   │       ├── roberta_model/         # Fine-tuned RoBERTa (comparison only)
│   │       ├── skill_model.pkl        # Logistic Regression skill detector
│   │       ├── best_baseline.pkl      # SVM TF-IDF fallback classifier
│   │       ├── tfidf_vectorizer.pkl   # TF-IDF vectoriser for baseline
│   │       └── scaler.pkl             # StandardScaler for structural features
│   ├── requirements.txt
│   ├── .env                           # ANTHROPIC_API_KEY, DATABASE_URL, SECRET_KEY
│   └── Dockerfile
│
├── frontend/
│   └── src/
│       ├── app/
│       │   ├── page.tsx               # Landing page
│       │   ├── layout.tsx
│       │   ├── analyze/page.tsx       # Main UI — editor + results (main UI)
│       │   └── dashboard/page.tsx     # Progress, history, lessons
│       ├── components/
│       │   ├── CodeEditor.tsx         # Monaco Editor wrapper
│       │   ├── BugReport.tsx          # Bug list with type, severity, line number
│       │   ├── FeedbackPanel.tsx      # LLM streamed feedback display
│       │   ├── SkillBadge.tsx         # Novice / Intermediate / Professional badge
│       │   ├── LessonCard.tsx         # Recommended lesson with difficulty + link
│       │   └── ProgressChart.tsx      # Recharts: error score over sessions
│       └── lib/
│           ├── api.ts                 # Axios client, all endpoint functions
│           └── useAnalysis.ts         # Custom hook: submit code, manage state
│
├── ml_pipeline/
│   ├── data_prep.py                   # 4-class mapping + stratified split
│   ├── feature_extraction.py          # 40 AST + token + behaviour features
│   ├── feature_histograms.py          # Histogram charts for all features
│   ├── baseline_classifiers_v2.py     # 10 models, 2 pipelines, GridSearchCV
│   ├── transformer_classifier.py      # BERT + RoBERTa + CodeBERT fine-tuning
│   ├── data/
│   │   ├── raw/                       # Original downloaded datasets
│   │   └── processed/                 # train/val/test.parquet + feature files
│   └── logs/                          # Per-model training logs + hyperparams
│
├── docker-compose.yml
├── .github/workflows/deploy.yml
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- NVIDIA GPU recommended for transformer inference (CPU fallback available)

### Clone and install

```bash
git clone https://github.com/<your-username>/code-analyzer-ai.git
cd code-analyzer-ai
```

### Backend setup

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file in `backend/`:

```env
ANTHROPIC_API_KEY=your_key_here
DATABASE_URL=sqlite:///./tutor.db
SECRET_KEY=your_secret_here

# Optional — defaults to backend/app/ml_models/codebert_model
BUG_CLASSIFIER_PATH=backend/app/ml_models/codebert_model

# Set to true for fast development without loading CodeBERT
USE_MOCK_BUG_CLASSIFIER=false
```

### Frontend setup

```bash
cd frontend
npm install
```

Create `.env.local` in `frontend/`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## Running the Backend

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Check it is running:

```bash
curl http://localhost:8000/health
```

**Quick test with mock mode** (no GPU needed):

```bash
USE_MOCK_BUG_CLASSIFIER=true uvicorn app.main:app --reload
```

Test the /analyze endpoint:

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def add(a, b):\n    return a - b",
    "role": "student",
    "language": "python"
  }'
```

### Running with Docker

```bash
docker-compose up --build
```

---

## Running the ML Pipeline

Run these scripts in order to reproduce the trained models from scratch.

> **Note:** All scripts save output to `ml_pipeline/data/processed/` and trained models to `backend/app/ml_models/`.

```bash
cd ml_pipeline

# Step 1 — Data preparation and 4-class mapping
python data_prep.py

# Step 2 — Feature extraction (AST + token + behaviour)
python feature_extraction.py

# Step 3 — Feature histogram charts
python feature_histograms.py

# Step 4 — Baseline classifiers (10 models, GridSearchCV tuning)
# Requires: pip install xgboost
python baseline_classifiers_v2.py

# Step 5 — Transformer fine-tuning (BERT + RoBERTa + CodeBERT)
# Requires: pip install transformers torch accelerate
# Recommended: NVIDIA GPU (RTX 3060 or above, ~30 min)
# CPU fallback available but takes ~3 hours
python transformer_classifier.py
```

Dataset paths are configured at the top of each script — update `BUGGED_LC_PATH` and similar variables to match your local dataset location.

---

## API Reference

### POST /analyze

Accepts Python code and returns bug classification, skill prediction, feedback, and lesson recommendations.

**Request body:**

```json
{
  "code": "def twoSum(nums, target):\n    for i in nums\n        pass",
  "role": "student",
  "language": "python"
}
```

**Response:**

```json
{
  "bug": {
    "bug_type": "syntax_error",
    "confidence": 1.0,
    "line_number": 2,
    "description": "Syntax error at line 2: expected ':'"
  },
  "skill": {
    "skill_level": "novice",
    "confidence": 0.82,
    "source": "model",
    "description": "Skill level inferred from code behaviour features."
  },
  "feedback": {
    "summary": "Syntax issue detected around line 2. Classifier confidence: 1.00.",
    "explanation": "Python could not read the code structure correctly. This usually means a missing colon, bracket, comma, quote, or invalid expression.",
    "next_steps": [
      "Read the error line and the line above it.",
      "Check missing colons, brackets, quotes, and commas.",
      "Run the code again after fixing the first syntax issue."
    ],
    "tone": "beginner"
  },
  "lessons": [
    {
      "lesson_id": "syntax_01",
      "title": "Python Syntax Basics",
      "description": "Learn colons, parentheses, and common syntax rules.",
      "difficulty": "beginner",
      "url": "/lessons/syntax_01"
    }
  ]
}
```

### GET /health

Returns backend status. Used by Docker health checks and frontend connection validation.

---

## Model Performance Summary

### Bug Classifier — All models compared

| Model | Test accuracy | Macro F1 | Notes |
|---|---:|---:|---|
| DT (structural) | 0.5281 | 0.5319 | Weakest — shallow tree |
| SVM (structural) | 0.5815 | 0.5812 | Best structural-only |
| XGB (structural) | 0.5923 | 0.5969 | Overfits — train 0.763 vs test 0.592 |
| RF / DT (TF-IDF) | ~0.80 | ~0.81 | Tree models weak on sparse TF-IDF |
| XGB (TF-IDF) | 0.8880 | 0.8996 | Good but 100× slower than SVM |
| LR (TF-IDF) | 0.8942 | 0.9023 | Strong linear baseline |
| SVM (TF-IDF) | 0.9183 | 0.9278 | **Best classical baseline** |
| BERT | 0.9740 | 0.9778 | +0.0500 vs SVM TF-IDF |
| RoBERTa | 0.9835 | 0.9841 | +0.0563 vs SVM TF-IDF |
| **CodeBERT** ★ | **0.9849** | **0.9855** | **+0.0577 vs SVM TF-IDF — selected** |

**PyBugHive generalisation (real-world, out-of-domain):** Macro F1=0.2833 — domain shift from synthetic LeetCode snippets to real open-source patch fragments. See limitations section.

### Skill Detector

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Dummy majority | 0.5015 | 0.3340 |
| Decision Tree | 0.8047 | 0.8025 |
| **Logistic Regression** ★ | **0.8542** | **0.8542** |

---

## Configuration

| Environment variable | Default | Description |
|---|---|---|
| `USE_MOCK_BUG_CLASSIFIER` | `false` | Use lightweight heuristic instead of CodeBERT |
| `BUG_CLASSIFIER_PATH` | `backend/app/ml_models/codebert_model` | Path to HuggingFace model directory |
| `SKILL_MODEL_DIR` | `backend/app/ml_models` | Directory containing `skill_model.pkl` |
| `ANTHROPIC_API_KEY` | — | Required for LLM-powered feedback (future) |
| `DATABASE_URL` | `sqlite:///./tutor.db` | SQLAlchemy-compatible database URL |
| `SECRET_KEY` | — | JWT signing secret for auth |

---

## Future Work

The following items are identified as the next priorities, in order:

**1. Frontend integration (highest priority)**
Build or complete the Next.js analysis page. Required components: Monaco code editor, role selector (student/worker), analyze button, bug type + confidence display, feedback panel, skill-level badge, recommended lesson cards, and session progress chart.

**2. Clean-code detection gate**
Add a binary buggy/clean classifier or confidence threshold before the multi-class CodeBERT model. Currently, syntactically valid code with no bug is forced into one of three bug classes. This is the most important production quality improvement.

**3. Domain adaptation for generalisation**
Fine-tune CodeBERT on a small set of PyBugHive examples (few-shot or continued pretraining) to bridge the domain gap between synthetic LeetCode bugs and real open-source code. The PyBugHive generalisation result (macro F1=0.2833) shows this gap is significant.

**4. Learning loop and progress tracking**
Implement the iterative resubmission flow: after receiving feedback, the user revises and resubmits code. The system re-evaluates and tracks which error types decrease over time. Store session history in the CodeSession and BugDetection SQLAlchemy models.

**5. LLM-powered feedback (Anthropic API)**
Replace the current template-based feedback with LLM-generated explanations using the Anthropic API (feedback_generator.py is already wired for this). The LLM should be given the bug type, code snippet, and role to generate personalised explanations.

**6. Lesson content**
Build the actual lesson pages referenced in the lesson catalogue (lesson_recommender.py). Currently the URLs point to /lessons/syntax_01 etc. which do not yet have content.

**7. User authentication and persistence**
Implement the auth.py and users.py routers (JWT login/register) and connect to the SQLAlchemy user and session models for persistent progress tracking across sessions.

**8. GraphCodeBERT comparison**
GraphCodeBERT incorporates data-flow information and may improve logic/variable misuse separation further. This would strengthen the thesis ML contribution with one additional model comparison point.

---

## Limitations

- **No `no_bug` class:** Syntactically valid, semantically correct code is currently forced into one of three bug categories. A clean-code gate is planned.
- **Domain shift on PyBugHive:** The CodeBERT model achieves macro F1=0.9855 on synthetic LeetCode-style bugs but only F1=0.2833 on real open-source patch data, reflecting the significant difference between training and real-world code format.
- **Skill detector proxy labels:** The novice/professional classifier uses LeetCode difficulty as a proxy. It calibrates feedback tone effectively but should not be used as a definitive ability judgement.
- **Python only:** The system is scoped to Python in this phase. Extending to other languages would require retraining on multi-language datasets.

---

## Acknowledgements

- [BuggedPythonLeetCode](https://huggingface.co/datasets/NeuroDragon/BuggedPythonLeetCode) — primary training dataset
- [LeetCodeDataset](https://huggingface.co/datasets/newfacade/LeetCodeDataset) — clean code source and skill detector proxy
- [PyBugHive](https://github.com/pybughive/pybughive) — real-world generalisation evaluation
- [microsoft/codebert-base](https://huggingface.co/microsoft/codebert-base) — pretrained CodeBERT model
- [roberta-base](https://huggingface.co/roberta-base) — pretrained RoBERTa model
- [bert-base-uncased](https://huggingface.co/bert-base-uncased) — pretrained BERT model
