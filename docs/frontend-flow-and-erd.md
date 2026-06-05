# Frontend Flow And ERD Plan

## Product Goal

CodeTutor AI is a web-based personalised programming tutor for Python learners. The frontend should
first demonstrate the complete user journey with mock data, then connect each flow to the FastAPI
backend and database.

## Environment Setup

The project uses two environments:

1. Python virtual environment (`.venv`)
   - Used for the backend and ML pipeline.
   - Runs FastAPI, CodeBERT inference, skill detector, dataset processing, and training scripts.
   - Typical backend command:

```bash
cd backend
..\.venv\Scripts\activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

2. Node.js frontend environment
   - Used only for the Next.js website.
   - This is separate from Python `.venv`.
   - Typical frontend command:

```bash
cd frontend
npm install
npm run dev
```

## Frontend Flow

1. Landing page
   - Introduces the project, ML contribution, and main feature set.
   - Primary action sends the user to authentication or dashboard demo mode.

2. Authentication
   - Login/register screen for students and professional workers.
   - Later connects to `backend/app/routers/auth.py`.

3. Dashboard
   - Main project demo screen.
   - Shows AI tutor workspace first.
   - Summarises lessons, progress, badges, and next demo milestone.

4. AI Program Tutor
   - Left panel: Python code editor.
   - Controls: role selector and analyze button.
   - Right panel: bug report, skill signal, feedback, and lesson recommendations.
   - Later connects to `POST /analyze`.

5. Lesson Learning
   - W3Schools-style modules for syntax, indentation, logic, and variable misuse.
   - Later expands into lesson detail pages, examples, quizzes, and exercises.

6. Progress Chart
   - Shows weekly improvement and analysis statistics.
   - Later uses saved `CodeSession` and `BugDetection` records.

7. Badges And Achievements
   - Shows earned, active, and locked achievements.
   - Later calculates badge state from lesson completion and analysis history.

8. Profile
   - Stores user identity, default role, and learning summary.

9. Settings
   - Stores role preference, mock/API mode, and lesson reminder preferences.

## ERD

```mermaid
erDiagram
    USER ||--o{ CODE_SESSION : creates
    USER ||--o{ LESSON_PROGRESS : studies
    USER ||--o{ USER_ACHIEVEMENT : earns
    USER ||--|| USER_SETTING : configures

    CODE_SESSION ||--o{ BUG_DETECTION : contains
    CODE_SESSION ||--o{ FEEDBACK_RECORD : produces
    BUG_DETECTION }o--|| LESSON : recommends

    LESSON ||--o{ LESSON_PROGRESS : tracked_by
    ACHIEVEMENT ||--o{ USER_ACHIEVEMENT : awarded_as

    USER {
        int id PK
        string name
        string email UK
        string password_hash
        string role
        datetime created_at
    }

    USER_SETTING {
        int id PK
        int user_id FK
        string default_role
        boolean lesson_reminders
        boolean use_mock_api
        string theme
    }

    CODE_SESSION {
        int id PK
        int user_id FK
        text submitted_code
        string language
        string declared_role
        string skill_level
        float skill_confidence
        datetime created_at
    }

    BUG_DETECTION {
        int id PK
        int session_id FK
        string bug_type
        float confidence
        int line_number
        text description
    }

    FEEDBACK_RECORD {
        int id PK
        int session_id FK
        text summary
        text explanation
        text next_steps_json
        string tone
    }

    LESSON {
        int id PK
        string lesson_code UK
        string title
        string bug_type
        string difficulty
        text content
        text example_code
    }

    LESSON_PROGRESS {
        int id PK
        int user_id FK
        int lesson_id FK
        int progress_percent
        boolean completed
        datetime updated_at
    }

    ACHIEVEMENT {
        int id PK
        string achievement_code UK
        string title
        text description
        string rule_type
        int target_value
    }

    USER_ACHIEVEMENT {
        int id PK
        int user_id FK
        int achievement_id FK
        string status
        int current_value
        datetime earned_at
    }
```

## Step-By-Step Implementation Timeline

### Phase 1: Frontend Prototype

- Build static landing, auth, dashboard, AI tutor, lessons, progress, badges, profile, and settings pages.
- Use mock analysis responses that match the backend schema.
- Make the UI demo-ready before backend wiring.

### Phase 2: Backend Wiring

- Connect AI tutor form to `POST /analyze`.
- Add loading and error states for real API requests.
- Keep `NEXT_PUBLIC_USE_MOCK_API=true` available for fast demo fallback.

### Phase 3: Persistence

- Implement authentication and user session storage.
- Save every submitted code analysis as a `CodeSession`.
- Store bug detections, feedback, lesson recommendations, and progress.

### Phase 4: Learning Features

- Create real lesson detail pages.
- Add quizzes or exercises for each bug type.
- Update lesson progress when users finish lesson sections.

### Phase 5: Achievements

- Define achievement rules.
- Calculate badge progress from sessions and lesson completion.
- Show earned, active, and locked states in the dashboard.

### Phase 6: Final Report And Demo

- Add screenshots of every main page.
- Explain the ML pipeline, backend architecture, frontend flow, and ERD.
- Prepare scripted test cases for syntax, indentation, logic, variable misuse, and clean code.
