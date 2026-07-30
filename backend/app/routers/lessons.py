from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.models.database import LessonProgress, User, get_db
from app.models.schemas import LessonItem
from app.services import lesson_recommender, learning_loop

router = APIRouter(prefix="/lessons", tags=["lessons"])

DB = Annotated[Session, Depends(get_db)]


# ── Inline lesson content (matches the catalogue in lesson_recommender.py) ────

LESSON_DETAIL: dict[str, dict] = {
    "syntax_01": {
        "lesson_id": "syntax_01",
        "title": "Python Syntax Basics",
        "description": "Learn colons, parentheses, and common syntax rules.",
        "difficulty": "beginner",
        "url": "/lessons/syntax_01",
        "content": (
            "Python is strict about syntax. Every `if`, `for`, `def`, and `class` "
            "statement must end with a colon. Opening parentheses, brackets, and braces "
            "must each be closed. Missing any of these causes a SyntaxError before your "
            "code runs."
        ),
        "example": "# Wrong\nif x > 0\n    print(x)\n\n# Correct\nif x > 0:\n    print(x)",
    },
    "syntax_02": {
        "lesson_id": "syntax_02",
        "title": "Reading Python Error Messages",
        "description": "Understand what SyntaxError messages are telling you.",
        "difficulty": "beginner",
        "url": "/lessons/syntax_02",
        "content": (
            "When Python raises SyntaxError it always shows a file name, a line number, "
            "and a caret (^) pointing at the problem. The error is often on the previous "
            "line — a missing colon or bracket on line 3 shows up as line 4."
        ),
        "example": "# SyntaxError: unexpected EOF while parsing\n"
                   "def greet(name:\n    print('Hello', name)",
    },
    "indent_01": {
        "lesson_id": "indent_01",
        "title": "Python Indentation Rules",
        "description": "Why indentation matters and how to get it right.",
        "difficulty": "beginner",
        "url": "/lessons/indent_01",
        "content": (
            "Python uses indentation to define blocks of code. Each nested block must be "
            "indented by exactly four spaces (PEP 8 recommendation). Mixing tabs and "
            "spaces causes IndentationError."
        ),
        "example": "def add(a, b):\n    result = a + b   # 4-space indent\n    return result",
    },
    "logic_01": {
        "lesson_id": "logic_01",
        "title": "Debugging with Print Statements",
        "description": "Step through your code manually to find logic mistakes.",
        "difficulty": "beginner",
        "url": "/lessons/logic_01",
        "content": (
            "Inserting print statements at key points lets you see what values your "
            "variables hold at each step. Compare the actual output with your expected "
            "output to isolate where the behaviour diverges."
        ),
        "example": "for i in range(5):\n    print(f'i = {i}, total = {total}')\n    total += i",
    },
    "logic_02": {
        "lesson_id": "logic_02",
        "title": "Understanding Loops & Conditions",
        "description": "Common mistakes: off-by-one, infinite loops, wrong conditions.",
        "difficulty": "beginner",
        "url": "/lessons/logic_02",
        "content": (
            "Off-by-one errors happen when loop boundaries are slightly wrong — using "
            "< instead of <=, or starting at 1 instead of 0. Always test your loop "
            "with a concrete small example and verify the expected number of iterations."
        ),
        "example": "# Off-by-one: misses last element\nfor i in range(len(arr) - 1):\n    ...\n\n"
                   "# Correct\nfor i in range(len(arr)):\n    ...",
    },
    "var_01": {
        "lesson_id": "var_01",
        "title": "Variables & Scope in Python",
        "description": "Understand local vs global scope and NameError causes.",
        "difficulty": "beginner",
        "url": "/lessons/var_01",
        "content": (
            "A variable defined inside a function is local — it cannot be read outside. "
            "If you need to share state, pass values as arguments and return them. "
            "Accidentally referencing a variable before it is assigned raises NameError."
        ),
        "example": "total = 0\n\ndef add_to_total(n):\n    global total   # required to modify the outer name\n    total += n",
    },
    "logic_pro_01": {
        "lesson_id": "logic_pro_01",
        "title": "Unit Testing with pytest",
        "description": "Write tests to catch logic errors before they reach production.",
        "difficulty": "advanced",
        "url": "/lessons/logic_pro_01",
        "content": (
            "pytest discovers test files automatically. A test is any function whose name "
            "starts with `test_`. Use assert statements directly — pytest rewrites them "
            "for clearer failure messages."
        ),
        "example": "# test_math.py\ndef test_add():\n    assert add(2, 3) == 5\n    assert add(-1, 1) == 0",
    },
    "var_pro_01": {
        "lesson_id": "var_pro_01",
        "title": "Type Hints & mypy",
        "description": "Use type annotations and mypy to catch variable misuse statically.",
        "difficulty": "advanced",
        "url": "/lessons/var_pro_01",
        "content": (
            "Type hints tell both human readers and static analysis tools what type each "
            "variable, argument, and return value should have. mypy checks these at "
            "analysis time so you find bugs without running the code."
        ),
        "example": "def greet(name: str) -> str:\n    return 'Hello, ' + name\n\n"
                   "greet(42)   # mypy: Argument 1 has incompatible type int; expected str",
    },
    "syntax_pro_01": {
        "lesson_id": "syntax_pro_01",
        "title": "Linting & Static Analysis",
        "description": "Use Ruff or Flake8 to catch syntax issues before runtime.",
        "difficulty": "intermediate",
        "url": "/lessons/syntax_pro_01",
        "content": (
            "Professional Python projects run a linter in CI so code with syntax or style "
            "errors never merges. Ruff is the fastest modern option — it catches syntax "
            "errors, unused imports, and PEP 8 violations in milliseconds."
        ),
        "example": "# Run locally before every commit\nruff check .\n\n"
                   "# Or auto-fix what it can\nruff check --fix .",
    },
    "indent_pro_01": {
        "lesson_id": "indent_pro_01",
        "title": "Editor Config & PEP 8 Compliance",
        "description": "Set up your editor to enforce 4-space indentation automatically.",
        "difficulty": "intermediate",
        "url": "/lessons/indent_pro_01",
        "content": (
            "Indentation issues in team projects usually come from editors mixing tabs "
            "and spaces. An .editorconfig file committed at the project root enforces "
            "consistent settings across every editor and contributor automatically."
        ),
        "example": "# .editorconfig\n[*.py]\nindent_style = space\nindent_size = 4",
    },
    "logic_pro_02": {
        "lesson_id": "logic_pro_02",
        "title": "Using the Python Debugger (pdb)",
        "description": "Step through execution with breakpoints to isolate logic bugs.",
        "difficulty": "intermediate",
        "url": "/lessons/logic_pro_02",
        "content": (
            "pdb lets you pause execution, inspect variables, step line-by-line, and "
            "evaluate expressions interactively. Use breakpoint() (Python 3.7+) to drop "
            "into the debugger at any line."
        ),
        "example": "def find_max(nums):\n    breakpoint()   # pause here\n    ...\n\n"
                   "# Commands: n=next, s=step into, p var=print, c=continue, q=quit",
    },
    "good_01": {
        "lesson_id": "good_01",
        "title": "Keep it Up — Next Challenges",
        "description": "Your code looks clean! Try a harder exercise.",
        "difficulty": "beginner",
        "url": "/lessons/good_01",
        "content": (
            "No major bug was detected — great work. The next step is harder problems "
            "that stress-test edge cases, efficiency, and clarity. Prefer Python "
            "built-ins (in, any, all, sum) over manual loops where possible."
        ),
        "example": "# Manual loop\nfor n in nums:\n    if n == target:\n        return True\n"
                   "return False\n\n# Pythonic\nreturn target in nums",
    },
    "good_pro_01": {
        "lesson_id": "good_pro_01",
        "title": "Code Review Best Practices",
        "description": "Great code passes the linter — now learn to write reviewable code.",
        "difficulty": "advanced",
        "url": "/lessons/good_pro_01",
        "content": (
            "Passing the linter is necessary but not sufficient. Good review checks "
            "correctness at the boundary, clarity for the next reader, and testability. "
            "Never interpolate user input into SQL strings — use parameterised queries."
        ),
        "example": "# Reviewable: typed, parameterised, explicit\n"
                   "def get_user(db: Session, user_id: int) -> User | None:\n"
                   "    return db.query(User).filter(User.id == user_id).first()",
    },
}


# ── Schemas ───────────────────────────────────────────────────────────────────

class LessonDetail(LessonItem):
    content: str = ""
    example: str = ""


class ProgressPayload(BaseModel):
    user_id: int | None = None   # if provided and no auth, use this (demo mode)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("", response_model=list[LessonDetail])
def list_lessons(
    role: str = Query(default="student", description="student or worker"),
    bug_type: str | None = Query(default=None, description="Filter by bug type"),
) -> list[dict]:
    """Return all lessons, optionally filtered by bug_type."""
    catalogue = lesson_recommender.LESSON_CATALOGUE
    all_items: list[LessonDetail] = []

    for bt, roles in catalogue.items():
        if bug_type and bt != bug_type:
            continue
        lessons = roles.get(role, roles.get("student", []))
        for lesson in lessons:
            detail = LESSON_DETAIL.get(lesson.lesson_id, {})
            all_items.append(
                LessonDetail(
                    lesson_id=lesson.lesson_id,
                    title=lesson.title,
                    description=lesson.description,
                    difficulty=lesson.difficulty,
                    url=lesson.url,
                    content=detail.get("content", ""),
                    example=detail.get("example", ""),
                )
            )

    return all_items


@router.get("/recommend/{bug_type}", response_model=list[LessonItem])
def recommend_lessons(
    bug_type: str,
    role: str = Query(default="student"),
) -> list[LessonItem]:
    """Convenience endpoint — same as lesson_recommender.recommend()."""
    return lesson_recommender.recommend(bug_type, role)


@router.get("/progress")
def lesson_progress(
    db: DB,
    user_id: int = Query(default=None, description="User ID (demo mode, no auth)"),
) -> dict[str, str]:
    """Return this user's per-lesson status as {lesson_id: 'started'|'completed'}."""
    if not user_id:
        return {}
    rows = db.query(LessonProgress).filter_by(user_id=user_id).all()
    return {row.lesson_id: row.status for row in rows}


@router.get("/{lesson_id}", response_model=LessonDetail)
def get_lesson(lesson_id: str) -> LessonDetail:
    detail = LESSON_DETAIL.get(lesson_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Lesson '{lesson_id}' not found.")
    return LessonDetail(**detail)


@router.post("/{lesson_id}/start")
def start_lesson(
    lesson_id: str,
    db: DB,
    user_id: int = Query(default=None, description="User ID (demo mode, no auth)"),
) -> dict:
    if lesson_id not in LESSON_DETAIL:
        raise HTTPException(status_code=404, detail=f"Lesson '{lesson_id}' not found.")
    if user_id:
        learning_loop.record_lesson(db, user_id, lesson_id, "started")
    return {"status": "started", "lesson_id": lesson_id}


@router.post("/{lesson_id}/complete")
def complete_lesson(
    lesson_id: str,
    db: DB,
    user_id: int = Query(default=None, description="User ID (demo mode, no auth)"),
) -> dict:
    if lesson_id not in LESSON_DETAIL:
        raise HTTPException(status_code=404, detail=f"Lesson '{lesson_id}' not found.")
    if user_id:
        learning_loop.record_lesson(db, user_id, lesson_id, "completed")
    return {"status": "completed", "lesson_id": lesson_id}
