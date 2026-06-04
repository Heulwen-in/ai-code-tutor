"""
lesson_recommender.py
---------------------
Maps detected bug_type + user role (student / worker) to relevant lessons.

Students get foundational lessons explaining the concept.
Workers get advanced lessons focused on best practices and efficiency.
"""

from app.models.schemas import LessonItem

# ── Lesson catalogue ────────────────────────────────────────────────────────
# Each lesson has a student version and a worker version.
# Extend this dictionary as you add lesson pages to the frontend.

LESSON_CATALOGUE: dict[str, dict] = {
    # Syntax errors
    "syntax_error": {
        "student": [
            LessonItem(
                lesson_id="syntax_01",
                title="Python Syntax Basics",
                description="Learn colons, parentheses, and common syntax rules.",
                difficulty="beginner",
                url="/lessons/syntax_01",
            ),
            LessonItem(
                lesson_id="syntax_02",
                title="Reading Python Error Messages",
                description="Understand what SyntaxError messages are telling you.",
                difficulty="beginner",
                url="/lessons/syntax_02",
            ),
        ],
        "worker": [
            LessonItem(
                lesson_id="syntax_pro_01",
                title="Linting & Static Analysis",
                description="Use Ruff or Flake8 to catch syntax issues before runtime.",
                difficulty="intermediate",
                url="/lessons/syntax_pro_01",
            ),
        ],
    },

    # Indentation errors
    "indentation_error": {
        "student": [
            LessonItem(
                lesson_id="indent_01",
                title="Python Indentation Rules",
                description="Why indentation matters and how to get it right.",
                difficulty="beginner",
                url="/lessons/indent_01",
            ),
        ],
        "worker": [
            LessonItem(
                lesson_id="indent_pro_01",
                title="Editor Config & PEP 8 Compliance",
                description="Set up your editor to enforce 4-space indentation automatically.",
                difficulty="intermediate",
                url="/lessons/indent_pro_01",
            ),
        ],
    },

    # Logic errors
    "logic_error": {
        "student": [
            LessonItem(
                lesson_id="logic_01",
                title="Debugging with Print Statements",
                description="Step through your code manually to find logic mistakes.",
                difficulty="beginner",
                url="/lessons/logic_01",
            ),
            LessonItem(
                lesson_id="logic_02",
                title="Understanding Loops & Conditions",
                description="Common loop mistakes: off-by-one, infinite loops, wrong conditions.",
                difficulty="beginner",
                url="/lessons/logic_02",
            ),
        ],
        "worker": [
            LessonItem(
                lesson_id="logic_pro_01",
                title="Unit Testing with pytest",
                description="Write tests to catch logic errors before they reach production.",
                difficulty="advanced",
                url="/lessons/logic_pro_01",
            ),
            LessonItem(
                lesson_id="logic_pro_02",
                title="Using the Python Debugger (pdb)",
                description="Step through execution with breakpoints to isolate logic bugs.",
                difficulty="intermediate",
                url="/lessons/logic_pro_02",
            ),
        ],
    },

    # Variable misuse
    "variable_misuse": {
        "student": [
            LessonItem(
                lesson_id="var_01",
                title="Variables & Scope in Python",
                description="Understand local vs global scope and NameError causes.",
                difficulty="beginner",
                url="/lessons/var_01",
            ),
        ],
        "worker": [
            LessonItem(
                lesson_id="var_pro_01",
                title="Type Hints & mypy",
                description="Use type annotations and mypy to catch variable misuse statically.",
                difficulty="advanced",
                url="/lessons/var_pro_01",
            ),
        ],
    },

    # No bug — reinforce good habits
    "no_bug": {
        "student": [
            LessonItem(
                lesson_id="good_01",
                title="Keep it Up — Next Challenges",
                description="Your code looks clean! Try a harder exercise.",
                difficulty="beginner",
                url="/lessons/good_01",
            ),
        ],
        "worker": [
            LessonItem(
                lesson_id="good_pro_01",
                title="Code Review Best Practices",
                description="Great code passes the linter — now learn to write reviewable code.",
                difficulty="advanced",
                url="/lessons/good_pro_01",
            ),
        ],
    },
}


def recommend(bug_type: str, role: str) -> list[LessonItem]:
    """
    Returns a list of LessonItem recommended for the given bug_type and role.

    Args:
        bug_type: one of the 5 classifier labels
        role: "student" or "worker"

    Returns:
        List of LessonItem (max 3)
    """
    bucket = LESSON_CATALOGUE.get(bug_type, {})
    lessons = bucket.get(role, bucket.get("student", []))
    return lessons[:3]