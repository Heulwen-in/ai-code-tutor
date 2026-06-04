from __future__ import annotations

from app.models.schemas import FeedbackResponse


BUG_TITLES = {
    "syntax_error": "Syntax issue detected",
    "indentation_error": "Indentation issue detected",
    "logic_error": "Logic issue detected",
    "variable_misuse": "Variable misuse detected",
    "no_bug": "No obvious bug detected",
}


STUDENT_EXPLANATIONS = {
    "syntax_error": (
        "Python could not read the code structure correctly. This usually means a "
        "missing colon, bracket, comma, quote, or invalid expression."
    ),
    "indentation_error": (
        "Python uses indentation to understand code blocks. One or more lines are "
        "not aligned with the block they belong to."
    ),
    "logic_error": (
        "The code can run, but the steps may not match the intended algorithm. "
        "Check conditions, loop boundaries, return timing, and edge cases."
    ),
    "variable_misuse": (
        "A variable may be updated, referenced, or initialized incorrectly. Trace "
        "each variable from creation to final use."
    ),
    "no_bug": (
        "No common issue was detected by the current classifier. It is still worth "
        "testing with edge cases and reviewing the expected output."
    ),
}


WORKER_EXPLANATIONS = {
    "syntax_error": (
        "The parser reports invalid Python syntax. Run a formatter or linter first, "
        "then inspect the reported line and the line immediately above it."
    ),
    "indentation_error": (
        "The submission fails before semantic analysis because block indentation is "
        "invalid. Normalize tabs/spaces and verify nested control-flow blocks."
    ),
    "logic_error": (
        "The model predicts a semantic/control-flow defect. Focus review on branch "
        "conditions, loop invariants, off-by-one boundaries, and return paths."
    ),
    "variable_misuse": (
        "The model predicts incorrect variable use. Inspect initialization, mutation, "
        "scope, aliases, and accidental index/name substitutions."
    ),
    "no_bug": (
        "The current classifier did not flag a major defect. Continue with unit tests, "
        "static analysis, and edge-case review."
    ),
}


NEXT_STEPS = {
    "syntax_error": [
        "Read the error line and the line above it.",
        "Check missing colons, brackets, quotes, and commas.",
        "Run the code again after fixing the first syntax issue.",
    ],
    "indentation_error": [
        "Use four spaces consistently for nested blocks.",
        "Check every line after def, if, for, while, and else.",
        "Avoid mixing tabs and spaces.",
    ],
    "logic_error": [
        "Write down the expected output for one small example.",
        "Trace each loop iteration by hand or with print/debug statements.",
        "Add tests for boundary cases such as empty input and one-element input.",
    ],
    "variable_misuse": [
        "List each variable and its intended meaning.",
        "Check where each variable is first assigned and later updated.",
        "Look for similar names, wrong indexes, or stale values.",
    ],
    "no_bug": [
        "Run the provided sample tests.",
        "Add at least two edge-case tests.",
        "Review whether the algorithm matches the problem constraints.",
    ],
}


def generate_feedback(
    bug_type: str,
    role: str,
    skill_level: str,
    confidence: float,
    line_number: int | None = None,
) -> FeedbackResponse:
    """Generate deterministic tutor feedback from classifier outputs."""
    is_worker = role == "worker"
    explanations = WORKER_EXPLANATIONS if is_worker else STUDENT_EXPLANATIONS
    tone = "advanced" if is_worker or skill_level == "professional" else "beginner"

    line_hint = f" around line {line_number}" if line_number else ""
    summary = (
        f"{BUG_TITLES.get(bug_type, 'Issue detected')}{line_hint}. "
        f"Classifier confidence: {confidence:.2f}."
    )

    explanation = explanations.get(bug_type, STUDENT_EXPLANATIONS["no_bug"])
    if tone == "beginner":
        explanation += " Start with one small test case and make the code correct before optimizing it."
    else:
        explanation += " Prioritize a minimal reproduction, then validate with targeted tests."

    return FeedbackResponse(
        summary=summary,
        explanation=explanation,
        next_steps=NEXT_STEPS.get(bug_type, NEXT_STEPS["no_bug"]),
        tone=tone,
    )
