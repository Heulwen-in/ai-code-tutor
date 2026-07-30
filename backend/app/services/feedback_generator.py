from __future__ import annotations

from app.models.schemas import FeedbackResponse


def generate_feedback(
    bug_type: str,
    role: str,
    skill_level: str,
    confidence: float,
    line_number: int | None = None,
    bug_subtype: str | None = None,
    code: str = "",
) -> FeedbackResponse:
    """
    Generate tutor feedback via Ollama (Qwen2.5-Coder 7b).
    If Ollama is unreachable or returns malformed JSON, a visible error
    response is returned so the UI surfaces the problem clearly instead of
    silently showing boilerplate text.
    """
    if bug_type == "no_bug":
        return FeedbackResponse(
            summary="No significant bug pattern detected in this code.",
            explanation=(
                "The classifier did not find a common bug pattern. "
                "This does not guarantee the code is fully correct — "
                "test it with several inputs, including edge cases."
            ),
            next_steps=[
                "Run the code with a few representative inputs to confirm it works.",
                "Add edge-case tests (empty input, single element, large values).",
                "Review the algorithm logic against the problem constraints.",
            ],
            tone="beginner",
            source="system",
        )

    if code:
        from app.services.llm_feedback import generate_llm_feedback

        llm_response = generate_llm_feedback(
            code=code,
            bug_type=bug_type,
            bug_subtype=bug_subtype,
            line_number=line_number,
            confidence=confidence,
            role=role,
            skill_level=skill_level,
        )
        if llm_response is not None:
            return llm_response

    return FeedbackResponse(
        summary="AI feedback unavailable — Ollama is not running.",
        explanation=(
            "The local Qwen2.5-Coder model could not be reached. "
            "Start Ollama with `ollama serve` and ensure the model is loaded."
        ),
        next_steps=[
            "Run: ollama serve",
            "Run: ollama pull qwen2.5-coder:7b",
            "Restart the backend server, then analyse again.",
        ],
        tone="beginner",
        source="error",
    )
