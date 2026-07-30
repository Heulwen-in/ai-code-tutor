from __future__ import annotations

import json
import os

from app.models.schemas import FeedbackResponse

LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "20"))
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def build_prompt(
    code: str,
    bug_type: str,
    bug_subtype: str | None,
    line_number: int | None,
    confidence: float,
    role: str,
    skill_level: str,
) -> str:
    """Shared prompt for all evaluation runs — identical wording keeps the
    provider comparison in ml_pipeline/llm_feedback_comparison.py fair."""
    subtype_part = (
        f" The specific bug pattern detected is: {bug_subtype.replace('_', ' ')}."
        if bug_subtype else ""
    )
    line_part = f" The issue is likely around line {line_number}." if line_number else ""

    if role == "worker":
        style = (
            "The user is a professional developer. Be direct and technical: state "
            "the likely defect, why it breaks, and how to verify the fix."
        )
    else:
        style = (
            "The user is a student learning Python. Do NOT hand them the corrected "
            "code. Guide them with hints and questions so they find the fix "
            "themselves. Keep language simple"
            + (" and beginner-friendly." if skill_level == "novice" else ".")
        )

    return f"""You are an encouraging Python programming tutor inside an AI code-analysis tool.

A bug classifier analysed the user's code and predicted the bug category
"{bug_type}" with confidence {confidence:.2f}.{subtype_part}{line_part}

{style}

The user's code:
```python
{code}
```

Respond with ONLY a JSON object (no markdown fences, no extra text) in exactly this shape:
{{"summary": "<one-sentence summary of the issue>",
 "explanation": "<2-4 sentence explanation of what is wrong and why>",
 "next_steps": ["<step 1>", "<step 2>", "<step 3>"],
 "tone": "{'advanced' if role == 'worker' or skill_level == 'professional' else 'beginner'}"}}"""


def _parse_feedback_json(text: str, source: str) -> FeedbackResponse | None:
    """Validate the LLM's JSON into a FeedbackResponse; None on any mismatch."""
    text = text.strip()
    if text.startswith("```"):
        lines = [l for l in text.splitlines() if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        data = json.loads(text)
        return FeedbackResponse(
            summary=str(data["summary"]),
            explanation=str(data["explanation"]),
            next_steps=[str(s) for s in data["next_steps"]][:3],
            tone=str(data.get("tone", "beginner")),
            source=source,
        )
    except Exception:
        return None


class OllamaProvider:
    name = "ollama-qwen2.5-coder"

    def generate(self, prompt: str) -> FeedbackResponse | None:
        import requests

        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            },
            timeout=max(LLM_TIMEOUT_SECONDS, 60),
        )
        resp.raise_for_status()
        return _parse_feedback_json(resp.json()["response"], self.name)


def generate_llm_feedback(
    code: str,
    bug_type: str,
    bug_subtype: str | None,
    line_number: int | None,
    confidence: float,
    role: str,
    skill_level: str,
) -> FeedbackResponse | None:
    """
    Generate feedback via Ollama (Qwen2.5-Coder 7b).
    Returns None on ANY failure — the caller surfaces a visible error rather
    than silently falling back to template boilerplate.
    """
    try:
        prompt = build_prompt(
            code, bug_type, bug_subtype, line_number, confidence, role, skill_level)
        return OllamaProvider().generate(prompt)
    except Exception as exc:
        print(f"[LLMFeedback] Ollama failed: {exc}")
        return None
