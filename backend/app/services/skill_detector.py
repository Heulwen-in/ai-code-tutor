"""
skill_detector.py
-----------------
Infers a broad skill level from Python code behaviour features.

The trained model is a lightweight proxy classifier:
    Easy LeetCode solutions -> novice
    Hard LeetCode solutions -> professional

Use this as a feedback-calibration signal, not as a final judgement of a user.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import joblib


SKILL_LABELS = ["novice", "professional"]
DEFAULT_FEATURE_NAMES = [
    "max_indent_depth",
    "mean_indent_depth",
    "single_letter_vars",
    "comment_ratio",
    "has_type_hints",
    "has_list_comp",
    "has_generator",
    "has_recursion",
    "magic_numbers",
    "builtins_used",
    "descriptive_ratio",
    "code_len",
    "line_count",
    "avg_line_len",
]


@dataclass
class SkillResult:
    skill_level: str
    confidence: float
    source: str = "model"
    description: str = ""


_model = None
_feature_names: list[str] = DEFAULT_FEATURE_NAMES.copy()


def _default_model_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "ml_models"


def load_model() -> None:
    """Load the skill detector model once at startup."""
    global _model, _feature_names

    model_dir = Path(os.getenv("SKILL_MODEL_DIR", str(_default_model_dir())))
    model_path = model_dir / "skill_model.pkl"
    feature_path = model_dir / "skill_feature_names.json"

    if feature_path.exists():
        _feature_names = json.loads(feature_path.read_text(encoding="utf-8"))

    if not model_path.exists() or model_path.stat().st_size == 0:
        print("[SkillDetector] No trained model found; using heuristic fallback.")
        _model = None
        return

    _model = joblib.load(model_path)
    print(f"[SkillDetector] Model loaded from {model_path}")


def extract_behaviour_features(code: str) -> dict[str, float]:
    """Extract the same code behaviour features used during training."""
    code = str(code or "")
    lines = code.splitlines() or [""]
    non_empty = [line for line in lines if line.strip()]

    indent_depths = []
    for line in non_empty:
        stripped = line.lstrip()
        if stripped:
            indent_depths.append(len(line) - len(stripped))
    max_indent = max(indent_depths) if indent_depths else 0
    mean_indent = round(sum(indent_depths) / len(indent_depths), 3) if indent_depths else 0.0

    single_letter = len(re.findall(r"\b[a-zA-Z]\b", code))
    comment_lines = sum(1 for line in lines if line.strip().startswith("#"))
    comment_ratio = round(comment_lines / max(len(lines), 1), 3)

    has_type_hints = int(bool(re.search(r"\)\s*->\s*[\w\[\], .|]+", code)))
    has_list_comp = int(bool(re.search(r"\[[^\]]+\bfor\b.+\bin\b", code, flags=re.S)))
    has_generator = int(bool(re.search(r"\([^)]+\bfor\b.+\bin\b", code, flags=re.S)))

    function_names = re.findall(r"\bdef\s+([A-Za-z_]\w*)\s*\(", code)
    has_recursion = int(
        any(re.search(rf"\b{name}\s*\(", code.split(f"def {name}", 1)[-1]) for name in function_names)
    )

    magic_numbers = len(re.findall(r"(?<!\w)\d{2,}(?!\w)", code))
    builtins_used = len(
        re.findall(
            r"\b(map|filter|zip|enumerate|sorted|reversed|any|all|sum|max|min|len|range|set|dict|list)\s*\(",
            code,
        )
    )

    keywords = {
        "True", "False", "None", "and", "or", "not", "in", "is", "if", "else",
        "elif", "for", "while", "with", "return", "def", "class", "import",
        "from", "pass", "break", "continue", "self",
    }
    names = re.findall(r"\b([A-Za-z_]\w*)\b", code)
    descriptive = sum(1 for name in names if len(name) > 4 and name not in keywords)
    descriptive_ratio = round(descriptive / max(len(names), 1), 3)

    code_len = len(code)
    line_count = len(lines)
    avg_line_len = round(sum(len(line) for line in lines) / max(line_count, 1), 3)

    return {
        "max_indent_depth": max_indent,
        "mean_indent_depth": mean_indent,
        "single_letter_vars": single_letter,
        "comment_ratio": comment_ratio,
        "has_type_hints": has_type_hints,
        "has_list_comp": has_list_comp,
        "has_generator": has_generator,
        "has_recursion": has_recursion,
        "magic_numbers": magic_numbers,
        "builtins_used": builtins_used,
        "descriptive_ratio": descriptive_ratio,
        "code_len": code_len,
        "line_count": line_count,
        "avg_line_len": avg_line_len,
    }


def _feature_vector(code: str) -> list[list[float]]:
    features = extract_behaviour_features(code)
    return [[float(features.get(name, 0.0)) for name in _feature_names]]


def _heuristic_detect(code: str) -> SkillResult:
    features = extract_behaviour_features(code)
    score = 0
    score += 1 if features["has_type_hints"] else 0
    score += 1 if features["has_list_comp"] else 0
    score += 1 if features["has_generator"] else 0
    score += 1 if features["builtins_used"] >= 4 else 0
    score += 1 if features["line_count"] >= 20 else 0
    score -= 1 if features["single_letter_vars"] >= 12 else 0

    if score >= 2:
        return SkillResult(
            skill_level="professional",
            confidence=0.65,
            source="heuristic",
            description="Code uses several advanced style or complexity signals.",
        )
    return SkillResult(
        skill_level="novice",
        confidence=0.65,
        source="heuristic",
        description="Code is short or uses fewer advanced style signals.",
    )


def detect_skill(code: str) -> SkillResult:
    """
    Return broad skill-level prediction for submitted code.

    The model returns "novice" or "professional" with a confidence score.
    """
    if _model is None:
        load_model()

    if _model is None:
        return _heuristic_detect(code)

    vector = _feature_vector(code)
    pred = int(_model.predict(vector)[0])
    confidence = 0.75

    if hasattr(_model, "predict_proba"):
        probs = _model.predict_proba(vector)[0]
        confidence = float(max(probs))

    return SkillResult(
        skill_level=SKILL_LABELS[pred],
        confidence=round(confidence, 4),
        source="model",
        description="Skill level inferred from code behaviour features.",
    )


def classify(code: str) -> SkillResult:
    """Alias for routers that use classify-style service names."""
    return detect_skill(code)
