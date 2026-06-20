"""
bug_classifier.py
-----------------
Loads the trained PyTorch / HuggingFace model and classifies submitted Python
code into one of the ML-trained categories:

    syntax_error | logic_error | variable_misuse

Rule-based parse checks run first and may return:

    indentation_error | syntax_error

Set USE_MOCK_BUG_CLASSIFIER=true to use the lightweight development stub.
"""

from __future__ import annotations

import ast
import os
import random
from dataclasses import dataclass
from pathlib import Path


USE_MOCK = os.getenv("USE_MOCK_BUG_CLASSIFIER", "false").lower() == "true"

# Import the heavy native ML stack eagerly at module load — i.e. before the ASGI
# server starts its event loop. On Windows, importing native extensions such as
# pyarrow (pulled in lazily via transformers -> sklearn -> pandas) for the first
# time inside the async lifespan raises OSError [WinError 6714] during the
# directory scan. Importing up front makes the later lazy imports cache hits.
if not USE_MOCK:
    import torch  # noqa: F401
    from transformers import (  # noqa: F401
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

ML_BUG_LABELS = [
    "syntax_error",
    "logic_error",
    "variable_misuse",
]

SYSTEM_BUG_LABELS = [
    *ML_BUG_LABELS,
    "indentation_error",
    "no_bug",
]


@dataclass
class ClassificationResult:
    bug_type: str
    confidence: float
    line_number: int | None = None
    description: str = ""


_model = None
_tokenizer = None
_device = None


def _default_model_path() -> str:
    return str(Path(__file__).resolve().parents[1] / "ml_models" / "codebert_model")


def detect_parse_error(code: str) -> ClassificationResult | None:
    """
    Detect parse-level problems before the ML model runs.

    Indentation errors are not represented in the transformer training data, so
    the rule-based detector owns that class. Plain SyntaxError is also handled
    here because malformed user submissions should not be sent to the model.
    """
    try:
        ast.parse(code)
    except IndentationError as e:
        return ClassificationResult(
            bug_type="indentation_error",
            confidence=1.0,
            line_number=e.lineno,
            description=f"Indentation problem at line {e.lineno}: {e.msg}",
        )
    except SyntaxError as e:
        return ClassificationResult(
            bug_type="syntax_error",
            confidence=1.0,
            line_number=e.lineno,
            description=f"Syntax error at line {e.lineno}: {e.msg}",
        )
    return None


def detect_indentation_error(code: str) -> ClassificationResult | None:
    """Backward-compatible helper for earlier tests."""
    result = detect_parse_error(code)
    if result is not None and result.bug_type == "indentation_error":
        return result
    return None


def load_model() -> None:
    """Load the CodeBERT classifier once at app startup."""
    global _model, _tokenizer, _device
    if USE_MOCK:
        print("[BugClassifier] Running in MOCK mode; no model loaded.")
        return
    if _model is not None and _tokenizer is not None:
        return

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_path = os.getenv("BUG_CLASSIFIER_PATH", _default_model_path())
    _tokenizer = AutoTokenizer.from_pretrained(model_path)
    _model = AutoModelForSequenceClassification.from_pretrained(model_path)
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model.to(_device)
    _model.eval()
    print(f"[BugClassifier] Model loaded from {model_path} on {_device}")


def _real_classify(code: str) -> ClassificationResult:
    import torch

    if _model is None or _tokenizer is None:
        load_model()
    if _model is None or _tokenizer is None:
        return _mock_classify(code)

    inputs = _tokenizer(
        code,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True,
    )
    inputs = {key: value.to(_device) for key, value in inputs.items()}

    with torch.no_grad():
        logits = _model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        idx = probs.argmax().item()

    return ClassificationResult(
        bug_type=ML_BUG_LABELS[idx],
        confidence=round(probs[idx].item(), 4),
    )


def _mock_classify(code: str) -> ClassificationResult:
    """
    Lightweight heuristic classifier for fast development.
    Parse errors are handled before this function is called.
    """
    if "while True" in code and "break" not in code:
        return ClassificationResult(
            bug_type="logic_error",
            confidence=0.80,
            description="Possible infinite loop: 'while True' with no 'break' statement.",
        )

    return ClassificationResult(
        bug_type="no_bug",
        confidence=round(random.uniform(0.75, 0.95), 4),
        description="No common bugs detected.",
    )


def classify(code: str) -> ClassificationResult:
    """
    Main entry point. Returns parse errors first, then mock or real classifier.
    """
    parse_result = detect_parse_error(code)
    if parse_result is not None:
        return parse_result

    if USE_MOCK:
        return _mock_classify(code)
    return _real_classify(code)
