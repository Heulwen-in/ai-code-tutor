from __future__ import annotations

import ast
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path


USE_MOCK = os.getenv("USE_MOCK_BUG_CLASSIFIER", "false").lower() == "true"

# Minimum confidence required to report a bug class.
# If CodeBERT's top-class probability is below this, no_bug is returned instead.
NO_BUG_THRESHOLD = float(os.getenv("NO_BUG_THRESHOLD", "0.60"))

# Minimum Stage 2 confidence to trust a bug classification.
# When Stage 1 is a false positive on fixed/clean code, Stage 2 is typically
# uncertain (< 0.65). If Stage 2 can't confirm a specific subtype, the whole
# prediction is discarded and no_bug is returned instead.
SUBTYPE_CONFIDENCE_THRESHOLD = float(os.getenv("SUBTYPE_CONFIDENCE_THRESHOLD", "0.65"))

# Stage 2: fine-grained subtype detection within the coarse bug category.
# Disable with ENABLE_STAGE2=false; a missing model degrades gracefully too.
ENABLE_STAGE2 = os.getenv("ENABLE_STAGE2", "true").lower() == "true"

# Line detection: a separate token-classification model that localises the bug
# to a line. Optional; disable with ENABLE_LINE_DETECTION=false. A missing model
# degrades gracefully (line_number stays None for ML-detected bugs).
ENABLE_LINE_DETECTION = os.getenv("ENABLE_LINE_DETECTION", "true").lower() == "true"

# Minimum mean per-token buggy probability for a line to be reported. Below this,
# the model isn't confident enough to point at a line, so line_number stays None.
LINE_DETECTION_THRESHOLD = float(os.getenv("LINE_DETECTION_THRESHOLD", "0.5"))

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

# After retraining with 4-class data (codebert_retrain_4class.py),
# the model predicts index 3 = no_bug directly — no threshold gate needed.
# For the original 3-class model, the threshold gate in _real_classify() applies.
ML_BUG_LABELS = [
    "syntax_error",
    "logic_error",
    "variable_misuse",
    "no_bug",   # index 3 — only predicted by the 4-class retrained model
]

SYSTEM_BUG_LABELS = [
    "syntax_error",
    "logic_error",
    "variable_misuse",
    "indentation_error",
    "no_bug",
]


@dataclass
class ClassificationResult:
    bug_type: str
    confidence: float
    line_number: int | None = None
    description: str = ""
    bug_subtype: str | None = None
    subtype_confidence: float | None = None


_model = None
_tokenizer = None
_device = None

# Stage 2 (14 fine-grained subtypes) — optional, loaded alongside Stage 1
_stage2_model = None
_stage2_tokenizer = None
_stage2_mapping = None  # subtype_mapping.json contents

# Line detection (token classification) — optional, loaded alongside Stage 1
_line_model = None
_line_tokenizer = None


def _default_model_path() -> str:
    return str(Path(__file__).resolve().parents[1] / "ml_models" / "codebert_model")


def _default_stage2_path() -> str:
    return str(Path(__file__).resolve().parents[1] / "ml_models" / "codebert_stage2_model")


def _default_line_path() -> str:
    return str(Path(__file__).resolve().parents[1] / "ml_models" / "codebert_line_detection_model")


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

    if ENABLE_STAGE2:
        _load_stage2()
    if ENABLE_LINE_DETECTION:
        _load_line_detector()


def _load_stage2() -> None:
    """Load the optional Stage 2 subtype model; degrade to coarse-only on failure."""
    global _stage2_model, _stage2_tokenizer, _stage2_mapping
    if _stage2_model is not None:
        return

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    stage2_path = os.getenv("BUG_SUBTYPE_MODEL_PATH", _default_stage2_path())
    try:
        with open(os.path.join(stage2_path, "subtype_mapping.json"), encoding="utf-8") as f:
            _stage2_mapping = json.load(f)
        _stage2_tokenizer = AutoTokenizer.from_pretrained(stage2_path)
        _stage2_model = AutoModelForSequenceClassification.from_pretrained(stage2_path)
        _stage2_model.to(_device)
        _stage2_model.eval()
        print(f"[BugClassifier] Stage 2 subtype model loaded from {stage2_path}")
    except Exception as exc:
        _stage2_model = _stage2_tokenizer = _stage2_mapping = None
        print(f"[BugClassifier] Stage 2 unavailable ({exc}); coarse-only mode.")


def _load_line_detector() -> None:
    """Load the optional token-classification line detector; degrade to no
    line-number on failure."""
    global _line_model, _line_tokenizer
    if _line_model is not None:
        return

    from transformers import AutoModelForTokenClassification, AutoTokenizer

    line_path = os.getenv("BUG_LINE_MODEL_PATH", _default_line_path())
    try:
        _line_tokenizer = AutoTokenizer.from_pretrained(line_path)
        _line_model = AutoModelForTokenClassification.from_pretrained(line_path)
        _line_model.to(_device)
        _line_model.eval()
        print(f"[BugClassifier] Line detection model loaded from {line_path}")
    except Exception as exc:
        _line_model = _line_tokenizer = None
        print(f"[BugClassifier] Line detection unavailable ({exc}); no line numbers.")


def _char_to_line(code: str) -> list[int]:
    """Map each character index -> 1-indexed line number (mirrors training)."""
    mapping, line = [], 1
    for ch in code:
        mapping.append(line)
        if ch == "\n":
            line += 1
    mapping.append(line)
    return mapping


def _detect_bug_line(code: str) -> int | None:
    """
    Token-classification line localisation. Runs the line model, aggregates
    per-token buggy probabilities into a mean score per line, and returns the
    top-scoring line (1-indexed) if it clears LINE_DETECTION_THRESHOLD.
    Returns None when the model is unavailable or not confident enough.
    """
    import torch

    if _line_model is None or _line_tokenizer is None:
        return None

    enc = _line_tokenizer(
        code,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True,
        return_offsets_mapping=True,
    )
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(_device) for k, v in enc.items()}

    with torch.no_grad():
        logits = _line_model(**enc).logits[0]
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()

    c2l = _char_to_line(code)
    line_scores: dict[int, list[float]] = {}
    for prob, (start, end) in zip(probs, offsets):
        if start == end:  # special / padding token
            continue
        ln = c2l[start] if start < len(c2l) else c2l[-1]
        line_scores.setdefault(ln, []).append(prob)

    if not line_scores:
        return None
    top_line = max(line_scores, key=lambda k: sum(line_scores[k]) / len(line_scores[k]))
    top_score = sum(line_scores[top_line]) / len(line_scores[top_line])
    if top_score < LINE_DETECTION_THRESHOLD:
        return None
    return top_line


def _classify_subtype(code: str, coarse_class: str) -> tuple[str | None, float | None]:
    """
    Stage 2: predict the fine-grained bug subtype within the coarse category.
    Logits are masked to the subtypes belonging to `coarse_class` (hierarchical
    classification) before the softmax, so the prediction can never contradict
    Stage 1. Returns (human_name, confidence) or (None, None).
    """
    import torch

    if _stage2_model is None or _stage2_mapping is None:
        return None, None
    group_ids = _stage2_mapping["coarse_groups"].get(coarse_class)
    if not group_ids:
        return None, None

    inputs = _stage2_tokenizer(
        code,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True,
    )
    inputs = {key: value.to(_device) for key, value in inputs.items()}

    with torch.no_grad():
        logits = _stage2_model(**inputs).logits[0]
        mask = torch.full_like(logits, float("-inf"))
        mask[group_ids] = logits[group_ids]
        probs = torch.softmax(mask, dim=-1)
        idx = probs.argmax().item()
        conf = probs[idx].item()

    transformer_name = _stage2_mapping["int_to_subtype"][str(idx)]
    human_name = _stage2_mapping["human_names"].get(transformer_name, transformer_name)
    return human_name, round(conf, 4)


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
        top_confidence = probs[idx].item()

    if top_confidence < NO_BUG_THRESHOLD:
        return ClassificationResult(
            bug_type="no_bug",
            confidence=round(top_confidence, 4),
            description="No bug detected with sufficient confidence.",
        )

    bug_type = ML_BUG_LABELS[idx]
    subtype, subtype_conf = (None, None)
    if bug_type in ("syntax_error", "logic_error", "variable_misuse"):
        subtype, subtype_conf = _classify_subtype(code, bug_type)
        # Guard 1: low Stage 2 confidence → Stage 1 likely false-positive on clean code.
        if subtype_conf is not None and subtype_conf < SUBTYPE_CONFIDENCE_THRESHOLD:
            return ClassificationResult(
                bug_type="no_bug",
                confidence=round(top_confidence, 4),
                description="Low subtype confidence - no specific bug pattern confirmed.",
            )

    # Localise the bug to a line (token-classification model). Optional — stays
    # None if the line detector is unavailable or not confident enough.
    line_number = _detect_bug_line(code) if ENABLE_LINE_DETECTION else None

    return ClassificationResult(
        bug_type=bug_type,
        confidence=round(top_confidence, 4),
        line_number=line_number,
        bug_subtype=subtype,
        subtype_confidence=subtype_conf,
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
