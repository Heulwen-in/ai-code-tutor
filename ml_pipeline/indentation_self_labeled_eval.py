"""
Self-labeled validation for parse-level indentation detection.

The transformer model is trained on syntax_error, logic_error, and variable_misuse.
Indentation is handled before ML inference by Python's parser in
backend/app/services/bug_classifier.py.

Run:
  python ml_pipeline/indentation_self_labeled_eval.py

Output:
  ml_pipeline/data/processed/indentation_self_labeled_eval.csv
  ml_pipeline/data/processed/indentation_self_labeled_eval_summary.txt
"""

from __future__ import annotations

import csv
import os
import sys
from collections import Counter


BASE_DIR = r"F:\Heulwen\IT GREENWICH\Final Project\Tutor_AI_code"
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
OUTPUT_DIR = os.path.join(BASE_DIR, "ml_pipeline", "data", "processed")
sys.path.insert(0, BACKEND_DIR)

from app.services.bug_classifier import detect_parse_error  # noqa: E402


SELF_LABELED_CASES = [
    {
        "id": "indent_missing_function_body",
        "expected": "indentation_error",
        "reason": "A function definition must be followed by an indented block.",
        "code": "def greet():\nprint('hello')",
    },
    {
        "id": "indent_missing_if_body",
        "expected": "indentation_error",
        "reason": "An if statement must be followed by an indented block.",
        "code": "if True:\nprint('yes')",
    },
    {
        "id": "indent_bad_dedent",
        "expected": "indentation_error",
        "reason": "The return line is dedented while still inside the function structure.",
        "code": "def total(values):\n    result = 0\n        result += 1\n    return result",
    },
    {
        "id": "indent_mixed_tabs_spaces",
        "expected": "indentation_error",
        "reason": "Mixed tab and spaces can trigger inconsistent indentation.",
        "code": "def f():\n\tprint('tab')\n    print('spaces')",
    },
    {
        "id": "valid_nested_blocks",
        "expected": "none",
        "reason": "The code has consistent indentation and should pass parse-level detection.",
        "code": "def f(items):\n    for item in items:\n        if item:\n            print(item)\n    return len(items)",
    },
    {
        "id": "syntax_missing_colon",
        "expected": "syntax_error",
        "reason": "This is syntax, not indentation: the if statement is missing a colon.",
        "code": "if True\n    print('yes')",
    },
    {
        "id": "syntax_unclosed_parenthesis",
        "expected": "syntax_error",
        "reason": "This is syntax, not indentation: an opening parenthesis is not closed.",
        "code": "print('hello'",
    },
    {
        "id": "valid_compact_loop",
        "expected": "none",
        "reason": "Valid one-line loop body should not be labeled as indentation_error.",
        "code": "for i in range(3): print(i)",
    },
    {
        "id": "unexpected_indent_top_level",
        "expected": "indentation_error",
        "reason": "A top-level line cannot start with an unexpected indent.",
        "code": "    print('unexpected')",
    },
    {
        "id": "valid_class_method",
        "expected": "none",
        "reason": "Class and method blocks are consistently indented.",
        "code": "class Counter:\n    def __init__(self):\n        self.value = 0\n    def add(self):\n        self.value += 1",
    },
]


def normalize(result):
    if result is None:
        return "none", None, ""
    return result.bug_type, result.line_number, result.description


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = []

    for case in SELF_LABELED_CASES:
        predicted, line_number, description = normalize(detect_parse_error(case["code"]))
        rows.append(
            {
                "id": case["id"],
                "expected": case["expected"],
                "predicted": predicted,
                "correct": predicted == case["expected"],
                "line_number": line_number or "",
                "reason": case["reason"],
                "detector_description": description,
            }
        )

    total = len(rows)
    correct = sum(1 for row in rows if row["correct"])
    by_expected = Counter(row["expected"] for row in rows)
    mistakes = [row for row in rows if not row["correct"]]

    csv_path = os.path.join(OUTPUT_DIR, "indentation_self_labeled_eval.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = os.path.join(OUTPUT_DIR, "indentation_self_labeled_eval_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("Indentation Self-Labeled Evaluation\n")
        file.write("=" * 50 + "\n\n")
        file.write(f"Total cases: {total}\n")
        file.write(f"Correct: {correct}\n")
        file.write(f"Accuracy: {correct / total:.4f}\n")
        file.write(f"Expected label counts: {dict(by_expected)}\n\n")
        file.write("Labeling mechanism:\n")
        file.write(
            "The detector calls ast.parse(code). Python raises IndentationError for malformed block "
            "indentation and SyntaxError for other parse failures. Valid snippets return no parse-level "
            "bug and are passed to the ML classifier.\n\n"
        )
        file.write("Incorrect classifications:\n")
        if not mistakes:
            file.write("None in this self-labeled set.\n")
        else:
            for row in mistakes:
                file.write(
                    f"- {row['id']}: expected={row['expected']} predicted={row['predicted']} "
                    f"line={row['line_number']} description={row['detector_description']}\n"
                )

    print(f"Saved: {csv_path}")
    print(f"Saved: {summary_path}")
    print(f"Accuracy: {correct}/{total} = {correct / total:.4f}")


if __name__ == "__main__":
    main()
