"""
Phase 2 - Skill Detector Training

Trains a lightweight novice/professional classifier from code behaviour
features.

Dataset limitation:
The project does not currently have human-labelled novice/professional code.
This script therefore uses LeetCode difficulty as a transparent proxy:

  Easy -> novice
  Hard -> professional
  Medium -> excluded from training/evaluation

This detector should be described as a calibration aid for feedback style, not
as a definitive judgement of the user's ability.

Outputs:
  backend/app/ml_models/skill_model.pkl
  backend/app/ml_models/skill_feature_names.json
  ml_pipeline/data/processed/skill_detector_results.txt
  ml_pipeline/data/processed/skill_detector_confusion_matrix.png
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(r"F:\Heulwen\IT GREENWICH\Final Project")
LEETCODE_TRAIN = ROOT / "Datasets" / "LeetCodeDataset" / "LeetCodeDataset-train.jsonl"
LEETCODE_TEST = ROOT / "Datasets" / "LeetCodeDataset" / "LeetCodeDataset-test.jsonl"
MODEL_DIR = ROOT / "Tutor_AI_code" / "backend" / "app" / "ml_models"
OUTPUT_DIR = ROOT / "Tutor_AI_code" / "ml_pipeline" / "data" / "processed"

LABELS = ["novice", "professional"]
LABEL_TO_ID = {"novice": 0, "professional": 1}

FEATURE_NAMES = [
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


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def extract_answer_code(text: str) -> str:
    """Extract the first fenced Python code block from a model answer."""
    if not isinstance(text, str):
        return ""
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.I | re.S)
    if match:
        return match.group(1).strip()
    return text.strip()


def extract_behaviour_features(code: str) -> dict[str, float]:
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
    has_recursion = int(any(re.search(rf"\b{name}\s*\(", code.split(f"def {name}", 1)[-1]) for name in function_names))

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


def build_dataset() -> pd.DataFrame:
    train_df = load_jsonl(LEETCODE_TRAIN)
    test_df = load_jsonl(LEETCODE_TEST)
    df = pd.concat([train_df, test_df], ignore_index=True)
    df = df[df["difficulty"].isin(["Easy", "Hard"])].copy()

    df["skill_label"] = df["difficulty"].map({"Easy": "novice", "Hard": "professional"})
    df["skill_id"] = df["skill_label"].map(LABEL_TO_ID)
    df["code"] = df["completion"].apply(extract_answer_code)

    feature_rows = [extract_behaviour_features(code) for code in df["code"]]
    feat_df = pd.DataFrame(feature_rows)
    out = pd.concat([df[["task_id", "difficulty", "skill_label", "skill_id", "code"]].reset_index(drop=True), feat_df], axis=1)
    return out


def train_and_evaluate(df: pd.DataFrame) -> tuple[str, object, dict, pd.DataFrame]:
    x = df[FEATURE_NAMES].astype(float).to_numpy()
    y = df["skill_id"].astype(int).to_numpy()

    x_train, x_test, y_train, y_test, meta_train, meta_test = train_test_split(
        x,
        y,
        df[["task_id", "difficulty", "skill_label", "code"]],
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    candidates = {
        "Dummy majority": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, class_weight="balanced", random_state=42),
    }

    results = {}
    fitted = {}
    for name, model in candidates.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        results[name] = {
            "accuracy": accuracy_score(y_test, pred),
            "macro_f1": f1_score(y_test, pred, average="macro"),
            "y_pred": pred,
            "y_true": y_test,
        }
        fitted[name] = model

    best_name = max(results, key=lambda name: results[name]["macro_f1"])
    best_model = fitted[best_name]

    pred_df = meta_test.reset_index(drop=True).copy()
    pred_df["true_id"] = y_test
    pred_df["pred_id"] = results[best_name]["y_pred"]
    pred_df["pred_skill"] = [LABELS[i] for i in pred_df["pred_id"]]
    pred_df["correct"] = pred_df["true_id"] == pred_df["pred_id"]

    return best_name, best_model, results, pred_df


def feature_importance(model) -> list[tuple[str, float]]:
    if isinstance(model, Pipeline):
        clf = model.named_steps["clf"]
        values = getattr(clf, "coef_", [[0] * len(FEATURE_NAMES)])[0]
        return sorted(zip(FEATURE_NAMES, map(float, values)), key=lambda x: abs(x[1]), reverse=True)
    values = getattr(model, "feature_importances_", [0] * len(FEATURE_NAMES))
    return sorted(zip(FEATURE_NAMES, map(float, values)), key=lambda x: x[1], reverse=True)


def save_outputs(best_name: str, best_model, results: dict, pred_df: pd.DataFrame, df: pd.DataFrame) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "skill_model.pkl"
    feature_path = MODEL_DIR / "skill_feature_names.json"
    result_path = OUTPUT_DIR / "skill_detector_results.txt"
    pred_path = OUTPUT_DIR / "skill_detector_predictions.csv"
    cm_path = OUTPUT_DIR / "skill_detector_confusion_matrix.png"

    joblib.dump(best_model, model_path)
    feature_path.write_text(json.dumps(FEATURE_NAMES, indent=2), encoding="utf-8")
    pred_df.drop(columns=["code"]).to_csv(pred_path, index=False)

    best = results[best_name]
    report = classification_report(best["y_true"], best["y_pred"], target_names=LABELS, digits=4)
    cm = confusion_matrix(best["y_true"], best["y_pred"], labels=[0, 1])

    lines = [
        "Skill Detector Training Results",
        "=" * 60,
        "",
        "Task: novice vs professional code-style detection",
        "Proxy labels: Easy -> novice, Hard -> professional; Medium excluded",
        "",
        f"Rows used: {len(df)}",
        "Class distribution:",
        *[f"  {idx}: {count}" for idx, count in df["skill_label"].value_counts().items()],
        "",
        "Candidate models:",
    ]
    for name, metrics in results.items():
        lines.append(f"  {name:20s} accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")
    lines.extend(
        [
            "",
            f"Selected model: {best_name}",
            f"Accuracy      : {best['accuracy']:.4f}",
            f"Macro F1      : {best['macro_f1']:.4f}",
            "",
            "Classification Report:",
            report,
            "Confusion Matrix:",
            "  Labels: ['novice', 'professional']",
        ]
    )
    for row in cm:
        lines.append(f"  {list(row)}")
    lines.extend(["", "Top feature signals:"])
    for name, value in feature_importance(best_model)[:10]:
        lines.append(f"  {name:20s} {value:.4f}")
    lines.extend(
        [
            "",
            "Methodology limitation:",
            "  This is a proxy detector because no human-labelled novice/professional",
            "  dataset is available. It should calibrate feedback tone and lesson",
            "  difficulty, not make a final judgement about the user.",
        ]
    )
    result_path.write_text("\n".join(lines), encoding="utf-8")

    sns.set_theme(style="whitegrid", palette="Greens")
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=LABELS, yticklabels=LABELS)
    plt.title(f"Skill Detector - {best_name}\nMacro F1={best['macro_f1']:.4f}")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(result_path)
    print(pred_path)
    print(cm_path)
    print(model_path)
    print(feature_path)
    print("\nFinal skill detector summary")
    print(f"Selected model: {best_name}")
    print(f"Accuracy: {best['accuracy']:.4f}")
    print(f"Macro F1: {best['macro_f1']:.4f}")
    print(report)


def main() -> None:
    df = build_dataset()
    print(f"Rows after Easy/Hard filtering: {len(df)}")
    print(df["skill_label"].value_counts().to_string())
    best_name, best_model, results, pred_df = train_and_evaluate(df)
    save_outputs(best_name, best_model, results, pred_df, df)


if __name__ == "__main__":
    main()
