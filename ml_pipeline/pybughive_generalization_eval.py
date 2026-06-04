"""
Phase 2 - PyBugHive Generalisation Evaluation

Evaluates the selected CodeBERT classifier on labelled PyBugHive issues.

Important methodology note:
PyBugHive stores real bug-fix patches, not the same clean "bugged_code"
column used by BuggedPythonLeetCode. For evaluation, this script reconstructs
a compact code snippet per issue from Python patch lines:

  1. Prefer deleted lines ("-") because they are closer to the buggy code.
  2. Fall back to added lines ("+") when no deleted Python lines exist.

The transformer was trained on three classes only:
syntax_error, logic_error, variable_misuse.
indentation_error is excluded here because it is handled by the separate
rule-based detector in the backend.

Outputs:
  ml_pipeline/data/processed/pybughive_codebert_generalization.txt
  ml_pipeline/data/processed/pybughive_codebert_predictions.csv
  ml_pipeline/data/processed/pybughive_codebert_confusion_matrix.png
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(r"F:\Heulwen\IT GREENWICH\Final Project")
LABELLED_CSV = ROOT / "Datasets" / "eda_output" / "pybughive_labelled.csv"
PYBUGHIVE_FILES = [
    ROOT / "Datasets" / "Pybughive" / "pybughive" / "dataset" / "pybughive_current.json",
    ROOT / "Datasets" / "Pybughive" / "pybughive" / "dataset" / "pybughive_small.json",
    ROOT / "Datasets" / "Pybughive" / "pybughive" / "dataset_pybughivex" / "pybughivex_current.json",
]
MODEL_DIR = ROOT / "Tutor_AI_code" / "backend" / "app" / "ml_models" / "codebert_model"
OUTPUT_DIR = ROOT / "Tutor_AI_code" / "ml_pipeline" / "data" / "processed"

CLASS_NAMES = ["syntax_error", "logic_error", "variable_misuse"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
MAX_LENGTH = 256
BATCH_SIZE = 16


class CodeDataset(Dataset):
    def __init__(self, codes: list[str], tokenizer):
        self.codes = [str(code) for code in codes]
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.codes)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.codes[idx],
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


def source_name(path: Path) -> str:
    return path.stem


def clean_patch_lines(lines: list[str]) -> str:
    cleaned = []
    for line in lines:
        text = line.rstrip()
        if not text.strip():
            continue
        # Drop common non-code patch noise, but keep real comments/docstrings.
        if text.lstrip().startswith(("@@", "+++", "---")):
            continue
        cleaned.append(text)
    return "\n".join(cleaned).strip()


def collect_issue_code() -> dict[tuple[str, str, int], dict[str, str]]:
    issue_code = {}

    for path in PYBUGHIVE_FILES:
        with path.open("r", encoding="utf-8") as f:
            projects = json.load(f)

        src = source_name(path)
        for project in projects:
            repo = project.get("repository", "")
            for issue in project.get("issues", []):
                deleted_lines: list[str] = []
                added_lines: list[str] = []

                for commit in issue.get("commits", []):
                    files = commit.get("stat", {}).get("files", [])
                    for file_info in files:
                        filename = str(file_info.get("filename", ""))
                        if not filename.endswith(".py"):
                            continue
                        patch = file_info.get("patch", "") or ""
                        for raw_line in patch.splitlines():
                            if raw_line.startswith("-") and not raw_line.startswith("---"):
                                deleted_lines.append(raw_line[1:])
                            elif raw_line.startswith("+") and not raw_line.startswith("+++"):
                                added_lines.append(raw_line[1:])

                deleted_code = clean_patch_lines(deleted_lines)
                added_code = clean_patch_lines(added_lines)
                if deleted_code:
                    code_text = deleted_code
                    code_source = "deleted_lines_buggy_preferred"
                elif added_code:
                    code_text = added_code
                    code_source = "added_lines_fallback"
                else:
                    code_text = ""
                    code_source = "no_python_patch_code"

                key = (src, repo, int(issue.get("id")))
                issue_code[key] = {
                    "code_text": code_text,
                    "code_source": code_source,
                    "code_chars": str(len(code_text)),
                    "code_lines": str(code_text.count("\n") + 1 if code_text else 0),
                }

    return issue_code


def load_eval_frame() -> pd.DataFrame:
    labelled = pd.read_csv(LABELLED_CSV)
    issue_code = collect_issue_code()

    rows = []
    for _, row in labelled.iterrows():
        label = row["error_class"]
        if label not in CLASS_TO_ID:
            continue

        key = (str(row["source"]), str(row["repo"]), int(row["issue_id"]))
        code_info = issue_code.get(key, {})
        code_text = code_info.get("code_text", "")
        if not code_text:
            continue

        out = row.to_dict()
        out.update(code_info)
        out["label"] = CLASS_TO_ID[label]
        rows.append(out)

    return pd.DataFrame(rows)


def predict_codes(df: pd.DataFrame) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    dataset = CodeDataset(df["code_text"].astype(str).tolist(), tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    preds: list[int] = []
    confs: list[float] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            preds.extend(pred.cpu().tolist())
            confs.extend(conf.cpu().tolist())

    df = df.copy()
    df["pred_label"] = preds
    df["pred_error_class"] = [CLASS_NAMES[i] for i in preds]
    df["confidence"] = [round(x, 4) for x in confs]
    return df


def save_outputs(df: pd.DataFrame) -> None:
    y_true = df["label"].astype(int).to_numpy()
    y_pred = df["pred_label"].astype(int).to_numpy()

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    pred_csv = OUTPUT_DIR / "pybughive_codebert_predictions.csv"
    result_txt = OUTPUT_DIR / "pybughive_codebert_generalization.txt"
    cm_png = OUTPUT_DIR / "pybughive_codebert_confusion_matrix.png"

    save_cols = [
        "repo",
        "source",
        "issue_id",
        "title",
        "raw_labels",
        "error_class",
        "pred_error_class",
        "confidence",
        "code_source",
        "code_chars",
        "code_lines",
        "label_source",
    ]
    df[save_cols].to_csv(pred_csv, index=False)

    class_counts = df["error_class"].value_counts().reindex(CLASS_NAMES, fill_value=0)
    source_counts = df["code_source"].value_counts()

    lines = [
        "PyBugHive Generalisation Evaluation - CodeBERT",
        "=" * 60,
        "",
        "Dataset: PyBugHive labelled issues reconstructed from Python patch lines",
        "Model: backend/app/ml_models/codebert_model",
        "Excluded labels: unknown, indentation_error",
        "",
        f"Evaluation rows: {len(df)}",
        "Class distribution:",
    ]
    for cls, count in class_counts.items():
        lines.append(f"  {cls:16s}: {count}")
    lines.extend(
        [
            "",
            "Code reconstruction source:",
            *[f"  {idx}: {val}" for idx, val in source_counts.items()],
            "",
            "Metrics:",
            f"  Accuracy   : {acc:.4f}",
            f"  Macro F1   : {macro_f1:.4f}",
            f"  Weighted F1: {weighted_f1:.4f}",
            "",
            "Classification Report:",
            report,
            "Confusion Matrix:",
            f"  Labels: {CLASS_NAMES}",
        ]
    )
    for row in cm:
        lines.append(f"  {list(row)}")
    lines.extend(
        [
            "",
            "Methodology limitation:",
            "  PyBugHive contains real bug-fix patches, not the same synthetic bugged_code",
            "  format used for transformer training. Deleted Python patch lines are used as",
            "  the closest available buggy-code signal, with added lines as fallback.",
            "  Results should therefore be interpreted as external generalisation evidence,",
            "  not as a perfectly matched benchmark.",
        ]
    )
    result_txt.write_text("\n".join(lines), encoding="utf-8")

    sns.set_theme(style="whitegrid", palette="Blues")
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["syntax", "logic", "variable"],
        yticklabels=["syntax", "logic", "variable"],
    )
    plt.title(f"PyBugHive Generalisation - CodeBERT\nMacro F1={macro_f1:.4f}")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(cm_png, dpi=150, bbox_inches="tight")
    plt.close()

    print(result_txt)
    print(pred_csv)
    print(cm_png)
    print("\nFinal PyBugHive summary")
    print(f"Rows: {len(df)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(report)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_eval_frame()
    print(f"Evaluation rows after filtering: {len(df)}")
    print(df["error_class"].value_counts().to_string())
    print(df["code_source"].value_counts().to_string())
    if df.empty:
        raise RuntimeError("No PyBugHive rows available for evaluation.")

    pred_df = predict_codes(df)
    save_outputs(pred_df)


if __name__ == "__main__":
    main()
