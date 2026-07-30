"""
Phase 5 — Step 8: PyBugHive External Generalisation v3 (4-class grouped model)
=============================================================================
Project : code-analyzer-ai
Author  : Nguyen Ngoc Gia Han (GCS230054)

Cross-dataset test of the grouped-split Stage 1 model (codebert_4class_v3_model)
on real bug-fix patches from PyBugHive. This is the apples-to-(near)-apples
counterpart of pybughive_generalization_eval.py, which evaluated the OLD leaky
3-class model. Two questions:

  1. Does removing problem-context leakage improve external generalisation
     (macro-F1 on the 3 real bug classes)?
  2. Does the model still COLLAPSE to no_bug on real defective code? The
     "no_bug rate" reported here is the headline symptom metric — the deployed
     4-class model previously returned no_bug on external buggy code with high
     confidence. A lower no_bug rate after the grouped retrain is direct
     evidence the collapse is reduced.

PyBugHive is labelled with the 3 bug classes only (no no_bug ground truth), so
a no_bug prediction (class 3) counts as an error for the 3-class metrics AND is
tracked separately as the collapse rate.

Override the model with env BUG_CLASSIFIER_V3 (default codebert_4class_v3_model).

HOW TO RUN:
  1. Train the model:  python ml_pipeline/codebert_train_4class_v3.py
  2. python ml_pipeline/pybughive_generalization_eval_v3.py

Output (ml_pipeline/data/processed/):
  pybughive_v3_generalization.txt / pybughive_v3_predictions.csv
  pybughive_v3_confusion_matrix.png
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(r"F:\Heulwen\IT GREENWICH\Final Project")
LABELLED_CSV = ROOT / "Datasets" / "eda_output" / "pybughive_labelled.csv"
PYBUGHIVE_FILES = [
    ROOT / "Datasets" / "Pybughive" / "pybughive" / "dataset" / "pybughive_current.json",
    ROOT / "Datasets" / "Pybughive" / "pybughive" / "dataset" / "pybughive_small.json",
    ROOT / "Datasets" / "Pybughive" / "pybughive" / "dataset_pybughivex" / "pybughivex_current.json",
]
MODEL_DIR = ROOT / "Tutor_AI_code" / "backend" / "app" / "ml_models" / "codebert_4class_v3_model"
OUTPUT_DIR = ROOT / "Tutor_AI_code" / "ml_pipeline" / "data" / "processed"

# 4-class model label order; the last (no_bug) has no ground truth in PyBugHive.
MODEL_CLASSES = ["syntax_error", "logic_error", "variable_misuse", "no_bug"]
BUG_CLASSES   = ["syntax_error", "logic_error", "variable_misuse"]
BUG_TO_ID     = {name: i for i, name in enumerate(BUG_CLASSES)}
MAX_LENGTH = 256
BATCH_SIZE = 16


class CodeDataset(Dataset):
    def __init__(self, codes, tokenizer):
        self.codes = [str(c) for c in codes]; self.tokenizer = tokenizer

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.codes[idx], max_length=MAX_LENGTH, truncation=True,
                             padding="max_length", return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0)}


def clean_patch_lines(lines):
    cleaned = []
    for line in lines:
        text = line.rstrip()
        if not text.strip():
            continue
        if text.lstrip().startswith(("@@", "+++", "---")):
            continue
        cleaned.append(text)
    return "\n".join(cleaned).strip()


def collect_issue_code():
    issue_code = {}
    for path in PYBUGHIVE_FILES:
        with path.open("r", encoding="utf-8") as f:
            projects = json.load(f)
        src = path.stem
        for project in projects:
            repo = project.get("repository", "")
            for issue in project.get("issues", []):
                deleted, added = [], []
                for commit in issue.get("commits", []):
                    for file_info in commit.get("stat", {}).get("files", []):
                        if not str(file_info.get("filename", "")).endswith(".py"):
                            continue
                        for raw in (file_info.get("patch", "") or "").splitlines():
                            if raw.startswith("-") and not raw.startswith("---"):
                                deleted.append(raw[1:])
                            elif raw.startswith("+") and not raw.startswith("+++"):
                                added.append(raw[1:])
                deleted_code, added_code = clean_patch_lines(deleted), clean_patch_lines(added)
                if deleted_code:
                    code_text, code_source = deleted_code, "deleted_lines_buggy_preferred"
                elif added_code:
                    code_text, code_source = added_code, "added_lines_fallback"
                else:
                    code_text, code_source = "", "no_python_patch_code"
                issue_code[(src, repo, int(issue.get("id")))] = {
                    "code_text": code_text, "code_source": code_source,
                    "code_chars": str(len(code_text)),
                    "code_lines": str(code_text.count("\n") + 1 if code_text else 0)}
    return issue_code


def load_eval_frame():
    labelled = pd.read_csv(LABELLED_CSV)
    issue_code = collect_issue_code()
    rows = []
    for _, row in labelled.iterrows():
        label = row["error_class"]
        if label not in BUG_TO_ID:
            continue
        info = issue_code.get((str(row["source"]), str(row["repo"]), int(row["issue_id"])), {})
        if not info.get("code_text", ""):
            continue
        out = row.to_dict(); out.update(info)
        out["label"] = BUG_TO_ID[label]
        rows.append(out)
    return pd.DataFrame(rows)


def predict(df):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}   Model: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()
    loader = DataLoader(CodeDataset(df["code_text"].tolist(), tokenizer), batch_size=BATCH_SIZE)
    preds, confs = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device); mask = batch["attention_mask"].to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            probs = torch.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            preds.extend(pred.cpu().tolist()); confs.extend(conf.cpu().tolist())
    df = df.copy()
    df["pred_label"] = preds
    df["pred_error_class"] = [MODEL_CLASSES[i] for i in preds]
    df["confidence"] = [round(x, 4) for x in confs]
    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_eval_frame()
    print(f"Evaluation rows: {len(df)}")
    if df.empty:
        raise RuntimeError("No PyBugHive rows available.")

    df = predict(df)
    y_true = df["label"].astype(int).to_numpy()          # 0..2
    y_pred = df["pred_label"].astype(int).to_numpy()      # 0..3 (3 = no_bug)

    # Headline symptom metric: collapse-to-no_bug rate on real buggy code
    n_nobug = int((y_pred == 3).sum())
    nobug_rate = n_nobug / len(df)

    # 3-class metrics: a no_bug prediction (3) is an error (never equals a true 0..2)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=[0, 1, 2], average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, labels=[0, 1, 2],
                                   target_names=BUG_CLASSES, digits=4, zero_division=0)
    # 3 (true) x 4 (pred) confusion so the no_bug column is visible
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])[:3, :]

    df[["repo", "source", "issue_id", "error_class", "pred_error_class",
        "confidence", "code_source"]].to_csv(
        OUTPUT_DIR / "pybughive_v3_predictions.csv", index=False)

    lines = [
        "PyBugHive Generalisation v3 — CodeBERT 4-class (grouped split)",
        "=" * 60, "",
        f"Model: {MODEL_DIR}",
        f"Evaluation rows: {len(df)}",
        "",
        "HEADLINE symptom metric:",
        f"  no_bug (collapse) rate on real bugs : {nobug_rate:.4f}  ({n_nobug}/{len(df)})",
        "  (lower is better — these are all genuinely buggy snippets)",
        "",
        "3-class metrics (no_bug prediction counts as an error):",
        f"  Accuracy    : {acc:.4f}",
        f"  Macro F1    : {macro_f1:.4f}   <-- compare v2 leaky model = 0.2833",
        f"  Weighted F1 : {weighted_f1:.4f}",
        "",
        "Classification report:",
        report,
        "Confusion matrix (rows=true 3 classes, cols=pred incl. no_bug):",
        f"  cols: {MODEL_CLASSES}",
    ]
    for name, row in zip(BUG_CLASSES, cm):
        lines.append(f"  {name:16s}: {list(int(x) for x in row)}")
    lines += [
        "",
        "Interpretation:",
        "  - Lower no_bug rate + higher macro-F1 than the v2 model => grouped",
        "    training reduced the rote-driven collapse and improved real-bug",
        "    detection. If macro-F1 stays near 0.28 the residual gap is domain",
        "    shift (synthetic vs real bugs), which grouping alone cannot fix.",
    ]
    (OUTPUT_DIR / "pybughive_v3_generalization.txt").write_text("\n".join(lines), encoding="utf-8")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["syntax", "logic", "variable", "no_bug"],
                yticklabels=["syntax", "logic", "variable"])
    plt.title(f"PyBugHive v3 — CodeBERT 4-class (grouped)\n"
              f"Macro F1={macro_f1:.4f}  |  no_bug rate={nobug_rate:.3f}")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pybughive_v3_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nRows: {len(df)}   Acc: {acc:.4f}   Macro F1: {macro_f1:.4f}   "
          f"no_bug rate: {nobug_rate:.4f}")
    print(report)
    print(f"[saved] pybughive_v3_generalization.txt / _predictions.csv / _confusion_matrix.png")


if __name__ == "__main__":
    main()
