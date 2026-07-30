"""
Phase 5 — Step 9: Standalone Evaluation of the Trained v3 Models
===============================================================
Project : code-analyzer-ai
Author  : Nguyen Ngoc Gia Han (GCS230054)

Why this script exists:
  The training scripts already evaluate on the test split, but they do it as
  part of a long GPU training run. This script loads the SAVED v3 checkpoints
  and re-computes accuracy / macro-F1 on the val and test splits in ~1 minute,
  so the reported numbers can be verified independently (and re-run any time)
  without retraining anything.

  IMPORTANT — which split to report:
    val_*  was used during training for early stopping / best-checkpoint
           selection, so the model has indirectly seen it. It is NOT a clean
           estimate and must not be the headline number.
    test_* was never used for any decision. This is the honest estimate and the
           figure to quote in the thesis.

  Both are printed so the val-vs-test gap is visible (a large gap would indicate
  the checkpoint was over-selected on val).

Models evaluated (from backend/app/ml_models/):
  codebert_4class_v3_model         on val4_v3  / test4_v3
  codebert_stage2_v3_model         on val14_v3 / test14_v3
  codebert_line_detection_v3_model on line_val_v3 / line_test_v3

HOW TO RUN:
  python ml_pipeline/evaluate_v3_models.py

Output (ml_pipeline/data/processed/):
  evaluate_v3_models_results.txt
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification,
)
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, precision_recall_fscore_support,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")

# ── CONFIG ─────────────────────────────────────────────────────────────
PROCESSED_DIR  = r"F:\Heulwen\IT GREENWICH\Final Project\Tutor_AI_code\ml_pipeline\data\processed"
MODEL_BASE_DIR = r"F:\Heulwen\IT GREENWICH\Final Project\Tutor_AI_code\backend\app\ml_models"

STAGE1_DIR = os.path.join(MODEL_BASE_DIR, "codebert_4class_v3_model")
STAGE2_DIR = os.path.join(MODEL_BASE_DIR, "codebert_stage2_v3_model")
LINE_DIR   = os.path.join(MODEL_BASE_DIR, "codebert_line_detection_v3_model")

MAX_LENGTH = 256
BATCH_SIZE = 32
CLASS_NAMES = ["syntax_error", "logic_error", "variable_misuse", "no_bug"]
# ───────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
report_lines = []


def section(title):
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}")
    report_lines.append(f"\n{bar}\n  {title}\n{bar}")


def log(msg):
    print(msg)
    report_lines.append(msg)


class SeqDataset(Dataset):
    def __init__(self, codes, labels, tok):
        self.codes = [str(c) for c in codes]; self.labels = labels; self.tok = tok

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, i):
        e = self.tok(self.codes[i], max_length=MAX_LENGTH, padding="max_length",
                     truncation=True, return_tensors="pt")
        return {"input_ids": e["input_ids"].squeeze(0),
                "attention_mask": e["attention_mask"].squeeze(0),
                "label": torch.tensor(self.labels[i], dtype=torch.long)}


def eval_sequence_model(model, tok, df, label_col):
    ds = SeqDataset(df["code"].values, df[label_col].values.astype(int), tok)
    loader = DataLoader(ds, batch_size=BATCH_SIZE)
    preds, labs = [], []
    model.eval()
    with torch.no_grad():
        for b in loader:
            out = model(input_ids=b["input_ids"].to(device),
                        attention_mask=b["attention_mask"].to(device))
            preds.extend(out.logits.argmax(1).cpu().numpy())
            labs.extend(b["label"].numpy())
    return np.array(labs), np.array(preds)


# ════════════════════════════════════════════════════════════════
#  STAGE 1 — 4-class
# ════════════════════════════════════════════════════════════════
section("STAGE 1 — 4-class classifier (codebert_4class_v3_model)")
tok1 = AutoTokenizer.from_pretrained(STAGE1_DIR)
m1 = AutoModelForSequenceClassification.from_pretrained(STAGE1_DIR).to(device)

stage1_summary = {}
for split in ("val", "test"):
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, f"{split}4_v3.parquet"))
    y, p = eval_sequence_model(m1, tok1, df, "label")
    acc = accuracy_score(y, p)
    mf1 = f1_score(y, p, average="macro", zero_division=0)
    wf1 = f1_score(y, p, average="weighted", zero_division=0)
    stage1_summary[split] = (acc, mf1)
    tag = "(model-selection split — NOT the headline)" if split == "val" else "(HONEST held-out estimate)"
    log(f"\n{split.upper()}  n={len(df):,}  {tag}")
    log(f"  Accuracy    : {acc:.4f}")
    log(f"  Macro F1    : {mf1:.4f}")
    log(f"  Weighted F1 : {wf1:.4f}")
    if split == "test":
        log("\n" + classification_report(y, p, target_names=CLASS_NAMES, digits=4, zero_division=0))
        # no_bug recall by source — shows how the clean-code class behaves per origin
        for src in ("leetcode_original", "flytech"):
            m = (df["label"].values == 3) & (df["source"].values == src)
            if m.sum():
                log(f"  no_bug recall ({src:17s}): {(p[m] == 3).mean():.4f}  (n={int(m.sum())})")

del m1
if device.type == "cuda":
    torch.cuda.empty_cache()

# ════════════════════════════════════════════════════════════════
#  STAGE 2 — 14 subtypes
# ════════════════════════════════════════════════════════════════
section("STAGE 2 — 14-subtype classifier (codebert_stage2_v3_model)")
with open(os.path.join(PROCESSED_DIR, "subtype_mapping_v3.json"), encoding="utf-8") as f:
    mapping = json.load(f)
SUBTYPES = [mapping["int_to_subtype"][str(i)] for i in range(14)]
HUMAN = [mapping["human_names"][s] for s in SUBTYPES]
SUB_TO_COARSE = {i: c for c, ids in mapping["coarse_groups"].items() for i in ids}

tok2 = AutoTokenizer.from_pretrained(STAGE2_DIR)
m2 = AutoModelForSequenceClassification.from_pretrained(STAGE2_DIR).to(device)

stage2_summary = {}
for split in ("val", "test"):
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, f"{split}14_v3.parquet"))
    y, p = eval_sequence_model(m2, tok2, df, "sub_label")
    acc = accuracy_score(y, p)
    mf1 = f1_score(y, p, average="macro", zero_division=0)
    cff = np.mean([SUB_TO_COARSE[a] == SUB_TO_COARSE[b] for a, b in zip(y, p)])
    stage2_summary[split] = (acc, mf1)
    tag = "(model-selection split)" if split == "val" else "(HONEST held-out estimate)"
    log(f"\n{split.upper()}  n={len(df):,}  {tag}")
    log(f"  Accuracy             : {acc:.4f}")
    log(f"  Macro F1             : {mf1:.4f}")
    log(f"  Coarse-from-fine acc : {cff:.4f}")
    if split == "test":
        log("\n" + classification_report(y, p, target_names=HUMAN, digits=4, zero_division=0))

del m2
if device.type == "cuda":
    torch.cuda.empty_cache()

# ════════════════════════════════════════════════════════════════
#  LINE DETECTION — token classification
# ════════════════════════════════════════════════════════════════
section("LINE DETECTION — token classifier (codebert_line_detection_v3_model)")


def char_to_line(code):
    mapping, line = [], 1
    for ch in code:
        mapping.append(line)
        if ch == "\n":
            line += 1
    mapping.append(line)
    return mapping


tok3 = AutoTokenizer.from_pretrained(LINE_DIR)
m3 = AutoModelForTokenClassification.from_pretrained(LINE_DIR).to(device)
m3.eval()

line_summary = {}
for split in ("val", "test"):
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, f"line_{split}_v3.parquet"))
    tok_preds, tok_labs, hits, scorable = [], [], 0, 0
    with torch.no_grad():
        for start in range(0, len(df), BATCH_SIZE):
            chunk = df.iloc[start:start + BATCH_SIZE]
            codes = [str(c) for c in chunk["code"].tolist()]
            enc = tok3(codes, max_length=MAX_LENGTH, padding="max_length", truncation=True,
                       return_offsets_mapping=True, return_tensors="pt")
            offs = enc.pop("offset_mapping").numpy()
            logits = m3(input_ids=enc["input_ids"].to(device),
                        attention_mask=enc["attention_mask"].to(device)).logits
            probs = torch.softmax(logits, dim=-1)[:, :, 1].cpu().numpy()
            preds = logits.argmax(-1).cpu().numpy()
            for bi, (_, row) in enumerate(chunk.iterrows()):
                code = str(row["code"]); gold = set(int(x) for x in row["buggy_lines"])
                c2l = char_to_line(code)
                scores = {}
                for ti, (s, e) in enumerate(offs[bi]):
                    if s == e:
                        continue
                    ln = c2l[s] if s < len(c2l) else c2l[-1]
                    tok_labs.append(1 if ln in gold else 0)
                    tok_preds.append(int(preds[bi, ti]))
                    scores.setdefault(ln, []).append(probs[bi, ti])
                if scores:
                    top = max(scores, key=lambda k: np.mean(scores[k]))
                    hits += 1 if top in gold else 0
                    scorable += 1
    pr, rc, f1, _ = precision_recall_fscore_support(tok_labs, tok_preds, labels=[1],
                                                    average="binary", zero_division=0)
    hit1 = hits / scorable if scorable else 0.0
    line_summary[split] = (f1, hit1)
    tag = "(model-selection split)" if split == "val" else "(HONEST held-out estimate)"
    log(f"\n{split.upper()}  n={len(df):,}  {tag}")
    log(f"  Token precision : {pr:.4f}")
    log(f"  Token recall    : {rc:.4f}")
    log(f"  Token F1        : {f1:.4f}")
    log(f"  Line hit@1      : {hit1:.4f}")

# ════════════════════════════════════════════════════════════════
#  SUMMARY
# ════════════════════════════════════════════════════════════════
section("SUMMARY — val vs test (a large gap would mean over-selection on val)")
log(f"{'Model':22s} {'Metric':14s} {'VAL':>10s} {'TEST':>10s}")
log("-" * 60)
log(f"{'Stage 1 (4-class)':22s} {'Accuracy':14s} {stage1_summary['val'][0]:>10.4f} {stage1_summary['test'][0]:>10.4f}")
log(f"{'':22s} {'Macro F1':14s} {stage1_summary['val'][1]:>10.4f} {stage1_summary['test'][1]:>10.4f}")
log(f"{'Stage 2 (14-class)':22s} {'Accuracy':14s} {stage2_summary['val'][0]:>10.4f} {stage2_summary['test'][0]:>10.4f}")
log(f"{'':22s} {'Macro F1':14s} {stage2_summary['val'][1]:>10.4f} {stage2_summary['test'][1]:>10.4f}")
log(f"{'Line detection':22s} {'Token F1':14s} {line_summary['val'][0]:>10.4f} {line_summary['test'][0]:>10.4f}")
log(f"{'':22s} {'Line hit@1':14s} {line_summary['val'][1]:>10.4f} {line_summary['test'][1]:>10.4f}")
log("")
log("Report the TEST column in the thesis. These splits are problem-grouped, so")
log("every test problem is unseen in training (0 cross-split problems asserted in")
log("data_prep_grouped_v3.py). Reference: leaky v2 Stage 1 test macro-F1 = 0.9840.")

out = os.path.join(PROCESSED_DIR, "evaluate_v3_models_results.txt")
with open(out, "w", encoding="utf-8") as f:
    f.write("Standalone Evaluation of Trained v3 Models (grouped, leakage-free splits)\n")
    f.write("=" * 60 + "\n")
    f.write("\n".join(report_lines))
print(f"\n[saved] {out}")
