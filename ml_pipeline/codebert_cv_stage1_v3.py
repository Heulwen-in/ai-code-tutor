"""
Phase 5 — Step 7: Stage 1 Grouped Cross-Validation v3
=====================================================
Project : code-analyzer-ai
Author  : Nguyen Ngoc Gia Han (GCS230054)

Why this script exists:
  A single grouped train/test split proves the headline number is not a lucky
  cut only weakly. This runs k-fold GROUPED cross-validation (StratifiedGroupKFold,
  groups = problem_id) over the whole 4-class dataset, so every fold's validation
  set contains entirely unseen problems. Reporting mean +/- std macro-F1 across
  folds is the strongest single piece of evidence that the model generalises
  across problems (i.e. is NOT rote-memorising), independent of any one split.

  Each fold fine-tunes a fresh `microsoft/codebert-base` (override via env
  CODEBERT_BASE). Fold models are NOT persisted (only metrics), to save disk.

  This is expensive: k folds x CV_EPOCHS. Defaults: 5 folds, 3 epochs. Tune via
  env N_SPLITS and CV_EPOCHS. Fold models are discarded; the deployed Stage 1
  model comes from codebert_train_4class_v3.py, not from here.

HOW TO RUN:
  1. python ml_pipeline/data_prep_grouped_v3.py
  2. python ml_pipeline/codebert_cv_stage1_v3.py
     (optionally: N_SPLITS=5 CV_EPOCHS=3 python ...)

Output (ml_pipeline/data/processed/):
  codebert_cv_stage1_v3_results.txt / codebert_cv_stage1_v3.png
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")

PROCESSED_DIR = r"F:\Heulwen\IT GREENWICH\Final Project\Tutor_AI_code\ml_pipeline\data\processed"
SOURCE_CHECKPOINT = "microsoft/codebert-base"

# CV uses a FIXED epoch count (no early stopping): early-stopping on a fold's own
# validation set would bias the very metric the fold reports. Keep it modest to
# bound GPU time — this is a robustness check, not the deployed model.
N_SPLITS      = 5
CV_EPOCHS     = 4
LEARNING_RATE = 2e-5
BATCH_SIZE    = 16
MAX_LENGTH    = 256
NUM_CLASSES = 4
RANDOM_SEED = 42
CLASS_NAMES = ["syntax_error", "logic_error", "variable_misuse", "no_bug"]

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
sns.set_theme(style="whitegrid", palette="muted")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}  |  {N_SPLITS}-fold grouped CV  |  {CV_EPOCHS} epochs/fold")


def section(t):
    print("\n" + "=" * 60 + f"\n  {t}\n" + "=" * 60)


class CodeDataset(Dataset):
    def __init__(self, codes, labels, tok, mx):
        self.codes = [str(c) for c in codes]; self.labels = labels; self.tok = tok; self.mx = mx

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, i):
        e = self.tok(self.codes[i], max_length=self.mx, padding="max_length",
                     truncation=True, return_tensors="pt")
        return {"input_ids": e["input_ids"].squeeze(0), "attention_mask": e["attention_mask"].squeeze(0),
                "label": torch.tensor(self.labels[i], dtype=torch.long)}


# ── Combine the grouped v3 4-class splits into one CV pool ─────────────
section("Loading + pooling 4-class v3 data for CV")
parts = [pd.read_parquet(os.path.join(PROCESSED_DIR, f"{s}4_v3.parquet"))
         for s in ("train", "val", "test")]
data = pd.concat(parts, ignore_index=True)
codes = data["code"].astype(str).values
y = data["label"].values.astype(int)
groups = data["problem_id"].astype(str).values
print(f"Pooled rows: {len(data):,}  unique problems: {pd.Series(groups).nunique():,}")
print(f"Class counts: {{ {', '.join(f'{c}:{int((y==i).sum())}' for i,c in enumerate(CLASS_NAMES))} }}")

tokenizer = AutoTokenizer.from_pretrained(SOURCE_CHECKPOINT)


def run_fold(tr_idx, va_idx):
    # Class weights from this fold's training labels
    counts = np.bincount(y[tr_idx], minlength=NUM_CLASSES)
    w = torch.tensor([len(tr_idx) / (NUM_CLASSES * max(1, c)) for c in counts],
                     dtype=torch.float).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)

    model = AutoModelForSequenceClassification.from_pretrained(
        SOURCE_CHECKPOINT, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True).to(device)

    tr_loader = DataLoader(CodeDataset(codes[tr_idx], y[tr_idx], tokenizer, MAX_LENGTH),
                           batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(CodeDataset(codes[va_idx], y[va_idx], tokenizer, MAX_LENGTH),
                           batch_size=BATCH_SIZE)

    steps = len(tr_loader) * CV_EPOCHS
    opt = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(steps * 0.1), steps)

    for ep in range(CV_EPOCHS):
        model.train()
        for batch in tr_loader:
            ids = batch["input_ids"].to(device); mask = batch["attention_mask"].to(device)
            lab = batch["label"].to(device)
            opt.zero_grad()
            out = model(input_ids=ids, attention_mask=mask)
            loss = loss_fn(out.logits, lab)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()

    model.eval()
    preds, labs = [], []
    with torch.no_grad():
        for batch in va_loader:
            ids = batch["input_ids"].to(device); mask = batch["attention_mask"].to(device)
            out = model(input_ids=ids, attention_mask=mask)
            preds.extend(out.logits.argmax(1).cpu().numpy()); labs.extend(batch["label"].numpy())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return (accuracy_score(labs, preds),
            f1_score(labs, preds, average="macro", zero_division=0),
            f1_score(labs, preds, average="weighted", zero_division=0))


sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
fold_rows = []
t0 = time.time()
for k, (tr_idx, va_idx) in enumerate(sgkf.split(codes, y, groups), start=1):
    # guard: the fold's train/val problems must be disjoint
    assert not (set(groups[tr_idx]) & set(groups[va_idx])), f"fold {k} group leak"
    section(f"Fold {k}/{N_SPLITS}  (train {len(tr_idx):,} / val {len(va_idx):,})")
    acc, mf1, wf1 = run_fold(tr_idx, va_idx)
    print(f"  Fold {k}: acc={acc:.4f}  macroF1={mf1:.4f}  weightedF1={wf1:.4f}")
    fold_rows.append({"fold": k, "acc": acc, "macro_f1": mf1, "weighted_f1": wf1,
                      "n_val": len(va_idx)})

df_cv = pd.DataFrame(fold_rows)
mean_f1, std_f1 = df_cv["macro_f1"].mean(), df_cv["macro_f1"].std()
mean_acc, std_acc = df_cv["acc"].mean(), df_cv["acc"].std()
total_time = round(time.time() - t0, 1)

section("Cross-validation summary")
print(df_cv.to_string(index=False))
print(f"\n  Macro-F1: {mean_f1:.4f} ± {std_f1:.4f}   Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
print(f"  Total CV time: {total_time}s")

res_path = os.path.join(PROCESSED_DIR, "codebert_cv_stage1_v3_results.txt")
with open(res_path, "w", encoding="utf-8") as f:
    f.write(f"Stage 1 Grouped {N_SPLITS}-Fold CV v3 (StratifiedGroupKFold on problem_id)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Backbone: {SOURCE_CHECKPOINT}   epochs/fold: {CV_EPOCHS}\n")
    f.write("Every fold's validation problems are unseen in that fold's training.\n\n")
    f.write(df_cv.to_string(index=False))
    f.write(f"\n\nMacro-F1: {mean_f1:.4f} +/- {std_f1:.4f}\n")
    f.write(f"Accuracy: {mean_acc:.4f} +/- {std_acc:.4f}\n")
    f.write(f"Total CV time: {total_time}s\n\n")
    f.write("Interpretation: a tight, high mean across folds = the model generalises\n")
    f.write("across problems (not rote). A large drop vs the v2 leaky 0.9840 = the\n")
    f.write("original figure was inflated by problem-context leakage.\n")
print(f"[saved] {res_path}")

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(df_cv["fold"].astype(str), df_cv["macro_f1"], color="#4C72B0", edgecolor="white")
ax.axhline(mean_f1, color="green", ls="--", label=f"mean {mean_f1:.4f} ± {std_f1:.4f}")
ax.axhline(0.9840, color="gray", ls=":", label="v2 leaky 0.984")
for b, v in zip(bars, df_cv["macro_f1"]):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
ax.set_ylim(0, 1.05); ax.set_xlabel("Fold"); ax.set_ylabel("Macro F1")
ax.set_title(f"Stage 1 — {N_SPLITS}-Fold Grouped CV (v3)", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "codebert_cv_stage1_v3.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[saved] codebert_cv_stage1_v3.png")
print(f"\nCV COMPLETE — Macro-F1 {mean_f1:.4f} ± {std_f1:.4f}")
