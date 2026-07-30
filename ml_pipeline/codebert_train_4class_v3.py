"""
Phase 5 — Step 4: Stage 1 CodeBERT Training v3 (4-class, grouped split)
======================================================================
Project : code-analyzer-ai
Author  : Nguyen Ngoc Gia Han (GCS230054)

Why this script exists:
  Retrains the Stage 1 4-class classifier on the leakage-free grouped splits
  (train4_v3 / val4_v3 / test4_v3) from data_prep_grouped_v3.py.

  Two deliberate differences from codebert_retrain_4class_v2.py:
    1. Fine-tunes from the CLEAN pretrained backbone `microsoft/codebert-base`
       (not the old local codebert_model, which was trained on the leaky split
       and would carry leaked weights forward).
    2. Trains/evaluates on problem-grouped splits, so the test macro-F1 is an
       HONEST estimate of generalisation to unseen problems. Compare it against
       the v2 leaky test macro-F1 (0.9840): a drop quantifies the leakage /
       rote-memorisation that inflated the original figure.

  Trains up to 10 epochs with early stopping (patience 3 on validation macro-F1)
  and keeps the best checkpoint, so the model cannot overfit past its peak.

HOW TO RUN:
  1. python ml_pipeline/data_prep_grouped_v3.py
  2. python ml_pipeline/codebert_train_4class_v3.py
     (GPU strongly recommended; downloads codebert-base ~500MB on first run)

Output (backend/app/ml_models/): codebert_4class_v3_model/
Output (ml_pipeline/data/processed/):
  codebert_4class_v3_results.txt / _history.png / _confusion_matrix.png
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
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
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

SOURCE_CHECKPOINT = "microsoft/codebert-base"
SAVE_DIR          = os.path.join(MODEL_BASE_DIR, "codebert_4class_v3_model")

LEARNING_RATE       = 2e-5
BATCH_SIZE          = 16
NUM_EPOCHS          = 10
EARLY_STOP_PATIENCE = 3     # stop if no val-F1 improvement for this many epochs
MAX_LENGTH          = 256
NUM_CLASSES         = 4
RANDOM_SEED         = 42

CLASS_NAMES = ["syntax_error", "logic_error", "variable_misuse", "no_bug"]
# ───────────────────────────────────────────────────────────────────────

os.makedirs(SAVE_DIR, exist_ok=True)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
sns.set_theme(style="whitegrid", palette="muted")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice : {device}")
if device.type == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: no GPU — training will be slow.")


def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


class CodeDataset(Dataset):
    def __init__(self, codes, labels, tokenizer, max_length):
        self.codes = [str(c) for c in codes]
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.codes[idx], max_length=self.max_length,
                             padding="max_length", truncation=True, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Load data ──────────────────────────────────────────────────────────
section("Loading 4-class v3 (grouped) splits")
df_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "train4_v3.parquet"))
df_val   = pd.read_parquet(os.path.join(PROCESSED_DIR, "val4_v3.parquet"))
df_test  = pd.read_parquet(os.path.join(PROCESSED_DIR, "test4_v3.parquet"))
print(f"Train {len(df_train):,}  Val {len(df_val):,}  Test {len(df_test):,}")
for i, c in enumerate(CLASS_NAMES):
    print(f"  {c:16s}: train {int((df_train['label']==i).sum()):,}")

class_weights_list = [1.0] * NUM_CLASSES
with open(os.path.join(PROCESSED_DIR, "class_weights4_v3.txt")) as f:
    for line in f:
        if line.startswith("#"):
            continue
        parts = line.strip().split(",")
        if len(parts) == 3:
            class_weights_list[int(parts[0])] = float(parts[2])
weights_tensor = torch.tensor(class_weights_list, dtype=torch.float).to(device)
loss_fn = torch.nn.CrossEntropyLoss(weight=weights_tensor)
print(f"Class weights: {[round(w,3) for w in class_weights_list]}")

# ── Model ──────────────────────────────────────────────────────────────
section(f"Loading backbone: {SOURCE_CHECKPOINT}")
tokenizer = AutoTokenizer.from_pretrained(SOURCE_CHECKPOINT)
model = AutoModelForSequenceClassification.from_pretrained(
    SOURCE_CHECKPOINT, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True)
model.config.id2label = {i: c for i, c in enumerate(CLASS_NAMES)}
model.config.label2id = {c: i for i, c in enumerate(CLASS_NAMES)}
model.to(device)

train_ds = CodeDataset(df_train["code"].values, df_train["label"].values.astype(int), tokenizer, MAX_LENGTH)
val_ds   = CodeDataset(df_val["code"].values,   df_val["label"].values.astype(int),   tokenizer, MAX_LENGTH)
test_ds  = CodeDataset(df_test["code"].values,  df_test["label"].values.astype(int),  tokenizer, MAX_LENGTH)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)


def train_one_epoch(model, loader, optimizer, scheduler):
    model.train()
    total, correct, n = 0.0, 0, 0
    for batch in loader:
        ids = batch["input_ids"].to(device); mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        out = model(input_ids=ids, attention_mask=mask)
        loss = loss_fn(out.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        total += loss.item()
        correct += (out.logits.argmax(1) == labels).sum().item(); n += labels.size(0)
    return total / len(loader), correct / n


def evaluate(model, loader):
    model.eval()
    total, preds, labs = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device); mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            out = model(input_ids=ids, attention_mask=mask)
            total += loss_fn(out.logits, labels).item()
            preds.extend(out.logits.argmax(1).cpu().numpy()); labs.extend(labels.cpu().numpy())
    return (total / len(loader), accuracy_score(labs, preds),
            f1_score(labs, preds, average="macro", zero_division=0), preds, labs)


# ── Train ──────────────────────────────────────────────────────────────
section(f"Training — up to {NUM_EPOCHS} epochs  lr={LEARNING_RATE:.0e}  batch={BATCH_SIZE}  "
        f"early-stop patience={EARLY_STOP_PATIENCE}")
total_steps = len(train_loader) * NUM_EPOCHS
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)

history, best_f1, best_epoch, t_start = [], 0.0, 1, time.time()
epochs_no_improve = 0
for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scheduler)
    v_loss, v_acc, v_f1, _, _ = evaluate(model, val_loader)
    print(f"  Epoch {epoch}/{NUM_EPOCHS}  train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
          f"val_loss={v_loss:.4f} val_acc={v_acc:.4f} val_f1={v_f1:.4f}  [{round(time.time()-t0,1)}s]")
    history.append({"epoch": epoch, "train_loss": round(tr_loss, 4), "train_acc": round(tr_acc, 4),
                    "val_loss": round(v_loss, 4), "val_acc": round(v_acc, 4), "val_f1": round(v_f1, 4)})
    if v_f1 > best_f1:
        best_f1, best_epoch = v_f1, epoch
        epochs_no_improve = 0
        model.save_pretrained(SAVE_DIR); tokenizer.save_pretrained(SAVE_DIR)
        print(f"  ✓ best checkpoint saved (val_f1={v_f1:.4f})")
    else:
        epochs_no_improve += 1
        print(f"    no val-F1 improvement ({epochs_no_improve}/{EARLY_STOP_PATIENCE})")
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping at epoch {epoch} — best epoch {best_epoch} (val_f1={best_f1:.4f})")
            break
total_time = round(time.time() - t_start, 1)
print(f"\nTraining complete: {total_time}s  (best epoch {best_epoch}/{NUM_EPOCHS})")

# ── Test ───────────────────────────────────────────────────────────────
section("Test evaluation (best checkpoint) — HONEST unseen-problem estimate")
best = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR, num_labels=NUM_CLASSES).to(device)
_, test_acc, test_f1, test_preds, test_labels = evaluate(best, test_loader)
test_wf1 = f1_score(test_labels, test_preds, average="weighted", zero_division=0)
report = classification_report(test_labels, test_preds, target_names=CLASS_NAMES, output_dict=True)
print(f"  Test accuracy : {test_acc:.4f}")
print(f"  Test macro F1 : {test_f1:.4f}   (compare v2 leaky = 0.9840)")
print(f"  Weighted F1   : {test_wf1:.4f}")
print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES))

# Per-source no_bug recall (grouped): leetcode_original vs flytech
preds_arr = np.array(test_preds)
nobug_mask = df_test["label"].values == 3
src_lines = []
for src in ("leetcode_original", "flytech"):
    m = nobug_mask & (df_test["source"].values == src)
    if m.sum() > 0:
        src_lines.append(f"  no_bug recall ({src:17s}): {(preds_arr[m]==3).mean():.4f}  (n={int(m.sum())})")
print("\nPer-source no_bug recall (test):")
for l in src_lines:
    print(l)

# ── Save results ───────────────────────────────────────────────────────
res_path = os.path.join(PROCESSED_DIR, "codebert_4class_v3_results.txt")
with open(res_path, "w", encoding="utf-8") as f:
    f.write("CodeBERT 4-class v3 Results (grouped split, from codebert-base)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Backbone      : {SOURCE_CHECKPOINT}\n")
    f.write(f"Saved model   : {SAVE_DIR}\n")
    f.write(f"Best epoch    : {best_epoch}/{NUM_EPOCHS}  (val F1 {best_f1:.4f})\n")
    f.write(f"Train time    : {total_time}s\n\n")
    f.write("HONEST test (unseen problems):\n")
    f.write(f"  Accuracy    : {test_acc:.4f}\n")
    f.write(f"  Macro F1    : {test_f1:.4f}   <-- compare v2 leaky 0.9840\n")
    f.write(f"  Weighted F1 : {test_wf1:.4f}\n\n")
    f.write("Per-class F1:\n")
    for c in CLASS_NAMES:
        f.write(f"  {c:16s}: {report[c]['f1-score']:.4f}\n")
    f.write("\nPer-source no_bug recall:\n")
    for l in src_lines:
        f.write(l + "\n")
    f.write("\nClassification report:\n")
    f.write(classification_report(test_labels, test_preds, target_names=CLASS_NAMES))
    f.write("\nTraining history:\n")
    for h in history:
        f.write(f"  epoch={h['epoch']} val_f1={h['val_f1']:.4f} val_acc={h['val_acc']:.4f} train_loss={h['train_loss']:.4f}\n")
print(f"[saved] {res_path}")

# ── Charts ─────────────────────────────────────────────────────────────
epochs = [h["epoch"] for h in history]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(epochs, [h["train_loss"] for h in history], "b-o", label="Train loss")
axes[0].plot(epochs, [h["val_loss"] for h in history], "r-o", label="Val loss")
axes[0].axvline(best_epoch, color="green", ls="--", alpha=0.6, label=f"Best {best_epoch}")
axes[0].set_title("CodeBERT 4-class v3 — Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
axes[1].plot(epochs, [h["val_f1"] for h in history], "g-o", label="Val macro F1")
axes[1].axhline(test_f1, color="red", ls="--", alpha=0.7, label=f"Test F1={test_f1:.4f}")
axes[1].axhline(0.9840, color="gray", ls=":", alpha=0.7, label="v2 leaky 0.984")
axes[1].set_title("CodeBERT 4-class v3 — Val F1"); axes[1].set_xlabel("Epoch"); axes[1].set_ylim(0, 1.05); axes[1].legend()
fig.suptitle("CodeBERT 4-class v3 (grouped) — Training History", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "codebert_4class_v3_history.png"), dpi=150, bbox_inches="tight")
plt.close()

cm = confusion_matrix(test_labels, test_preds)
fig2, ax2 = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(cm, display_labels=["syntax", "logic", "variable", "no_bug"]).plot(ax=ax2, cmap="Blues")
ax2.set_title(f"CodeBERT 4-class v3 — Confusion (Test)\nMacro F1={test_f1:.4f}", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "codebert_4class_v3_confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[saved] codebert_4class_v3_history.png / _confusion_matrix.png")

print(f"""
{'=' * 60}
  CODEBERT 4-CLASS v3 COMPLETE
{'=' * 60}
  Honest test macro-F1: {test_f1:.4f}   (v2 leaky was 0.9840)
  Model: {SAVE_DIR}

  Next: codebert_train_stage2_v3.py, codebert_train_line_detection_v3.py,
        codebert_cv_stage1_v3.py, pybughive_generalization_eval_v3.py
{'=' * 60}
""")
