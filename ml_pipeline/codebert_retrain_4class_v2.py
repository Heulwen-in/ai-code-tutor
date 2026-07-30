"""
Phase 4 — Step 2: CodeBERT Retraining v2 (4-class, expanded no_bug pool)
========================================================================
Project : code-analyzer-ai
Author  : Nguyen Ngoc Gia Han (GCS230054)

Why this script exists:
  The v1 4-class model saw no_bug examples from a single source (clean
  LeetCode solutions). The v2 splits (data_prep_4class_v2.py) add a second
  no_bug source (flytech/python-codes-25k, instructional-style code), so
  this retrain teaches the model to recognise clean code across styles.

  Same recipe as codebert_retrain_4class.py: backbone continued from the
  3-class CodeBERT checkpoint, classification head reinitialised for 4
  labels, weighted cross-entropy, best checkpoint by validation macro-F1.

  New in v2: per-source no_bug recall on the test set (leetcode vs flytech)
  is reported — this quantifies domain shift for the thesis.

HOW TO RUN:
  1. python ml_pipeline/data_prep_4class_v2.py   (creates *_v2 parquets)
  2. python ml_pipeline/codebert_retrain_4class_v2.py

  GPU strongly recommended (~20-35 min on RTX 2070). CPU fallback: ~3 hrs.

Output (backend/app/ml_models/):
  codebert_4class_v2_model/

Output (ml_pipeline/data/processed/):
  codebert_4class_v2_results.txt
  codebert_4class_v2_history.png
  codebert_4class_v2_confusion_matrix.png
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
    RobertaTokenizer,
    RobertaForSequenceClassification,
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
LOG_DIR        = r"F:\Heulwen\IT GREENWICH\Final Project\Tutor_AI_code\ml_pipeline\logs"

# Source checkpoint — the 3-class CodeBERT backbone will be reused.
SOURCE_CHECKPOINT = os.path.join(MODEL_BASE_DIR, "codebert_model")
SAVE_DIR          = os.path.join(MODEL_BASE_DIR, "codebert_4class_v2_model")

LEARNING_RATE  = 2e-5
BATCH_SIZE     = 16
NUM_EPOCHS     = 5
MAX_LENGTH     = 256
NUM_CLASSES    = 4
RANDOM_SEED    = 42

CLASS_NAMES = ["syntax_error", "logic_error", "variable_misuse", "no_bug"]
# ───────────────────────────────────────────────────────────────────────

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
sns.set_theme(style="whitegrid", palette="muted")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice : {device}")
if device.type == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM   : {vram:.1f} GB")
else:
    print("WARNING: No GPU — training will take ~3 hours on CPU.")


def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ════════════════════════════════════════════════════════════════
#  DATASET
# ════════════════════════════════════════════════════════════════

class CodeDataset(Dataset):
    def __init__(self, codes, labels, tokenizer, max_length):
        self.codes      = [str(c) for c in codes]
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.codes[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ════════════════════════════════════════════════════════════════
#  LOAD DATA
# ════════════════════════════════════════════════════════════════
section("Loading 4-class v2 data splits")

df_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "train4_v2.parquet"))
df_val   = pd.read_parquet(os.path.join(PROCESSED_DIR, "val4_v2.parquet"))
df_test  = pd.read_parquet(os.path.join(PROCESSED_DIR, "test4_v2.parquet"))

print(f"Train4_v2: {len(df_train):,}  Val4_v2: {len(df_val):,}  Test4_v2: {len(df_test):,}")
print("\nTrain class distribution:")
for i, cls in enumerate(CLASS_NAMES):
    cnt = int((df_train["label"] == i).sum())
    print(f"  {cls:25s}: {cnt:,}")

weight_path = os.path.join(PROCESSED_DIR, "class_weights4_v2.txt")
class_weights_list = [1.0] * NUM_CLASSES
with open(weight_path) as f:
    for line in f:
        if line.startswith("#"):
            continue
        parts = line.strip().split(",")
        if len(parts) == 3:
            class_weights_list[int(parts[0])] = float(parts[2])

weights_tensor = torch.tensor(class_weights_list, dtype=torch.float).to(device)
loss_fn        = torch.nn.CrossEntropyLoss(weight=weights_tensor)
print(f"\nClass weights: {[round(w, 4) for w in class_weights_list]}")

code_train = df_train["code"].values
code_val   = df_val["code"].values
code_test  = df_test["code"].values
y_train    = df_train["label"].values.astype(int)
y_val      = df_val["label"].values.astype(int)
y_test     = df_test["label"].values.astype(int)


# ════════════════════════════════════════════════════════════════
#  LOAD MODEL — continue from existing CodeBERT checkpoint
# ════════════════════════════════════════════════════════════════
section("Loading CodeBERT checkpoint (backbone preserved, head reinitialised)")

tokenizer = RobertaTokenizer.from_pretrained(SOURCE_CHECKPOINT)

model = RobertaForSequenceClassification.from_pretrained(
    SOURCE_CHECKPOINT,
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True,
)
model.config.id2label = {i: c for i, c in enumerate(CLASS_NAMES)}
model.config.label2id = {c: i for i, c in enumerate(CLASS_NAMES)}
model.to(device)

print(f"Source checkpoint : {SOURCE_CHECKPOINT}")
print(f"Save destination  : {SAVE_DIR}")
print(f"Num labels        : {NUM_CLASSES}")


# ════════════════════════════════════════════════════════════════
#  DATA LOADERS
# ════════════════════════════════════════════════════════════════

train_ds = CodeDataset(code_train, y_train, tokenizer, MAX_LENGTH)
val_ds   = CodeDataset(code_val,   y_val,   tokenizer, MAX_LENGTH)
test_ds  = CodeDataset(code_test,  y_test,  tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# ════════════════════════════════════════════════════════════════
#  TRAINING HELPERS
# ════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        ids    = batch["input_ids"].to(device)
        mask   = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        out  = model(input_ids=ids, attention_mask=mask)
        loss = loss_fn(out.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        preds = torch.argmax(out.logits, dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return total_loss / len(loader), correct / total


def evaluate_loader(model, loader):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            ids    = batch["input_ids"].to(device)
            mask   = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            out    = model(input_ids=ids, attention_mask=mask)
            loss   = loss_fn(out.logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(out.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / len(loader), acc, macro_f1, all_preds, all_labels


# ════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ════════════════════════════════════════════════════════════════
section(f"Training — {NUM_EPOCHS} epochs  lr={LEARNING_RATE:.0e}  batch={BATCH_SIZE}")

total_steps  = len(train_loader) * NUM_EPOCHS
warmup_steps = int(total_steps * 0.1)
optimizer    = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

history      = []
best_val_f1  = 0.0
best_epoch   = 1
train_start  = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler)
    val_loss, val_acc, val_f1, _, _ = evaluate_loader(model, val_loader)
    epoch_time = round(time.time() - t0, 1)

    print(
        f"  Epoch {epoch}/{NUM_EPOCHS}  "
        f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
        f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
        f"val_f1={val_f1:.4f}  [{epoch_time}s]"
    )

    history.append({
        "epoch":       epoch,
        "train_loss":  round(train_loss, 4),
        "train_acc":   round(train_acc, 4),
        "val_loss":    round(val_loss, 4),
        "val_acc":     round(val_acc, 4),
        "val_f1":      round(val_f1, 4),
        "epoch_time_s": epoch_time,
    })

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch  = epoch
        model.save_pretrained(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)
        print(f"  ✓ Best checkpoint saved (val_f1={val_f1:.4f})")

total_time = round(time.time() - train_start, 1)
print(f"\nTraining complete: {total_time}s total")


# ════════════════════════════════════════════════════════════════
#  TEST EVALUATION
# ════════════════════════════════════════════════════════════════
section("Test evaluation (best checkpoint)")

best_model = RobertaForSequenceClassification.from_pretrained(
    SAVE_DIR, num_labels=NUM_CLASSES).to(device)
_, test_acc, test_macro_f1, test_preds, test_labels = evaluate_loader(
    best_model, test_loader)

test_weighted_f1 = f1_score(test_labels, test_preds, average="weighted", zero_division=0)
report = classification_report(
    test_labels, test_preds, target_names=CLASS_NAMES, output_dict=True)

print(f"\n  Test accuracy    : {test_acc:.4f}")
print(f"  Test macro F1    : {test_macro_f1:.4f}")
print(f"  Test weighted F1 : {test_weighted_f1:.4f}")
print(f"\n  Per-class F1:")
for cls in CLASS_NAMES:
    print(f"    {cls:25s}: {report[cls]['f1-score']:.4f}")

print(f"\n  Full classification report:")
print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES))

# ── Per-source no_bug recall (domain-shift check, new in v2) ──
test_preds_arr = np.array(test_preds)
nobug_mask     = df_test["label"].values == 3
source_lines   = []
for src in ("leetcode", "flytech"):
    m = nobug_mask & (df_test["source"].values == src)
    if m.sum() > 0:
        recall = (test_preds_arr[m] == 3).mean()
        source_lines.append(f"  no_bug recall ({src:8s}): {recall:.4f}  (n={int(m.sum())})")
print("\n  Per-source no_bug recall (test):")
for line in source_lines:
    print(line)


# ════════════════════════════════════════════════════════════════
#  SAVE RESULTS TEXT
# ════════════════════════════════════════════════════════════════
section("Saving results")

res_path = os.path.join(PROCESSED_DIR, "codebert_4class_v2_results.txt")
with open(res_path, "w", encoding="utf-8") as f:
    f.write("CodeBERT 4-class v2 Retraining Results (expanded no_bug pool)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Source checkpoint : {SOURCE_CHECKPOINT}\n")
    f.write(f"Saved model       : {SAVE_DIR}\n")
    f.write(f"Best epoch        : {best_epoch} / {NUM_EPOCHS}\n")
    f.write(f"Best val F1       : {best_val_f1:.4f}\n")
    f.write(f"Train time        : {total_time}s\n\n")
    f.write(f"Test Results:\n")
    f.write(f"  Accuracy    : {test_acc:.4f}\n")
    f.write(f"  Macro F1    : {test_macro_f1:.4f}\n")
    f.write(f"  Weighted F1 : {test_weighted_f1:.4f}\n\n")
    f.write("Per-class F1:\n")
    for cls in CLASS_NAMES:
        f.write(f"  {cls:25s}: {report[cls]['f1-score']:.4f}\n")
    f.write("\nPer-source no_bug recall (domain-shift check):\n")
    for line in source_lines:
        f.write(line + "\n")
    f.write("\nClassification Report:\n")
    f.write(classification_report(test_labels, test_preds, target_names=CLASS_NAMES))
    f.write("\nTraining History:\n")
    for h in history:
        f.write(f"  epoch={h['epoch']}  val_f1={h['val_f1']:.4f}  "
                f"val_acc={h['val_acc']:.4f}  train_loss={h['train_loss']:.4f}\n")
print(f"[saved] {res_path}")


# ════════════════════════════════════════════════════════════════
#  CHARTS
# ════════════════════════════════════════════════════════════════
section("Generating charts")

epochs     = [h["epoch"] for h in history]
train_loss = [h["train_loss"] for h in history]
val_loss   = [h["val_loss"] for h in history]
val_f1_h   = [h["val_f1"] for h in history]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(epochs, train_loss, "b-o", label="Train loss", linewidth=2)
axes[0].plot(epochs, val_loss,   "r-o", label="Val loss",   linewidth=2)
axes[0].axvline(best_epoch, color="green", linestyle="--", alpha=0.6,
                label=f"Best epoch {best_epoch}")
axes[0].set_title("CodeBERT 4-class v2 — Loss per epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].plot(epochs, val_f1_h, "g-o", linewidth=2, label="Val macro F1")
axes[1].axhline(test_macro_f1, color="red", linestyle="--", alpha=0.7,
                label=f"Test F1 = {test_macro_f1:.4f}")
axes[1].axvline(best_epoch, color="green", linestyle="--", alpha=0.6)
axes[1].set_title("CodeBERT 4-class v2 — Val F1 per epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Macro F1")
axes[1].set_ylim(0, 1.05)
axes[1].legend()

fig.suptitle("CodeBERT 4-class v2 Retraining — Training History",
             fontsize=13, fontweight="bold")
plt.tight_layout()
hist_path = os.path.join(PROCESSED_DIR, "codebert_4class_v2_history.png")
plt.savefig(hist_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[saved] {hist_path}")

cm   = confusion_matrix(test_labels, test_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=["syntax", "logic", "variable", "no_bug"])
fig2, ax2 = plt.subplots(figsize=(7, 6))
disp.plot(ax=ax2, colorbar=True, cmap="Blues")
ax2.set_title(
    f"CodeBERT 4-class v2 — Confusion Matrix (Test)\nMacro F1 = {test_macro_f1:.4f}",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
cm_path = os.path.join(PROCESSED_DIR, "codebert_4class_v2_confusion_matrix.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[saved] {cm_path}")


print(f"""
{'=' * 60}
  CODEBERT 4-CLASS v2 RETRAINING COMPLETE
{'=' * 60}
  New model saved to:
    {SAVE_DIR}

  To activate this model, set in backend/.env:
    BUG_CLASSIFIER_PATH=app/ml_models/codebert_4class_v2_model

  Next: run data_prep_stage2.py (fine-grained 14-class splits)
{'=' * 60}
""")
