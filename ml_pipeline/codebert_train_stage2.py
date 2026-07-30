"""
Phase 4 — Step 4: Stage 2 CodeBERT Training (14 fine-grained bug types)
=======================================================================
Project : code-analyzer-ai
Author  : Nguyen Ngoc Gia Han (GCS230054)

Why this script exists:
  Stage 1 predicts the coarse bug category (syntax / logic / variable_misuse
  / no_bug). Stage 2 (this model) predicts the SPECIFIC bug type within the
  category — e.g. logic_error -> off_by_one_index. At inference time the
  backend masks Stage 2 logits to the subtypes belonging to Stage 1's
  predicted coarse class (hierarchical classification).

  Training recipe mirrors codebert_retrain_4class.py: continue from the
  3-class CodeBERT checkpoint (its backbone already learned bug-discriminative
  features), reinitialise the head for 14 outputs, weighted cross-entropy
  (14-class imbalance is ~15:1), best checkpoint by validation macro-F1.

HOW TO RUN:
  1. python ml_pipeline/data_prep_stage2.py   (creates train14/val14/test14)
  2. python ml_pipeline/codebert_train_stage2.py

  GPU strongly recommended (~25-40 min on RTX 2070).

Output (backend/app/ml_models/):
  codebert_stage2_model/   (includes subtype_mapping.json for the backend)

Output (ml_pipeline/data/processed/):
  codebert_stage2_results.txt
  codebert_stage2_history.png
  codebert_stage2_confusion_matrix.png
"""

import json
import os
import shutil
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
    accuracy_score, f1_score, classification_report, confusion_matrix,
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

SOURCE_CHECKPOINT = os.path.join(MODEL_BASE_DIR, "codebert_model")   # 3-class fine-tune
SAVE_DIR          = os.path.join(MODEL_BASE_DIR, "codebert_stage2_model")
MAPPING_PATH      = os.path.join(PROCESSED_DIR, "subtype_mapping.json")

LEARNING_RATE  = 2e-5
BATCH_SIZE     = 16
NUM_EPOCHS     = 5
MAX_LENGTH     = 256
NUM_CLASSES    = 14
RANDOM_SEED    = 42
# ───────────────────────────────────────────────────────────────────────

os.makedirs(SAVE_DIR, exist_ok=True)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
sns.set_theme(style="whitegrid", palette="muted")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice : {device}")
if device.type == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

with open(MAPPING_PATH, encoding="utf-8") as f:
    MAPPING = json.load(f)

SUBTYPE_LIST  = [MAPPING["int_to_subtype"][str(i)] for i in range(NUM_CLASSES)]
HUMAN_NAMES   = [MAPPING["human_names"][bt] for bt in SUBTYPE_LIST]
COARSE_GROUPS = MAPPING["coarse_groups"]

# subtype id -> coarse name (for the coarse-from-fine metric)
SUB_TO_COARSE = {}
for coarse, ids in COARSE_GROUPS.items():
    for i in ids:
        SUB_TO_COARSE[i] = coarse


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
section("Loading Stage 2 (14-class) data splits")

df_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "train14.parquet"))
df_val   = pd.read_parquet(os.path.join(PROCESSED_DIR, "val14.parquet"))
df_test  = pd.read_parquet(os.path.join(PROCESSED_DIR, "test14.parquet"))

print(f"Train14: {len(df_train):,}  Val14: {len(df_val):,}  Test14: {len(df_test):,}")

weight_path = os.path.join(PROCESSED_DIR, "class_weights14.txt")
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
print(f"Class weights: {[round(w, 3) for w in class_weights_list]}")

code_train = df_train["code"].values
code_val   = df_val["code"].values
code_test  = df_test["code"].values
y_train    = df_train["sub_label"].values.astype(int)
y_val      = df_val["sub_label"].values.astype(int)
y_test     = df_test["sub_label"].values.astype(int)


# ════════════════════════════════════════════════════════════════
#  LOAD MODEL
# ════════════════════════════════════════════════════════════════
section("Loading CodeBERT checkpoint (backbone preserved, head -> 14 outputs)")

tokenizer = RobertaTokenizer.from_pretrained(SOURCE_CHECKPOINT)
model = RobertaForSequenceClassification.from_pretrained(
    SOURCE_CHECKPOINT,
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True,
)
model.config.id2label = {i: bt for i, bt in enumerate(SUBTYPE_LIST)}
model.config.label2id = {bt: i for i, bt in enumerate(SUBTYPE_LIST)}
model.to(device)

print(f"Source checkpoint : {SOURCE_CHECKPOINT}")
print(f"Save destination  : {SAVE_DIR}")


# ════════════════════════════════════════════════════════════════
#  DATA LOADERS + TRAINING HELPERS
# ════════════════════════════════════════════════════════════════

train_ds = CodeDataset(code_train, y_train, tokenizer, MAX_LENGTH)
val_ds   = CodeDataset(code_val,   y_val,   tokenizer, MAX_LENGTH)
test_ds  = CodeDataset(code_test,  y_test,  tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


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

# Ship the subtype mapping with the model so the backend loads it from one place
shutil.copy2(MAPPING_PATH, os.path.join(SAVE_DIR, "subtype_mapping.json"))
print(f"[copied] subtype_mapping.json -> {SAVE_DIR}")


# ════════════════════════════════════════════════════════════════
#  TEST EVALUATION
# ════════════════════════════════════════════════════════════════
section("Test evaluation (best checkpoint)")

best_model = RobertaForSequenceClassification.from_pretrained(
    SAVE_DIR, num_labels=NUM_CLASSES).to(device)
_, test_acc, test_macro_f1, test_preds, test_labels = evaluate_loader(
    best_model, test_loader)

test_weighted_f1 = f1_score(test_labels, test_preds, average="weighted", zero_division=0)
report_txt = classification_report(
    test_labels, test_preds, target_names=HUMAN_NAMES, digits=4, zero_division=0)

print(f"\n  Test accuracy    : {test_acc:.4f}")
print(f"  Test macro F1    : {test_macro_f1:.4f}")
print(f"  Test weighted F1 : {test_weighted_f1:.4f}")
print(f"\n{report_txt}")

# Coarse-from-fine accuracy: map predicted subtype back to its coarse class.
# High value = subtype confusions stay within the coarse category, which
# justifies the hierarchical masking design.
pred_coarse = np.array([SUB_TO_COARSE[p] for p in test_preds])
gold_coarse = np.array([SUB_TO_COARSE[l] for l in test_labels])
coarse_from_fine_acc = (pred_coarse == gold_coarse).mean()
print(f"  Coarse-from-fine accuracy: {coarse_from_fine_acc:.4f}  "
      f"(subtype prediction mapped back to coarse class)")


# ════════════════════════════════════════════════════════════════
#  SAVE RESULTS TEXT
# ════════════════════════════════════════════════════════════════
section("Saving results")

res_path = os.path.join(PROCESSED_DIR, "codebert_stage2_results.txt")
with open(res_path, "w", encoding="utf-8") as f:
    f.write("CodeBERT Stage 2 Training Results (14 fine-grained bug types)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Source checkpoint : {SOURCE_CHECKPOINT}\n")
    f.write(f"Saved model       : {SAVE_DIR}\n")
    f.write(f"Best epoch        : {best_epoch} / {NUM_EPOCHS}\n")
    f.write(f"Best val F1       : {best_val_f1:.4f}\n")
    f.write(f"Train time        : {total_time}s\n\n")
    f.write(f"Test Results:\n")
    f.write(f"  Accuracy               : {test_acc:.4f}\n")
    f.write(f"  Macro F1               : {test_macro_f1:.4f}\n")
    f.write(f"  Weighted F1            : {test_weighted_f1:.4f}\n")
    f.write(f"  Coarse-from-fine acc   : {coarse_from_fine_acc:.4f}\n")
    f.write("    (predicted subtype mapped back to coarse class — high value\n")
    f.write("     means confusions stay within the coarse category)\n\n")
    f.write("Classification Report (support column shows per-class test size;\n")
    f.write("small classes e.g. wrong_comparison_target have wide CIs):\n")
    f.write(report_txt)
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
axes[0].set_title("CodeBERT Stage 2 — Loss per epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].plot(epochs, val_f1_h, "g-o", linewidth=2, label="Val macro F1")
axes[1].axhline(test_macro_f1, color="red", linestyle="--", alpha=0.7,
                label=f"Test F1 = {test_macro_f1:.4f}")
axes[1].axvline(best_epoch, color="green", linestyle="--", alpha=0.6)
axes[1].set_title("CodeBERT Stage 2 — Val F1 per epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Macro F1")
axes[1].set_ylim(0, 1.05)
axes[1].legend()

fig.suptitle("CodeBERT Stage 2 (14-class) — Training History",
             fontsize=13, fontweight="bold")
plt.tight_layout()
hist_path = os.path.join(PROCESSED_DIR, "codebert_stage2_history.png")
plt.savefig(hist_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[saved] {hist_path}")

# 14x14 confusion matrix — subtype ids are grouped by coarse class, so a
# healthy model shows a block-diagonal structure.
cm = confusion_matrix(test_labels, test_preds, labels=list(range(NUM_CLASSES)))
fig2, ax2 = plt.subplots(figsize=(12, 11))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
            xticklabels=HUMAN_NAMES, yticklabels=HUMAN_NAMES, ax=ax2,
            annot_kws={"fontsize": 8})
ax2.set_xlabel("Predicted subtype")
ax2.set_ylabel("True subtype")
ax2.set_title(
    f"CodeBERT Stage 2 — Confusion Matrix (Test)\n"
    f"Macro F1 = {test_macro_f1:.4f}  |  ids grouped: syntax [0-1], "
    f"logic [2-8], variable [9-13]",
    fontsize=12, fontweight="bold")
plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", fontsize=8)
plt.setp(ax2.get_yticklabels(), rotation=0, fontsize=8)
plt.tight_layout()
cm_path = os.path.join(PROCESSED_DIR, "codebert_stage2_confusion_matrix.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[saved] {cm_path}")


print(f"""
{'=' * 60}
  CODEBERT STAGE 2 TRAINING COMPLETE
{'=' * 60}
  Model saved to:
    {SAVE_DIR}
  (includes subtype_mapping.json for backend inference)

  Next: run hierarchical_eval.py (end-to-end Stage1 -> Stage2)
{'=' * 60}
""")
