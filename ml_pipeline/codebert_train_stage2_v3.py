"""
Phase 5 — Step 5: Stage 2 CodeBERT Training v3 (14 subtypes, grouped split)
==========================================================================
Project : code-analyzer-ai
Author  : Nguyen Ngoc Gia Han (GCS230054)

Retrains the Stage 2 fine-grained (14-class) subtype model on the grouped,
leakage-free splits (train14_v3 / val14_v3 / test14_v3), fine-tuning from the
clean pretrained backbone `microsoft/codebert-base`. Same recipe as
codebert_train_stage2.py otherwise (weighted CE, best checkpoint by val macro-F1,
coarse-from-fine consistency metric, subtype_mapping shipped with the model).

Override the backbone with env CODEBERT_BASE if offline.

HOW TO RUN:
  1. python ml_pipeline/data_prep_grouped_v3.py
  2. python ml_pipeline/codebert_train_stage2_v3.py

Output (backend/app/ml_models/): codebert_stage2_v3_model/  (with subtype_mapping.json)
Output (ml_pipeline/data/processed/): codebert_stage2_v3_results.txt / _history.png / _confusion_matrix.png
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
    AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

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
SAVE_DIR          = os.path.join(MODEL_BASE_DIR, "codebert_stage2_v3_model")
MAPPING_PATH      = os.path.join(PROCESSED_DIR, "subtype_mapping_v3.json")

LEARNING_RATE       = 2e-5
BATCH_SIZE          = 16
NUM_EPOCHS          = 10
EARLY_STOP_PATIENCE = 3     # stop if no val-F1 improvement for this many epochs
MAX_LENGTH          = 256
NUM_CLASSES         = 14
RANDOM_SEED         = 42
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
SUB_TO_COARSE = {i: coarse for coarse, ids in COARSE_GROUPS.items() for i in ids}


def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


class CodeDataset(Dataset):
    def __init__(self, codes, labels, tokenizer, max_length):
        self.codes = [str(c) for c in codes]; self.labels = labels
        self.tokenizer = tokenizer; self.max_length = max_length

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.codes[idx], max_length=self.max_length,
                             padding="max_length", truncation=True, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": torch.tensor(self.labels[idx], dtype=torch.long)}


section("Loading Stage 2 v3 (14-class, grouped) splits")
df_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "train14_v3.parquet"))
df_val   = pd.read_parquet(os.path.join(PROCESSED_DIR, "val14_v3.parquet"))
df_test  = pd.read_parquet(os.path.join(PROCESSED_DIR, "test14_v3.parquet"))
print(f"Train {len(df_train):,}  Val {len(df_val):,}  Test {len(df_test):,}")

class_weights_list = [1.0] * NUM_CLASSES
with open(os.path.join(PROCESSED_DIR, "class_weights14_v3.txt")) as f:
    for line in f:
        if line.startswith("#"):
            continue
        parts = line.strip().split(",")
        if len(parts) == 3:
            class_weights_list[int(parts[0])] = float(parts[2])
weights_tensor = torch.tensor(class_weights_list, dtype=torch.float).to(device)
loss_fn = torch.nn.CrossEntropyLoss(weight=weights_tensor)
print(f"Class weights: {[round(w,2) for w in class_weights_list]}")

section(f"Loading backbone: {SOURCE_CHECKPOINT}  (head -> 14 outputs)")
tokenizer = AutoTokenizer.from_pretrained(SOURCE_CHECKPOINT)
model = AutoModelForSequenceClassification.from_pretrained(
    SOURCE_CHECKPOINT, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True)
model.config.id2label = {i: bt for i, bt in enumerate(SUBTYPE_LIST)}
model.config.label2id = {bt: i for i, bt in enumerate(SUBTYPE_LIST)}
model.to(device)

train_ds = CodeDataset(df_train["code"].values, df_train["sub_label"].values.astype(int), tokenizer, MAX_LENGTH)
val_ds   = CodeDataset(df_val["code"].values,   df_val["sub_label"].values.astype(int),   tokenizer, MAX_LENGTH)
test_ds  = CodeDataset(df_test["code"].values,  df_test["sub_label"].values.astype(int),  tokenizer, MAX_LENGTH)
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
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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


section(f"Training — up to {NUM_EPOCHS} epochs  early-stop patience={EARLY_STOP_PATIENCE}")
total_steps = len(train_loader) * NUM_EPOCHS
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)

history, best_f1, best_epoch, t_start = [], 0.0, 1, time.time()
epochs_no_improve = 0
for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scheduler)
    v_loss, v_acc, v_f1, _, _ = evaluate(model, val_loader)
    print(f"  Epoch {epoch}/{NUM_EPOCHS}  train_loss={tr_loss:.4f} val_acc={v_acc:.4f} val_f1={v_f1:.4f}  [{round(time.time()-t0,1)}s]")
    history.append({"epoch": epoch, "train_loss": round(tr_loss, 4), "val_loss": round(v_loss, 4),
                    "val_acc": round(v_acc, 4), "val_f1": round(v_f1, 4)})
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
shutil.copy2(MAPPING_PATH, os.path.join(SAVE_DIR, "subtype_mapping.json"))
print(f"\nTraining complete: {total_time}s  (subtype_mapping.json copied to model dir)")

section("Test evaluation (best checkpoint)")
best = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR, num_labels=NUM_CLASSES).to(device)
_, test_acc, test_f1, test_preds, test_labels = evaluate(best, test_loader)
test_wf1 = f1_score(test_labels, test_preds, average="weighted", zero_division=0)
report_txt = classification_report(test_labels, test_preds, target_names=HUMAN_NAMES, digits=4, zero_division=0)
pred_coarse = np.array([SUB_TO_COARSE[p] for p in test_preds])
gold_coarse = np.array([SUB_TO_COARSE[l] for l in test_labels])
coarse_from_fine = (pred_coarse == gold_coarse).mean()
print(f"  Test accuracy : {test_acc:.4f}")
print(f"  Test macro F1 : {test_f1:.4f}")
print(f"  Coarse-from-fine acc: {coarse_from_fine:.4f}")
print(report_txt)

res_path = os.path.join(PROCESSED_DIR, "codebert_stage2_v3_results.txt")
with open(res_path, "w", encoding="utf-8") as f:
    f.write("CodeBERT Stage 2 v3 Results (14-class, grouped split, from codebert-base)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Backbone    : {SOURCE_CHECKPOINT}\nSaved model : {SAVE_DIR}\n")
    f.write(f"Best epoch  : {best_epoch}/{NUM_EPOCHS}  (val F1 {best_f1:.4f})\nTrain time  : {total_time}s\n\n")
    f.write(f"Test accuracy        : {test_acc:.4f}\nTest macro F1        : {test_f1:.4f}\n")
    f.write(f"Test weighted F1     : {test_wf1:.4f}\nCoarse-from-fine acc : {coarse_from_fine:.4f}\n\n")
    f.write(report_txt)
    f.write("\nHistory:\n")
    for h in history:
        f.write(f"  epoch={h['epoch']} val_f1={h['val_f1']:.4f} val_acc={h['val_acc']:.4f}\n")
print(f"[saved] {res_path}")

epochs = [h["epoch"] for h in history]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(epochs, [h["train_loss"] for h in history], "b-o", label="Train loss")
axes[0].plot(epochs, [h["val_loss"] for h in history], "r-o", label="Val loss")
axes[0].axvline(best_epoch, color="green", ls="--", alpha=0.6); axes[0].set_title("Stage 2 v3 — Loss")
axes[0].set_xlabel("Epoch"); axes[0].legend()
axes[1].plot(epochs, [h["val_f1"] for h in history], "g-o", label="Val macro F1")
axes[1].axhline(test_f1, color="red", ls="--", alpha=0.7, label=f"Test F1={test_f1:.4f}")
axes[1].set_title("Stage 2 v3 — Val F1"); axes[1].set_xlabel("Epoch"); axes[1].set_ylim(0, 1.05); axes[1].legend()
fig.suptitle("CodeBERT Stage 2 v3 — Training History", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "codebert_stage2_v3_history.png"), dpi=150, bbox_inches="tight")
plt.close()

cm = confusion_matrix(test_labels, test_preds, labels=list(range(NUM_CLASSES)))
fig2, ax2 = plt.subplots(figsize=(12, 11))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=HUMAN_NAMES, yticklabels=HUMAN_NAMES,
            ax=ax2, annot_kws={"fontsize": 8})
ax2.set_xlabel("Predicted"); ax2.set_ylabel("True")
ax2.set_title(f"Stage 2 v3 — Confusion (Test)  Macro F1={test_f1:.4f}", fontweight="bold")
plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", fontsize=8)
plt.setp(ax2.get_yticklabels(), rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "codebert_stage2_v3_confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[saved] codebert_stage2_v3_history.png / _confusion_matrix.png")
print(f"\nSTAGE 2 v3 COMPLETE — model: {SAVE_DIR}")
