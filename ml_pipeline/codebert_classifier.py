"""
Phase 2 — Step 4: CodeBERT Fine-Tuning
=======================================
Project : code-analyzer-ai
Author  : Nguyen Ngoc Gia Han (GCS230054)

Fine-tunes microsoft/codebert-base for 3-class Python error classification:
    0 → syntax_error
    1 → logic_error
    2 → variable_misuse

Baseline to beat: SVM (TF-IDF)  macro F1 = 0.9060

HOW TO RUN:
  pip install transformers torch datasets scikit-learn matplotlib seaborn

  # CPU only (slow ~2-3 hrs):
  python ml_pipeline/codebert_classifier.py

  # GPU recommended (10-20 min) — auto-detected if available

Output (ml_pipeline/data/processed/):
  - codebert_results.txt
  - codebert_vs_baseline.png
  - codebert_confusion_matrix.png

Output (backend/app/ml_models/):
  - codebert_model/   ← HuggingFace format, loaded by bug_classifier.py
"""

import os, json, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup
)

warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────────
PROCESSED_DIR = r"F:\Heulwen\IT GREENWICH\Final Project\Tutor_AI_code\ml_pipeline\data\processed"
MODEL_SAVE_DIR = r"F:\Heulwen\IT GREENWICH\Final Project\Tutor_AI_code\backend\app\ml_models\codebert_model"
OUTPUT_DIR     = PROCESSED_DIR

# Training hyperparameters
MAX_LENGTH   = 256     # token limit — covers 95%+ of samples
BATCH_SIZE   = 16      # reduce to 8 if out of memory
NUM_EPOCHS   = 5
LEARNING_RATE = 2e-5
WARMUP_RATIO  = 0.1    # 10% of steps for warm-up
RANDOM_SEED   = 42

NUM_CLASSES  = 3
CLASS_NAMES  = ["syntax_error", "logic_error", "variable_misuse"]
INT_TO_CLASS = {0: "syntax_error", 1: "logic_error", 2: "variable_misuse"}

# Class weights (from data prep step)
CLASS_WEIGHTS = [1.336174, 0.623233, 0.607143]

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Device setup ────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
if device.type == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("NOTE   : Running on CPU — training will take ~2-3 hours.")
    print("         Reduce NUM_EPOCHS to 3 if time is limited.")


# ════════════════════════════════════════════════════════════════
#  DATASET CLASS
# ════════════════════════════════════════════════════════════════

class CodeDataset(Dataset):
    def __init__(self, codes, labels, tokenizer, max_length=256):
        self.codes      = codes
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.codes[idx]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids"     : encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label"         : torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ════════════════════════════════════════════════════════════════
#  LOAD DATA
# ════════════════════════════════════════════════════════════════

def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

section("Loading data")
df_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "train.parquet"))
df_val   = pd.read_parquet(os.path.join(PROCESSED_DIR, "val.parquet"))
df_test  = pd.read_parquet(os.path.join(PROCESSED_DIR, "test.parquet"))

print(f"Train: {len(df_train):,}  Val: {len(df_val):,}  Test: {len(df_test):,}")


# ════════════════════════════════════════════════════════════════
#  TOKENIZER + MODEL
# ════════════════════════════════════════════════════════════════

section("Loading CodeBERT tokenizer and model")
print("Downloading microsoft/codebert-base (this may take a moment)...")

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model     = RobertaForSequenceClassification.from_pretrained(
    "microsoft/codebert-base",
    num_labels=NUM_CLASSES
)
model = model.to(device)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters    : {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")


# ════════════════════════════════════════════════════════════════
#  DATALOADERS
# ════════════════════════════════════════════════════════════════

section("Building dataloaders")

train_dataset = CodeDataset(
    df_train["code"].values, df_train["label"].values, tokenizer, MAX_LENGTH)
val_dataset   = CodeDataset(
    df_val["code"].values,   df_val["label"].values,   tokenizer, MAX_LENGTH)
test_dataset  = CodeDataset(
    df_test["code"].values,  df_test["label"].values,  tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train batches : {len(train_loader)}")
print(f"Val batches   : {len(val_loader)}")
print(f"Max token length: {MAX_LENGTH}")


# ════════════════════════════════════════════════════════════════
#  TRAINING SETUP
# ════════════════════════════════════════════════════════════════

section("Training setup")

total_steps   = len(train_loader) * NUM_EPOCHS
warmup_steps  = int(total_steps * WARMUP_RATIO)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

# Weighted loss to handle class imbalance
weights_tensor = torch.tensor(CLASS_WEIGHTS, dtype=torch.float).to(device)
loss_fn        = torch.nn.CrossEntropyLoss(weight=weights_tensor)

print(f"Epochs         : {NUM_EPOCHS}")
print(f"Total steps    : {total_steps}")
print(f"Warmup steps   : {warmup_steps}")
print(f"Learning rate  : {LEARNING_RATE}")
print(f"Batch size     : {BATCH_SIZE}")
print(f"Class weights  : {CLASS_WEIGHTS}")


# ════════════════════════════════════════════════════════════════
#  TRAINING & EVALUATION FUNCTIONS
# ════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, scheduler, loss_fn, device):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits  = outputs.logits
        loss    = loss_fn(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, macro_f1


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs.logits
            loss    = loss_fn(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, macro_f1, all_preds, all_labels


# ════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ════════════════════════════════════════════════════════════════

section("Training CodeBERT")

history = []
best_val_f1   = 0.0
best_epoch    = 0
train_start   = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_start = time.time()
    print(f"\n  Epoch {epoch}/{NUM_EPOCHS}")

    train_loss, train_f1 = train_epoch(
        model, train_loader, optimizer, scheduler, loss_fn, device)

    val_loss, val_f1, val_preds, val_labels = evaluate(
        model, val_loader, loss_fn, device)

    epoch_time = round(time.time() - epoch_start, 1)
    print(f"  Train loss={train_loss:.4f}  train_f1={train_f1:.4f}")
    print(f"  Val   loss={val_loss:.4f}  val_f1={val_f1:.4f}  [{epoch_time}s]")

    history.append({
        "epoch": epoch,
        "train_loss": train_loss, "train_f1": train_f1,
        "val_loss": val_loss,     "val_f1": val_f1
    })

    # Save best checkpoint
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch  = epoch
        model.save_pretrained(MODEL_SAVE_DIR)
        tokenizer.save_pretrained(MODEL_SAVE_DIR)
        print(f"  ✓ New best — saved to {MODEL_SAVE_DIR}")

total_time = round(time.time() - train_start, 1)
print(f"\nTraining complete in {total_time}s")
print(f"Best val macro F1: {best_val_f1:.4f} (epoch {best_epoch})")


# ════════════════════════════════════════════════════════════════
#  FINAL TEST EVALUATION (best checkpoint)
# ════════════════════════════════════════════════════════════════

section("Final test evaluation (best checkpoint)")

# Reload best model
best_model = RobertaForSequenceClassification.from_pretrained(MODEL_SAVE_DIR)
best_model = best_model.to(device)

test_loss, test_f1, test_preds, test_labels = evaluate(
    best_model, test_loader, loss_fn, device)

test_acc     = accuracy_score(test_labels, test_preds)
test_macro_f1 = f1_score(test_labels, test_preds, average="macro")
test_weighted_f1 = f1_score(test_labels, test_preds, average="weighted")

print(f"\nTest accuracy      : {test_acc:.4f}")
print(f"Test macro F1      : {test_macro_f1:.4f}")
print(f"Test weighted F1   : {test_weighted_f1:.4f}")
print(f"\nFull classification report:")
print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES))

# Baseline comparison
BASELINE_MACRO_F1 = 0.9060  # SVM TF-IDF
improvement = test_macro_f1 - BASELINE_MACRO_F1
print(f"\nBaseline (SVM TF-IDF) macro F1 : {BASELINE_MACRO_F1:.4f}")
print(f"CodeBERT macro F1              : {test_macro_f1:.4f}")
print(f"Improvement                    : {improvement:+.4f}")


# ════════════════════════════════════════════════════════════════
#  SAVE RESULTS
# ════════════════════════════════════════════════════════════════

results = {
    "model": "CodeBERT (microsoft/codebert-base)",
    "best_epoch": best_epoch,
    "best_val_macro_f1": round(best_val_f1, 4),
    "test_accuracy": round(test_acc, 4),
    "test_macro_f1": round(test_macro_f1, 4),
    "test_weighted_f1": round(test_weighted_f1, 4),
    "baseline_macro_f1": BASELINE_MACRO_F1,
    "improvement": round(improvement, 4),
    "hyperparameters": {
        "max_length": MAX_LENGTH, "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS, "lr": LEARNING_RATE
    },
    "training_history": history
}

results_path = os.path.join(OUTPUT_DIR, "codebert_results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[saved] {results_path}")

results_txt = os.path.join(OUTPUT_DIR, "codebert_results.txt")
with open(results_txt, "w") as f:
    f.write("CodeBERT Fine-Tuning Results\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Best epoch        : {best_epoch}\n")
    f.write(f"Test accuracy     : {test_acc:.4f}\n")
    f.write(f"Test macro F1     : {test_macro_f1:.4f}\n")
    f.write(f"Test weighted F1  : {test_weighted_f1:.4f}\n")
    f.write(f"Improvement vs SVM TF-IDF: {improvement:+.4f}\n\n")
    f.write(classification_report(test_labels, test_preds, target_names=CLASS_NAMES))
print(f"[saved] {results_txt}")


# ════════════════════════════════════════════════════════════════
#  VISUALISATIONS
# ════════════════════════════════════════════════════════════════

section("Generating charts")

# 1. Training history
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
epochs = [h["epoch"] for h in history]

axes[0].plot(epochs, [h["train_loss"] for h in history], "b-o", label="Train loss")
axes[0].plot(epochs, [h["val_loss"]   for h in history], "r-o", label="Val loss")
axes[0].axvline(best_epoch, color="green", linestyle="--", alpha=0.5, label=f"Best epoch {best_epoch}")
axes[0].set_title("Training & validation loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].plot(epochs, [h["train_f1"] for h in history], "b-o", label="Train macro F1")
axes[1].plot(epochs, [h["val_f1"]   for h in history], "r-o", label="Val macro F1")
axes[1].axhline(BASELINE_MACRO_F1, color="orange", linestyle="--", alpha=0.7,
                label=f"Best baseline F1={BASELINE_MACRO_F1}")
axes[1].axvline(best_epoch, color="green", linestyle="--", alpha=0.5, label=f"Best epoch {best_epoch}")
axes[1].set_title("Training & validation macro F1")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Macro F1")
axes[1].legend()

fig.suptitle("CodeBERT Fine-Tuning — Training History", fontsize=13, fontweight="bold")
plt.tight_layout()
hist_path = os.path.join(OUTPUT_DIR, "codebert_training_history.png")
plt.savefig(hist_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[saved] {hist_path}")

# 2. Confusion matrix
fig, ax = plt.subplots(figsize=(7, 5))
cm = confusion_matrix(test_labels, test_preds)
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(cm, display_labels=["syntax","logic","variable"])
disp.plot(ax=ax, colorbar=True, cmap="Blues")
ax.set_title(f"CodeBERT — Confusion Matrix (Test Set)\nMacro F1={test_macro_f1:.4f}")
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "codebert_confusion_matrix.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[saved] {cm_path}")

# 3. CodeBERT vs all baselines comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# All models macro F1
all_models = [
    ("DecisionTree\n(structural)", 0.4942, "#CCCCCC"),
    ("LR\n(structural)",           0.5719, "#A9C4E0"),
    ("SVM\n(structural)",          0.5812, "#7AADD4"),
    ("DecisionTree\n(TF-IDF)",     0.5716, "#CCCCCC"),
    ("LR\n(TF-IDF)",               0.8633, "#7DC8A5"),
    ("SVM\n(TF-IDF)",              0.9060, "#55A868"),
    ("CodeBERT",                   test_macro_f1, "#C44E52"),
]
names   = [m[0] for m in all_models]
f1s     = [m[1] for m in all_models]
colors  = [m[2] for m in all_models]

bars = axes[0].bar(range(len(names)), f1s, color=colors, edgecolor="white")
axes[0].set_xticks(range(len(names)))
axes[0].set_xticklabels(names, fontsize=8)
axes[0].set_title("All models — Macro F1 comparison")
axes[0].set_ylabel("Macro F1")
axes[0].set_ylim(0, 1)
axes[0].axhline(BASELINE_MACRO_F1, color="orange", linestyle="--", alpha=0.5, label="Best baseline")
for bar, v in zip(bars, f1s):
    axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.01,
                 f"{v:.3f}", ha="center", fontsize=8)

# Per-class F1 — SVM TF-IDF vs CodeBERT
report = classification_report(test_labels, test_preds, target_names=CLASS_NAMES, output_dict=True)
per_class_cb  = [report[c]["f1-score"] for c in CLASS_NAMES]
per_class_svm = [0.9531, 0.8799, 0.8849]  # from baseline results

x = np.arange(3)
w = 0.35
axes[1].bar(x - w/2, per_class_svm, w, label="SVM (TF-IDF)", color="#55A868", edgecolor="white")
axes[1].bar(x + w/2, per_class_cb,  w, label="CodeBERT",     color="#C44E52", edgecolor="white")
axes[1].set_xticks(x)
axes[1].set_xticklabels(["Syntax","Logic","Variable\nMisuse"])
axes[1].set_title("Per-class F1: SVM (TF-IDF) vs CodeBERT")
axes[1].set_ylabel("F1 Score")
axes[1].set_ylim(0, 1)
axes[1].legend()
for i, (sv, cb) in enumerate(zip(per_class_svm, per_class_cb)):
    axes[1].text(i - w/2, sv + 0.01, f"{sv:.3f}", ha="center", fontsize=8)
    axes[1].text(i + w/2, cb + 0.01, f"{cb:.3f}", ha="center", fontsize=8)

fig.suptitle("CodeBERT vs Baseline — Final Comparison", fontsize=13, fontweight="bold")
plt.tight_layout()
comp_path = os.path.join(OUTPUT_DIR, "codebert_vs_baseline.png")
plt.savefig(comp_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[saved] {comp_path}")

print(f"""
{'=' * 60}
  CODEBERT FINE-TUNING COMPLETE
{'=' * 60}
  Best epoch    : {best_epoch}
  Test macro F1 : {test_macro_f1:.4f}
  vs Baseline   : {improvement:+.4f}

  Model saved to:
    {MODEL_SAVE_DIR}

  Load in bug_classifier.py with:
    from transformers import RobertaTokenizer, RobertaForSequenceClassification
    tokenizer = RobertaTokenizer.from_pretrained("path/to/codebert_model")
    model = RobertaForSequenceClassification.from_pretrained("path/to/codebert_model")

  Next: ml_pipeline/pybughive_evaluation.py
{'=' * 60}
""")