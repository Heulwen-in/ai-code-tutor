"""
Phase 4 — Line Detection Step 2: CodeBERT Token-Classification Training
======================================================================
Project : code-analyzer-ai
Author  : Nguyen Ngoc Gia Han (GCS230054)

Why this script exists:
  Stage 1 / Stage 2 do whole-sequence classification and cannot localise a bug
  to a line. This trains a SEPARATE token-classification model: for every token
  in the (bugged) code it predicts buggy (1) or clean (0). At inference the
  per-token buggy probabilities are aggregated per line and the top line is
  reported as the bug location.

  Backbone: the existing local CodeBERT checkpoint (codebert_model) — encoder
  weights are reused, a fresh 2-way token-classification head is initialised.
  This runs fully offline (no HuggingFace download).

Labels come from data_prep_line_detection.py (difflib-derived buggy lines).
Token labels are assigned by mapping each token's character offset to a line
number; special/padding tokens get -100 (ignored by the loss).

Metrics:
  token_f1     — F1 on the buggy token class (label 1)
  line_hit@1   — fraction of samples where the single highest-scoring line is
                 one of the gold buggy lines (the product-facing metric)
Best checkpoint is chosen by val line_hit@1.

HOW TO RUN:
  1. python ml_pipeline/data_prep_line_detection.py
  2. python ml_pipeline/codebert_train_line_detection.py

Output (backend/app/ml_models/):
  codebert_line_detection_model/

Output (ml_pipeline/data/processed/):
  line_detection_results.txt
  line_detection_history.png
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
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import precision_recall_fscore_support

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")

# ── CONFIG ─────────────────────────────────────────────────────────────
PROCESSED_DIR  = r"F:\Heulwen\IT GREENWICH\Final Project\Tutor_AI_code\ml_pipeline\data\processed"
MODEL_BASE_DIR = r"F:\Heulwen\IT GREENWICH\Final Project\Tutor_AI_code\backend\app\ml_models"

SOURCE_CHECKPOINT = os.path.join(MODEL_BASE_DIR, "codebert_model")
SAVE_DIR          = os.path.join(MODEL_BASE_DIR, "codebert_line_detection_model")

LEARNING_RATE = 2e-5
BATCH_SIZE    = 16
NUM_EPOCHS    = 4
MAX_LENGTH    = 256
RANDOM_SEED   = 42

ID2LABEL = {0: "O", 1: "BUG"}
LABEL2ID = {"O": 0, "BUG": 1}
# ───────────────────────────────────────────────────────────────────────

os.makedirs(SAVE_DIR, exist_ok=True)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
sns.set_theme(style="whitegrid", palette="muted")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice : {device}")
if device.type == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")


def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def char_to_line(code: str) -> list[int]:
    """Map each character index -> 1-indexed line number."""
    mapping = []
    line = 1
    for ch in code:
        mapping.append(line)
        if ch == "\n":
            line += 1
    mapping.append(line)  # for offset == len(code)
    return mapping


# ════════════════════════════════════════════════════════════════
#  DATASET — token labels via offset mapping
# ════════════════════════════════════════════════════════════════

class LineTokenDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.codes       = [str(c) for c in df["code"].tolist()]
        self.buggy_lines = [set(int(x) for x in ls) for ls in df["buggy_lines"].tolist()]
        self.tokenizer   = tokenizer
        self.max_length  = max_length

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        code  = self.codes[idx]
        buggy = self.buggy_lines[idx]
        enc = self.tokenizer(
            code,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = enc["offset_mapping"][0].tolist()
        c2l = char_to_line(code)

        labels = []
        for (start, end) in offsets:
            if start == end:            # special token / padding
                labels.append(-100)
            else:
                ln = c2l[start] if start < len(c2l) else c2l[-1]
                labels.append(1 if ln in buggy else 0)

        return {
            "input_ids":      enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "offset_mapping": enc["offset_mapping"][0],
            "labels":         torch.tensor(labels, dtype=torch.long),
            "idx":            idx,
        }


# ════════════════════════════════════════════════════════════════
#  LOAD DATA
# ════════════════════════════════════════════════════════════════
section("Loading line-detection splits")

df_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "line_train.parquet"))
df_val   = pd.read_parquet(os.path.join(PROCESSED_DIR, "line_val.parquet"))
df_test  = pd.read_parquet(os.path.join(PROCESSED_DIR, "line_test.parquet"))
print(f"Train: {len(df_train):,}  Val: {len(df_val):,}  Test: {len(df_test):,}")

tokenizer = AutoTokenizer.from_pretrained(SOURCE_CHECKPOINT)

train_ds = LineTokenDataset(df_train, tokenizer, MAX_LENGTH)
val_ds   = LineTokenDataset(df_val,   tokenizer, MAX_LENGTH)
test_ds  = LineTokenDataset(df_test,  tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Class weights from token label counts (buggy tokens are the minority)
n_bug, n_clean = 0, 0
for i in range(len(train_ds)):
    lbl = train_ds[i]["labels"]
    n_bug   += int((lbl == 1).sum())
    n_clean += int((lbl == 0).sum())
    if i >= 500:   # sample 500 for a fast estimate
        break
ratio = max(1.0, n_clean / max(1, n_bug))
class_weights = torch.tensor([1.0, min(ratio, 10.0)], dtype=torch.float).to(device)
print(f"Token balance (sampled): clean={n_clean:,} buggy={n_bug:,}  "
      f"-> buggy weight = {class_weights[1].item():.2f}")

loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)


# ════════════════════════════════════════════════════════════════
#  MODEL — token classification head on CodeBERT backbone
# ════════════════════════════════════════════════════════════════
section("Loading CodeBERT backbone (fresh token-classification head)")

model = AutoModelForTokenClassification.from_pretrained(
    SOURCE_CHECKPOINT,
    num_labels=2,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    ignore_mismatched_sizes=True,
)
model.to(device)
print(f"Source : {SOURCE_CHECKPOINT}")
print(f"Save   : {SAVE_DIR}")


# ════════════════════════════════════════════════════════════════
#  METRICS
# ════════════════════════════════════════════════════════════════

def line_hit_at_1(probs_bug, offsets, code, gold_lines) -> int | None:
    """Aggregate token buggy-probs per line (mean); return 1 if the top line is
    in gold_lines, 0 otherwise, None if no scorable tokens."""
    c2l = char_to_line(code)
    line_scores: dict[int, list[float]] = {}
    for p, (start, end) in zip(probs_bug, offsets):
        if start == end:
            continue
        ln = c2l[start] if start < len(c2l) else c2l[-1]
        line_scores.setdefault(ln, []).append(p)
    if not line_scores:
        return None
    top_line = max(line_scores, key=lambda k: np.mean(line_scores[k]))
    return 1 if top_line in gold_lines else 0


def evaluate(model, loader, dataset, df):
    model.eval()
    total_loss = 0.0
    all_tok_preds, all_tok_labels = [], []
    hits, scorable = 0, 0
    with torch.no_grad():
        for batch in loader:
            ids    = batch["input_ids"].to(device)
            mask   = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            loss   = loss_fn(logits.view(-1, 2), labels.view(-1))
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=-1)[:, :, 1].cpu().numpy()
            preds = logits.argmax(-1).cpu().numpy()
            labs  = labels.cpu().numpy()
            offs  = batch["offset_mapping"].numpy()
            idxs  = batch["idx"].numpy()

            for b in range(ids.size(0)):
                # token metrics (ignore -100)
                for t in range(labs.shape[1]):
                    if labs[b, t] != -100:
                        all_tok_preds.append(preds[b, t])
                        all_tok_labels.append(labs[b, t])
                # line hit@1
                gold = set(int(x) for x in df.iloc[int(idxs[b])]["buggy_lines"])
                code = dataset.codes[int(idxs[b])]
                h = line_hit_at_1(probs[b], offs[b].tolist(), code, gold)
                if h is not None:
                    hits += h
                    scorable += 1

    p, r, f1, _ = precision_recall_fscore_support(
        all_tok_labels, all_tok_preds, labels=[1],
        average="binary", zero_division=0)
    line_hit = hits / scorable if scorable else 0.0
    return total_loss / len(loader), f1, line_hit, p, r


# ════════════════════════════════════════════════════════════════
#  TRAIN
# ════════════════════════════════════════════════════════════════
section(f"Training — {NUM_EPOCHS} epochs  lr={LEARNING_RATE:.0e}  batch={BATCH_SIZE}")

total_steps  = len(train_loader) * NUM_EPOCHS
optimizer    = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler    = get_linear_schedule_with_warmup(
    optimizer, int(total_steps * 0.1), total_steps)

history       = []
best_line_hit = 0.0
best_epoch    = 1
train_start   = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    t0, running = time.time(), 0.0
    for batch in train_loader:
        ids    = batch["input_ids"].to(device)
        mask   = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        logits = model(input_ids=ids, attention_mask=mask).logits
        loss   = loss_fn(logits.view(-1, 2), labels.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        running += loss.item()
    train_loss = running / len(train_loader)

    val_loss, val_f1, val_hit, val_p, val_r = evaluate(model, val_loader, val_ds, df_val)
    dt = round(time.time() - t0, 1)
    print(f"  Epoch {epoch}/{NUM_EPOCHS}  train_loss={train_loss:.4f}  "
          f"val_loss={val_loss:.4f}  token_f1={val_f1:.4f}  "
          f"line_hit@1={val_hit:.4f}  [{dt}s]")

    history.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                    "val_loss": round(val_loss, 4), "token_f1": round(val_f1, 4),
                    "line_hit": round(val_hit, 4)})

    if val_hit > best_line_hit:
        best_line_hit = val_hit
        best_epoch = epoch
        model.save_pretrained(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)
        print(f"    * best checkpoint saved (line_hit@1={val_hit:.4f})")

print(f"\nTraining complete: {round(time.time() - train_start, 1)}s")


# ════════════════════════════════════════════════════════════════
#  TEST
# ════════════════════════════════════════════════════════════════
section("Test evaluation (best checkpoint)")

best = AutoModelForTokenClassification.from_pretrained(SAVE_DIR).to(device)
test_loss, test_f1, test_hit, test_p, test_r = evaluate(best, test_loader, test_ds, df_test)
print(f"  token precision : {test_p:.4f}")
print(f"  token recall    : {test_r:.4f}")
print(f"  token F1        : {test_f1:.4f}")
print(f"  line hit@1      : {test_hit:.4f}")

res_path = os.path.join(PROCESSED_DIR, "line_detection_results.txt")
with open(res_path, "w", encoding="utf-8") as f:
    f.write("CodeBERT Line-Detection (token classification) Results\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Backbone     : {SOURCE_CHECKPOINT}\n")
    f.write(f"Saved model  : {SAVE_DIR}\n")
    f.write(f"Best epoch   : {best_epoch}/{NUM_EPOCHS}  (val line_hit@1={best_line_hit:.4f})\n\n")
    f.write("Test:\n")
    f.write(f"  token precision : {test_p:.4f}\n")
    f.write(f"  token recall    : {test_r:.4f}\n")
    f.write(f"  token F1        : {test_f1:.4f}\n")
    f.write(f"  line hit@1      : {test_hit:.4f}\n\n")
    f.write("History:\n")
    for h in history:
        f.write(f"  epoch={h['epoch']}  token_f1={h['token_f1']:.4f}  "
                f"line_hit@1={h['line_hit']:.4f}  val_loss={h['val_loss']:.4f}\n")
print(f"\n[saved] {res_path}")

# History chart
epochs = [h["epoch"] for h in history]
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].plot(epochs, [h["train_loss"] for h in history], "b-o", label="train loss")
ax[0].plot(epochs, [h["val_loss"] for h in history], "r-o", label="val loss")
ax[0].set_title("Line Detection — Loss"); ax[0].set_xlabel("Epoch"); ax[0].legend()
ax[1].plot(epochs, [h["token_f1"] for h in history], "m-o", label="token F1")
ax[1].plot(epochs, [h["line_hit"] for h in history], "g-o", label="line hit@1")
ax[1].axhline(test_hit, color="green", ls="--", alpha=0.6, label=f"test hit@1={test_hit:.3f}")
ax[1].set_title("Line Detection — Metrics"); ax[1].set_xlabel("Epoch")
ax[1].set_ylim(0, 1.05); ax[1].legend()
fig.suptitle("CodeBERT Line Detection — Training History", fontsize=13, fontweight="bold")
plt.tight_layout()
chart = os.path.join(PROCESSED_DIR, "line_detection_history.png")
plt.savefig(chart, dpi=150, bbox_inches="tight"); plt.close()
print(f"[saved] {chart}")

print(f"""
{'=' * 60}
  LINE DETECTION TRAINING COMPLETE
{'=' * 60}
  Model saved to: {SAVE_DIR}
  Test line hit@1: {test_hit:.4f}

  To activate, set in backend/.env:
    BUG_LINE_MODEL_PATH=app/ml_models/codebert_line_detection_model
    ENABLE_LINE_DETECTION=true
{'=' * 60}
""")
