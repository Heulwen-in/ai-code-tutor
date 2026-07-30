"""
Phase 5 — Step 6: Line-Detection CodeBERT Training v3 (grouped split)
====================================================================
Project : code-analyzer-ai
Author  : Nguyen Ngoc Gia Han (GCS230054)

Retrains the token-classification line detector on the grouped, leakage-free
line splits (line_train_v3 / line_val_v3 / line_test_v3), fine-tuning from the
clean pretrained backbone `microsoft/codebert-base`. Same objective and metrics
as codebert_train_line_detection.py (per-token buggy/clean, token-F1 +
line_hit@1, best checkpoint by val line_hit@1).

Override the backbone with env CODEBERT_BASE if offline.

HOW TO RUN:
  1. python ml_pipeline/data_prep_grouped_v3.py
  2. python ml_pipeline/codebert_train_line_detection_v3.py

Output (backend/app/ml_models/): codebert_line_detection_v3_model/
Output (ml_pipeline/data/processed/): line_detection_v3_results.txt / _history.png
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
    AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup,
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

SOURCE_CHECKPOINT = "microsoft/codebert-base"
SAVE_DIR          = os.path.join(MODEL_BASE_DIR, "codebert_line_detection_v3_model")

LEARNING_RATE       = 2e-5
BATCH_SIZE          = 16
NUM_EPOCHS          = 10
EARLY_STOP_PATIENCE = 3     # stop if no line_hit@1 improvement for this many epochs
MAX_LENGTH          = 256
RANDOM_SEED         = 42
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


def char_to_line(code):
    mapping, line = [], 1
    for ch in code:
        mapping.append(line)
        if ch == "\n":
            line += 1
    mapping.append(line)
    return mapping


class LineTokenDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.codes = [str(c) for c in df["code"].tolist()]
        self.buggy_lines = [set(int(x) for x in ls) for ls in df["buggy_lines"].tolist()]
        self.tokenizer = tokenizer; self.max_length = max_length

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        code, buggy = self.codes[idx], self.buggy_lines[idx]
        enc = self.tokenizer(code, max_length=self.max_length, padding="max_length",
                             truncation=True, return_offsets_mapping=True, return_tensors="pt")
        offsets = enc["offset_mapping"][0].tolist()
        c2l = char_to_line(code)
        labels = []
        for (start, end) in offsets:
            if start == end:
                labels.append(-100)
            else:
                ln = c2l[start] if start < len(c2l) else c2l[-1]
                labels.append(1 if ln in buggy else 0)
        return {"input_ids": enc["input_ids"][0], "attention_mask": enc["attention_mask"][0],
                "offset_mapping": enc["offset_mapping"][0],
                "labels": torch.tensor(labels, dtype=torch.long), "idx": idx}


section("Loading line-detection v3 splits")
df_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "line_train_v3.parquet"))
df_val   = pd.read_parquet(os.path.join(PROCESSED_DIR, "line_val_v3.parquet"))
df_test  = pd.read_parquet(os.path.join(PROCESSED_DIR, "line_test_v3.parquet"))
print(f"Train {len(df_train):,}  Val {len(df_val):,}  Test {len(df_test):,}")

tokenizer = AutoTokenizer.from_pretrained(SOURCE_CHECKPOINT)
train_ds = LineTokenDataset(df_train, tokenizer, MAX_LENGTH)
val_ds   = LineTokenDataset(df_val,   tokenizer, MAX_LENGTH)
test_ds  = LineTokenDataset(df_test,  tokenizer, MAX_LENGTH)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

n_bug, n_clean = 0, 0
for i in range(min(len(train_ds), 501)):
    lbl = train_ds[i]["labels"]
    n_bug += int((lbl == 1).sum()); n_clean += int((lbl == 0).sum())
ratio = max(1.0, n_clean / max(1, n_bug))
class_weights = torch.tensor([1.0, min(ratio, 10.0)], dtype=torch.float).to(device)
print(f"Token balance (sampled): clean={n_clean:,} buggy={n_bug:,} -> buggy weight {class_weights[1].item():.2f}")
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

section(f"Loading backbone: {SOURCE_CHECKPOINT}  (fresh token-classification head)")
model = AutoModelForTokenClassification.from_pretrained(
    SOURCE_CHECKPOINT, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID,
    ignore_mismatched_sizes=True).to(device)


def line_hit_at_1(probs_bug, offsets, code, gold):
    c2l = char_to_line(code)
    scores = {}
    for p, (s, e) in zip(probs_bug, offsets):
        if s == e:
            continue
        ln = c2l[s] if s < len(c2l) else c2l[-1]
        scores.setdefault(ln, []).append(p)
    if not scores:
        return None
    top = max(scores, key=lambda k: np.mean(scores[k]))
    return 1 if top in gold else 0


def evaluate(model, loader, dataset, df):
    model.eval()
    total, tok_preds, tok_labs, hits, scorable = 0.0, [], [], 0, 0
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device); mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            total += loss_fn(logits.view(-1, 2), labels.view(-1)).item()
            probs = torch.softmax(logits, dim=-1)[:, :, 1].cpu().numpy()
            preds = logits.argmax(-1).cpu().numpy(); labs = labels.cpu().numpy()
            offs = batch["offset_mapping"].numpy(); idxs = batch["idx"].numpy()
            for b in range(ids.size(0)):
                for t in range(labs.shape[1]):
                    if labs[b, t] != -100:
                        tok_preds.append(preds[b, t]); tok_labs.append(labs[b, t])
                gold = set(int(x) for x in df.iloc[int(idxs[b])]["buggy_lines"])
                h = line_hit_at_1(probs[b], offs[b].tolist(), dataset.codes[int(idxs[b])], gold)
                if h is not None:
                    hits += h; scorable += 1
    p, r, f1, _ = precision_recall_fscore_support(tok_labs, tok_preds, labels=[1], average="binary", zero_division=0)
    return total / len(loader), f1, (hits / scorable if scorable else 0.0), p, r


section(f"Training — up to {NUM_EPOCHS} epochs  early-stop patience={EARLY_STOP_PATIENCE}")
total_steps = len(train_loader) * NUM_EPOCHS
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)

history, best_hit, best_epoch, t_start = [], 0.0, 1, time.time()
epochs_no_improve = 0
for epoch in range(1, NUM_EPOCHS + 1):
    model.train(); t0, running = time.time(), 0.0
    for batch in train_loader:
        ids = batch["input_ids"].to(device); mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        logits = model(input_ids=ids, attention_mask=mask).logits
        loss = loss_fn(logits.view(-1, 2), labels.view(-1))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        running += loss.item()
    tr_loss = running / len(train_loader)
    v_loss, v_f1, v_hit, v_p, v_r = evaluate(model, val_loader, val_ds, df_val)
    print(f"  Epoch {epoch}/{NUM_EPOCHS}  train_loss={tr_loss:.4f} token_f1={v_f1:.4f} line_hit@1={v_hit:.4f}  [{round(time.time()-t0,1)}s]")
    history.append({"epoch": epoch, "train_loss": round(tr_loss, 4), "val_loss": round(v_loss, 4),
                    "token_f1": round(v_f1, 4), "line_hit": round(v_hit, 4)})
    if v_hit > best_hit:
        best_hit, best_epoch = v_hit, epoch
        epochs_no_improve = 0
        model.save_pretrained(SAVE_DIR); tokenizer.save_pretrained(SAVE_DIR)
        print(f"    * best checkpoint saved (line_hit@1={v_hit:.4f})")
    else:
        epochs_no_improve += 1
        print(f"    no line_hit@1 improvement ({epochs_no_improve}/{EARLY_STOP_PATIENCE})")
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping at epoch {epoch} — best epoch {best_epoch} (line_hit@1={best_hit:.4f})")
            break
print(f"\nTraining complete: {round(time.time()-t_start,1)}s  (best epoch {best_epoch})")

section("Test evaluation (best checkpoint)")
best = AutoModelForTokenClassification.from_pretrained(SAVE_DIR).to(device)
test_loss, test_f1, test_hit, test_p, test_r = evaluate(best, test_loader, test_ds, df_test)
print(f"  token precision {test_p:.4f}  recall {test_r:.4f}  F1 {test_f1:.4f}  line_hit@1 {test_hit:.4f}")

res_path = os.path.join(PROCESSED_DIR, "line_detection_v3_results.txt")
with open(res_path, "w", encoding="utf-8") as f:
    f.write("CodeBERT Line-Detection v3 (grouped split, from codebert-base)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Backbone    : {SOURCE_CHECKPOINT}\nSaved model : {SAVE_DIR}\n")
    f.write(f"Best epoch  : {best_epoch}/{NUM_EPOCHS} (val line_hit@1={best_hit:.4f})\n\n")
    f.write(f"Test token precision : {test_p:.4f}\nTest token recall    : {test_r:.4f}\n")
    f.write(f"Test token F1        : {test_f1:.4f}\nTest line hit@1      : {test_hit:.4f}\n\n")
    f.write("History:\n")
    for h in history:
        f.write(f"  epoch={h['epoch']} token_f1={h['token_f1']:.4f} line_hit@1={h['line_hit']:.4f}\n")
print(f"[saved] {res_path}")

epochs = [h["epoch"] for h in history]
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].plot(epochs, [h["train_loss"] for h in history], "b-o", label="train loss")
ax[0].plot(epochs, [h["val_loss"] for h in history], "r-o", label="val loss")
ax[0].set_title("Line Detection v3 — Loss"); ax[0].set_xlabel("Epoch"); ax[0].legend()
ax[1].plot(epochs, [h["token_f1"] for h in history], "m-o", label="token F1")
ax[1].plot(epochs, [h["line_hit"] for h in history], "g-o", label="line hit@1")
ax[1].axhline(test_hit, color="green", ls="--", alpha=0.6, label=f"test hit@1={test_hit:.3f}")
ax[1].set_title("Line Detection v3 — Metrics"); ax[1].set_xlabel("Epoch"); ax[1].set_ylim(0, 1.05); ax[1].legend()
fig.suptitle("CodeBERT Line Detection v3 — Training History", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "line_detection_v3_history.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[saved] line_detection_v3_history.png")
print(f"\nLINE DETECTION v3 COMPLETE — model: {SAVE_DIR}")
