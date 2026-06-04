"""
Phase 2 — Step 5: Transformer Fine-Tuning

Fine-tunes 3 transformer models for 3-class Python error classification:
  - BERT       (bert-base-uncased)         — general language baseline
  - RoBERTa    (roberta-base)              — improved BERT pretraining
  - CodeBERT   (microsoft/codebert-base)  — code + language specialist

Pipeline:
  Step 1 → Hyperparameter sweep (LR × batch_size, 1 epoch each)
  Step 2 → Fine-tune all 3 models with best hyperparams (5 epochs)
  Step 3 → Evaluate + compare vs best baseline (SVM TF-IDF F1=0.9278)

HOW TO RUN:
  pip install transformers torch accelerate scikit-learn
  python ml_pipeline/transformer_classifier.py

  Verify GPU before running (activate .venv first):
  python -c "import torch; print('cuda:', torch.cuda.is_available());
  print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only — install CUDA torch')"

  If cuda is False, reinstall PyTorch with CUDA (RTX 2070 — use cu124):
  pip uninstall -y torch torchvision torchaudio
  pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
  (torch 2.11.0 is not published for cu124; use cu126 index if you need 2.11+)

Output (ml_pipeline/data/processed/):
  transformer_results.txt
  transformer_training_history.png
  transformer_comparison.png
  transformer_confusion_matrices.png

Output (backend/app/ml_models/):
  bert_model/
  roberta_model/
  codebert_model/
"""

import os, json, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizer,        BertForSequenceClassification,
    RobertaTokenizer,     RobertaForSequenceClassification,
    get_linear_schedule_with_warmup
)
warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────────
PROCESSED_DIR  = r"F:\Heulwen\IT GREENWICH\Final Project\Tutor_AI_code\ml_pipeline\data\processed"
LOG_DIR        = r"F:\Heulwen\IT GREENWICH\Final Project\Tutor_AI_code\ml_pipeline\logs"
MODEL_BASE_DIR = r"F:\Heulwen\IT GREENWICH\Final Project\Tutor_AI_code\backend\app\ml_models"
OUTPUT_DIR     = PROCESSED_DIR

# Training settings
NUM_EPOCHS      = 5
MAX_LENGTH      = 256
NUM_CLASSES     = 3
RANDOM_SEED     = 42
CLASS_WEIGHTS   = [1.336174, 0.623233, 0.607143]  # from data_prep.py
BASELINE_F1     = 0.9278   # SVM (TF-IDF) — target to beat

# Hyperparameter sweep candidates
SWEEP_LRS    = [1e-5, 2e-5, 3e-5]
SWEEP_BATCH  = [8, 16]

CLASS_NAMES  = ["syntax_error", "logic_error", "variable_misuse"]

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_BASE_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Device ──────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice : {device}")
if device.type == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM   : {vram:.1f} GB")
    if vram < 6:
        print("NOTE   : Low VRAM — reducing batch size to 8 and max_length to 128")
        SWEEP_BATCH = [4, 8]
        MAX_LENGTH  = 128
else:
    print("WARNING: No GPU found — training will be very slow (2-4 hrs per model)")
    print("         Consider using Google Colab with GPU runtime instead")


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
            return_tensors="pt"
        )
        return {
            "input_ids"     : enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label"         : torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ════════════════════════════════════════════════════════════════
#  LOAD DATA
# ════════════════════════════════════════════════════════════════

section("Loading data")
df_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "train.parquet"))
df_val   = pd.read_parquet(os.path.join(PROCESSED_DIR, "val.parquet"))
df_test  = pd.read_parquet(os.path.join(PROCESSED_DIR, "test.parquet"))

code_train  = df_train["code"].values
code_val    = df_val["code"].values
code_test   = df_test["code"].values
y_train     = df_train["label"].values
y_val       = df_val["label"].values
y_test      = df_test["label"].values

print(f"Train: {len(y_train):,}  Val: {len(y_val):,}  Test: {len(y_test):,}")

weights_tensor = torch.tensor(CLASS_WEIGHTS, dtype=torch.float).to(device)
loss_fn        = torch.nn.CrossEntropyLoss(weight=weights_tensor)


# ════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════

def make_loaders(tokenizer, max_length, batch_size):
    train_ds = CodeDataset(code_train, y_train, tokenizer, max_length)
    val_ds   = CodeDataset(code_val,   y_val,   tokenizer, max_length)
    test_ds  = CodeDataset(code_test,  y_test,  tokenizer, max_length)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0),
    )


def train_one_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in loader:
        ids   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        out   = model(input_ids=ids, attention_mask=mask)
        loss  = loss_fn(out.logits, labels)
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
    total_loss, all_preds, all_labels = 0, [], []
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
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return total_loss / len(loader), acc, macro_f1, all_preds, all_labels


def save_model_log(model_name, history, test_labels, test_preds,
                   best_params, best_epoch):
    report = classification_report(test_labels, test_preds,
                                   target_names=CLASS_NAMES)
    cm     = confusion_matrix(test_labels, test_preds)
    log_path = os.path.join(LOG_DIR, f"transformer_{model_name.replace('/','_')}.txt")
    with open(log_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Best hyperparameters:\n")
        f.write(f"  learning_rate : {best_params['lr']}\n")
        f.write(f"  batch_size    : {best_params['batch_size']}\n")
        f.write(f"  best_epoch    : {best_epoch}\n\n")
        f.write("Training history (per epoch):\n")
        f.write(f"{'Epoch':>6}  {'TrainLoss':>10}  {'TrainAcc':>9}  "
                f"{'ValLoss':>8}  {'ValAcc':>7}  {'ValF1':>7}\n")
        for h in history:
            f.write(f"{h['epoch']:>6}  {h['train_loss']:>10.4f}  "
                    f"{h['train_acc']:>9.4f}  {h['val_loss']:>8.4f}  "
                    f"{h['val_acc']:>7.4f}  {h['val_f1']:>7.4f}\n")
        f.write(f"\nTest results:\n")
        f.write(f"  Accuracy   : {accuracy_score(test_labels, test_preds):.4f}\n")
        f.write(f"  Macro F1   : {f1_score(test_labels, test_preds, average='macro'):.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        for row in cm:
            f.write(f"  {list(row)}\n")
    print(f"  [log] {log_path}")
    return log_path


# ════════════════════════════════════════════════════════════════
#  MODEL DEFINITIONS
# ════════════════════════════════════════════════════════════════

MODELS = {
    "BERT": {
        "hf_name"   : "bert-base-uncased",
        "tokenizer_cls": BertTokenizer,
        "model_cls" : BertForSequenceClassification,
        "save_dir"  : os.path.join(MODEL_BASE_DIR, "bert_model"),
        "description": "General English language model (Google, 2018)"
    },
    "RoBERTa": {
        "hf_name"   : "roberta-base",
        "tokenizer_cls": RobertaTokenizer,
        "model_cls" : RobertaForSequenceClassification,
        "save_dir"  : os.path.join(MODEL_BASE_DIR, "roberta_model"),
        "description": "Improved BERT with better pretraining (Facebook, 2019)"
    },
    "CodeBERT": {
        "hf_name"   : "microsoft/codebert-base",
        "tokenizer_cls": RobertaTokenizer,
        "model_cls" : RobertaForSequenceClassification,
        "save_dir"  : os.path.join(MODEL_BASE_DIR, "codebert_model"),
        "description": "Code + language bimodal model (Microsoft, 2020)"
    },
}

all_results   = {}
all_histories = {}


# ════════════════════════════════════════════════════════════════
#  STEP 1 — HYPERPARAMETER SWEEP
# ════════════════════════════════════════════════════════════════

section("Step 1 — Hyperparameter Sweep (1 epoch per combination)")
print(f"  Models   : {list(MODELS.keys())}")
print(f"  LR values: {SWEEP_LRS}")
print(f"  Batches  : {SWEEP_BATCH}")
print(f"  Combinations per model: {len(SWEEP_LRS) * len(SWEEP_BATCH)}")

sweep_results = {}

for model_name, cfg in MODELS.items():
    print(f"\n  Sweeping: {model_name} ({cfg['hf_name']})")
    best_val_f1 = 0
    best_params = {"lr": 2e-5, "batch_size": 16}

    tokenizer = cfg["tokenizer_cls"].from_pretrained(cfg["hf_name"])

    for lr in SWEEP_LRS:
        for bs in SWEEP_BATCH:
            model = cfg["model_cls"].from_pretrained(
                cfg["hf_name"], num_labels=NUM_CLASSES).to(device)

            train_loader, val_loader, _ = make_loaders(tokenizer, MAX_LENGTH, bs)
            total_steps  = len(train_loader)
            warmup_steps = max(1, int(total_steps * 0.1))
            optimizer    = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
            scheduler    = get_linear_schedule_with_warmup(
                optimizer, warmup_steps, total_steps)

            # Train 1 epoch
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, scheduler)
            val_loss, val_acc, val_f1, _, _ = evaluate_loader(model, val_loader)

            print(f"    lr={lr:.0e}  bs={bs:2d}  "
                  f"train_loss={train_loss:.3f}  val_f1={val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_params = {"lr": lr, "batch_size": bs}

            del model
            torch.cuda.empty_cache() if device.type == "cuda" else None

    sweep_results[model_name] = best_params
    print(f"  ★ Best for {model_name}: lr={best_params['lr']:.0e}  "
          f"batch={best_params['batch_size']}  val_f1={best_val_f1:.4f}")

# Save sweep results
sweep_path = os.path.join(LOG_DIR, "transformer_sweep_results.json")
with open(sweep_path, "w") as f:
    json.dump({k: {kk: str(vv) for kk, vv in v.items()}
               for k, v in sweep_results.items()}, f, indent=2)
print(f"\n[saved] {sweep_path}")


# ════════════════════════════════════════════════════════════════
#  STEP 2 — FULL TRAINING (5 EPOCHS WITH BEST PARAMS)
# ════════════════════════════════════════════════════════════════

section("Step 2 — Full Training (5 epochs, best hyperparams)")

for model_name, cfg in MODELS.items():
    section(f"Training: {model_name}")
    print(f"  Model      : {cfg['hf_name']}")
    print(f"  Description: {cfg['description']}")

    best_params  = sweep_results[model_name]
    lr           = best_params["lr"]
    batch_size   = best_params["batch_size"]

    print(f"  LR={lr:.0e}  batch={batch_size}  epochs={NUM_EPOCHS}")
    os.makedirs(cfg["save_dir"], exist_ok=True)

    tokenizer    = cfg["tokenizer_cls"].from_pretrained(cfg["hf_name"])
    model        = cfg["model_cls"].from_pretrained(
        cfg["hf_name"], num_labels=NUM_CLASSES).to(device)

    train_loader, val_loader, test_loader = make_loaders(
        tokenizer, MAX_LENGTH, batch_size)

    total_steps  = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * 0.1)
    optimizer    = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps)

    history      = []
    best_val_f1  = 0
    best_epoch   = 1
    train_start  = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler)
        val_loss, val_acc, val_f1, _, _ = evaluate_loader(model, val_loader)
        epoch_time = round(time.time() - t0, 1)

        print(f"  Epoch {epoch}/{NUM_EPOCHS}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
              f"val_f1={val_f1:.4f}  [{epoch_time}s]")

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc" : round(train_acc, 4),
            "val_loss"  : round(val_loss, 4),
            "val_acc"   : round(val_acc, 4),
            "val_f1"    : round(val_f1, 4),
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            model.save_pretrained(cfg["save_dir"])
            tokenizer.save_pretrained(cfg["save_dir"])
            print(f"  ✓ Best checkpoint saved (epoch {epoch})")

    total_time = round(time.time() - train_start, 1)
    print(f"\n  Training complete: {total_time}s total")

    # Test evaluation with best checkpoint
    best_model = cfg["model_cls"].from_pretrained(
        cfg["save_dir"], num_labels=NUM_CLASSES).to(device)
    _, test_acc, test_macro_f1, test_preds, test_labels = evaluate_loader(
        best_model, test_loader)
    test_weighted_f1 = f1_score(test_labels, test_preds, average="weighted")
    report = classification_report(test_labels, test_preds,
                                   target_names=CLASS_NAMES, output_dict=True)

    print(f"\n  Test results (best checkpoint epoch {best_epoch}):")
    print(f"    Accuracy    : {test_acc:.4f}")
    print(f"    Macro F1    : {test_macro_f1:.4f}")
    print(f"    vs Baseline : {test_macro_f1 - BASELINE_F1:+.4f}")
    print(f"    Syntax F1   : {report['syntax_error']['f1-score']:.4f}")
    print(f"    Logic F1    : {report['logic_error']['f1-score']:.4f}")
    print(f"    Variable F1 : {report['variable_misuse']['f1-score']:.4f}")

    all_results[model_name] = {
        "hf_name"       : cfg["hf_name"],
        "description"   : cfg["description"],
        "best_epoch"    : best_epoch,
        "best_val_f1"   : round(best_val_f1, 4),
        "test_acc"      : round(test_acc, 4),
        "test_macro_f1" : round(test_macro_f1, 4),
        "test_weighted_f1": round(test_weighted_f1, 4),
        "syntax_f1"     : round(report["syntax_error"]["f1-score"], 4),
        "logic_f1"      : round(report["logic_error"]["f1-score"], 4),
        "variable_f1"   : round(report["variable_misuse"]["f1-score"], 4),
        "vs_baseline"   : round(test_macro_f1 - BASELINE_F1, 4),
        "train_time_s"  : total_time,
        "best_params"   : best_params,
        "y_pred"        : test_preds,
        "y_true"        : test_labels,
    }
    all_histories[model_name] = history

    save_model_log(model_name, history, test_labels, test_preds,
                   best_params, best_epoch)

    del model, best_model
    torch.cuda.empty_cache() if device.type == "cuda" else None


# ════════════════════════════════════════════════════════════════
#  SAVE RESULTS
# ════════════════════════════════════════════════════════════════

section("Saving results")

# Text summary
res_path = os.path.join(OUTPUT_DIR, "transformer_results.txt")
with open(res_path, "w") as f:
    f.write("Transformer Fine-Tuning Results\n")
    f.write("(BERT | RoBERTa | CodeBERT)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Best baseline (SVM TF-IDF): macro F1 = {BASELINE_F1}\n\n")
    for name, r in all_results.items():
        f.write(f"Model: {name} ({r['hf_name']})\n")
        f.write(f"  {r['description']}\n")
        f.write(f"  Best params  : lr={r['best_params']['lr']:.0e}  "
                f"batch={r['best_params']['batch_size']}\n")
        f.write(f"  Best epoch   : {r['best_epoch']}\n")
        f.write(f"  Test accuracy: {r['test_acc']:.4f}\n")
        f.write(f"  Macro F1     : {r['test_macro_f1']:.4f}  "
                f"(vs baseline: {r['vs_baseline']:+.4f})\n")
        f.write(f"  Syntax F1    : {r['syntax_f1']:.4f}\n")
        f.write(f"  Logic F1     : {r['logic_f1']:.4f}\n")
        f.write(f"  Variable F1  : {r['variable_f1']:.4f}\n\n")
print(f"[saved] {res_path}")

# JSON
json_path = os.path.join(LOG_DIR, "transformer_results.json")
safe = {k: {kk: vv for kk, vv in v.items() if kk not in ["y_pred","y_true"]}
        for k, v in all_results.items()}
with open(json_path, "w") as f:
    json.dump(safe, f, indent=2)
print(f"[saved] {json_path}")


# ════════════════════════════════════════════════════════════════
#  VISUALISATIONS
# ════════════════════════════════════════════════════════════════

section("Generating charts")
COLORS_M = {"BERT": "#4C72B0", "RoBERTa": "#CCB974", "CodeBERT": "#C44E52"}

# 1. Training history (loss + accuracy per epoch, 2 rows × 3 cols)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for col, (mname, hist) in enumerate(all_histories.items()):
    epochs      = [h["epoch"] for h in hist]
    train_loss  = [h["train_loss"] for h in hist]
    val_loss    = [h["val_loss"]   for h in hist]
    train_acc   = [h["train_acc"]  for h in hist]
    val_acc     = [h["val_acc"]    for h in hist]
    best_ep     = all_results[mname]["best_epoch"]

    # Loss
    axes[0][col].plot(epochs, train_loss, "b-o", label="Train loss", linewidth=2)
    axes[0][col].plot(epochs, val_loss,   "r-o", label="Val loss",   linewidth=2)
    axes[0][col].axvline(best_ep, color="green", linestyle="--", alpha=0.6,
                         label=f"Best epoch {best_ep}")
    axes[0][col].set_title(f"{mname}\nTrain / Val Loss", fontsize=11, fontweight="bold")
    axes[0][col].set_xlabel("Epoch")
    axes[0][col].set_ylabel("Loss")
    axes[0][col].legend(fontsize=8)

    # Accuracy
    axes[1][col].plot(epochs, train_acc, "b-o", label="Train acc", linewidth=2)
    axes[1][col].plot(epochs, val_acc,   "r-o", label="Val acc",   linewidth=2)
    axes[1][col].axhline(all_results[mname]["test_acc"], color="green",
                         linestyle="--", alpha=0.7,
                         label=f"Test acc={all_results[mname]['test_acc']:.4f}")
    axes[1][col].axvline(best_ep, color="green", linestyle="--", alpha=0.4)
    axes[1][col].set_title(f"{mname}\nTrain / Val Accuracy", fontsize=11, fontweight="bold")
    axes[1][col].set_xlabel("Epoch")
    axes[1][col].set_ylabel("Accuracy")
    axes[1][col].set_ylim(0, 1.05)
    axes[1][col].legend(fontsize=8)

fig.suptitle("Transformer Models — Training History (Loss & Accuracy per Epoch)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
hist_path = os.path.join(OUTPUT_DIR, "transformer_training_history.png")
plt.savefig(hist_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[saved] {hist_path}")

# 2. Full model comparison (all baselines + all transformers)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# All models macro F1
baseline_models = [
    ("DT\n(struct)", 0.5319, "#CCCCCC"),
    ("RF\n(struct)", 0.5698, "#BBBBBB"),
    ("LR\n(struct)", 0.5715, "#AAAAAA"),
    ("SVM\n(struct)",0.5812, "#999999"),
    ("XGB\n(struct)",0.5969, "#888888"),
    ("RF\n(TF-IDF)", 0.8128, "#A8D4B8"),
    ("DT\n(TF-IDF)", 0.8130, "#90CCA4"),
    ("XGB\n(TF-IDF)",0.8996, "#70BE8C"),
    ("LR\n(TF-IDF)", 0.9023, "#55A868"),
    ("SVM\n(TF-IDF)",0.9278, "#2E8B57"),
]
tf_models = [(f"{n}", r["test_macro_f1"], COLORS_M[n])
             for n, r in all_results.items()]

all_models = baseline_models + tf_models
names_all  = [m[0] for m in all_models]
f1_all     = [m[1] for m in all_models]
cols_all   = [m[2] for m in all_models]

bars = axes[0].bar(range(len(names_all)), f1_all, color=cols_all, edgecolor="white")
axes[0].set_xticks(range(len(names_all)))
axes[0].set_xticklabels(names_all, fontsize=8)
axes[0].set_title("All Models — Macro F1 Comparison", fontsize=12)
axes[0].set_ylabel("Macro F1")
axes[0].set_ylim(0, 1.05)
axes[0].axhline(BASELINE_F1, color="orange", linestyle="--",
                alpha=0.7, label=f"Best baseline ({BASELINE_F1})")
axes[0].legend(fontsize=9)
# Add SVM TF-IDF divider line
axes[0].axvline(9.5, color="red", linestyle=":", alpha=0.4)
axes[0].text(9.6, 0.3, "Transformers →", fontsize=8, color="red", alpha=0.6)
for bar, v in zip(bars, f1_all):
    axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.005,
                 f"{v:.3f}", ha="center", fontsize=7, rotation=90)

# Per-class F1 comparison (SVM TF-IDF vs all transformers)
svm_per_class   = [0.9708, 0.9026, 0.9099]
model_names_pc  = ["SVM\n(TF-IDF)"] + list(all_results.keys())
per_class_data  = [svm_per_class] + [
    [r["syntax_f1"], r["logic_f1"], r["variable_f1"]]
    for r in all_results.values()
]
x   = np.arange(3)
w   = 0.2
bar_colors = ["#2E8B57"] + list(COLORS_M.values())
for i, (mname, vals, col) in enumerate(zip(model_names_pc, per_class_data, bar_colors)):
    offset = (i - len(model_names_pc)/2 + 0.5) * w
    b = axes[1].bar(x + offset, vals, w, label=mname, color=col, edgecolor="white")

axes[1].set_xticks(x)
axes[1].set_xticklabels(["Syntax", "Logic", "Variable\nMisuse"])
axes[1].set_title("Per-class F1: SVM (TF-IDF) vs Transformers", fontsize=12)
axes[1].set_ylabel("F1 Score")
axes[1].set_ylim(0, 1.1)
axes[1].legend(fontsize=9)

fig.suptitle("Full Model Comparison — Baselines vs Transformers",
             fontsize=13, fontweight="bold")
plt.tight_layout()
comp_path = os.path.join(OUTPUT_DIR, "transformer_comparison.png")
plt.savefig(comp_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[saved] {comp_path}")

# 3. Transformer confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for i, (mname, r) in enumerate(all_results.items()):
    cm   = confusion_matrix(r["y_true"], r["y_pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["syntax","logic","variable"])
    disp.plot(ax=axes[i], colorbar=False, cmap="Blues")
    axes[i].set_title(f"{mname}\nMacro F1={r['test_macro_f1']:.4f}",
                      fontsize=11, fontweight="bold")

fig.suptitle("Transformer Models — Confusion Matrices (Test Set)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "transformer_confusion_matrices.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[saved] {cm_path}")


# ════════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ════════════════════════════════════════════════════════════════

section("Final Summary — All Transformers vs Best Baseline")

print(f"\n{'Model':<12} {'Macro F1':>10} {'Accuracy':>10} "
      f"{'vs Baseline':>12} {'Best Epoch':>11}")
print("-" * 58)
print(f"{'SVM(TF-IDF)':<12} {BASELINE_F1:>10.4f}  {'0.9183':>9}  "
      f"{'(baseline)':>11}  {'N/A':>10}")
for mname, r in all_results.items():
    print(f"{mname:<12} {r['test_macro_f1']:>10.4f}  {r['test_acc']:>9.4f}  "
          f"{r['vs_baseline']:>+11.4f}  {r['best_epoch']:>10}")

best_transformer = max(all_results.items(), key=lambda x: x[1]["test_macro_f1"])
print(f"\n★  Best transformer: {best_transformer[0]}  "
      f"macro_f1={best_transformer[1]['test_macro_f1']:.4f}")

print(f"""
{'=' * 60}
  TRANSFORMER FINE-TUNING COMPLETE
{'=' * 60}
  Saved charts:
    transformer_training_history.png
    transformer_comparison.png
    transformer_confusion_matrices.png

  Saved logs (ml_pipeline/logs/):
    transformer_BERT.txt
    transformer_RoBERTa.txt
    transformer_CodeBERT.txt

  Saved models (backend/app/ml_models/):
    bert_model/
    roberta_model/
    codebert_model/

  Load in bug_classifier.py:
    from transformers import RobertaTokenizer, RobertaForSequenceClassification
    model = RobertaForSequenceClassification.from_pretrained("path/to/codebert_model")
{'=' * 60}
""")