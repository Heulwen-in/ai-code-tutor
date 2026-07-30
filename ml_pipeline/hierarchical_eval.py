"""
Phase 4 — Step 5: Hierarchical Evaluation (Stage 1 -> Stage 2 end-to-end)
=========================================================================
Project : code-analyzer-ai
Author  : Nguyen Ngoc Gia Han (GCS230054)

Why this script exists:
  Stage 1 (4-class) and Stage 2 (14-class) are trained and evaluated
  separately, but in production they run as a pipeline: Stage 1 picks the
  coarse category, then Stage 2's logits are MASKED to the subtypes of that
  category before the subtype argmax. This script measures the pipeline as
  the user experiences it:

    1. Stage 1 coarse metrics on the full v2 test set (buggy + no_bug)
    2. Stage 2 subtype accuracy CONDITIONAL on Stage 1 being correct
       (isolates Stage 2 quality)
    3. END-TO-END subtype accuracy — a wrong coarse prediction counts as a
       wrong subtype (the headline number)
    4. Ablation: masked vs unmasked Stage 2 (quantifies the hierarchy's value)
    5. False-negative rate: buggy code that Stage 1 calls no_bug
       (these never reach Stage 2)

HOW TO RUN (after both models are trained):
  python ml_pipeline/hierarchical_eval.py

Output (ml_pipeline/data/processed/):
  hierarchical_eval_results.txt
  hierarchical_confusion_matrix.png
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
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

STAGE1_DIR   = os.path.join(MODEL_BASE_DIR, "codebert_4class_v2_model")
STAGE2_DIR   = os.path.join(MODEL_BASE_DIR, "codebert_stage2_model")
MAPPING_PATH = os.path.join(STAGE2_DIR, "subtype_mapping.json")

BATCH_SIZE = 32
MAX_LENGTH = 256

COARSE_NAMES = ["syntax_error", "logic_error", "variable_misuse", "no_bug"]
# ───────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice : {device}")
sns.set_theme(style="whitegrid", palette="muted")

with open(MAPPING_PATH, encoding="utf-8") as f:
    MAPPING = json.load(f)

NUM_SUBTYPES  = len(MAPPING["subtype_to_int"])
SUBTYPE_LIST  = [MAPPING["int_to_subtype"][str(i)] for i in range(NUM_SUBTYPES)]
HUMAN_NAMES   = [MAPPING["human_names"][bt] for bt in SUBTYPE_LIST]
COARSE_GROUPS = MAPPING["coarse_groups"]


def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


class CodeDataset(Dataset):
    def __init__(self, codes, tokenizer, max_length):
        self.codes      = [str(c) for c in codes]
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
        }


def predict_logits(model, tokenizer, codes):
    """Run a model over a list of code strings, return the full logits array."""
    ds = CodeDataset(codes, tokenizer, MAX_LENGTH)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model.eval()
    chunks = []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out  = model(input_ids=ids, attention_mask=mask)
            chunks.append(out.logits.cpu().numpy())
    return np.concatenate(chunks, axis=0)


# ════════════════════════════════════════════════════════════════
#  STEP 1 — Load models + test data
# ════════════════════════════════════════════════════════════════
section("STEP 1 — Loading models and test data")

tok1 = RobertaTokenizer.from_pretrained(STAGE1_DIR)
m1   = RobertaForSequenceClassification.from_pretrained(STAGE1_DIR).to(device)
tok2 = RobertaTokenizer.from_pretrained(STAGE2_DIR)
m2   = RobertaForSequenceClassification.from_pretrained(STAGE2_DIR).to(device)
print(f"Stage 1: {STAGE1_DIR}")
print(f"Stage 2: {STAGE2_DIR}")

df_buggy = pd.read_parquet(os.path.join(PROCESSED_DIR, "test14.parquet"))
df_test4 = pd.read_parquet(os.path.join(PROCESSED_DIR, "test4_v2.parquet"))
df_nobug = df_test4[df_test4["label"] == 3].reset_index(drop=True)
print(f"Buggy test samples : {len(df_buggy):,} (with gold subtype)")
print(f"no_bug test samples: {len(df_nobug):,}")


# ════════════════════════════════════════════════════════════════
#  STEP 2 — Stage 1 predictions on the combined test set
# ════════════════════════════════════════════════════════════════
section("STEP 2 — Stage 1 coarse predictions")

all_codes  = df_buggy["code"].tolist() + df_nobug["code"].tolist()
gold_coarse = np.concatenate([
    df_buggy["label"].values.astype(int),      # 0/1/2
    np.full(len(df_nobug), 3, dtype=int),      # no_bug
])

logits1     = predict_logits(m1, tok1, all_codes)
pred_coarse = logits1.argmax(axis=1)

s1_acc = accuracy_score(gold_coarse, pred_coarse)
s1_f1  = f1_score(gold_coarse, pred_coarse, average="macro", zero_division=0)
print(f"Stage 1 accuracy (buggy + no_bug): {s1_acc:.4f}")
print(f"Stage 1 macro F1                 : {s1_f1:.4f}")

# False negatives: buggy code predicted no_bug — never reaches Stage 2
n_buggy   = len(df_buggy)
buggy_pred_coarse = pred_coarse[:n_buggy]
fn_mask   = buggy_pred_coarse == 3
fn_rate   = fn_mask.mean()
print(f"Buggy predicted as no_bug (FN)   : {fn_mask.sum()} / {n_buggy} = {fn_rate:.4f}")


# ════════════════════════════════════════════════════════════════
#  STEP 3 — Stage 2 predictions (masked + unmasked) on buggy samples
# ════════════════════════════════════════════════════════════════
section("STEP 3 — Stage 2 subtype predictions")

gold_sub = df_buggy["sub_label"].values.astype(int)
logits2  = predict_logits(m2, tok2, df_buggy["code"].tolist())

# Unmasked: plain 14-way argmax (ablation baseline)
pred_sub_unmasked = logits2.argmax(axis=1)

# Masked: restrict to the subtypes of Stage 1's predicted coarse class.
# Buggy samples that Stage 1 called no_bug get no subtype (-1).
pred_sub_masked = np.full(n_buggy, -1, dtype=int)
for coarse_name, ids in COARSE_GROUPS.items():
    coarse_id = COARSE_NAMES.index(coarse_name)
    rows = np.where(buggy_pred_coarse == coarse_id)[0]
    if len(rows) == 0:
        continue
    masked = np.full((len(rows), NUM_SUBTYPES), -np.inf)
    masked[:, ids] = logits2[rows][:, ids]
    pred_sub_masked[rows] = masked.argmax(axis=1)


# ════════════════════════════════════════════════════════════════
#  STEP 4 — Metrics
# ════════════════════════════════════════════════════════════════
section("STEP 4 — Hierarchical metrics")

# (a) End-to-end: wrong coarse (incl. FN -> -1) counts as wrong subtype
e2e_acc = (pred_sub_masked == gold_sub).mean()

# (b) Conditional: subtype accuracy where Stage 1 got the coarse class right
cond_mask = buggy_pred_coarse == df_buggy["label"].values.astype(int)
cond_acc  = (pred_sub_masked[cond_mask] == gold_sub[cond_mask]).mean()

# (c) Ablation: unmasked (flat 14-way) accuracy on all buggy samples
flat_acc = (pred_sub_unmasked == gold_sub).mean()

# (d) Stage 1 coarse accuracy on buggy samples only
s1_buggy_acc = cond_mask.mean()

print(f"Stage 1 coarse accuracy (buggy only)      : {s1_buggy_acc:.4f}")
print(f"Stage 2 conditional subtype accuracy      : {cond_acc:.4f}  "
      f"(where Stage 1 coarse was correct, n={int(cond_mask.sum())})")
print(f"END-TO-END subtype accuracy (headline)    : {e2e_acc:.4f}")
print(f"Ablation — flat 14-way (no hierarchy)     : {flat_acc:.4f}")
print(f"Plausibility: s1_buggy_acc * cond_acc = {s1_buggy_acc * cond_acc:.4f} "
      f"~ e2e {e2e_acc:.4f}")

e2e_macro_f1 = f1_score(gold_sub, pred_sub_masked, average="macro",
                        labels=list(range(NUM_SUBTYPES)), zero_division=0)
print(f"End-to-end macro F1 (over 14 subtypes)    : {e2e_macro_f1:.4f}")

report_txt = classification_report(
    gold_sub, pred_sub_masked, labels=list(range(NUM_SUBTYPES)),
    target_names=HUMAN_NAMES, digits=4, zero_division=0)


# ════════════════════════════════════════════════════════════════
#  STEP 5 — Save results
# ════════════════════════════════════════════════════════════════
section("STEP 5 — Saving results")

res_path = os.path.join(PROCESSED_DIR, "hierarchical_eval_results.txt")
with open(res_path, "w", encoding="utf-8") as f:
    f.write("Hierarchical Evaluation — Stage 1 (coarse) -> Stage 2 (subtype)\n")
    f.write("=" * 64 + "\n\n")
    f.write(f"Stage 1 model : {STAGE1_DIR}\n")
    f.write(f"Stage 2 model : {STAGE2_DIR}\n")
    f.write(f"Test data     : test14.parquet ({n_buggy:,} buggy) + "
            f"test4_v2 no_bug ({len(df_nobug):,})\n\n")
    f.write("Stage 1 (combined buggy + no_bug test set):\n")
    f.write(f"  Accuracy : {s1_acc:.4f}\n")
    f.write(f"  Macro F1 : {s1_f1:.4f}\n")
    f.write(f"  Buggy predicted as no_bug (false negatives): "
            f"{int(fn_mask.sum())} / {n_buggy} = {fn_rate:.4f}\n\n")
    f.write("Hierarchical pipeline (buggy samples):\n")
    f.write(f"  Stage 1 coarse accuracy (buggy only)  : {s1_buggy_acc:.4f}\n")
    f.write(f"  Stage 2 conditional subtype accuracy  : {cond_acc:.4f}\n")
    f.write(f"  END-TO-END subtype accuracy           : {e2e_acc:.4f}\n")
    f.write(f"  End-to-end macro F1                   : {e2e_macro_f1:.4f}\n\n")
    f.write("Ablation — value of hierarchical masking:\n")
    f.write(f"  Flat 14-way argmax (no Stage 1 mask)  : {flat_acc:.4f}\n")
    f.write(f"  Masked by Stage 1 coarse prediction   : {e2e_acc:.4f}\n\n")
    f.write("End-to-end classification report (wrong coarse => wrong subtype;\n")
    f.write("samples Stage 1 called no_bug have prediction -1, counted wrong):\n")
    f.write(report_txt)
print(f"[saved] {res_path}")


# ════════════════════════════════════════════════════════════════
#  STEP 6 — End-to-end confusion matrix
# ════════════════════════════════════════════════════════════════
section("STEP 6 — Confusion matrix chart")

# Rows: gold subtype. Columns: predicted subtype + a "no_bug (FN)" column.
labels_ext = list(range(NUM_SUBTYPES)) + [-1]
cm = confusion_matrix(gold_sub, pred_sub_masked, labels=labels_ext)
cm = cm[:NUM_SUBTYPES]  # gold never = -1

col_names = HUMAN_NAMES + ["pred no_bug (FN)"]
fig, ax = plt.subplots(figsize=(13, 11))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
            xticklabels=col_names, yticklabels=HUMAN_NAMES, ax=ax,
            annot_kws={"fontsize": 8})
ax.set_xlabel("Predicted (end-to-end)")
ax.set_ylabel("True subtype")
ax.set_title(
    f"End-to-End Hierarchical Confusion Matrix (Test)\n"
    f"Stage1 -> masked Stage2  |  end-to-end acc = {e2e_acc:.4f}",
    fontsize=12, fontweight="bold")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
plt.tight_layout()
cm_path = os.path.join(PROCESSED_DIR, "hierarchical_confusion_matrix.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[saved] {cm_path}")

print(f"""
{'=' * 60}
  HIERARCHICAL EVALUATION COMPLETE
{'=' * 60}
  Headline numbers for the thesis:
    Stage 1 accuracy (full test) : {s1_acc:.4f}
    End-to-end subtype accuracy  : {e2e_acc:.4f}
    Masked vs flat ablation      : {e2e_acc:.4f} vs {flat_acc:.4f}
{'=' * 60}
""")
