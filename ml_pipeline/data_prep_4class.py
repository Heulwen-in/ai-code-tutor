"""
Phase 3 — Step 1: 4-Class Data Preparation (adds no_bug class)
==============================================================
Project : code-analyzer-ai
Author  : Nguyen Ngoc Gia Han (GCS230054)

Why this script exists:
  The original 3-class model (syntax/logic/variable_misuse) was trained entirely
  on bugged code. It never saw clean code, so it always predicts a bug class even
  for correct submissions. This script adds no_bug examples from the LeetCodeDataset
  (correct Python solutions) as class 3, enabling the model to actually say "no bug".

What this script does:
  1. Loads existing 3-class train/val/test splits
  2. Loads LeetCodeDataset-train.jsonl — extracts clean code from "completion" field
  3. Filters: valid Python AST, length >= 20 chars
  4. Adds a balanced no_bug sample matching ~1/3 of the clean pool per split
  5. Saves new 4-class train4/val4/test4 parquet files

Output (ml_pipeline/data/processed/):
  train4.parquet   — 4-class training set
  val4.parquet     — 4-class validation set
  test4.parquet    — 4-class test set
  class_weights4.txt
  label_distribution4.png

HOW TO RUN:
  python ml_pipeline/data_prep_4class.py
"""

import ast
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── CONFIG ─────────────────────────────────────────────────────────────
PROCESSED_DIR   = r"F:\Heulwen\IT GREENWICH\Final Project\Tutor_AI_code\ml_pipeline\data\processed"
LC_TRAIN_JSONL  = r"F:\Heulwen\IT GREENWICH\Final Project\Datasets\LeetCodeDataset\LeetCodeDataset-train.jsonl"
LC_TEST_JSONL   = r"F:\Heulwen\IT GREENWICH\Final Project\Datasets\LeetCodeDataset\LeetCodeDataset-test.jsonl"
OUTPUT_DIR      = PROCESSED_DIR
# ───────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")

CLASS_TO_INT = {
    "syntax_error":     0,
    "logic_error":      1,
    "variable_misuse":  2,
    "no_bug":           3,
}
INT_TO_CLASS = {v: k for k, v in CLASS_TO_INT.items()}
CLASS_NAMES  = ["syntax_error", "logic_error", "variable_misuse", "no_bug"]

def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ════════════════════════════════════════════════════════════════
#  STEP 1 — Load existing 3-class splits
# ════════════════════════════════════════════════════════════════
section("STEP 1 — Loading existing 3-class splits")

df_train3 = pd.read_parquet(os.path.join(PROCESSED_DIR, "train.parquet"))
df_val3   = pd.read_parquet(os.path.join(PROCESSED_DIR, "val.parquet"))
df_test3  = pd.read_parquet(os.path.join(PROCESSED_DIR, "test.parquet"))

print(f"Train3 : {len(df_train3):,} rows")
print(f"Val3   : {len(df_val3):,} rows")
print(f"Test3  : {len(df_test3):,} rows")
print(f"Columns: {list(df_train3.columns)}")


# ════════════════════════════════════════════════════════════════
#  STEP 2 — Load clean LeetCode completions as no_bug source
# ════════════════════════════════════════════════════════════════
section("STEP 2 — Loading clean LeetCode code (no_bug source)")

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

lc_train_records = load_jsonl(LC_TRAIN_JSONL)
lc_test_records  = load_jsonl(LC_TEST_JSONL)
all_lc = lc_train_records + lc_test_records
print(f"LeetCodeDataset total records: {len(all_lc):,}")

def is_valid_python(code_str):
    try:
        ast.parse(code_str)
        return True
    except Exception:
        return False

def clean_completion(raw):
    """Extract just the function/class code from completion field."""
    if not isinstance(raw, str):
        return ""
    # Strip markdown code fences if present
    code = raw.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        # Remove first and last fence lines
        inner = [l for l in lines if not l.strip().startswith("```")]
        code = "\n".join(inner).strip()
    return code

clean_codes = []
for rec in all_lc:
    raw = rec.get("completion", "") or ""
    code = clean_completion(raw)
    if len(code) >= 20 and is_valid_python(code):
        clean_codes.append(code)

print(f"Valid clean Python completions: {len(clean_codes):,}")

# ════════════════════════════════════════════════════════════════
#  STEP 3 — Balance no_bug sample to match existing class sizes
# ════════════════════════════════════════════════════════════════
section("STEP 3 — Balancing no_bug samples")

# Target: match roughly the per-class count in the existing training data
# Existing 3-class distribution
train3_counts = df_train3["error_class"].value_counts()
print("Existing train class distribution:")
for cls, cnt in train3_counts.items():
    print(f"  {cls:25s}: {cnt:,}")

avg_per_class = int(train3_counts.mean())
print(f"\nAverage samples per existing class: {avg_per_class:,}")

# Use all available clean codes if fewer than target, otherwise cap
n_nobug = min(len(clean_codes), avg_per_class)
print(f"no_bug samples to use: {n_nobug:,} (from {len(clean_codes):,} available)")

# Shuffle and select
rng = np.random.default_rng(42)
selected_codes = rng.choice(clean_codes, size=n_nobug, replace=False).tolist()

# Split no_bug pool into train / val / test (70/15/15)
nb_train_codes, nb_temp = train_test_split(selected_codes, test_size=0.30, random_state=42)
nb_val_codes, nb_test_codes = train_test_split(nb_temp, test_size=0.50, random_state=42)

print(f"\nno_bug split:")
print(f"  Train : {len(nb_train_codes):,}")
print(f"  Val   : {len(nb_val_codes):,}")
print(f"  Test  : {len(nb_test_codes):,}")


# ════════════════════════════════════════════════════════════════
#  STEP 4 — Build 4-class DataFrames
# ════════════════════════════════════════════════════════════════
section("STEP 4 — Building 4-class DataFrames")

def make_nobug_df(codes):
    return pd.DataFrame({
        "code":        codes,
        "error_class": "no_bug",
        "label":       3,
        "code_len":    [len(c) for c in codes],
        "line_count":  [c.count("\n") + 1 for c in codes],
        "is_parseable": True,
    })

# Ensure existing splits have consistent label dtype
for df in [df_train3, df_val3, df_test3]:
    df["label"] = df["label"].astype(int)

df_train4 = pd.concat([df_train3, make_nobug_df(nb_train_codes)], ignore_index=True).sample(
    frac=1, random_state=42).reset_index(drop=True)
df_val4   = pd.concat([df_val3,   make_nobug_df(nb_val_codes)],   ignore_index=True).sample(
    frac=1, random_state=42).reset_index(drop=True)
df_test4  = pd.concat([df_test3,  make_nobug_df(nb_test_codes)],  ignore_index=True).sample(
    frac=1, random_state=42).reset_index(drop=True)

print(f"\n4-class split sizes:")
print(f"  Train4: {len(df_train4):,}")
print(f"  Val4  : {len(df_val4):,}")
print(f"  Test4 : {len(df_test4):,}")

print(f"\nTrain4 class distribution:")
for cls in CLASS_NAMES:
    cnt = (df_train4["error_class"] == cls).sum()
    pct = cnt / len(df_train4) * 100
    print(f"  {cls:25s}: {cnt:,}  ({pct:.1f}%)")


# ════════════════════════════════════════════════════════════════
#  STEP 5 — Compute class weights
# ════════════════════════════════════════════════════════════════
section("STEP 5 — Class weights for imbalanced training")

class_counts  = df_train4["label"].value_counts().sort_index()
total         = len(df_train4)
class_weights = {i: total / (len(CLASS_TO_INT) * cnt) for i, cnt in class_counts.items()}

for i, w in class_weights.items():
    print(f"  {INT_TO_CLASS[i]:25s} (label {i}): weight = {w:.4f}")


# ════════════════════════════════════════════════════════════════
#  STEP 6 — Save
# ════════════════════════════════════════════════════════════════
section("STEP 6 — Saving 4-class splits")

df_train4.to_parquet(os.path.join(OUTPUT_DIR, "train4.parquet"), index=False)
df_val4.to_parquet(os.path.join(OUTPUT_DIR,   "val4.parquet"),   index=False)
df_test4.to_parquet(os.path.join(OUTPUT_DIR,  "test4.parquet"),  index=False)

weights_path = os.path.join(OUTPUT_DIR, "class_weights4.txt")
with open(weights_path, "w", encoding="utf-8") as f:
    f.write("# Class weights for 4-class training (includes no_bug)\n")
    f.write("# Format: label,class_name,weight\n")
    for i, w in class_weights.items():
        f.write(f"{i},{INT_TO_CLASS[i]},{w:.6f}\n")

print(f"  [saved] {OUTPUT_DIR}/train4.parquet  ({len(df_train4):,} rows)")
print(f"  [saved] {OUTPUT_DIR}/val4.parquet    ({len(df_val4):,} rows)")
print(f"  [saved] {OUTPUT_DIR}/test4.parquet   ({len(df_test4):,} rows)")
print(f"  [saved] {weights_path}")


# ════════════════════════════════════════════════════════════════
#  STEP 7 — Visualise
# ════════════════════════════════════════════════════════════════
section("STEP 7 — Visualising class distribution")

COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Train distribution
counts = [int((df_train4["error_class"] == c).sum()) for c in CLASS_NAMES]
short  = ["syntax", "logic", "variable\nmisuse", "no_bug"]
bars = axes[0].bar(short, counts, color=COLORS, edgecolor="white")
axes[0].set_title("4-class training set distribution")
axes[0].set_ylabel("Count")
for bar, v in zip(bars, counts):
    axes[0].text(bar.get_x() + bar.get_width() / 2, v + 30,
                 f"{v:,}", ha="center", fontsize=10)

# Split sizes
splits = ["Train4", "Val4", "Test4"]
sizes  = [len(df_train4), len(df_val4), len(df_test4)]
axes[1].bar(splits, sizes, color=["#4C72B0", "#CCB974", "#55A868"], edgecolor="white")
axes[1].set_title("Dataset split sizes (4-class)")
axes[1].set_ylabel("Samples")
for i, (name, v) in enumerate(zip(splits, sizes)):
    axes[1].text(i, v + 30, f"{v:,}", ha="center", fontsize=10)

fig.suptitle("Phase 3 — 4-Class Data Preparation (adds no_bug)", fontsize=13, fontweight="bold")
plt.tight_layout()
chart_path = os.path.join(OUTPUT_DIR, "label_distribution4.png")
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  [saved] {chart_path}")

print(f"""
{'=' * 60}
  4-CLASS DATA PREP COMPLETE
{'=' * 60}
  New class mapping:
    0 -> syntax_error      (ML model)
    1 -> logic_error       (ML model)
    2 -> variable_misuse   (ML model)
    3 -> no_bug            (NEW — from LeetCode clean code)

  indentation_error stays rule-based (ast.parse), not in ML.

  Next: run codebert_retrain_4class.py
{'=' * 60}
""")
