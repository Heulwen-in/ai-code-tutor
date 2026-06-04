"""
Phase 2 — Step 1: Data Preparation & 4-Class Mapping
=====================================================
Project : code-analyzer-ai
Author  : Nguyen Ngoc Gia Han (GCS230054)

What this script does:
  1. Loads BuggedPythonLeetCode dataset (primary training data)
  2. Maps 15 bug types → 4 error classes
  3. Drops unmappable / severely imbalanced samples
  4. Cleans and validates the code column
  5. Extracts basic features for EDA confirmation
  6. Saves train / val / test splits ready for Phase 2 model training

Output files (saved to ml_pipeline/data/processed/):
  - train.parquet
  - val.parquet
  - test.parquet
  - label_distribution.png
  - class_mapping_summary.txt

HOW TO RUN:
  pip install pandas pyarrow scikit-learn matplotlib seaborn
  python ml_pipeline/data_prep.py

UPDATE the path below to your BuggedPythonLeetCode parquet file.
"""

import os
import re
import sys
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter

# Force UTF-8 output so that arrows (→) and other unicode chars print
# correctly on Windows terminals that default to cp1252.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── CONFIG ────────────────────────────────────────────────────────────
BUGGED_LC_PATH = r"F:\Heulwen\IT GREENWICH\Final Project\Datasets\BuggedPythonLeetCode Dataset\Train\0000.parquet"
OUTPUT_DIR     = r"F:\Heulwen\IT GREENWICH\Final Project\Tutor_AI_code\ml_pipeline\data\processed"
# ─────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")

# ════════════════════════════════════════════════════════════════════
#  STEP 1 — Load dataset
# ════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  STEP 1 — Loading BuggedPythonLeetCode")
print("=" * 60)

df = pd.read_parquet(BUGGED_LC_PATH)
print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")

# ════════════════════════════════════════════════════════════════════
#  STEP 2 — 4-class mapping
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 2 — Mapping 15 bug types → 4 error classes")
print("=" * 60)

# Mapping dictionary: bug_type (from dataset) → error_class
BUG_TYPE_MAP = {
    # ── Variable misuse ──────────────────────────────────────────
    "ForgettingToUpdateVariableTransformer"         : "variable_misuse",
    "VariableNameTypoTransformer"                   : "variable_misuse",
    "MutableDefaultArgumentTransformer"             : "variable_misuse",
    "IncorrectVariableInitializationTransformer"    : "variable_misuse",
    "UseBeforeDefinitionTransformer"                : "variable_misuse",

    # ── Logic error ──────────────────────────────────────────────
    "ReturningEarlyTransformer"                     : "logic_error",
    "ComparisonSwapTransformer"                     : "logic_error",
    "OffByKIndexTransformer"                        : "logic_error",
    "SwapForTransformer"                            : "logic_error",
    "NonExistingMethodTransformer"                  : "logic_error",
    "InfiniteWhileTransformer"                      : "logic_error",
    "ComparisonTargetTransformer"                   : "logic_error",

    # ── Syntax error ─────────────────────────────────────────────
    "MissingArgumentTransformer"                    : "syntax_error",
    "IncorrectTypeTransformer"                      : "syntax_error",

    # ── DROPPED — only 4 samples, not enough to train on ────────
    # "IncorrectExceptionHandlerTransformer"        : dropped
}

# Label encoding for model training
CLASS_TO_INT = {
    "syntax_error"    : 0,
    "logic_error"     : 1,
    "variable_misuse" : 2,
    "indentation_error": 3,   # reserved for rule-based detector
}
INT_TO_CLASS = {v: k for k, v in CLASS_TO_INT.items()}

# Print original distribution
print("\nOriginal bug_type distribution:")
for bt, cnt in df["bug_type"].value_counts().items():
    mapped = BUG_TYPE_MAP.get(bt, "DROP")
    print(f"  {bt}: {cnt:,}  →  {mapped}")

# Apply mapping
df["error_class"] = df["bug_type"].map(BUG_TYPE_MAP)

# Drop unmapped (IncorrectExceptionHandlerTransformer = 4 samples)
dropped = df["error_class"].isna().sum()
print(f"\nDropped (unmappable): {dropped} samples")
df = df.dropna(subset=["error_class"]).reset_index(drop=True)

# Integer label
df["label"] = df["error_class"].map(CLASS_TO_INT)

print(f"\nFinal class distribution after mapping:")
cls_counts = df["error_class"].value_counts()
for cls, cnt in cls_counts.items():
    pct = cnt / len(df) * 100
    print(f"  {cls:25s}: {cnt:,}  ({pct:.1f}%)")

print(f"\nTotal usable samples: {len(df):,}")

# ════════════════════════════════════════════════════════════════════
#  STEP 3 — Clean the code column
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 3 — Cleaning code (bugged_code column)")
print("=" * 60)

def clean_code(code_str):
    """Normalise escape sequences and strip leading/trailing whitespace."""
    if not isinstance(code_str, str):
        return ""
    # Convert literal \n \t back to real newlines/tabs
    code_str = code_str.replace("\\n", "\n").replace("\\t", "    ")
    return code_str.strip()

def is_valid_python(code_str):
    """Return True if code parses without SyntaxError."""
    try:
        ast.parse(code_str)
        return True
    except SyntaxError:
        return False

df["code"] = df["bugged_code"].apply(clean_code)

# Remove empty
before = len(df)
df = df[df["code"].str.len() > 0].reset_index(drop=True)
print(f"Removed {before - len(df)} empty code rows")

# Validate Python syntax
# NOTE: bugged code is intentionally broken so syntax errors WILL fail parse.
# We only drop rows where the code is completely empty or non-Python garbage.
df["code_len"]   = df["code"].apply(len)
df["line_count"] = df["code"].apply(lambda x: x.count("\n") + 1)
df["is_parseable"] = df["code"].apply(is_valid_python)

parse_rate = df["is_parseable"].mean() * 100
print(f"Parseable (valid Python AST): {parse_rate:.1f}%  "
      f"({df['is_parseable'].sum():,} / {len(df):,})")
print("Note: ~50% expected to fail parse — these ARE the buggy samples.")

# Remove rows that are clearly not Python (len < 10 chars)
df = df[df["code_len"] >= 10].reset_index(drop=True)
print(f"After removing trivial rows: {len(df):,} samples")

# ════════════════════════════════════════════════════════════════════
#  STEP 4 — Feature preview (for confirmation)
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 4 — Feature preview per class")
print("=" * 60)

stats = df.groupby("error_class").agg(
    count       = ("code",       "count"),
    mean_chars  = ("code_len",   "mean"),
    median_chars= ("code_len",   "median"),
    mean_lines  = ("line_count", "mean"),
    median_lines= ("line_count", "median"),
    parseable_pct=("is_parseable","mean"),
).round(2)
stats["parseable_pct"] = (stats["parseable_pct"] * 100).round(1)
print(stats.to_string())

# ════════════════════════════════════════════════════════════════════
#  STEP 5 — Train / val / test split  (70 / 15 / 15, stratified)
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 5 — Stratified train / val / test split (70/15/15)")
print("=" * 60)

# Keep only columns needed downstream
keep_cols = ["code", "error_class", "label", "code_len", "line_count", "is_parseable"]
df_clean  = df[keep_cols].copy()

X = df_clean.drop(columns=["label"])
y = df_clean["label"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

df_train = X_train.copy(); df_train["label"] = y_train.values
df_val   = X_val.copy();   df_val["label"]   = y_val.values
df_test  = X_test.copy();  df_test["label"]  = y_test.values

print(f"Train : {len(df_train):,} rows")
print(f"Val   : {len(df_val):,} rows")
print(f"Test  : {len(df_test):,} rows")

print(f"\nClass balance check (train):")
for cls, cnt in df_train["error_class"].value_counts().items():
    print(f"  {cls:25s}: {cnt:,}  ({cnt/len(df_train)*100:.1f}%)")

# Compute class weights for imbalanced training
class_counts  = df_train["label"].value_counts().sort_index()
total         = len(df_train)
class_weights = {i: total / (len(CLASS_TO_INT) * cnt)
                 for i, cnt in class_counts.items()}
print(f"\nClass weights for imbalanced training:")
for i, w in class_weights.items():
    print(f"  {INT_TO_CLASS[i]:25s} (label {i}): weight = {w:.3f}")

# ════════════════════════════════════════════════════════════════════
#  STEP 6 — Save splits
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 6 — Saving processed splits")
print("=" * 60)

df_train.to_parquet(os.path.join(OUTPUT_DIR, "train.parquet"), index=False)
df_val.to_parquet(os.path.join(OUTPUT_DIR, "val.parquet"),     index=False)
df_test.to_parquet(os.path.join(OUTPUT_DIR, "test.parquet"),   index=False)

print(f"  [saved] {OUTPUT_DIR}/train.parquet  ({len(df_train):,} rows)")
print(f"  [saved] {OUTPUT_DIR}/val.parquet    ({len(df_val):,} rows)")
print(f"  [saved] {OUTPUT_DIR}/test.parquet   ({len(df_test):,} rows)")

# Save class weights for use in training scripts
weights_path = os.path.join(OUTPUT_DIR, "class_weights.txt")
with open(weights_path, "w", encoding="utf-8") as f:
    f.write("# Class weights for imbalanced training\n")
    f.write("# Format: label,class_name,weight\n")
    for i, w in class_weights.items():
        f.write(f"{i},{INT_TO_CLASS[i]},{w:.6f}\n")
print(f"  [saved] {weights_path}")

# Save mapping summary
summary_path = os.path.join(OUTPUT_DIR, "class_mapping_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("4-Class Mapping Summary\n")
    f.write("=" * 40 + "\n\n")
    f.write("INT_TO_CLASS:\n")
    for i, cls in INT_TO_CLASS.items():
        f.write(f"  {i} → {cls}\n")
    f.write("\nBUG_TYPE → ERROR_CLASS:\n")
    for bt, cls in BUG_TYPE_MAP.items():
        f.write(f"  {bt} → {cls}\n")
print(f"  [saved] {summary_path}")

# ════════════════════════════════════════════════════════════════════
#  STEP 7 — Visualisation
# ════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
COLORS = ["#4C72B0","#55A868","#C44E52","#8172B2"]
CLASS_LABELS = ["syntax_error","logic_error","variable_misuse","indentation_error"]

# 1. Final class distribution
counts = [df_train["error_class"].value_counts().get(c, 0) for c in CLASS_LABELS[:3]]
bars = axes[0].bar(["syntax","logic","variable\nmisuse"], counts,
                   color=COLORS[:3], edgecolor="white")
axes[0].set_title("Training set class distribution")
axes[0].set_ylabel("Count")
for bar, v in zip(bars, counts):
    axes[0].text(bar.get_x() + bar.get_width()/2, v + 50,
                 f"{v:,}", ha="center", fontsize=10)

# 2. Code length per class
data_by_class = [df_train[df_train["error_class"]==c]["code_len"].clip(
    upper=df_train["code_len"].quantile(0.95)).values
    for c in CLASS_LABELS[:3]]
bp = axes[1].boxplot(data_by_class, labels=["syntax","logic","variable\nmisuse"],
                     patch_artist=True)
for patch, color in zip(bp["boxes"], COLORS[:3]):
    patch.set_facecolor(color); patch.set_alpha(0.7)
axes[1].set_title("Code length (chars) by class")
axes[1].set_ylabel("Characters")

# 3. Split sizes
split_names = ["Train","Val","Test"]
split_sizes  = [len(df_train), len(df_val), len(df_test)]
axes[2].bar(split_names, split_sizes,
            color=["#4C72B0","#CCB974","#55A868"], edgecolor="white")
axes[2].set_title("Dataset split sizes")
axes[2].set_ylabel("Samples")
for i, (name, v) in enumerate(zip(split_names, split_sizes)):
    axes[2].text(i, v + 50, f"{v:,}", ha="center", fontsize=10)

fig.suptitle("Phase 2 — Data Preparation Summary", fontsize=13, fontweight="bold")
plt.tight_layout()
chart_path = os.path.join(OUTPUT_DIR, "label_distribution.png")
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  [saved] {chart_path}")

print(f"""
{'=' * 60}
  DATA PREP COMPLETE
{'=' * 60}
  Ready for Phase 2 Step 2: Feature Extraction
  Next script: ml_pipeline/feature_extraction.py

  Files to use in training:
    ml_pipeline/data/processed/train.parquet
    ml_pipeline/data/processed/val.parquet
    ml_pipeline/data/processed/test.parquet
    ml_pipeline/data/processed/class_weights.txt

  Class mapping:
    0 → syntax_error
    1 → logic_error
    2 → variable_misuse
    3 → indentation_error  (rule-based, not in training data)
{'=' * 60}
""")