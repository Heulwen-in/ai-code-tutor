"""
Phase 4 — Step 3: Stage 2 Data Preparation (14 fine-grained bug types)
======================================================================
Project : code-analyzer-ai
Author  : Nguyen Ngoc Gia Han (GCS230054)

Why this script exists:
  Stage 1 classifies code into 3 coarse bug categories + no_bug. The original
  BuggedPythonLeetCode dataset has 15 fine-grained bug types ("Transformers")
  which data_prep.py combined into the 3 coarse classes — and then DROPPED the
  bug_type column before saving. Stage 2 needs a model that predicts the
  specific bug type WITHIN the coarse category, so this script re-derives the
  fine-grained labels.

  CRITICAL: the train/val/test split here must be row-for-row identical to
  the existing Stage 1 split, otherwise Stage 2 test samples could appear in
  Stage 1 training (leakage). data_prep.py's split is deterministic (same
  input order, random_state=42, stratified on the coarse label), so we
  reproduce the exact same processing pipeline and then HARD-ASSERT that the
  resulting code multisets match the saved train/val/test parquets.

Output (ml_pipeline/data/processed/):
  train14.parquet / val14.parquet / test14.parquet
  subtype_mapping.json     — single source of truth for subtype label ids
  class_weights14.txt
  label_distribution14.png

HOW TO RUN:
  python ml_pipeline/data_prep_stage2.py
"""

import ast
import json
import os
import sys
from collections import Counter

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
BUGGED_LC_PATH = r"F:\Heulwen\IT GREENWICH\Final Project\Datasets\BuggedPythonLeetCode Dataset\Train\0000.parquet"
PROCESSED_DIR  = r"F:\Heulwen\IT GREENWICH\Final Project\Tutor_AI_code\ml_pipeline\data\processed"
OUTPUT_DIR     = PROCESSED_DIR
RANDOM_SEED    = 42
# ───────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")

# Same mapping as data_prep.py — must stay in sync
BUG_TYPE_MAP = {
    "ForgettingToUpdateVariableTransformer"      : "variable_misuse",
    "VariableNameTypoTransformer"                : "variable_misuse",
    "MutableDefaultArgumentTransformer"          : "variable_misuse",
    "IncorrectVariableInitializationTransformer" : "variable_misuse",
    "UseBeforeDefinitionTransformer"             : "variable_misuse",

    "ReturningEarlyTransformer"                  : "logic_error",
    "ComparisonSwapTransformer"                  : "logic_error",
    "OffByKIndexTransformer"                     : "logic_error",
    "SwapForTransformer"                         : "logic_error",
    "NonExistingMethodTransformer"               : "logic_error",
    "InfiniteWhileTransformer"                   : "logic_error",
    "ComparisonTargetTransformer"                : "logic_error",

    "MissingArgumentTransformer"                 : "syntax_error",
    "IncorrectTypeTransformer"                   : "syntax_error",
    # IncorrectExceptionHandlerTransformer — dropped (4 samples)
}

CLASS_TO_INT = {
    "syntax_error"     : 0,
    "logic_error"      : 1,
    "variable_misuse"  : 2,
    "indentation_error": 3,   # reserved, rule-based — same as data_prep.py
}

# Subtype ids: grouped by coarse class (syntax, logic, variable), alphabetical
# within each group. This ordering makes the 14x14 confusion matrix block-
# diagonal by coarse category.
COARSE_ORDER = ["syntax_error", "logic_error", "variable_misuse"]
SUBTYPES_BY_COARSE = {
    coarse: sorted(bt for bt, c in BUG_TYPE_MAP.items() if c == coarse)
    for coarse in COARSE_ORDER
}
SUBTYPE_LIST   = [bt for coarse in COARSE_ORDER for bt in SUBTYPES_BY_COARSE[coarse]]
SUBTYPE_TO_INT = {bt: i for i, bt in enumerate(SUBTYPE_LIST)}
COARSE_GROUPS  = {
    coarse: [SUBTYPE_TO_INT[bt] for bt in SUBTYPES_BY_COARSE[coarse]]
    for coarse in COARSE_ORDER
}

# Short readable names for plots, backend responses, and feedback prompts
HUMAN_NAMES = {
    "MissingArgumentTransformer"                 : "missing_argument",
    "IncorrectTypeTransformer"                   : "incorrect_type",
    "ComparisonSwapTransformer"                  : "swapped_comparison_operands",
    "ComparisonTargetTransformer"                : "wrong_comparison_target",
    "InfiniteWhileTransformer"                   : "infinite_while_loop",
    "NonExistingMethodTransformer"               : "non_existing_method",
    "OffByKIndexTransformer"                     : "off_by_one_index",
    "ReturningEarlyTransformer"                  : "returning_early",
    "SwapForTransformer"                         : "swapped_for_range",
    "ForgettingToUpdateVariableTransformer"      : "forgotten_variable_update",
    "IncorrectVariableInitializationTransformer" : "incorrect_initialization",
    "MutableDefaultArgumentTransformer"          : "mutable_default_argument",
    "UseBeforeDefinitionTransformer"             : "use_before_definition",
    "VariableNameTypoTransformer"                : "variable_name_typo",
}


def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ════════════════════════════════════════════════════════════════
#  STEP 1 — Reproduce data_prep.py's pipeline (keeping bug_type)
# ════════════════════════════════════════════════════════════════
section("STEP 1 — Loading + reprocessing BuggedPythonLeetCode")

df = pd.read_parquet(BUGGED_LC_PATH)
print(f"Loaded: {df.shape[0]:,} rows")

# Identical order of operations to data_prep.py STEP 2-3
df["error_class"] = df["bug_type"].map(BUG_TYPE_MAP)
dropped = df["error_class"].isna().sum()
df = df.dropna(subset=["error_class"]).reset_index(drop=True)
print(f"Dropped unmappable: {dropped}")

df["label"] = df["error_class"].map(CLASS_TO_INT)

def clean_code(code_str):
    if not isinstance(code_str, str):
        return ""
    code_str = code_str.replace("\\n", "\n").replace("\\t", "    ")
    return code_str.strip()

def is_valid_python(code_str):
    try:
        ast.parse(code_str)
        return True
    except SyntaxError:
        return False

df["code"] = df["bugged_code"].apply(clean_code)
df = df[df["code"].str.len() > 0].reset_index(drop=True)
df["code_len"]     = df["code"].apply(len)
df["line_count"]   = df["code"].apply(lambda x: x.count("\n") + 1)
df["is_parseable"] = df["code"].apply(is_valid_python)
df = df[df["code_len"] >= 10].reset_index(drop=True)
print(f"After cleaning: {len(df):,} samples")

# Fine-grained label
df["sub_label"] = df["bug_type"].map(SUBTYPE_TO_INT)


# ════════════════════════════════════════════════════════════════
#  STEP 2 — Identical split call (coarse-stratified, seed 42)
# ════════════════════════════════════════════════════════════════
section("STEP 2 — Reproducing the Stage 1 split")

# Same call shape as data_prep.py STEP 5, with bug_type/sub_label carried along
keep_cols = ["code", "error_class", "label", "code_len", "line_count",
             "is_parseable", "bug_type", "sub_label"]
df_clean = df[keep_cols].copy()

X = df_clean.drop(columns=["label"])
y = df_clean["label"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

df_train = X_train.copy(); df_train["label"] = y_train.values
df_val   = X_val.copy();   df_val["label"]   = y_val.values
df_test  = X_test.copy();  df_test["label"]  = y_test.values

print(f"Train: {len(df_train):,}  Val: {len(df_val):,}  Test: {len(df_test):,}")


# ════════════════════════════════════════════════════════════════
#  STEP 3 — HARD ASSERTION: split matches saved Stage 1 splits
# ════════════════════════════════════════════════════════════════
section("STEP 3 — Verifying split alignment with Stage 1 (leakage guard)")

for name, df_new in [("train", df_train), ("val", df_val), ("test", df_test)]:
    df_old = pd.read_parquet(os.path.join(PROCESSED_DIR, f"{name}.parquet"))
    new_codes = sorted(df_new["code"].tolist())
    old_codes = sorted(df_old["code"].tolist())
    assert new_codes == old_codes, (
        f"SPLIT MISMATCH in '{name}': reproduced split does not equal the saved "
        f"Stage 1 split ({len(new_codes)} vs {len(old_codes)} rows). "
        f"Stage 2 test data could leak into Stage 1 training — ABORTING."
    )
    print(f"  {name:5s}: {len(new_codes):,} rows — multiset of code matches ✓")

print("Alignment verified: Stage 2 splits are row-identical to Stage 1 splits.")


# ════════════════════════════════════════════════════════════════
#  STEP 4 — Per-type counts + class weights (14-class)
# ════════════════════════════════════════════════════════════════
section("STEP 4 — 14-class distribution and weights")

print(f"{'id':>3} {'subtype':45s} {'coarse':16s} {'train':>6} {'val':>5} {'test':>5}")
for bt in SUBTYPE_LIST:
    i = SUBTYPE_TO_INT[bt]
    tr = int((df_train["bug_type"] == bt).sum())
    va = int((df_val["bug_type"] == bt).sum())
    te = int((df_test["bug_type"] == bt).sum())
    print(f"{i:>3} {bt:45s} {BUG_TYPE_MAP[bt]:16s} {tr:>6,} {va:>5,} {te:>5,}")

class_counts  = df_train["sub_label"].value_counts().sort_index()
total         = len(df_train)
class_weights = {i: total / (len(SUBTYPE_LIST) * cnt) for i, cnt in class_counts.items()}

weights_path = os.path.join(OUTPUT_DIR, "class_weights14.txt")
with open(weights_path, "w", encoding="utf-8") as f:
    f.write("# Class weights for 14-class Stage 2 training\n")
    f.write("# Format: label,subtype_name,weight\n")
    for i, w in class_weights.items():
        f.write(f"{i},{SUBTYPE_LIST[i]},{w:.6f}\n")
print(f"\n  [saved] {weights_path}")

smallest = class_counts.idxmin()
print(f"  NOTE: smallest class = {SUBTYPE_LIST[smallest]} "
      f"({class_counts[smallest]} train samples) — expect wide CI on its test F1.")


# ════════════════════════════════════════════════════════════════
#  STEP 5 — Save splits + subtype mapping JSON
# ════════════════════════════════════════════════════════════════
section("STEP 5 — Saving Stage 2 artifacts")

save_cols = ["code", "bug_type", "error_class", "label", "sub_label",
             "code_len", "line_count", "is_parseable"]
df_train[save_cols].to_parquet(os.path.join(OUTPUT_DIR, "train14.parquet"), index=False)
df_val[save_cols].to_parquet(os.path.join(OUTPUT_DIR,   "val14.parquet"),   index=False)
df_test[save_cols].to_parquet(os.path.join(OUTPUT_DIR,  "test14.parquet"),  index=False)

mapping = {
    "subtype_to_int": SUBTYPE_TO_INT,
    "int_to_subtype": {str(i): bt for bt, i in SUBTYPE_TO_INT.items()},
    "coarse_groups":  COARSE_GROUPS,
    "human_names":    HUMAN_NAMES,
}
mapping_path = os.path.join(OUTPUT_DIR, "subtype_mapping.json")
with open(mapping_path, "w", encoding="utf-8") as f:
    json.dump(mapping, f, indent=2)

print(f"  [saved] train14.parquet ({len(df_train):,} rows)")
print(f"  [saved] val14.parquet   ({len(df_val):,} rows)")
print(f"  [saved] test14.parquet  ({len(df_test):,} rows)")
print(f"  [saved] {mapping_path}")


# ════════════════════════════════════════════════════════════════
#  STEP 6 — Visualise
# ════════════════════════════════════════════════════════════════
section("STEP 6 — Visualising 14-class distribution")

COARSE_COLORS = {"syntax_error": "#4C72B0", "logic_error": "#55A868",
                 "variable_misuse": "#C44E52"}

fig, ax = plt.subplots(figsize=(10, 7))
names  = [HUMAN_NAMES[bt] for bt in SUBTYPE_LIST]
counts = [int((df_train["bug_type"] == bt).sum()) for bt in SUBTYPE_LIST]
colors = [COARSE_COLORS[BUG_TYPE_MAP[bt]] for bt in SUBTYPE_LIST]

bars = ax.barh(range(len(names)), counts, color=colors, edgecolor="white")
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Training samples")
ax.set_title("Stage 2 — 14 fine-grained bug types (training set)",
             fontsize=12, fontweight="bold")
for bar, v in zip(bars, counts):
    ax.text(v + 8, bar.get_y() + bar.get_height() / 2, f"{v:,}",
            va="center", fontsize=8)

from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=c, label=l.replace("_", " "))
                   for l, c in COARSE_COLORS.items()], loc="lower right")

plt.tight_layout()
chart_path = os.path.join(OUTPUT_DIR, "label_distribution14.png")
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  [saved] {chart_path}")

print(f"""
{'=' * 60}
  STAGE 2 DATA PREP COMPLETE
{'=' * 60}
  14 fine-grained subtypes, grouped by coarse class:
    syntax_error    : ids {COARSE_GROUPS['syntax_error']}
    logic_error     : ids {COARSE_GROUPS['logic_error']}
    variable_misuse : ids {COARSE_GROUPS['variable_misuse']}

  Split alignment with Stage 1 verified — no leakage possible.

  Next: run codebert_train_stage2.py
{'=' * 60}
""")
