"""
Phase 4 — Line Detection Step 1: Data Preparation (bug-line labels)
===================================================================
Project : code-analyzer-ai
Author  : Nguyen Ngoc Gia Han (GCS230054)

Why this script exists:
  The Stage 1 / Stage 2 CodeBERT models do whole-sequence classification
  (code -> one label). They cannot say WHICH line the bug is on. To show a
  line number on arbitrary user code, we train a separate token-classification
  model. This script builds its training labels.

  The BuggedPythonLeetCode dataset has NO bug-location column, but it does
  provide both `original_code` (clean) and `bugged_code` (bug injected). The
  line(s) that a mutation changed = the bug location. We recover them by
  aligning the two versions with difflib.SequenceMatcher (handles inserted /
  deleted lines correctly, unlike a naive positional diff).

  IMPORTANT: the model input is `bugged_code` only (that is all we have at
  inference). `original_code` is used ONLY to derive labels, never as input.

What this script does:
  1. Load BuggedPythonLeetCode raw parquet (same file as data_prep.py)
  2. For each row, align original vs bugged lines -> buggy line numbers
  3. Keep low-noise samples (1-3 changed lines) so labels are clean
  4. Deduplicate by normalized bugged_code (prevents cross-split leakage)
  5. Split 70/15/15 stratified by bug_type, seed 42, disjointness asserted
  6. Save line_train/val/test.parquet + a distribution chart

Output (ml_pipeline/data/processed/):
  line_train.parquet / line_val.parquet / line_test.parquet
  line_detection_stats.txt
  line_detection_distribution.png

Each parquet row: code (bugged), bug_type, buggy_lines (list[int], 1-indexed),
line_count.

HOW TO RUN:
  python ml_pipeline/data_prep_line_detection.py
"""

import difflib
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
OUTPUT_DIR     = r"F:\Heulwen\IT GREENWICH\Final Project\Tutor_AI_code\ml_pipeline\data\processed"

MAX_BUGGY_LINES = 3     # drop samples where more than this many lines changed
MIN_CODE_LEN    = 20    # same threshold used elsewhere in the pipeline
RANDOM_SEED     = 42
# ───────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")


def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def normalize_key(code_str):
    """Whitespace-normalized key for exact-duplicate detection."""
    return "\n".join(line.rstrip() for line in code_str.strip().splitlines())


def buggy_line_numbers(original: str, bugged: str) -> list[int]:
    """
    Return the 1-indexed line numbers IN `bugged` that differ from `original`,
    using difflib.SequenceMatcher for a proper alignment. 'replace' and
    'insert' opcodes mark bugged-side lines that carry the mutation. 'delete'
    opcodes (a line removed from original) are attributed to the bugged line
    where the deletion occurs so the location is still reported.
    """
    o_lines = original.split("\n")
    b_lines = bugged.split("\n")
    sm = difflib.SequenceMatcher(a=o_lines, b=b_lines, autojunk=False)
    changed: set[int] = set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag in ("replace", "insert"):
            for j in range(j1, j2):
                changed.add(j + 1)  # 1-indexed on bugged side
        elif tag == "delete":
            # deletion: mark the bugged-side line at the deletion point
            changed.add(min(j1 + 1, len(b_lines)))
    return sorted(changed)


# ════════════════════════════════════════════════════════════════
#  STEP 1 — Load raw dataset
# ════════════════════════════════════════════════════════════════
section("STEP 1 — Loading BuggedPythonLeetCode")

df = pd.read_parquet(BUGGED_LC_PATH)
print(f"Loaded: {len(df):,} rows  columns: {list(df.columns)}")

required = {"original_code", "bugged_code", "bug_type"}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f"Dataset missing required columns: {missing}")


# ════════════════════════════════════════════════════════════════
#  STEP 2 — Derive buggy-line labels via difflib alignment
# ════════════════════════════════════════════════════════════════
section("STEP 2 — Deriving buggy-line labels (difflib alignment)")

records = []
n_empty, n_toolong, n_short, n_identical = 0, 0, 0, 0

for _, row in df.iterrows():
    original = row["original_code"]
    bugged   = row["bugged_code"]
    bug_type = row["bug_type"]

    if not isinstance(bugged, str) or not isinstance(original, str):
        continue
    if len(bugged) < MIN_CODE_LEN:
        n_short += 1
        continue
    if normalize_key(original) == normalize_key(bugged):
        n_identical += 1
        continue

    lines = buggy_line_numbers(original, bugged)
    if len(lines) == 0:
        n_empty += 1
        continue
    if len(lines) > MAX_BUGGY_LINES:
        n_toolong += 1
        continue

    records.append({
        "code":        bugged,
        "bug_type":    bug_type,
        "buggy_lines": lines,
        "line_count":  bugged.count("\n") + 1,
    })

data = pd.DataFrame(records)
print(f"Kept: {len(data):,} / {len(df):,} samples")
print(f"  dropped — identical (no diff)   : {n_identical:,}")
print(f"  dropped — empty diff            : {n_empty:,}")
print(f"  dropped — >{MAX_BUGGY_LINES} changed lines     : {n_toolong:,}")
print(f"  dropped — too short             : {n_short:,}")

dist = Counter(len(r) for r in data["buggy_lines"])
print("\nBuggy-lines-per-sample distribution:")
for k in sorted(dist):
    print(f"  {k} line(s): {dist[k]:,}")


# ════════════════════════════════════════════════════════════════
#  STEP 3 — Deduplicate by normalized bugged code
# ════════════════════════════════════════════════════════════════
section("STEP 3 — Deduplicating (prevents cross-split leakage)")

before = len(data)
data["norm_key"] = data["code"].apply(normalize_key)
data = data.drop_duplicates(subset="norm_key").reset_index(drop=True)
print(f"Unique samples: {len(data):,} (removed {before - len(data):,} duplicates)")


# ════════════════════════════════════════════════════════════════
#  STEP 4 — Split 70/15/15 stratified by bug_type
# ════════════════════════════════════════════════════════════════
section("STEP 4 — Splitting 70/15/15 (stratified by bug_type)")

# Stratify only on bug types with >=3 samples; rare ones fall back to unstratified.
type_counts = data["bug_type"].value_counts()
stratify_col = data["bug_type"] if (type_counts.min() >= 3) else None

train_df, temp_df = train_test_split(
    data, test_size=0.30, random_state=RANDOM_SEED,
    stratify=stratify_col)
temp_strat = temp_df["bug_type"] if (temp_df["bug_type"].value_counts().min() >= 2) else None
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=RANDOM_SEED,
    stratify=temp_strat)

for name, part in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    print(f"  {name:5s}: {len(part):,}")

# Leakage guard: normalized-key sets must be pairwise disjoint
ks = [set(p["norm_key"]) for p in (train_df, val_df, test_df)]
assert not (ks[0] & ks[1]), "LEAKAGE: train/val overlap"
assert not (ks[0] & ks[2]), "LEAKAGE: train/test overlap"
assert not (ks[1] & ks[2]), "LEAKAGE: val/test overlap"
print("  Disjointness assertion passed (train/val/test)")


# ════════════════════════════════════════════════════════════════
#  STEP 5 — Save splits
# ════════════════════════════════════════════════════════════════
section("STEP 5 — Saving splits")

drop = ["norm_key"]
train_out = train_df.drop(columns=drop).reset_index(drop=True)
val_out   = val_df.drop(columns=drop).reset_index(drop=True)
test_out  = test_df.drop(columns=drop).reset_index(drop=True)

train_out.to_parquet(os.path.join(OUTPUT_DIR, "line_train.parquet"), index=False)
val_out.to_parquet(os.path.join(OUTPUT_DIR,   "line_val.parquet"),   index=False)
test_out.to_parquet(os.path.join(OUTPUT_DIR,  "line_test.parquet"),  index=False)
print(f"  [saved] line_train.parquet ({len(train_out):,})")
print(f"  [saved] line_val.parquet   ({len(val_out):,})")
print(f"  [saved] line_test.parquet  ({len(test_out):,})")

stats_path = os.path.join(OUTPUT_DIR, "line_detection_stats.txt")
with open(stats_path, "w", encoding="utf-8") as f:
    f.write("Line Detection Data Preparation Stats\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Raw rows           : {len(df):,}\n")
    f.write(f"Kept after filters : {before:,}\n")
    f.write(f"Unique (deduped)   : {len(data):,}\n")
    f.write(f"Max buggy lines    : {MAX_BUGGY_LINES}\n\n")
    f.write(f"Split sizes: train {len(train_out):,}  val {len(val_out):,}  test {len(test_out):,}\n\n")
    f.write("Buggy-lines-per-sample distribution:\n")
    for k in sorted(dist):
        f.write(f"  {k} line(s): {dist[k]:,}\n")
    f.write("\nBug-type distribution (kept samples):\n")
    for bt, c in data["bug_type"].value_counts().items():
        f.write(f"  {bt:45s}: {c:,}\n")
print(f"  [saved] {stats_path}")


# ════════════════════════════════════════════════════════════════
#  STEP 6 — Distribution chart
# ════════════════════════════════════════════════════════════════
section("STEP 6 — Distribution chart")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ks_sorted = sorted(dist)
axes[0].bar([str(k) for k in ks_sorted], [dist[k] for k in ks_sorted],
            color="#4C72B0", edgecolor="white")
axes[0].set_title("Buggy lines per sample")
axes[0].set_xlabel("Number of changed lines")
axes[0].set_ylabel("Samples")

sizes = [len(train_out), len(val_out), len(test_out)]
axes[1].bar(["Train", "Val", "Test"], sizes,
            color=["#4C72B0", "#CCB974", "#55A868"], edgecolor="white")
axes[1].set_title("Split sizes")
axes[1].set_ylabel("Samples")
for i, v in enumerate(sizes):
    axes[1].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=10)

fig.suptitle("Line Detection — Data Preparation", fontsize=13, fontweight="bold")
plt.tight_layout()
chart_path = os.path.join(OUTPUT_DIR, "line_detection_distribution.png")
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  [saved] {chart_path}")

print(f"""
{'=' * 60}
  LINE DETECTION DATA PREP COMPLETE
{'=' * 60}
  Kept {len(data):,} clean-label samples ({MAX_BUGGY_LINES} lines max).
  Next: python ml_pipeline/codebert_train_line_detection.py
{'=' * 60}
""")
