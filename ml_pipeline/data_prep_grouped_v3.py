"""
Phase 5 — Step 1: Grouped Data Preparation v3 (leakage-free, problem-grouped)
============================================================================
Project : code-analyzer-ai
Author  : Nguyen Ngoc Gia Han (GCS230054)

Why this script exists
----------------------
The v1/v2 pipeline splits the data with a row-level, label-stratified
`train_test_split`. But BuggedPythonLeetCode generates up to 15 bugged variants
from EACH correct solution, so ~90%-token-identical siblings of one problem end
up on both sides of the split. Deduplication keys on the *bugged* string and
therefore cannot catch siblings (different injected bug => different string).
The result is train/test context leakage: the in-domain 0.984 Macro-F1 is
partly memorised problem context, which is absent from any external program —
hence the collapse to a single "safe" class on real code.

This script replaces the leaky split with a GROUPED split.

Key ideas
---------
1. problem_id = hash(normalised original_code). Every clean solution and ALL of
   its bugged variants share this id.
2. The 70/15/15 split is performed over unique problem_ids, so a whole problem
   (its clean copy + every bugged variant) lands in exactly one partition.
3. no_bug is drawn from `original_code` itself (deduped) — same distribution as
   the bug classes, which removes the provenance confound — and topped up with
   flytech/python-codes-25k for clean-code style diversity. flytech rows have no
   LeetCode sibling, so each is its own singleton group.
4. One split assignment drives all three model datasets (Stage 1 4-class,
   Stage 2 14-class, Line detection), so they are mutually leakage-free by
   construction (no cross-stage assertion needed).

Outputs (ml_pipeline/data/processed/)
  train4_v3 / val4_v3 / test4_v3.parquet        (Stage 1, 4-class)
  train14_v3 / val14_v3 / test14_v3.parquet     (Stage 2, 14 subtypes, bugs only)
  line_train_v3 / line_val_v3 / line_test_v3.parquet  (Line detection, bugs only)
  class_weights4_v3.txt / class_weights3_v3.txt / class_weights14_v3.txt
  subtype_mapping_v3.json
  grouped_split_stats_v3.txt
  label_distribution4_v3.png / label_distribution14_v3.png
  line_detection_distribution_v3.png / leakage_audit_v3.png

HOW TO RUN
  python ml_pipeline/data_prep_grouped_v3.py
"""

import ast
import difflib
import hashlib
import json
import os
import re
import sys
import urllib.request
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── CONFIG ─────────────────────────────────────────────────────────────
BUGGED_LC_PATH  = r"F:\Heulwen\IT GREENWICH\Final Project\Datasets\BuggedPythonLeetCode Dataset\Train\0000.parquet"
PROCESSED_DIR   = r"F:\Heulwen\IT GREENWICH\Final Project\Tutor_AI_code\ml_pipeline\data\processed"
FLYTECH_DIR     = r"F:\Heulwen\IT GREENWICH\Final Project\Datasets\flytech_python_codes_25k"
FLYTECH_PARQUET = os.path.join(FLYTECH_DIR, "flytech_train_0000.parquet")
FLYTECH_URL     = ("https://huggingface.co/datasets/flytech/python-codes-25k/"
                   "resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet")
OUTPUT_DIR      = PROCESSED_DIR

NOBUG_TARGET    = 4000   # total no_bug pool (original_code backbone + flytech top-up)
MIN_CODE_LEN    = 10     # Stage 1/2 threshold (matches data_prep.py)
MIN_LINE_LEN    = 20     # Line-detection threshold (matches data_prep_line_detection.py)
MAX_BUGGY_LINES = 3      # keep low-noise line labels only
TRAIN_FRAC      = 0.70
VAL_FRAC        = 0.15   # test gets the remaining 0.15
RANDOM_SEED     = 42
# ───────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FLYTECH_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")
rng = np.random.default_rng(RANDOM_SEED)

# ── Label schema (4-class convention: no_bug = 3) ──────────────────────
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
    # IncorrectExceptionHandlerTransformer — dropped (2 samples)
}

CLASS_TO_INT = {
    "syntax_error":    0,
    "logic_error":     1,
    "variable_misuse": 2,
    "no_bug":          3,
}
INT_TO_CLASS = {v: k for k, v in CLASS_TO_INT.items()}
CLASS_NAMES  = ["syntax_error", "logic_error", "variable_misuse", "no_bug"]

# ── Stage 2 subtype schema (grouped by coarse class, alphabetical) ─────
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


def clean_code(code_str):
    """Normalise literal escape sequences and strip (matches data_prep.py)."""
    if not isinstance(code_str, str):
        return ""
    return code_str.replace("\\n", "\n").replace("\\t", "    ").strip()


def is_valid_python(code_str):
    try:
        ast.parse(code_str)
        return True
    except Exception:
        return False


def normalize_key(code_str):
    """Whitespace-normalised key for hashing / dedup."""
    return "\n".join(line.rstrip() for line in code_str.strip().splitlines())


def problem_hash(original_clean):
    return hashlib.sha1(normalize_key(original_clean).encode("utf-8")).hexdigest()


def buggy_line_numbers(original: str, bugged: str) -> list:
    """1-indexed bugged-side line numbers that differ from original (difflib)."""
    o_lines = original.split("\n")
    b_lines = bugged.split("\n")
    sm = difflib.SequenceMatcher(a=o_lines, b=b_lines, autojunk=False)
    changed = set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag in ("replace", "insert"):
            for j in range(j1, j2):
                changed.add(j + 1)
        elif tag == "delete":
            changed.add(min(j1 + 1, len(b_lines)))
    return sorted(changed)


def assign_split(ids, train_frac, val_frac, seed):
    """Deterministically map a list of unique group ids -> {id: split}."""
    ids = list(ids)
    order = np.random.default_rng(seed).permutation(len(ids))
    n = len(ids)
    n_train = int(round(n * train_frac))
    n_val   = int(round(n * val_frac))
    out = {}
    for rank, idx in enumerate(order):
        if rank < n_train:
            out[ids[idx]] = "train"
        elif rank < n_train + n_val:
            out[ids[idx]] = "val"
        else:
            out[ids[idx]] = "test"
    return out


# ════════════════════════════════════════════════════════════════
#  STEP 1 — Load + map + clean BuggedPythonLeetCode
# ════════════════════════════════════════════════════════════════
section("STEP 1 — Loading + cleaning BuggedPythonLeetCode")

df = pd.read_parquet(BUGGED_LC_PATH)
print(f"Loaded: {len(df):,} rows  columns: {list(df.columns)}")

df["error_class"] = df["bug_type"].map(BUG_TYPE_MAP)
dropped = df["error_class"].isna().sum()
df = df.dropna(subset=["error_class"]).reset_index(drop=True)
print(f"Dropped unmappable (IncorrectExceptionHandler): {dropped}")

df["code"]      = df["bugged_code"].apply(clean_code)          # model input
df["orig_code"] = df["original_code"].apply(clean_code)        # labels + grouping only
df = df[df["code"].str.len() >= MIN_CODE_LEN].reset_index(drop=True)
df = df[df["orig_code"].str.len() > 0].reset_index(drop=True)

df["code_len"]     = df["code"].apply(len)
df["line_count"]   = df["code"].apply(lambda x: x.count("\n") + 1)
df["is_parseable"] = df["code"].apply(is_valid_python)
df["label"]        = df["error_class"].map(CLASS_TO_INT).astype(int)
df["sub_label"]    = df["bug_type"].map(SUBTYPE_TO_INT).astype(int)
df["problem_id"]   = df["orig_code"].apply(problem_hash)
print(f"After cleaning: {len(df):,} bug rows")
print(f"Unique problems (original_code): {df['problem_id'].nunique():,}")


# ════════════════════════════════════════════════════════════════
#  STEP 2 — Grouped 70/15/15 split over unique problem_ids
# ════════════════════════════════════════════════════════════════
section("STEP 2 — Grouped split over problems")

problem_ids = sorted(df["problem_id"].unique())
pid_to_split = assign_split(problem_ids, TRAIN_FRAC, VAL_FRAC, RANDOM_SEED)
df["split"]  = df["problem_id"].map(pid_to_split)
df["source"] = "bugged_leetcode"

n_by_split = Counter(pid_to_split.values())
print(f"Problems: {len(problem_ids):,}  ->  "
      f"train {n_by_split['train']:,} | val {n_by_split['val']:,} | test {n_by_split['test']:,}")
print("\nBug-row coarse-class distribution per split:")
for sp in ("train", "val", "test"):
    sub = df[df["split"] == sp]
    counts = {INT_TO_CLASS[i]: int((sub["label"] == i).sum()) for i in (0, 1, 2)}
    print(f"  {sp:5s} (n={len(sub):,}): {counts}")


# ════════════════════════════════════════════════════════════════
#  STEP 3 — no_bug backbone from original_code (deduped by problem)
# ════════════════════════════════════════════════════════════════
section("STEP 3 — Building no_bug backbone from original_code")

orig = (df.drop_duplicates(subset="problem_id")
          .loc[:, ["orig_code", "problem_id", "split"]]
          .reset_index(drop=True))
orig = orig[orig["orig_code"].str.len() >= MIN_CODE_LEN]
orig = orig[orig["orig_code"].apply(is_valid_python)].reset_index(drop=True)
print(f"Clean original_code no_bug rows: {len(orig):,} (one per unique problem)")

nobug_orig = pd.DataFrame({
    "code":         orig["orig_code"].tolist(),
    "error_class":  "no_bug",
    "label":        3,
    "code_len":     orig["orig_code"].apply(len).tolist(),
    "line_count":   orig["orig_code"].apply(lambda x: x.count("\n") + 1).tolist(),
    "is_parseable": True,
    "source":       "leetcode_original",
    "problem_id":   orig["problem_id"].tolist(),
    "split":        orig["split"].tolist(),   # inherits its bugged siblings' split
})
orig_keys = set(orig["orig_code"].apply(normalize_key))


# ════════════════════════════════════════════════════════════════
#  STEP 4 — no_bug top-up from flytech (style diversity, own groups)
# ════════════════════════════════════════════════════════════════
section("STEP 4 — Topping up no_bug with flytech")

FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

def extract_fenced(raw):
    if not isinstance(raw, str):
        return ""
    m = FENCE_RE.search(raw)
    return (m.group(1) if m else raw).strip()

n_needed = max(0, NOBUG_TARGET - len(nobug_orig))
nobug_fly = None

try:
    if not os.path.exists(FLYTECH_PARQUET):
        print(f"Downloading flytech parquet (~23 MB) ...")
        urllib.request.urlretrieve(FLYTECH_URL, FLYTECH_PARQUET)
    df_fly = pd.read_parquet(FLYTECH_PARQUET)
    df_fly["code"] = df_fly["output"].apply(extract_fenced)
    df_fly = df_fly[df_fly["code"].str.len() >= MIN_CODE_LEN]
    df_fly = df_fly[df_fly["code"].apply(is_valid_python)]
    df_fly["norm_key"] = df_fly["code"].apply(normalize_key)
    df_fly = df_fly.drop_duplicates(subset="norm_key")
    df_fly = df_fly[~df_fly["norm_key"].isin(orig_keys)].reset_index(drop=True)
    print(f"Usable unique flytech snippets: {len(df_fly):,}  (need {n_needed:,})")

    n_take = min(n_needed, len(df_fly))
    take_idx = np.random.default_rng(RANDOM_SEED).choice(len(df_fly), size=n_take, replace=False)
    fly = df_fly.iloc[sorted(take_idx)].reset_index(drop=True)

    # Each flytech row is its own singleton group -> ordinary grouped split.
    fly_ids = [f"fly_{i}" for i in range(len(fly))]
    fly_split = assign_split(fly_ids, TRAIN_FRAC, VAL_FRAC, RANDOM_SEED + 1)
    nobug_fly = pd.DataFrame({
        "code":         fly["code"].tolist(),
        "error_class":  "no_bug",
        "label":        3,
        "code_len":     fly["code"].apply(len).tolist(),
        "line_count":   fly["code"].apply(lambda x: x.count("\n") + 1).tolist(),
        "is_parseable": True,
        "source":       "flytech",
        "problem_id":   fly_ids,
        "split":        [fly_split[i] for i in fly_ids],
    })
    print(f"Added flytech no_bug: {len(nobug_fly):,}")
except Exception as exc:
    print(f"[warn] flytech unavailable ({exc}) — proceeding with original_code no_bug only.")

nobug = pd.concat([nobug_orig] + ([nobug_fly] if nobug_fly is not None else []),
                  ignore_index=True)
print(f"\nno_bug pool total: {len(nobug):,}  "
      f"(original {len(nobug_orig):,} + flytech {0 if nobug_fly is None else len(nobug_fly):,})")


# ════════════════════════════════════════════════════════════════
#  STEP 5 — Assemble the three model datasets
# ════════════════════════════════════════════════════════════════
section("STEP 5 — Assembling Stage 1 / Stage 2 / Line datasets")

STAGE1_COLS = ["code", "error_class", "label", "code_len", "line_count",
               "is_parseable", "source", "problem_id", "split"]

bug_stage1 = df[STAGE1_COLS].copy()
stage1 = pd.concat([bug_stage1, nobug[STAGE1_COLS]], ignore_index=True)

def split_of(frame, sp):
    return (frame[frame["split"] == sp]
            .sample(frac=1, random_state=RANDOM_SEED)
            .reset_index(drop=True))

train4, val4, test4 = (split_of(stage1, s) for s in ("train", "val", "test"))
print(f"Stage 1 (4-class): train {len(train4):,}  val {len(val4):,}  test {len(test4):,}")

# Stage 2 — bugs only, 14 subtypes
STAGE2_COLS = ["code", "bug_type", "error_class", "label", "sub_label",
               "code_len", "line_count", "is_parseable", "problem_id", "split"]
bug14 = df[STAGE2_COLS].copy()
train14, val14, test14 = (split_of(bug14, s) for s in ("train", "val", "test"))
print(f"Stage 2 (14-class): train {len(train14):,}  val {len(val14):,}  test {len(test14):,}")

# Line detection — bugs only, 1..MAX_BUGGY_LINES changed lines, deduped bugged code
section("STEP 5b — Deriving line-detection labels")
line_records = []
for _, row in df.iterrows():
    if len(row["code"]) < MIN_LINE_LEN:
        continue
    if normalize_key(row["orig_code"]) == normalize_key(row["code"]):
        continue
    lines = buggy_line_numbers(row["orig_code"], row["code"])
    if not (1 <= len(lines) <= MAX_BUGGY_LINES):
        continue
    line_records.append({
        "code":        row["code"],
        "bug_type":    row["bug_type"],
        "buggy_lines": lines,
        "line_count":  row["line_count"],
        "problem_id":  row["problem_id"],
        "split":       row["split"],
        "norm_key":    normalize_key(row["code"]),
    })
line_df = pd.DataFrame(line_records).drop_duplicates(subset="norm_key").reset_index(drop=True)
line_df = line_df.drop(columns=["norm_key"])
line_train, line_val, line_test = (split_of(line_df, s) for s in ("train", "val", "test"))
print(f"Line detection: train {len(line_train):,}  val {len(line_val):,}  test {len(line_test):,}")


# ════════════════════════════════════════════════════════════════
#  STEP 6 — Leakage audit (0 problems may span >1 split)
# ════════════════════════════════════════════════════════════════
section("STEP 6 — Leakage audit")

def audit(name, tr, va, te):
    s_tr, s_va, s_te = (set(x["problem_id"]) for x in (tr, va, te))
    inter = (s_tr & s_va) | (s_tr & s_te) | (s_va & s_te)
    assert not inter, f"LEAKAGE in {name}: {len(inter)} problem(s) span multiple splits"
    print(f"  {name:12s}: {len(s_tr):,}/{len(s_va):,}/{len(s_te):,} problems — disjoint ✓")
    return len(inter)

audit("Stage1",  train4,  val4,  test4)
audit("Stage2",  train14, val14, test14)
audit("Line",    line_train, line_val, line_test)

# Cross-dataset: a problem must map to the SAME split everywhere it appears
all_map = {}
cross = 0
for frame, sp in [(train4, "train"), (val4, "val"), (test4, "test")]:
    for pid in frame["problem_id"]:
        if all_map.setdefault(pid, sp) != sp:
            cross += 1
assert cross == 0, f"LEAKAGE: {cross} problem(s) assigned to >1 split across datasets"
print(f"  Cross-split problem assignments: {cross} (must be 0) ✓")


# ════════════════════════════════════════════════════════════════
#  STEP 7 — Class weights
# ════════════════════════════════════════════════════════════════
section("STEP 7 — Class weights")

def write_weights(path, counts_by_label, n_classes, name_fn, header):
    total = sum(counts_by_label.values())
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n# Format: label,name,weight\n")
        for i in sorted(counts_by_label):
            w = total / (n_classes * counts_by_label[i])
            f.write(f"{i},{name_fn(i)},{w:.6f}\n")
    print(f"  [saved] {path}")

# 4-class (Stage 1)
c4 = {i: int((train4["label"] == i).sum()) for i in range(4)}
write_weights(os.path.join(OUTPUT_DIR, "class_weights4_v3.txt"),
              c4, 4, lambda i: INT_TO_CLASS[i],
              "# 4-class v3 weights (grouped split, original_code+flytech no_bug)")
# 3-class (baselines, bugs only)
c3 = {i: int((train14["label"] == i).sum()) for i in (0, 1, 2)}
write_weights(os.path.join(OUTPUT_DIR, "class_weights3_v3.txt"),
              c3, 3, lambda i: INT_TO_CLASS[i], "# 3-class v3 weights (bugs only)")
# 14-class (Stage 2)
c14 = {i: int((train14["sub_label"] == i).sum()) for i in range(len(SUBTYPE_LIST))}
write_weights(os.path.join(OUTPUT_DIR, "class_weights14_v3.txt"),
              c14, len(SUBTYPE_LIST), lambda i: SUBTYPE_LIST[i], "# 14-class v3 weights")


# ════════════════════════════════════════════════════════════════
#  STEP 8 — Save splits + subtype mapping
# ════════════════════════════════════════════════════════════════
section("STEP 8 — Saving v3 datasets")

def save(frame, name):
    frame.to_parquet(os.path.join(OUTPUT_DIR, name), index=False)
    print(f"  [saved] {name} ({len(frame):,} rows)")

save(train4, "train4_v3.parquet"); save(val4, "val4_v3.parquet"); save(test4, "test4_v3.parquet")
save(train14, "train14_v3.parquet"); save(val14, "val14_v3.parquet"); save(test14, "test14_v3.parquet")
save(line_train, "line_train_v3.parquet"); save(line_val, "line_val_v3.parquet"); save(line_test, "line_test_v3.parquet")

mapping = {
    "subtype_to_int": SUBTYPE_TO_INT,
    "int_to_subtype": {str(i): bt for bt, i in SUBTYPE_TO_INT.items()},
    "coarse_groups":  COARSE_GROUPS,
    "human_names":    HUMAN_NAMES,
}
with open(os.path.join(OUTPUT_DIR, "subtype_mapping_v3.json"), "w", encoding="utf-8") as f:
    json.dump(mapping, f, indent=2)
print("  [saved] subtype_mapping_v3.json")


# ════════════════════════════════════════════════════════════════
#  STEP 9 — Stats report
# ════════════════════════════════════════════════════════════════
stats_path = os.path.join(OUTPUT_DIR, "grouped_split_stats_v3.txt")
with open(stats_path, "w", encoding="utf-8") as f:
    f.write("Grouped Data Preparation v3 — Stats\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Unique problems (original_code): {len(problem_ids):,}\n")
    f.write(f"  train/val/test problems: {n_by_split['train']:,}/{n_by_split['val']:,}/{n_by_split['test']:,}\n\n")
    f.write("Stage 1 (4-class) split sizes + class counts:\n")
    for nm, fr in [("train", train4), ("val", val4), ("test", test4)]:
        cc = {INT_TO_CLASS[i]: int((fr["label"] == i).sum()) for i in range(4)}
        f.write(f"  {nm:5s} n={len(fr):,}  {cc}\n")
    f.write("\nno_bug composition (Stage 1):\n")
    for nm, fr in [("train", train4), ("val", val4), ("test", test4)]:
        nb = fr[fr["label"] == 3]
        f.write(f"  {nm:5s}: leetcode_original {int((nb['source']=='leetcode_original').sum()):,}"
                f"  flytech {int((nb['source']=='flytech').sum()):,}\n")
    f.write(f"\nStage 2 (14-class) sizes: train {len(train14):,} val {len(val14):,} test {len(test14):,}\n")
    f.write(f"Line detection sizes:    train {len(line_train):,} val {len(line_val):,} test {len(line_test):,}\n")
    f.write("\nLeakage audit: 0 problems span >1 split (asserted). Grouped split is leakage-free.\n")
print(f"  [saved] {stats_path}")


# ════════════════════════════════════════════════════════════════
#  STEP 10 — Charts
# ════════════════════════════════════════════════════════════════
section("STEP 10 — Charts")

COLORS4 = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

# Chart 1 — 4-class distribution, split sizes, no_bug source composition
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
counts = [int((train4["error_class"] == c).sum()) for c in CLASS_NAMES]
bars = axes[0].bar(["syntax", "logic", "variable\nmisuse", "no_bug"], counts,
                   color=COLORS4, edgecolor="white")
axes[0].set_title("4-class v3 training distribution (grouped)")
axes[0].set_ylabel("Count")
for b, v in zip(bars, counts):
    axes[0].text(b.get_x() + b.get_width() / 2, v + 20, f"{v:,}", ha="center", fontsize=10)

sizes = [len(train4), len(val4), len(test4)]
axes[1].bar(["Train", "Val", "Test"], sizes,
            color=["#4C72B0", "#CCB974", "#55A868"], edgecolor="white")
axes[1].set_title("Split sizes (4-class v3)")
axes[1].set_ylabel("Samples")
for i, v in enumerate(sizes):
    axes[1].text(i, v + 20, f"{v:,}", ha="center", fontsize=10)

src = {s: [int(((fr["label"] == 3) & (fr["source"] == s)).sum())
           for fr in (train4, val4, test4)] for s in ("leetcode_original", "flytech")}
x = np.arange(3)
axes[2].bar(x - 0.2, src["leetcode_original"], 0.4, label="leetcode_original", color="#55A868")
axes[2].bar(x + 0.2, src["flytech"], 0.4, label="flytech", color="#DD8452")
axes[2].set_xticks(x); axes[2].set_xticklabels(["Train", "Val", "Test"])
axes[2].set_title("no_bug composition by source"); axes[2].set_ylabel("Samples"); axes[2].legend()
fig.suptitle("Phase 5 — Grouped Data Prep v3 (leakage-free)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "label_distribution4_v3.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  [saved] label_distribution4_v3.png")

# Chart 2 — 14 subtype distribution
COARSE_COLORS = {"syntax_error": "#4C72B0", "logic_error": "#55A868", "variable_misuse": "#C44E52"}
fig, ax = plt.subplots(figsize=(10, 7))
names  = [HUMAN_NAMES[bt] for bt in SUBTYPE_LIST]
cnts   = [int((train14["bug_type"] == bt).sum()) for bt in SUBTYPE_LIST]
cols   = [COARSE_COLORS[BUG_TYPE_MAP[bt]] for bt in SUBTYPE_LIST]
bars = ax.barh(range(len(names)), cnts, color=cols, edgecolor="white")
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9); ax.invert_yaxis()
ax.set_xlabel("Training samples"); ax.set_title("Stage 2 v3 — 14 subtypes (grouped)", fontweight="bold")
for b, v in zip(bars, cnts):
    ax.text(v + 4, b.get_y() + b.get_height() / 2, f"{v:,}", va="center", fontsize=8)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=c, label=l.replace("_", " ")) for l, c in COARSE_COLORS.items()],
          loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "label_distribution14_v3.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  [saved] label_distribution14_v3.png")

# Chart 3 — line-detection distribution
dist = Counter(len(x) for x in line_df["buggy_lines"])
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ks = sorted(dist)
axes[0].bar([str(k) for k in ks], [dist[k] for k in ks], color="#4C72B0", edgecolor="white")
axes[0].set_title("Buggy lines per sample (v3)"); axes[0].set_xlabel("Changed lines"); axes[0].set_ylabel("Samples")
lsz = [len(line_train), len(line_val), len(line_test)]
axes[1].bar(["Train", "Val", "Test"], lsz, color=["#4C72B0", "#CCB974", "#55A868"], edgecolor="white")
axes[1].set_title("Line-detection split sizes (v3)"); axes[1].set_ylabel("Samples")
for i, v in enumerate(lsz):
    axes[1].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=10)
fig.suptitle("Line Detection v3 — Data Preparation", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "line_detection_distribution_v3.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  [saved] line_detection_distribution_v3.png")

# Chart 4 — leakage audit (problems per split, 0 overlap)
fig, ax = plt.subplots(figsize=(7, 5))
pcounts = [n_by_split["train"], n_by_split["val"], n_by_split["test"]]
bars = ax.bar(["Train", "Val", "Test"], pcounts,
              color=["#4C72B0", "#CCB974", "#55A868"], edgecolor="white")
for b, v in zip(bars, pcounts):
    ax.text(b.get_x() + b.get_width() / 2, v + 5, f"{v:,}", ha="center", fontsize=10)
ax.set_title("Grouped split — unique problems per partition\n(0 problems shared across splits)",
             fontweight="bold")
ax.set_ylabel("Unique problems (original_code)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "leakage_audit_v3.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  [saved] leakage_audit_v3.png")

print(f"""
{'=' * 60}
  GROUPED DATA PREP v3 COMPLETE — leakage-free
{'=' * 60}
  Problems split 70/15/15: {n_by_split['train']:,}/{n_by_split['val']:,}/{n_by_split['test']:,}
  no_bug = original_code ({len(nobug_orig):,}) + flytech ({0 if nobug_fly is None else len(nobug_fly):,})
  Audit: 0 problems span >1 split.

  Next (Phase 2): feature_extraction_v3.py -> baseline_classifiers_v3.py
  Then (Phase 3, GPU): retrain Stage 1 / Stage 2 / Line on *_v3 splits.
{'=' * 60}
""")
