"""
Phase 4 — Step 1: 4-Class Data Preparation v2 (expanded no_bug pool)
====================================================================
Project : code-analyzer-ai
Author  : Nguyen Ngoc Gia Han (GCS230054)

Why this script exists:
  The v1 no_bug class (data_prep_4class.py) came entirely from clean LeetCode
  solutions — a single code style. This script adds a second no_bug source,
  the HuggingFace dataset flytech/python-codes-25k (instructional/task-style
  Python), so the model learns "clean code" across two different styles and
  does not shortcut on "LeetCode-looking = buggy".

  IMPORTANT: flytech is NOT guaranteed clean. Many samples contain broken
  quote nesting (e.g. input('type 'done' to finish')) which is a real syntax
  error, and the raw export contains ~2x duplicated rows. Every sample is
  therefore: fence-stripped -> ast.parse-filtered -> deduplicated (within
  flytech AND against the LeetCode pool) before use.

What this script does:
  1. Downloads/caches the flytech parquet (one-time, ~23 MB)
  2. Extracts code from markdown fences, filters by ast.parse + length
  3. Dedups (normalized code key) within flytech and cross-source
  4. Builds combined no_bug pool: ALL clean LeetCode + flytech fill-up
     to NOBUG_TARGET, each row tagged with a `source` column
  5. Splits no_bug 70/15/15 stratified by source, concats with the
     existing 3-class splits
  6. Saves v2 splits + class weights + distribution chart

Output (ml_pipeline/data/processed/):
  train4_v2.parquet / val4_v2.parquet / test4_v2.parquet
  class_weights4_v2.txt
  label_distribution4_v2.png

HOW TO RUN:
  python ml_pipeline/data_prep_4class_v2.py
"""

import ast
import json
import os
import re
import sys
import urllib.request

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
FLYTECH_DIR     = r"F:\Heulwen\IT GREENWICH\Final Project\Datasets\flytech_python_codes_25k"
FLYTECH_PARQUET = os.path.join(FLYTECH_DIR, "flytech_train_0000.parquet")
FLYTECH_URL     = ("https://huggingface.co/datasets/flytech/python-codes-25k/"
                   "resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet")
OUTPUT_DIR      = PROCESSED_DIR

NOBUG_TARGET    = 4000   # total no_bug pool size (all LeetCode + flytech fill-up)
MIN_CODE_LEN    = 20     # same threshold as data_prep_4class.py
RANDOM_SEED     = 42
# ───────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FLYTECH_DIR, exist_ok=True)
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


def is_valid_python(code_str):
    try:
        ast.parse(code_str)
        return True
    except Exception:
        return False


def normalize_key(code_str):
    """Whitespace-normalized key used for exact-duplicate detection."""
    return "\n".join(line.rstrip() for line in code_str.strip().splitlines())


# ════════════════════════════════════════════════════════════════
#  STEP 1 — Download / cache flytech parquet
# ════════════════════════════════════════════════════════════════
section("STEP 1 — Loading flytech/python-codes-25k")

if not os.path.exists(FLYTECH_PARQUET):
    print(f"Downloading flytech parquet (~23 MB) ...\n  {FLYTECH_URL}")
    urllib.request.urlretrieve(FLYTECH_URL, FLYTECH_PARQUET)
    print(f"  [saved] {FLYTECH_PARQUET}")
else:
    print(f"Using cached parquet: {FLYTECH_PARQUET}")

df_fly = pd.read_parquet(FLYTECH_PARQUET)
print(f"Raw flytech rows: {len(df_fly):,}  columns: {list(df_fly.columns)}")


# ════════════════════════════════════════════════════════════════
#  STEP 2 — Extract code from markdown fences + filter
# ════════════════════════════════════════════════════════════════
section("STEP 2 — Extracting + filtering flytech code")

FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

def extract_code(raw):
    """Pull the first fenced code block out of `output`; fall back to raw."""
    if not isinstance(raw, str):
        return ""
    m = FENCE_RE.search(raw)
    return (m.group(1) if m else raw).strip()

df_fly["code"] = df_fly["output"].apply(extract_code)

n_fenced = df_fly["output"].str.contains("```", na=False).sum()
print(f"Rows with markdown fences: {n_fenced:,} / {len(df_fly):,}")

before = len(df_fly)
df_fly = df_fly[df_fly["code"].str.len() >= MIN_CODE_LEN]
print(f"Length filter (>= {MIN_CODE_LEN} chars): kept {len(df_fly):,} / {before:,}")

before = len(df_fly)
df_fly = df_fly[df_fly["code"].apply(is_valid_python)]
rejected = before - len(df_fly)
print(f"ast.parse filter: kept {len(df_fly):,} / {before:,}  "
      f"(REJECTED {rejected:,} = {rejected / before * 100:.1f}% — broken quoting etc.)")

# Dedup within flytech (raw export is ~2x duplicated)
before = len(df_fly)
df_fly["norm_key"] = df_fly["code"].apply(normalize_key)
df_fly = df_fly.drop_duplicates(subset="norm_key").reset_index(drop=True)
print(f"Dedup within flytech: kept {len(df_fly):,} / {before:,} unique snippets")


# ════════════════════════════════════════════════════════════════
#  STEP 3 — Load clean LeetCode pool (same logic as data_prep_4class.py)
# ════════════════════════════════════════════════════════════════
section("STEP 3 — Loading clean LeetCode code")

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def clean_completion(raw):
    if not isinstance(raw, str):
        return ""
    code = raw.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        inner = [l for l in lines if not l.strip().startswith("```")]
        code = "\n".join(inner).strip()
    return code

all_lc = load_jsonl(LC_TRAIN_JSONL) + load_jsonl(LC_TEST_JSONL)
print(f"LeetCodeDataset records: {len(all_lc):,}")

lc_codes = []
for rec in all_lc:
    code = clean_completion(rec.get("completion", "") or "")
    if len(code) >= MIN_CODE_LEN and is_valid_python(code):
        lc_codes.append(code)
print(f"Valid clean LeetCode completions: {len(lc_codes):,}")

# Dedup within LeetCode (v1 did not — prevents cross-split leakage)
lc_keys_seen = set()
lc_unique = []
for code in lc_codes:
    k = normalize_key(code)
    if k not in lc_keys_seen:
        lc_keys_seen.add(k)
        lc_unique.append(code)
print(f"Unique LeetCode snippets: {len(lc_unique):,} "
      f"(removed {len(lc_codes) - len(lc_unique)} duplicates)")

# Cross-source dedup: drop flytech snippets colliding with LeetCode
before = len(df_fly)
df_fly = df_fly[~df_fly["norm_key"].isin(lc_keys_seen)].reset_index(drop=True)
print(f"Cross-source collisions removed from flytech: {before - len(df_fly)}")


# ════════════════════════════════════════════════════════════════
#  STEP 4 — Build combined no_bug pool
# ════════════════════════════════════════════════════════════════
section("STEP 4 — Building combined no_bug pool")

n_fly_needed = max(0, NOBUG_TARGET - len(lc_unique))
n_fly_take   = min(n_fly_needed, len(df_fly))
rng = np.random.default_rng(RANDOM_SEED)
fly_idx = rng.choice(len(df_fly), size=n_fly_take, replace=False)
fly_selected = df_fly.iloc[sorted(fly_idx)]["code"].tolist()

nobug_df = pd.DataFrame({
    "code":   lc_unique + fly_selected,
    "source": ["leetcode"] * len(lc_unique) + ["flytech"] * len(fly_selected),
})
print(f"no_bug pool: {len(nobug_df):,} total  "
      f"(leetcode {len(lc_unique):,} + flytech {len(fly_selected):,}, target {NOBUG_TARGET:,})")


# ════════════════════════════════════════════════════════════════
#  STEP 5 — Split no_bug 70/15/15 (stratified by source)
# ════════════════════════════════════════════════════════════════
section("STEP 5 — Splitting no_bug pool 70/15/15")

nb_train, nb_temp = train_test_split(
    nobug_df, test_size=0.30, random_state=RANDOM_SEED, stratify=nobug_df["source"])
nb_val, nb_test = train_test_split(
    nb_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=nb_temp["source"])

for name, part in [("Train", nb_train), ("Val", nb_val), ("Test", nb_test)]:
    counts = part["source"].value_counts()
    print(f"  {name:5s}: {len(part):,}  "
          f"(leetcode {counts.get('leetcode', 0):,}, flytech {counts.get('flytech', 0):,})")

# Leakage guard: normalized-key sets must be pairwise disjoint
key_sets = [set(part["code"].apply(normalize_key)) for part in (nb_train, nb_val, nb_test)]
assert not (key_sets[0] & key_sets[1]), "LEAKAGE: train/val no_bug overlap"
assert not (key_sets[0] & key_sets[2]), "LEAKAGE: train/test no_bug overlap"
assert not (key_sets[1] & key_sets[2]), "LEAKAGE: val/test no_bug overlap"
print("  Disjointness assertion passed (train/val/test no_bug keys)")


# ════════════════════════════════════════════════════════════════
#  STEP 6 — Concat with existing 3-class splits
# ════════════════════════════════════════════════════════════════
section("STEP 6 — Building 4-class v2 DataFrames")

df_train3 = pd.read_parquet(os.path.join(PROCESSED_DIR, "train.parquet"))
df_val3   = pd.read_parquet(os.path.join(PROCESSED_DIR, "val.parquet"))
df_test3  = pd.read_parquet(os.path.join(PROCESSED_DIR, "test.parquet"))
print(f"3-class splits: train {len(df_train3):,}  val {len(df_val3):,}  test {len(df_test3):,}")

def make_nobug_df(part):
    codes = part["code"].tolist()
    return pd.DataFrame({
        "code":        codes,
        "error_class": "no_bug",
        "label":       3,
        "code_len":    [len(c) for c in codes],
        "line_count":  [c.count("\n") + 1 for c in codes],
        "is_parseable": True,
        "source":      part["source"].tolist(),
    })

def combine(df_bug, nb_part):
    df_bug = df_bug.copy()
    df_bug["label"]  = df_bug["label"].astype(int)
    df_bug["source"] = "bugged_leetcode"
    combined = pd.concat([df_bug, make_nobug_df(nb_part)], ignore_index=True)
    return combined.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

df_train4 = combine(df_train3, nb_train)
df_val4   = combine(df_val3,   nb_val)
df_test4  = combine(df_test3,  nb_test)

print(f"\n4-class v2 split sizes: train {len(df_train4):,}  "
      f"val {len(df_val4):,}  test {len(df_test4):,}")
print(f"\nTrain4_v2 class distribution:")
for cls in CLASS_NAMES:
    cnt = (df_train4["error_class"] == cls).sum()
    print(f"  {cls:25s}: {cnt:,}  ({cnt / len(df_train4) * 100:.1f}%)")


# ════════════════════════════════════════════════════════════════
#  STEP 7 — Class weights
# ════════════════════════════════════════════════════════════════
section("STEP 7 — Class weights")

class_counts  = df_train4["label"].value_counts().sort_index()
total         = len(df_train4)
class_weights = {i: total / (len(CLASS_TO_INT) * cnt) for i, cnt in class_counts.items()}
for i, w in class_weights.items():
    print(f"  {INT_TO_CLASS[i]:25s} (label {i}): weight = {w:.4f}")


# ════════════════════════════════════════════════════════════════
#  STEP 8 — Save
# ════════════════════════════════════════════════════════════════
section("STEP 8 — Saving v2 splits")

df_train4.to_parquet(os.path.join(OUTPUT_DIR, "train4_v2.parquet"), index=False)
df_val4.to_parquet(os.path.join(OUTPUT_DIR,   "val4_v2.parquet"),   index=False)
df_test4.to_parquet(os.path.join(OUTPUT_DIR,  "test4_v2.parquet"),  index=False)

weights_path = os.path.join(OUTPUT_DIR, "class_weights4_v2.txt")
with open(weights_path, "w", encoding="utf-8") as f:
    f.write("# Class weights for 4-class v2 training (expanded no_bug pool)\n")
    f.write("# Format: label,class_name,weight\n")
    for i, w in class_weights.items():
        f.write(f"{i},{INT_TO_CLASS[i]},{w:.6f}\n")

print(f"  [saved] train4_v2.parquet ({len(df_train4):,} rows)")
print(f"  [saved] val4_v2.parquet   ({len(df_val4):,} rows)")
print(f"  [saved] test4_v2.parquet  ({len(df_test4):,} rows)")
print(f"  [saved] {weights_path}")


# ════════════════════════════════════════════════════════════════
#  STEP 9 — Visualise
# ════════════════════════════════════════════════════════════════
section("STEP 9 — Visualising class distribution")

COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

counts = [int((df_train4["error_class"] == c).sum()) for c in CLASS_NAMES]
short  = ["syntax", "logic", "variable\nmisuse", "no_bug"]
bars = axes[0].bar(short, counts, color=COLORS, edgecolor="white")
axes[0].set_title("4-class v2 training set distribution")
axes[0].set_ylabel("Count")
for bar, v in zip(bars, counts):
    axes[0].text(bar.get_x() + bar.get_width() / 2, v + 30, f"{v:,}",
                 ha="center", fontsize=10)

splits = ["Train4_v2", "Val4_v2", "Test4_v2"]
sizes  = [len(df_train4), len(df_val4), len(df_test4)]
axes[1].bar(splits, sizes, color=["#4C72B0", "#CCB974", "#55A868"], edgecolor="white")
axes[1].set_title("Dataset split sizes (4-class v2)")
axes[1].set_ylabel("Samples")
for i, v in enumerate(sizes):
    axes[1].text(i, v + 30, f"{v:,}", ha="center", fontsize=10)

# no_bug composition by source per split
src_counts = {
    src: [int((df["source"] == src).sum()) for df in (df_train4, df_val4, df_test4)]
    for src in ("leetcode", "flytech")
}
x = np.arange(3)
axes[2].bar(x - 0.2, src_counts["leetcode"], width=0.4, label="leetcode", color="#55A868")
axes[2].bar(x + 0.2, src_counts["flytech"],  width=0.4, label="flytech",  color="#DD8452")
axes[2].set_xticks(x)
axes[2].set_xticklabels(["Train", "Val", "Test"])
axes[2].set_title("no_bug composition by source")
axes[2].set_ylabel("Samples")
axes[2].legend()

fig.suptitle("Phase 4 — 4-Class Data Preparation v2 (expanded no_bug)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
chart_path = os.path.join(OUTPUT_DIR, "label_distribution4_v2.png")
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  [saved] {chart_path}")

print(f"""
{'=' * 60}
  4-CLASS DATA PREP v2 COMPLETE
{'=' * 60}
  no_bug pool: leetcode {len(lc_unique):,} + flytech {len(fly_selected):,}
  (flytech was fence-stripped, ast.parse-filtered, deduplicated)

  Next: run codebert_retrain_4class_v2.py
{'=' * 60}
""")
