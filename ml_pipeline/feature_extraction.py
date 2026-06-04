"""
Phase 2 — Step 2: Feature Extraction
=====================================
Project : code-analyzer-ai
Author  : Nguyen Ngoc Gia Han (GCS230054)

Extracts two feature sets from processed splits:

  A) Structural features (for baseline classifiers)
     - AST node type counts
     - Token-level statistics
     - Code complexity metrics

  B) User behaviour features (for skill_detector.py)
     - Coding style signals
     - Used to infer novice vs professional

Output (ml_pipeline/data/processed/):
  - train_features.parquet
  - val_features.parquet
  - test_features.parquet
  - feature_importance_preview.png
  - feature_names.txt

HOW TO RUN:
  pip install pandas pyarrow scikit-learn matplotlib seaborn
  python ml_pipeline/feature_extraction.py
"""

import os, ast, re, tokenize, io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── CONFIG ─────────────────────────────────────────────────────
PROCESSED_DIR = r"F:\Heulwen\IT GREENWICH\Final Project\Tutor_AI_code\ml_pipeline\data\processed"
OUTPUT_DIR    = PROCESSED_DIR   # save to same folder
# ───────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", palette="muted")

def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

# ════════════════════════════════════════════════════════════════
#  AST FEATURE EXTRACTOR
# ════════════════════════════════════════════════════════════════

class ASTFeatureExtractor(ast.NodeVisitor):
    """Walk the AST and count node types."""
    def __init__(self):
        self.counts = {
            "n_functions"   : 0,
            "n_classes"     : 0,
            "n_for_loops"   : 0,
            "n_while_loops" : 0,
            "n_if_stmts"    : 0,
            "n_return_stmts": 0,
            "n_assignments" : 0,
            "n_augassign"   : 0,
            "n_calls"       : 0,
            "n_comparisons" : 0,
            "n_try_except"  : 0,
            "n_list_comp"   : 0,
            "n_dict_comp"   : 0,
            "n_lambda"      : 0,
            "n_imports"     : 0,
            "n_assert"      : 0,
            "n_raise"       : 0,
            "max_depth"     : 0,
        }
        self._depth = 0

    def generic_visit(self, node):
        self._depth += 1
        self.counts["max_depth"] = max(self.counts["max_depth"], self._depth)
        super().generic_visit(node)
        self._depth -= 1

    def visit_FunctionDef(self, node):
        self.counts["n_functions"] += 1
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self.counts["n_functions"] += 1
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.counts["n_classes"] += 1
        self.generic_visit(node)

    def visit_For(self, node):
        self.counts["n_for_loops"] += 1
        self.generic_visit(node)

    def visit_While(self, node):
        self.counts["n_while_loops"] += 1
        self.generic_visit(node)

    def visit_If(self, node):
        self.counts["n_if_stmts"] += 1
        self.generic_visit(node)

    def visit_Return(self, node):
        self.counts["n_return_stmts"] += 1
        self.generic_visit(node)

    def visit_Assign(self, node):
        self.counts["n_assignments"] += 1
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        self.counts["n_augassign"] += 1
        self.generic_visit(node)

    def visit_Call(self, node):
        self.counts["n_calls"] += 1
        self.generic_visit(node)

    def visit_Compare(self, node):
        self.counts["n_comparisons"] += 1
        self.generic_visit(node)

    def visit_Try(self, node):
        self.counts["n_try_except"] += 1
        self.generic_visit(node)

    def visit_ListComp(self, node):
        self.counts["n_list_comp"] += 1
        self.generic_visit(node)

    def visit_DictComp(self, node):
        self.counts["n_dict_comp"] += 1
        self.generic_visit(node)

    def visit_Lambda(self, node):
        self.counts["n_lambda"] += 1
        self.generic_visit(node)

    def visit_Import(self, node):
        self.counts["n_imports"] += 1
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        self.counts["n_imports"] += 1
        self.generic_visit(node)

    def visit_Assert(self, node):
        self.counts["n_assert"] += 1
        self.generic_visit(node)

    def visit_Raise(self, node):
        self.counts["n_raise"] += 1
        self.generic_visit(node)


def extract_ast_features(code: str) -> dict:
    """Return AST feature dict. Returns zeros if code cannot be parsed."""
    extractor = ASTFeatureExtractor()
    try:
        tree = ast.parse(code)
        extractor.visit(tree)
    except SyntaxError:
        pass   # return zeros — bugged syntax code won't parse
    return extractor.counts


# ════════════════════════════════════════════════════════════════
#  TOKEN FEATURE EXTRACTOR
# ════════════════════════════════════════════════════════════════

def extract_token_features(code: str) -> dict:
    """Count token types using Python's tokenize module."""
    features = {
        "n_tokens"       : 0,
        "n_name_tokens"  : 0,   # identifiers
        "n_op_tokens"    : 0,   # operators
        "n_string_tokens": 0,
        "n_number_tokens": 0,
        "n_comment_tokens":0,
        "n_unique_names" : 0,   # vocabulary size
        "avg_name_length": 0.0, # naming style proxy
    }
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
        names = []
        for tok in tokens:
            ttype = tok.type
            features["n_tokens"] += 1
            if ttype == tokenize.NAME:
                features["n_name_tokens"] += 1
                names.append(tok.string)
            elif ttype == tokenize.OP:
                features["n_op_tokens"] += 1
            elif ttype == tokenize.STRING:
                features["n_string_tokens"] += 1
            elif ttype == tokenize.NUMBER:
                features["n_number_tokens"] += 1
            elif ttype == tokenize.COMMENT:
                features["n_comment_tokens"] += 1

        if names:
            features["n_unique_names"]  = len(set(names))
            features["avg_name_length"] = round(
                sum(len(n) for n in names) / len(names), 3)
    except tokenize.TokenError:
        pass
    return features


# ════════════════════════════════════════════════════════════════
#  USER BEHAVIOUR FEATURES  (→ skill_detector.py)
# ════════════════════════════════════════════════════════════════

def extract_behaviour_features(code: str) -> dict:
    """
    Signals that indicate coding experience level.
    These feed into skill_detector.py (novice / professional).
    """
    lines = code.split("\n")
    non_empty = [l for l in lines if l.strip()]

    # Indentation depth (novices often use inconsistent indentation)
    indent_depths = []
    for line in non_empty:
        stripped = line.lstrip()
        if stripped:
            indent_depths.append(len(line) - len(stripped))
    max_indent  = max(indent_depths) if indent_depths else 0
    mean_indent = round(sum(indent_depths) / len(indent_depths), 2) if indent_depths else 0

    # Variable naming: single-letter names = novice signal
    single_letter = len(re.findall(r'\b[a-zA-Z]\b', code))

    # Comment ratio
    comment_lines = sum(1 for l in lines if l.strip().startswith("#"))
    comment_ratio = round(comment_lines / len(lines), 3) if lines else 0

    # Type hint usage (professional signal)
    has_type_hints = int(bool(re.search(r'\)\s*->\s*\w+', code)))

    # List comprehension / generator usage (professional signal)
    has_list_comp  = int(bool(re.search(r'\[.+for.+in', code)))
    has_generator  = int(bool(re.search(r'\(.+for.+in', code)))

    # Recursion (base case pattern)
    has_recursion  = int(bool(re.search(r'def\s+(\w+).*:\s*(?:.*\n)*?.*\1\s*\(', code)))

    # Magic numbers (novice signal: hardcoded literals)
    magic_numbers  = len(re.findall(r'(?<!\w)\d{2,}(?!\w)', code))

    # Use of built-in functions (professional signal)
    builtins_used  = len(re.findall(
        r'\b(map|filter|zip|enumerate|sorted|reversed|any|all|sum|max|min|len)\s*\(', code))

    # Descriptive variable names (> 4 chars, not keywords)
    KEYWORDS = {"True","False","None","and","or","not","in","is",
                "if","else","elif","for","while","with","return",
                "def","class","import","from","pass","break","continue"}
    all_names = re.findall(r'\b([a-zA-Z_]\w*)\b', code)
    descriptive = sum(1 for n in all_names if len(n) > 4 and n not in KEYWORDS)
    descriptive_ratio = round(descriptive / max(len(all_names), 1), 3)

    return {
        "max_indent_depth"   : max_indent,
        "mean_indent_depth"  : mean_indent,
        "single_letter_vars" : single_letter,
        "comment_ratio"      : comment_ratio,
        "has_type_hints"     : has_type_hints,
        "has_list_comp"      : has_list_comp,
        "has_generator"      : has_generator,
        "has_recursion"      : has_recursion,
        "magic_numbers"      : magic_numbers,
        "builtins_used"      : builtins_used,
        "descriptive_ratio"  : descriptive_ratio,
    }


# ════════════════════════════════════════════════════════════════
#  COMBINED EXTRACTOR
# ════════════════════════════════════════════════════════════════

def extract_all_features(code: str) -> dict:
    """Run all three extractors and merge into one flat dict."""
    feats = {}
    feats.update(extract_ast_features(code))
    feats.update(extract_token_features(code))
    feats.update(extract_behaviour_features(code))

    # Raw code metrics
    feats["code_len"]   = len(code)
    feats["line_count"] = code.count("\n") + 1
    feats["avg_line_len"] = round(
        sum(len(l) for l in code.split("\n")) / max(code.count("\n") + 1, 1), 2)

    return feats


# ════════════════════════════════════════════════════════════════
#  PROCESS EACH SPLIT
# ════════════════════════════════════════════════════════════════

section("Loading processed splits")
df_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "train.parquet"))
df_val   = pd.read_parquet(os.path.join(PROCESSED_DIR, "val.parquet"))
df_test  = pd.read_parquet(os.path.join(PROCESSED_DIR, "test.parquet"))
print(f"Train: {len(df_train):,}  Val: {len(df_val):,}  Test: {len(df_test):,}")


def process_split(df, split_name):
    section(f"Extracting features — {split_name} ({len(df):,} rows)")
    print("  This may take 1-3 minutes...")

    feature_rows = []
    for i, code in enumerate(df["code"]):
        feature_rows.append(extract_all_features(str(code)))
        if (i + 1) % 1000 == 0:
            print(f"  {i+1:,} / {len(df):,} done")

    df_feats = pd.DataFrame(feature_rows)

    # Add label and class columns back
    df_out = df_feats.copy()
    df_out["label"]       = df["label"].values
    df_out["error_class"] = df["error_class"].values

    # Save
    out_path = os.path.join(OUTPUT_DIR, f"{split_name}_features.parquet")
    df_out.to_parquet(out_path, index=False)
    print(f"  [saved] {out_path}  ({len(df_out):,} rows × {len(df_out.columns)} cols)")

    return df_out


df_train_feats = process_split(df_train, "train")
df_val_feats   = process_split(df_val,   "val")
df_test_feats  = process_split(df_test,  "test")


# ════════════════════════════════════════════════════════════════
#  FEATURE ANALYSIS
# ════════════════════════════════════════════════════════════════

section("Feature analysis per class")

feature_cols = [c for c in df_train_feats.columns if c not in ["label","error_class"]]
print(f"Total features extracted: {len(feature_cols)}")
print(f"\nFeature list:")
for i, f in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {f}")

# Mean per class
print(f"\nTop discriminative features (mean per class):")
means = df_train_feats.groupby("error_class")[feature_cols].mean().round(3)
print(means.T.to_string())

# Save feature names
feat_names_path = os.path.join(OUTPUT_DIR, "feature_names.txt")
with open(feat_names_path, "w") as f:
    f.write("# Feature names for baseline classifiers\n")
    f.write("# AST features, token features, behaviour features, code metrics\n\n")
    for feat in feature_cols:
        f.write(feat + "\n")
print(f"\n[saved] {feat_names_path}")


# ════════════════════════════════════════════════════════════════
#  VISUALISATION
# ════════════════════════════════════════════════════════════════

# Pick top 10 most variable features to plot
variances = df_train_feats[feature_cols].var().sort_values(ascending=False)
top10 = variances.head(10).index.tolist()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
COLORS = {"syntax_error": "#4C72B0", "logic_error": "#55A868", "variable_misuse": "#C44E52"}

# 1. Feature variance (which features differ most between samples)
axes[0,0].barh(variances.head(15).index[::-1],
               variances.head(15).values[::-1],
               color="#4C72B0", edgecolor="white")
axes[0,0].set_title("Top 15 features by variance")
axes[0,0].set_xlabel("Variance")

# 2. Mean feature values per class (top 8)
top8 = variances.head(8).index.tolist()
means_top = df_train_feats.groupby("error_class")[top8].mean()
means_top.T.plot(kind="bar", ax=axes[0,1],
                 color=list(COLORS.values()), edgecolor="white")
axes[0,1].set_title("Mean feature values per class (top 8)")
axes[0,1].tick_params(axis="x", rotation=30)
axes[0,1].legend(fontsize=8)

# 3. Behaviour features — professional vs novice signals
beh_cols = ["has_type_hints","has_list_comp","has_recursion",
            "builtins_used","comment_ratio","single_letter_vars"]
beh_means = df_train_feats.groupby("error_class")[beh_cols].mean()
beh_means.T.plot(kind="bar", ax=axes[1,0],
                 color=list(COLORS.values()), edgecolor="white")
axes[1,0].set_title("Behaviour features per class")
axes[1,0].tick_params(axis="x", rotation=30)
axes[1,0].legend(fontsize=8)

# 4. Tokens vs code length scatter
for cls, color in COLORS.items():
    sub = df_train_feats[df_train_feats["error_class"] == cls].sample(
        min(300, len(df_train_feats[df_train_feats["error_class"]==cls])), random_state=42)
    axes[1,1].scatter(sub["code_len"], sub["n_tokens"],
                      alpha=0.3, s=10, color=color, label=cls)
axes[1,1].set_title("Code length vs token count by class")
axes[1,1].set_xlabel("Code length (chars)")
axes[1,1].set_ylabel("Token count")
axes[1,1].legend(fontsize=8)

fig.suptitle("Phase 2 — Feature Extraction Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
chart_path = os.path.join(OUTPUT_DIR, "feature_analysis.png")
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[saved] {chart_path}")

print(f"""
{'=' * 60}
  FEATURE EXTRACTION COMPLETE
{'=' * 60}
  Features extracted: {len(feature_cols)}
    - AST features   : 18
    - Token features  : 8
    - Behaviour feats : 11
    - Code metrics    : 3

  Files ready for baseline classifiers:
    train_features.parquet  ({len(df_train_feats):,} rows)
    val_features.parquet    ({len(df_val_feats):,} rows)
    test_features.parquet   ({len(df_test_feats):,} rows)

  Next script: ml_pipeline/baseline_classifiers.py
{'=' * 60}
""")