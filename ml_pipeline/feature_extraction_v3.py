"""
Phase 5 — Step 2: Feature Extraction v3 (grouped, leakage-free splits)
=====================================================================
Project : code-analyzer-ai
Author  : Nguyen Ngoc Gia Han (GCS230054)

Identical structural/token/behaviour feature set as feature_extraction.py, but
reads the grouped 3-class bug splits produced by data_prep_grouped_v3.py
(train14_v3 / val14_v3 / test14_v3) and writes *_features_v3.parquet.

Self-contained on purpose: importing feature_extraction.py would re-run the v2
extraction (that module executes at import time), so the extractors are copied
here unchanged to keep the two pipelines independent.

Output (ml_pipeline/data/processed/):
  train_features_v3.parquet / val_features_v3.parquet / test_features_v3.parquet
  feature_names_v3.txt

HOW TO RUN:
  python ml_pipeline/feature_extraction_v3.py
"""

import os, ast, re, tokenize, io, sys
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

PROCESSED_DIR = r"F:\Heulwen\IT GREENWICH\Final Project\Tutor_AI_code\ml_pipeline\data\processed"
OUTPUT_DIR    = PROCESSED_DIR


def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ── AST features (18) ──────────────────────────────────────────────────
class ASTFeatureExtractor(ast.NodeVisitor):
    def __init__(self):
        self.counts = {
            "n_functions": 0, "n_classes": 0, "n_for_loops": 0, "n_while_loops": 0,
            "n_if_stmts": 0, "n_return_stmts": 0, "n_assignments": 0, "n_augassign": 0,
            "n_calls": 0, "n_comparisons": 0, "n_try_except": 0, "n_list_comp": 0,
            "n_dict_comp": 0, "n_lambda": 0, "n_imports": 0, "n_assert": 0,
            "n_raise": 0, "max_depth": 0,
        }
        self._depth = 0

    def generic_visit(self, node):
        self._depth += 1
        self.counts["max_depth"] = max(self.counts["max_depth"], self._depth)
        super().generic_visit(node)
        self._depth -= 1

    def visit_FunctionDef(self, n):       self.counts["n_functions"] += 1;   self.generic_visit(n)
    def visit_AsyncFunctionDef(self, n):  self.counts["n_functions"] += 1;   self.generic_visit(n)
    def visit_ClassDef(self, n):          self.counts["n_classes"] += 1;     self.generic_visit(n)
    def visit_For(self, n):               self.counts["n_for_loops"] += 1;   self.generic_visit(n)
    def visit_While(self, n):             self.counts["n_while_loops"] += 1; self.generic_visit(n)
    def visit_If(self, n):                self.counts["n_if_stmts"] += 1;    self.generic_visit(n)
    def visit_Return(self, n):            self.counts["n_return_stmts"] += 1;self.generic_visit(n)
    def visit_Assign(self, n):            self.counts["n_assignments"] += 1; self.generic_visit(n)
    def visit_AugAssign(self, n):         self.counts["n_augassign"] += 1;   self.generic_visit(n)
    def visit_Call(self, n):              self.counts["n_calls"] += 1;       self.generic_visit(n)
    def visit_Compare(self, n):           self.counts["n_comparisons"] += 1; self.generic_visit(n)
    def visit_Try(self, n):               self.counts["n_try_except"] += 1;  self.generic_visit(n)
    def visit_ListComp(self, n):          self.counts["n_list_comp"] += 1;   self.generic_visit(n)
    def visit_DictComp(self, n):          self.counts["n_dict_comp"] += 1;   self.generic_visit(n)
    def visit_Lambda(self, n):            self.counts["n_lambda"] += 1;      self.generic_visit(n)
    def visit_Import(self, n):            self.counts["n_imports"] += 1;     self.generic_visit(n)
    def visit_ImportFrom(self, n):        self.counts["n_imports"] += 1;     self.generic_visit(n)
    def visit_Assert(self, n):            self.counts["n_assert"] += 1;      self.generic_visit(n)
    def visit_Raise(self, n):             self.counts["n_raise"] += 1;       self.generic_visit(n)


def extract_ast_features(code: str) -> dict:
    extractor = ASTFeatureExtractor()
    try:
        extractor.visit(ast.parse(code))
    except SyntaxError:
        pass   # unparseable (buggy) code -> zeros
    return extractor.counts


# ── Token features (8) ─────────────────────────────────────────────────
def extract_token_features(code: str) -> dict:
    features = {
        "n_tokens": 0, "n_name_tokens": 0, "n_op_tokens": 0, "n_string_tokens": 0,
        "n_number_tokens": 0, "n_comment_tokens": 0, "n_unique_names": 0,
        "avg_name_length": 0.0,
    }
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
        names = []
        for tok in tokens:
            features["n_tokens"] += 1
            if tok.type == tokenize.NAME:
                features["n_name_tokens"] += 1; names.append(tok.string)
            elif tok.type == tokenize.OP:      features["n_op_tokens"] += 1
            elif tok.type == tokenize.STRING:  features["n_string_tokens"] += 1
            elif tok.type == tokenize.NUMBER:  features["n_number_tokens"] += 1
            elif tok.type == tokenize.COMMENT: features["n_comment_tokens"] += 1
        if names:
            features["n_unique_names"]  = len(set(names))
            features["avg_name_length"] = round(sum(len(n) for n in names) / len(names), 3)
    except tokenize.TokenError:
        pass
    return features


# ── Behaviour features (11) ────────────────────────────────────────────
def extract_behaviour_features(code: str) -> dict:
    lines = code.split("\n")
    non_empty = [l for l in lines if l.strip()]
    indent_depths = [len(l) - len(l.lstrip()) for l in non_empty if l.lstrip()]
    max_indent  = max(indent_depths) if indent_depths else 0
    mean_indent = round(sum(indent_depths) / len(indent_depths), 2) if indent_depths else 0
    single_letter = len(re.findall(r'\b[a-zA-Z]\b', code))
    comment_lines = sum(1 for l in lines if l.strip().startswith("#"))
    comment_ratio = round(comment_lines / len(lines), 3) if lines else 0
    has_type_hints = int(bool(re.search(r'\)\s*->\s*\w+', code)))
    has_list_comp  = int(bool(re.search(r'\[.+for.+in', code)))
    has_generator  = int(bool(re.search(r'\(.+for.+in', code)))
    has_recursion  = int(bool(re.search(r'def\s+(\w+).*:\s*(?:.*\n)*?.*\1\s*\(', code)))
    magic_numbers  = len(re.findall(r'(?<!\w)\d{2,}(?!\w)', code))
    builtins_used  = len(re.findall(
        r'\b(map|filter|zip|enumerate|sorted|reversed|any|all|sum|max|min|len)\s*\(', code))
    KEYWORDS = {"True","False","None","and","or","not","in","is","if","else","elif",
                "for","while","with","return","def","class","import","from","pass",
                "break","continue"}
    all_names = re.findall(r'\b([a-zA-Z_]\w*)\b', code)
    descriptive = sum(1 for n in all_names if len(n) > 4 and n not in KEYWORDS)
    descriptive_ratio = round(descriptive / max(len(all_names), 1), 3)
    return {
        "max_indent_depth": max_indent, "mean_indent_depth": mean_indent,
        "single_letter_vars": single_letter, "comment_ratio": comment_ratio,
        "has_type_hints": has_type_hints, "has_list_comp": has_list_comp,
        "has_generator": has_generator, "has_recursion": has_recursion,
        "magic_numbers": magic_numbers, "builtins_used": builtins_used,
        "descriptive_ratio": descriptive_ratio,
    }


def extract_all_features(code: str) -> dict:
    feats = {}
    feats.update(extract_ast_features(code))
    feats.update(extract_token_features(code))
    feats.update(extract_behaviour_features(code))
    feats["code_len"]     = len(code)
    feats["line_count"]   = code.count("\n") + 1
    feats["avg_line_len"] = round(
        sum(len(l) for l in code.split("\n")) / max(code.count("\n") + 1, 1), 2)
    return feats


# ── Process the grouped 3-class bug splits ─────────────────────────────
section("Loading grouped v3 splits (bugs only)")
df_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "train14_v3.parquet"))
df_val   = pd.read_parquet(os.path.join(PROCESSED_DIR, "val14_v3.parquet"))
df_test  = pd.read_parquet(os.path.join(PROCESSED_DIR, "test14_v3.parquet"))
print(f"Train: {len(df_train):,}  Val: {len(df_val):,}  Test: {len(df_test):,}")


def process_split(df, name):
    section(f"Extracting features — {name} ({len(df):,} rows)")
    rows = []
    for i, code in enumerate(df["code"]):
        rows.append(extract_all_features(str(code)))
        if (i + 1) % 1000 == 0:
            print(f"  {i+1:,} / {len(df):,} done")
    out = pd.DataFrame(rows)
    out["label"]       = df["label"].values
    out["error_class"] = df["error_class"].values
    path = os.path.join(OUTPUT_DIR, f"{name}_features_v3.parquet")
    out.to_parquet(path, index=False)
    print(f"  [saved] {path}  ({len(out):,} rows x {len(out.columns)} cols)")
    return out


tr = process_split(df_train, "train")
process_split(df_val, "val")
process_split(df_test, "test")

feat_cols = [c for c in tr.columns if c not in ["label", "error_class"]]
with open(os.path.join(OUTPUT_DIR, "feature_names_v3.txt"), "w", encoding="utf-8") as f:
    f.write("# Feature names for v3 baseline classifiers (grouped split)\n\n")
    for c in feat_cols:
        f.write(c + "\n")
print(f"\n[saved] feature_names_v3.txt  ({len(feat_cols)} features)")
print(f"\nFEATURE EXTRACTION v3 COMPLETE — next: baseline_classifiers_v3.py")
