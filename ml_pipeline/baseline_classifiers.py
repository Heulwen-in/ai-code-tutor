"""
Phase 2 — Step 3: Baseline Classifiers
=======================================
Project : code-analyzer-ai
Author  : Nguyen Ngoc Gia Han (GCS230054)

Trains and evaluates 3 baseline classifiers on extracted features:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Decision Tree

Also trains a TF-IDF + classifier pipeline as Pipeline B.

Each model is evaluated with:
  - Accuracy, Macro F1, Weighted F1
  - Per-class Precision / Recall / F1
  - Confusion matrix
  - Training time

Output (ml_pipeline/data/processed/):
  - baseline_results.txt       — all metrics
  - confusion_matrices.png     — 2×3 grid of confusion matrices
  - model_comparison.png       — bar chart comparison
  - best_baseline.pkl          — best model saved for comparison with CodeBERT

HOW TO RUN:
  pip install pandas pyarrow scikit-learn matplotlib seaborn joblib
  python ml_pipeline/baseline_classifiers.py
"""

import os, time, joblib, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model    import LogisticRegression
from sklearn.svm             import LinearSVC
from sklearn.tree            import DecisionTreeClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics         import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────────
PROCESSED_DIR = r"F:\Heulwen\IT GREENWICH\Final Project\Tutor_AI_code\ml_pipeline\data\processed"
MODEL_DIR     = r"F:\Heulwen\IT GREENWICH\Final Project\Tutor_AI_code\backend\app\ml_models"
OUTPUT_DIR    = PROCESSED_DIR
os.makedirs(MODEL_DIR, exist_ok=True)
# ────────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", palette="muted")

CLASS_NAMES = ["syntax_error", "logic_error", "variable_misuse"]
INT_TO_CLASS = {0: "syntax_error", 1: "logic_error", 2: "variable_misuse"}

# Drop features that showed zero variance across classes
DROP_FEATURES = ["comment_ratio", "n_assert", "n_raise", "n_try_except"]

def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ════════════════════════════════════════════════════════════════
#  LOAD DATA
# ════════════════════════════════════════════════════════════════
section("Loading feature sets")

df_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "train_features.parquet"))
df_val   = pd.read_parquet(os.path.join(PROCESSED_DIR, "val_features.parquet"))
df_test  = pd.read_parquet(os.path.join(PROCESSED_DIR, "test_features.parquet"))

# Load raw code for TF-IDF pipeline
df_train_raw = pd.read_parquet(os.path.join(PROCESSED_DIR, "train.parquet"))
df_val_raw   = pd.read_parquet(os.path.join(PROCESSED_DIR, "val.parquet"))
df_test_raw  = pd.read_parquet(os.path.join(PROCESSED_DIR, "test.parquet"))

print(f"Train : {len(df_train):,} rows × {df_train.shape[1]} cols")
print(f"Val   : {len(df_val):,} rows")
print(f"Test  : {len(df_test):,} rows")

# Load class weights
weight_file = os.path.join(PROCESSED_DIR, "class_weights.txt")
class_weights = {}
with open(weight_file) as f:
    for line in f:
        if line.startswith("#"): continue
        parts = line.strip().split(",")
        if len(parts) == 3:
            class_weights[int(parts[0])] = float(parts[2])
print(f"\nClass weights loaded: {class_weights}")


# ════════════════════════════════════════════════════════════════
#  PREPARE FEATURE MATRIX
# ════════════════════════════════════════════════════════════════

feat_cols = [c for c in df_train.columns
             if c not in ["label","error_class"] + DROP_FEATURES]

X_train = df_train[feat_cols].values.astype(float)
y_train = df_train["label"].values
X_val   = df_val[feat_cols].values.astype(float)
y_val   = df_val["label"].values
X_test  = df_test[feat_cols].values.astype(float)
y_test  = df_test["label"].values

# Raw code for TF-IDF
code_train = df_train_raw["code"].astype(str).values
code_val   = df_val_raw["code"].astype(str).values
code_test  = df_test_raw["code"].astype(str).values

print(f"\nFeature matrix: {X_train.shape[1]} features (after dropping {len(DROP_FEATURES)} zero-variance)")
print(f"Dropped: {DROP_FEATURES}")


# ════════════════════════════════════════════════════════════════
#  EVALUATION HELPER
# ════════════════════════════════════════════════════════════════

def evaluate(model_name, y_true, y_pred, split="test"):
    acc       = accuracy_score(y_true, y_pred)
    macro_f1  = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    report    = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        output_dict=True
    )
    return {
        "model"        : model_name,
        "split"        : split,
        "accuracy"     : round(acc, 4),
        "macro_f1"     : round(macro_f1, 4),
        "weighted_f1"  : round(weighted_f1, 4),
        "syntax_f1"    : round(report["syntax_error"]["f1-score"], 4),
        "logic_f1"     : round(report["logic_error"]["f1-score"], 4),
        "variable_f1"  : round(report["variable_misuse"]["f1-score"], 4),
        "syntax_prec"  : round(report["syntax_error"]["precision"], 4),
        "logic_prec"   : round(report["logic_error"]["precision"], 4),
        "variable_prec": round(report["variable_misuse"]["precision"], 4),
        "syntax_rec"   : round(report["syntax_error"]["recall"], 4),
        "logic_rec"    : round(report["logic_error"]["recall"], 4),
        "variable_rec" : round(report["variable_misuse"]["recall"], 4),
    }


# ════════════════════════════════════════════════════════════════
#  PIPELINE A — STRUCTURAL FEATURES + CLASSICAL ML
# ════════════════════════════════════════════════════════════════
section("PIPELINE A — Structural features + Classical ML")

models_A = {
    "LR (structural)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            max_iter=1000, random_state=42,
            class_weight=class_weights, C=1.0))
    ]),
    "SVM (structural)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LinearSVC(
            max_iter=2000, random_state=42,
            class_weight=class_weights, C=1.0))
    ]),
    "DecisionTree (structural)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    DecisionTreeClassifier(
            random_state=42, max_depth=15,
            class_weight=class_weights))
    ]),
}

results = []
trained_models = {}

for name, model in models_A.items():
    print(f"\n  Training: {name}")
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = round(time.time() - t0, 2)

    y_val_pred  = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    val_metrics  = evaluate(name, y_val, y_val_pred, "val")
    test_metrics = evaluate(name, y_test, y_test_pred, "test")
    test_metrics["train_time_s"] = train_time

    results.append(val_metrics)
    results.append(test_metrics)
    trained_models[name] = (model, y_test_pred)

    print(f"  Train time : {train_time}s")
    print(f"  Val   accuracy={val_metrics['accuracy']:.4f}  macro_f1={val_metrics['macro_f1']:.4f}")
    print(f"  Test  accuracy={test_metrics['accuracy']:.4f}  macro_f1={test_metrics['macro_f1']:.4f}")
    print(f"  Per-class F1  syntax={test_metrics['syntax_f1']:.4f}  "
          f"logic={test_metrics['logic_f1']:.4f}  variable={test_metrics['variable_f1']:.4f}")


# ════════════════════════════════════════════════════════════════
#  PIPELINE B — TF-IDF + CLASSICAL ML
# ════════════════════════════════════════════════════════════════
section("PIPELINE B — TF-IDF on raw code + Classical ML")

# Code-aware tokenizer: split on Python operators and whitespace
def python_tokenizer(code):
    tokens = []
    for tok in code.replace("\n"," ").replace("(", " ( ").replace(
            ")", " ) ").replace(":", " : ").replace(",", " , ").split():
        if tok.strip():
            tokens.append(tok.strip())
    return tokens

models_B = {
    "LR (TF-IDF)": Pipeline([
        ("tfidf", TfidfVectorizer(
            tokenizer=python_tokenizer,
            token_pattern=None,
            max_features=5000,
            ngram_range=(1, 2),
            sublinear_tf=True)),
        ("clf", LogisticRegression(
            max_iter=1000, random_state=42,
            class_weight=class_weights, C=1.0))
    ]),
    "SVM (TF-IDF)": Pipeline([
        ("tfidf", TfidfVectorizer(
            tokenizer=python_tokenizer,
            token_pattern=None,
            max_features=5000,
            ngram_range=(1, 2),
            sublinear_tf=True)),
        ("clf", LinearSVC(
            max_iter=2000, random_state=42,
            class_weight=class_weights, C=1.0))
    ]),
    "DecisionTree (TF-IDF)": Pipeline([
        ("tfidf", TfidfVectorizer(
            tokenizer=python_tokenizer,
            token_pattern=None,
            max_features=5000,
            sublinear_tf=True)),
        ("clf", DecisionTreeClassifier(
            random_state=42, max_depth=15,
            class_weight=class_weights))
    ]),
}

for name, model in models_B.items():
    print(f"\n  Training: {name}")
    t0 = time.time()
    model.fit(code_train, y_train)
    train_time = round(time.time() - t0, 2)

    y_val_pred  = model.predict(code_val)
    y_test_pred = model.predict(code_test)

    val_metrics  = evaluate(name, y_val, y_val_pred, "val")
    test_metrics = evaluate(name, y_test, y_test_pred, "test")
    test_metrics["train_time_s"] = train_time

    results.append(val_metrics)
    results.append(test_metrics)
    trained_models[name] = (model, y_test_pred)

    print(f"  Train time : {train_time}s")
    print(f"  Val   accuracy={val_metrics['accuracy']:.4f}  macro_f1={val_metrics['macro_f1']:.4f}")
    print(f"  Test  accuracy={test_metrics['accuracy']:.4f}  macro_f1={test_metrics['macro_f1']:.4f}")
    print(f"  Per-class F1  syntax={test_metrics['syntax_f1']:.4f}  "
          f"logic={test_metrics['logic_f1']:.4f}  variable={test_metrics['variable_f1']:.4f}")


# ════════════════════════════════════════════════════════════════
#  FULL RESULTS TABLE
# ════════════════════════════════════════════════════════════════
section("Results Summary")

df_results = pd.DataFrame(results)
test_results = df_results[df_results["split"] == "test"].copy()
test_results = test_results.sort_values("macro_f1", ascending=False).reset_index(drop=True)

print("\nTest set results (sorted by macro F1):")
display_cols = ["model","accuracy","macro_f1","weighted_f1",
                "syntax_f1","logic_f1","variable_f1"]
print(test_results[display_cols].to_string(index=False))

# Best model
best_row  = test_results.iloc[0]
best_name = best_row["model"]
print(f"\n★  Best baseline: {best_name}")
print(f"   Macro F1  : {best_row['macro_f1']:.4f}")
print(f"   Accuracy  : {best_row['accuracy']:.4f}")

# Save best model
best_model = trained_models[best_name][0]
model_path = os.path.join(MODEL_DIR, "best_baseline.pkl")
joblib.dump(best_model, model_path)
print(f"   [saved] {model_path}")


# ════════════════════════════════════════════════════════════════
#  SAVE FULL RESULTS TEXT
# ════════════════════════════════════════════════════════════════
results_path = os.path.join(OUTPUT_DIR, "baseline_results.txt")
with open(results_path, "w") as f:
    f.write("Baseline Classifier Results\n")
    f.write("=" * 60 + "\n\n")
    for _, row in test_results.iterrows():
        f.write(f"Model: {row['model']}\n")
        f.write(f"  Accuracy    : {row['accuracy']:.4f}\n")
        f.write(f"  Macro F1    : {row['macro_f1']:.4f}\n")
        f.write(f"  Weighted F1 : {row['weighted_f1']:.4f}\n")
        f.write(f"  syntax_f1   : {row['syntax_f1']:.4f}\n")
        f.write(f"  logic_f1    : {row['logic_f1']:.4f}\n")
        f.write(f"  variable_f1 : {row['variable_f1']:.4f}\n")
        f.write("\n")
print(f"\n[saved] {results_path}")


# ════════════════════════════════════════════════════════════════
#  VISUALISATIONS
# ════════════════════════════════════════════════════════════════
section("Generating charts")

# 1. Confusion matrices (test set, 2 rows × 3 cols)
test_model_names = [n for n in trained_models.keys()]
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, (name, (model, y_pred)) in enumerate(trained_models.items()):
    if i >= 6: break
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["syntax","logic","variable"]
    )
    disp.plot(ax=axes[i], colorbar=False, cmap="Blues")
    axes[i].set_title(name, fontsize=10)
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("True")

fig.suptitle("Baseline Classifiers — Confusion Matrices (Test Set)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrices.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[saved] {cm_path}")

# 2. Model comparison bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Macro F1 comparison
models_short = [n.replace(" (structural)","").replace(" (TF-IDF)","")
                for n in test_results["model"]]
pipeline_type = ["Structural" if "structural" in n else "TF-IDF"
                 for n in test_results["model"]]
colors = ["#4C72B0" if p == "Structural" else "#55A868" for p in pipeline_type]

bars = axes[0].barh(
    [f"{m}\n({p})" for m, p in zip(models_short, pipeline_type)],
    test_results["macro_f1"].values,
    color=colors, edgecolor="white"
)
axes[0].set_title("Macro F1 — all baselines")
axes[0].set_xlabel("Macro F1")
axes[0].set_xlim(0, 1)
for bar, v in zip(bars, test_results["macro_f1"].values):
    axes[0].text(v + 0.005, bar.get_y() + bar.get_height()/2,
                 f"{v:.4f}", va="center", fontsize=9)

# Per-class F1 for best model
best_cls_f1 = [best_row["syntax_f1"], best_row["logic_f1"], best_row["variable_f1"]]
cls_colors  = ["#4C72B0","#55A868","#C44E52"]
bars2 = axes[1].bar(["Syntax","Logic","Variable\nMisuse"],
                    best_cls_f1, color=cls_colors, edgecolor="white")
axes[1].set_title(f"Per-class F1 — best model\n({best_name})")
axes[1].set_ylabel("F1 Score")
axes[1].set_ylim(0, 1)
for bar, v in zip(bars2, best_cls_f1):
    axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.01,
                 f"{v:.4f}", ha="center", fontsize=10)

fig.suptitle("Baseline Classifiers — Performance Comparison",
             fontsize=13, fontweight="bold")
plt.tight_layout()
comp_path = os.path.join(OUTPUT_DIR, "model_comparison.png")
plt.savefig(comp_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[saved] {comp_path}")


print(f"""
{'=' * 60}
  BASELINE CLASSIFIERS COMPLETE
{'=' * 60}
  Best baseline : {best_name}
  Macro F1      : {best_row['macro_f1']:.4f}
  Accuracy      : {best_row['accuracy']:.4f}

  Key observation for thesis:
  - If logic_f1 ≈ variable_f1 → structural features alone
    cannot distinguish these two classes → justifies CodeBERT

  Next script: ml_pipeline/codebert_classifier.py
  Best model saved to: backend/app/ml_models/best_baseline.pkl
{'=' * 60}
""")