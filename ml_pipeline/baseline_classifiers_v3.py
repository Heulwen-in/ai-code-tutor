"""
Phase 5 — Step 3: Baseline Classifiers v3 (grouped split, shortcut features removed)
===================================================================================
Project : code-analyzer-ai
Author  : Nguyen Ngoc Gia Han (GCS230054)

Same 10-model design as baseline_classifiers_v2.py (LR/SVM/DT/RF/XGB x
structural + TF-IDF, GridSearchCV), with TWO corrections for honest evaluation:

  1. Grouped, leakage-free splits (train_features_v3 / *_v3) from
     data_prep_grouped_v3.py — no problem-context leakage across train/test.
  2. Remediation B: the non-semantic length/provenance shortcuts
     (code_len, line_count, avg_line_len) are added to DROP_FEATURES, so the
     structural track cannot classify by snippet length instead of code meaning.

3-class task (syntax/logic/variable), identical to the v2 baseline, so v2 vs v3
is a like-for-like before/after comparison for the thesis.

Outputs (kept separate from v2 — deployed artefacts are NOT overwritten):
  ml_pipeline/data/processed/baseline_results_v3.txt
  ml_pipeline/data/processed/confusion_matrices_v3.png
  ml_pipeline/data/processed/model_comparison_v3.png
  ml_pipeline/logs/best_hyperparams_v3.json
  backend/app/ml_models/best_baseline_v3.pkl / scaler_v3.pkl / tfidf_vectorizer_v3.pkl

HOW TO RUN:
  python ml_pipeline/baseline_classifiers_v3.py
  python ml_pipeline/baseline_classifiers_v3.py --skip-tuning   (reuse saved params)
"""

import os, sys, time, json, joblib, warnings
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

SKIP_TUNING = "--skip-tuning" in sys.argv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model      import LogisticRegression
from sklearn.svm               import LinearSVC
from sklearn.tree              import DecisionTreeClassifier
from sklearn.ensemble          import RandomForestClassifier
from sklearn.preprocessing     import StandardScaler
from sklearn.model_selection   import GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics           import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────────
PROCESSED_DIR = r"F:\Heulwen\IT GREENWICH\Final Project\Tutor_AI_code\ml_pipeline\data\processed"
LOG_DIR       = r"F:\Heulwen\IT GREENWICH\Final Project\Tutor_AI_code\ml_pipeline\logs"
MODEL_DIR     = r"F:\Heulwen\IT GREENWICH\Final Project\Tutor_AI_code\backend\app\ml_models"
OUTPUT_DIR    = PROCESSED_DIR

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

CLASS_NAMES = ["syntax_error", "logic_error", "variable_misuse"]

# Remediation B: drop non-semantic shortcuts. The first four match v2; the last
# three (length/provenance) are the new exclusions that this study adds.
DROP_FEATURES = ["comment_ratio", "n_assert", "n_raise", "n_try_except",
                 "code_len", "line_count", "avg_line_len"]

sns.set_theme(style="whitegrid", palette="muted")


def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def save_model_log(name, y_test, y_pred, train_acc, val_acc, train_time, best_params):
    log_path = os.path.join(LOG_DIR, f"baseline_v3_{name.replace(' ','_').replace('/','_')}.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {name} (v3, grouped split)\n" + "=" * 50 + "\n\n")
        f.write("Best hyperparameters:\n")
        for k, v in best_params.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nTrain time    : {train_time}s\n")
        f.write(f"Train accuracy: {train_acc:.4f}\n")
        f.write(f"Val accuracy  : {val_acc:.4f}\n")
        f.write(f"Test accuracy : {accuracy_score(y_test, y_pred):.4f}\n")
        f.write(f"Macro F1      : {f1_score(y_test, y_pred, average='macro'):.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
        f.write("\nConfusion Matrix:\n  Labels: " + str(CLASS_NAMES) + "\n")
        for row in confusion_matrix(y_test, y_pred):
            f.write(f"  {list(row)}\n")
    return log_path


def evaluate_model(name, model, X_train, y_train, X_val, y_val, X_test, y_test, train_time, best_params):
    y_train_pred = model.predict(X_train)
    y_val_pred   = model.predict(X_val)
    y_test_pred  = model.predict(X_test)
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc   = accuracy_score(y_val,   y_val_pred)
    test_acc  = accuracy_score(y_test,  y_test_pred)
    report    = classification_report(y_test, y_test_pred,
                                      target_names=CLASS_NAMES, output_dict=True)
    log_path = save_model_log(name, y_test, y_test_pred, train_acc, val_acc, train_time, best_params)
    print(f"  [log saved] {log_path}")
    return {
        "model": name,
        "train_acc": round(train_acc, 4), "val_acc": round(val_acc, 4),
        "test_acc": round(test_acc, 4),
        "macro_f1": round(f1_score(y_test, y_test_pred, average="macro"), 4),
        "weighted_f1": round(f1_score(y_test, y_test_pred, average="weighted"), 4),
        "syntax_f1": round(report["syntax_error"]["f1-score"], 4),
        "logic_f1": round(report["logic_error"]["f1-score"], 4),
        "variable_f1": round(report["variable_misuse"]["f1-score"], 4),
        "train_time_s": train_time, "y_pred": y_test_pred,
    }


# ════════════════════════════════════════════════════════════════
#  LOAD DATA
# ════════════════════════════════════════════════════════════════
section("Loading grouped v3 data")

df_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "train_features_v3.parquet"))
df_val   = pd.read_parquet(os.path.join(PROCESSED_DIR, "val_features_v3.parquet"))
df_test  = pd.read_parquet(os.path.join(PROCESSED_DIR, "test_features_v3.parquet"))
df_train_raw = pd.read_parquet(os.path.join(PROCESSED_DIR, "train14_v3.parquet"))
df_val_raw   = pd.read_parquet(os.path.join(PROCESSED_DIR, "val14_v3.parquet"))
df_test_raw  = pd.read_parquet(os.path.join(PROCESSED_DIR, "test14_v3.parquet"))

feat_cols = [c for c in df_train.columns if c not in ["label", "error_class"] + DROP_FEATURES]

X_train_A = df_train[feat_cols].values.astype(float)
X_val_A   = df_val[feat_cols].values.astype(float)
X_test_A  = df_test[feat_cols].values.astype(float)
y_train   = df_train["label"].values
y_val     = df_val["label"].values
y_test    = df_test["label"].values

code_train = df_train_raw["code"].astype(str).values
code_val   = df_val_raw["code"].astype(str).values
code_test  = df_test_raw["code"].astype(str).values

class_weights = {}
with open(os.path.join(PROCESSED_DIR, "class_weights3_v3.txt")) as f:
    for line in f:
        if line.startswith("#"):
            continue
        parts = line.strip().split(",")
        if len(parts) == 3:
            class_weights[int(parts[0])] = float(parts[2])

print(f"Feature cols (shortcuts dropped): {len(feat_cols)}")
print(f"Dropped: {DROP_FEATURES}")
print(f"Train {len(y_train):,}  Val {len(y_val):,}  Test {len(y_test):,}")
print(f"Class weights: {class_weights}")

scaler_path = os.path.join(MODEL_DIR, "scaler_v3.pkl")
tfidf_path  = os.path.join(MODEL_DIR, "tfidf_vectorizer_v3.pkl")
hp_path     = os.path.join(LOG_DIR, "best_hyperparams_v3.json")


def python_tokenizer(code):
    tokens = []
    for tok in code.replace("\n", " ").replace("(", " ( ").replace(")", " ) ") \
                   .replace(":", " : ").replace(",", " , ").split():
        if tok.strip():
            tokens.append(tok.strip())
    return tokens


def build_models_from_params(p):
    return (
        LogisticRegression(max_iter=1000, random_state=42, class_weight=class_weights, **p["LR_structural"]),
        LinearSVC(max_iter=3000, random_state=42, class_weight=class_weights, **p["SVM_structural"]),
        DecisionTreeClassifier(random_state=42, class_weight=class_weights, **p["DT_structural"]),
        RandomForestClassifier(random_state=42, class_weight=class_weights, n_jobs=-1, **p["RF_structural"]),
        XGBClassifier(random_state=42, eval_metric="mlogloss", use_label_encoder=False, **p["XGB_structural"]),
        LogisticRegression(max_iter=1000, random_state=42, class_weight=class_weights, **p["LR_tfidf"]),
        LinearSVC(max_iter=3000, random_state=42, class_weight=class_weights, **p["SVM_tfidf"]),
        DecisionTreeClassifier(random_state=42, class_weight=class_weights, **p["DT_tfidf"]),
        RandomForestClassifier(random_state=42, class_weight=class_weights, n_jobs=-1, **p["RF_tfidf"]),
        XGBClassifier(random_state=42, eval_metric="mlogloss", use_label_encoder=False, **p["XGB_tfidf"]),
    )


# ════════════════════════════════════════════════════════════════
#  HYPERPARAMETER TUNING
# ════════════════════════════════════════════════════════════════
if SKIP_TUNING:
    section("Skipping GridSearchCV — loading saved v3 hyperparameters")
    if not os.path.isfile(hp_path):
        raise FileNotFoundError(f"Missing {hp_path}. Run without --skip-tuning first.")
    with open(hp_path, encoding="utf-8") as f:
        best_params_all = json.load(f)
    scaler = joblib.load(scaler_path)
    tfidf  = joblib.load(tfidf_path)
    X_train_scaled = scaler.transform(X_train_A)
    X_train_B = tfidf.transform(code_train)
    X_val_B   = tfidf.transform(code_val)
    X_test_B  = tfidf.transform(code_test)
    (best_lr_A, best_svm_A, best_dt_A, best_rf_A, best_xgb_A,
     best_lr_B, best_svm_B, best_dt_B, best_rf_B, best_xgb_B) = build_models_from_params(best_params_all)
    print(f"  Loaded: {hp_path}")
else:
    section("Hyperparameter Tuning (GridSearchCV, 3-fold CV)")
    print("  This will take ~10-20 minutes...")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_A)
    joblib.dump(scaler, scaler_path)

    def tune(name, estimator, grid, X, y):
        print(f"\n  Tuning: {name}")
        gs = GridSearchCV(estimator, grid, cv=cv, scoring="f1_macro", n_jobs=-1, refit=True)
        gs.fit(X, y)
        print(f"  Best params : {gs.best_params_}   Best CV F1: {gs.best_score_:.4f}")
        return gs.best_estimator_

    best_params_all = {}
    best_lr_A  = tune("LR (structural)", LogisticRegression(max_iter=1000, random_state=42, class_weight=class_weights),
                      {"C": [0.1, 1.0, 10.0]}, X_train_scaled, y_train)
    best_params_all["LR_structural"] = {"C": best_lr_A.C}
    best_svm_A = tune("SVM (structural)", LinearSVC(max_iter=3000, random_state=42, class_weight=class_weights),
                      {"C": [0.1, 1.0, 10.0]}, X_train_scaled, y_train)
    best_params_all["SVM_structural"] = {"C": best_svm_A.C}
    best_dt_A  = tune("DecisionTree (structural)", DecisionTreeClassifier(random_state=42, class_weight=class_weights),
                      {"max_depth": [5, 10, 15, 20]}, X_train_scaled, y_train)
    best_params_all["DT_structural"] = {"max_depth": best_dt_A.max_depth}
    best_rf_A  = tune("RandomForest (structural)", RandomForestClassifier(random_state=42, class_weight=class_weights, n_jobs=-1),
                      {"n_estimators": [100, 200], "max_depth": [10, 20, None]}, X_train_scaled, y_train)
    best_params_all["RF_structural"] = {"n_estimators": best_rf_A.n_estimators, "max_depth": best_rf_A.max_depth}
    best_xgb_A = tune("XGBoost (structural)", XGBClassifier(random_state=42, eval_metric="mlogloss", use_label_encoder=False),
                      {"learning_rate": [0.05, 0.1, 0.2], "n_estimators": [100, 200]}, X_train_scaled, y_train)
    best_params_all["XGB_structural"] = {"learning_rate": best_xgb_A.learning_rate, "n_estimators": best_xgb_A.n_estimators}

    tfidf = TfidfVectorizer(tokenizer=python_tokenizer, token_pattern=None,
                            max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
    X_train_B = tfidf.fit_transform(code_train)
    X_val_B   = tfidf.transform(code_val)
    X_test_B  = tfidf.transform(code_test)
    joblib.dump(tfidf, tfidf_path)

    best_lr_B  = tune("LR (TF-IDF)", LogisticRegression(max_iter=1000, random_state=42, class_weight=class_weights),
                      {"C": [0.1, 1.0, 10.0]}, X_train_B, y_train)
    best_params_all["LR_tfidf"] = {"C": best_lr_B.C}
    best_svm_B = tune("SVM (TF-IDF)", LinearSVC(max_iter=3000, random_state=42, class_weight=class_weights),
                      {"C": [0.1, 1.0, 10.0]}, X_train_B, y_train)
    best_params_all["SVM_tfidf"] = {"C": best_svm_B.C}
    best_dt_B  = tune("DecisionTree (TF-IDF)", DecisionTreeClassifier(random_state=42, class_weight=class_weights),
                      {"max_depth": [5, 10, 15, 20]}, X_train_B, y_train)
    best_params_all["DT_tfidf"] = {"max_depth": best_dt_B.max_depth}
    best_rf_B  = tune("RandomForest (TF-IDF)", RandomForestClassifier(random_state=42, class_weight=class_weights, n_jobs=-1),
                      {"n_estimators": [100, 200], "max_depth": [10, 20, None]}, X_train_B, y_train)
    best_params_all["RF_tfidf"] = {"n_estimators": best_rf_B.n_estimators, "max_depth": best_rf_B.max_depth}
    best_xgb_B = tune("XGBoost (TF-IDF)", XGBClassifier(random_state=42, eval_metric="mlogloss", use_label_encoder=False),
                      {"learning_rate": [0.05, 0.1, 0.2], "n_estimators": [100, 200]}, X_train_B, y_train)
    best_params_all["XGB_tfidf"] = {"learning_rate": best_xgb_B.learning_rate, "n_estimators": best_xgb_B.n_estimators}

    with open(hp_path, "w", encoding="utf-8") as f:
        json.dump(best_params_all, f, indent=2)
    print(f"\n[saved] {hp_path}")


# ════════════════════════════════════════════════════════════════
#  FINAL TRAINING
# ════════════════════════════════════════════════════════════════
section("Training all 10 models with best hyperparameters")

X_val_scaled  = scaler.transform(X_val_A)
X_test_scaled = scaler.transform(X_test_A)

results = []
all_preds = {}
models = {}


def train_eval(name, model, Xtr, Xv, Xte, params):
    t0 = time.time()
    model.fit(Xtr, y_train)
    t = round(time.time() - t0, 2)
    print(f"  ✓ {name}  ({t}s)")
    res = evaluate_model(name, model, Xtr, y_train, Xv, y_val, Xte, y_test, t, params)
    results.append(res)
    all_preds[name] = res.pop("y_pred")
    models[name] = model


print("\nPipeline A — Structural features (length shortcuts removed):")
train_eval("LR (structural)",  best_lr_A,  X_train_scaled, X_val_scaled, X_test_scaled, best_params_all["LR_structural"])
train_eval("SVM (structural)", best_svm_A, X_train_scaled, X_val_scaled, X_test_scaled, best_params_all["SVM_structural"])
train_eval("DT (structural)",  best_dt_A,  X_train_scaled, X_val_scaled, X_test_scaled, best_params_all["DT_structural"])
train_eval("RF (structural)",  best_rf_A,  X_train_scaled, X_val_scaled, X_test_scaled, best_params_all["RF_structural"])
train_eval("XGB (structural)", best_xgb_A, X_train_scaled, X_val_scaled, X_test_scaled, best_params_all["XGB_structural"])

print("\nPipeline B — TF-IDF:")
train_eval("LR (TF-IDF)",  best_lr_B,  X_train_B, X_val_B, X_test_B, best_params_all["LR_tfidf"])
train_eval("SVM (TF-IDF)", best_svm_B, X_train_B, X_val_B, X_test_B, best_params_all["SVM_tfidf"])
train_eval("DT (TF-IDF)",  best_dt_B,  X_train_B, X_val_B, X_test_B, best_params_all["DT_tfidf"])
train_eval("RF (TF-IDF)",  best_rf_B,  X_train_B, X_val_B, X_test_B, best_params_all["RF_tfidf"])
train_eval("XGB (TF-IDF)", best_xgb_B, X_train_B, X_val_B, X_test_B, best_params_all["XGB_tfidf"])


# ════════════════════════════════════════════════════════════════
#  RESULTS
# ════════════════════════════════════════════════════════════════
section("Results Summary")

df_res = pd.DataFrame(results).sort_values("macro_f1", ascending=False).reset_index(drop=True)
print(df_res[["model", "train_acc", "val_acc", "test_acc", "macro_f1",
              "syntax_f1", "logic_f1", "variable_f1", "train_time_s"]].to_string(index=False))

best_row = df_res.iloc[0]
best_name = best_row["model"]
print(f"\nBest model: {best_name}  macro_f1={best_row['macro_f1']:.4f}")

joblib.dump(models[best_name], os.path.join(MODEL_DIR, "best_baseline_v3.pkl"))
print(f"[saved] best_baseline_v3.pkl")

res_path = os.path.join(OUTPUT_DIR, "baseline_results_v3.txt")
with open(res_path, "w", encoding="utf-8") as f:
    f.write("Baseline Classifier Results v3 (grouped split, length shortcuts removed)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Dropped features: {DROP_FEATURES}\n")
    f.write(f"Structural feature count: {len(feat_cols)}\n\n")
    f.write(df_res[["model", "train_acc", "val_acc", "test_acc",
                    "macro_f1", "syntax_f1", "logic_f1", "variable_f1"]].to_string(index=False))
    f.write(f"\n\nBest model: {best_name}  macro_f1={best_row['macro_f1']:.4f}\n")
    f.write("\nNote: compare against baseline_results_v2.txt (leaky split) — a drop in\n")
    f.write("test macro-F1 here is expected and reflects removal of problem-context leakage.\n")
print(f"[saved] {res_path}")


# ════════════════════════════════════════════════════════════════
#  CHARTS
# ════════════════════════════════════════════════════════════════
section("Generating charts")

fig, axes = plt.subplots(2, 5, figsize=(26, 10))
for i, (name, preds) in enumerate(all_preds.items()):
    row, col = divmod(i, 5)
    ax = axes[row][col]
    disp = ConfusionMatrixDisplay(confusion_matrix(y_test, preds),
                                  display_labels=["syntax", "logic", "var"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"[{'A' if 'structural' in name else 'B'}] {name}", fontsize=8, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=7); ax.set_ylabel("True", fontsize=7); ax.tick_params(labelsize=7)
fig.suptitle("Baseline Classifiers v3 — Confusion Matrices (grouped split)\n"
             "[A] Structural (no length features)  |  [B] TF-IDF", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrices_v3.png"), dpi=130, bbox_inches="tight")
plt.close()
print("[saved] confusion_matrices_v3.png")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
names_short = df_res["model"].tolist()
f1s  = df_res["macro_f1"].tolist()
cols = ["#4C72B0" if "structural" in n else "#55A868" for n in names_short]
bars = axes[0].barh(names_short[::-1], f1s[::-1], color=cols[::-1], edgecolor="white")
axes[0].set_title("All 10 models — Macro F1 (v3)"); axes[0].set_xlabel("Macro F1"); axes[0].set_xlim(0, 1.05)
for b, v in zip(bars, f1s[::-1]):
    axes[0].text(v + 0.005, b.get_y() + b.get_height() / 2, f"{v:.4f}", va="center", fontsize=9)
from matplotlib.patches import Patch
axes[0].legend(handles=[Patch(facecolor="#4C72B0", label="Pipeline A (Structural)"),
                        Patch(facecolor="#55A868", label="Pipeline B (TF-IDF)")],
               loc="lower right", fontsize=9)
x = np.arange(len(df_res)); w = 0.25
axes[1].bar(x - w, df_res["train_acc"], w, label="Train", color="#4C72B0", edgecolor="white")
axes[1].bar(x,     df_res["val_acc"],   w, label="Val",   color="#CCB974", edgecolor="white")
axes[1].bar(x + w, df_res["test_acc"],  w, label="Test",  color="#55A868", edgecolor="white")
axes[1].set_xticks(x); axes[1].set_xticklabels([n.replace(" (", "\n(") for n in df_res["model"]], fontsize=7, rotation=30)
axes[1].set_title("Train / Val / Test Accuracy (v3)"); axes[1].set_ylabel("Accuracy"); axes[1].set_ylim(0, 1.1); axes[1].legend(fontsize=9)
fig.suptitle("Baseline Classifiers v3 — Performance (grouped split)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison_v3.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[saved] model_comparison_v3.png")

print(f"""
{'=' * 60}
  BASELINE CLASSIFIERS v3 COMPLETE
{'=' * 60}
  Best model : {best_name}   Macro F1: {best_row['macro_f1']:.4f}   Test acc: {best_row['test_acc']:.4f}
  Compare with v2 (leaky split) to quantify the leakage effect.
{'=' * 60}
""")
