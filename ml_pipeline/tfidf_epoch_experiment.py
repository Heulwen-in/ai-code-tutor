"""
TF-IDF epoch observation experiment.

Purpose:
  Tutor feedback asks for TF-IDF training with more epochs, observation, and
  early stopping. The existing SVM TF-IDF baseline is not epoch-based, so this
  script uses SGDClassifier on TF-IDF features to create a valid epoch-by-epoch
  experiment.

Run:
  python ml_pipeline/tfidf_epoch_experiment.py --epochs 15 --patience 3

Output:
  ml_pipeline/logs/tfidf_epoch_experiment_results.json
  ml_pipeline/data/processed/tfidf_epoch_experiment_results.txt
"""

from __future__ import annotations

import argparse
import json
import os
import time
from copy import deepcopy

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight


BASE_DIR = r"F:\Heulwen\IT GREENWICH\Final Project\Tutor_AI_code"
PROCESSED_DIR = os.path.join(BASE_DIR, "ml_pipeline", "data", "processed")
LOG_DIR = os.path.join(BASE_DIR, "ml_pipeline", "logs")
MODEL_DIR = os.path.join(BASE_DIR, "backend", "app", "ml_models")
CLASS_NAMES = ["syntax_error", "logic_error", "variable_misuse"]
CLASSES = np.array([0, 1, 2])


def python_tokenizer(code: str) -> list[str]:
    spaced = (
        str(code)
        .replace("\n", " ")
        .replace("(", " ( ")
        .replace(")", " ) ")
        .replace(":", " : ")
        .replace(",", " , ")
    )
    return [token.strip() for token in spaced.split() if token.strip()]


def load_data():
    train = pd.read_parquet(os.path.join(PROCESSED_DIR, "train.parquet"))
    val = pd.read_parquet(os.path.join(PROCESSED_DIR, "val.parquet"))
    test = pd.read_parquet(os.path.join(PROCESSED_DIR, "test.parquet"))
    return train, val, test


def train_epoch_model(name, loss, x_train, y_train, x_val, y_val, x_test, y_test, epochs, patience):
    weights = compute_class_weight(class_weight="balanced", classes=CLASSES, y=y_train)
    class_weight = {int(label): float(weight) for label, weight in zip(CLASSES, weights)}
    model = SGDClassifier(
        loss=loss,
        penalty="l2",
        alpha=1e-5,
        random_state=42,
        class_weight=class_weight,
        learning_rate="optimal",
    )

    history = []
    best_model = None
    best_val_f1 = -1.0
    best_epoch = 0
    no_improve = 0
    start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.partial_fit(x_train, y_train, classes=CLASSES)

        train_pred = model.predict(x_train)
        val_pred = model.predict(x_val)
        train_f1 = f1_score(y_train, train_pred, average="macro")
        val_f1 = f1_score(y_val, val_pred, average="macro")
        epoch_time = round(time.time() - epoch_start, 2)

        history.append(
            {
                "epoch": epoch,
                "train_f1": round(train_f1, 4),
                "val_f1": round(val_f1, 4),
                "epoch_time_s": epoch_time,
            }
        )
        print(f"{name} epoch {epoch}/{epochs}: train_f1={train_f1:.4f} val_f1={val_f1:.4f} [{epoch_time}s]")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_model = deepcopy(model)
            no_improve = 0
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                print(f"{name}: early stopping after {no_improve} epochs without improvement.")
                break

    total_time = round(time.time() - start, 2)
    final_model = best_model or model
    test_pred = final_model.predict(x_test)
    report = classification_report(y_test, test_pred, target_names=CLASS_NAMES, output_dict=True)

    result = {
        "model": name,
        "loss": loss,
        "best_epoch": best_epoch,
        "best_val_f1": round(best_val_f1, 4),
        "test_accuracy": round(accuracy_score(y_test, test_pred), 4),
        "test_macro_f1": round(f1_score(y_test, test_pred, average="macro"), 4),
        "syntax_f1": round(report["syntax_error"]["f1-score"], 4),
        "logic_f1": round(report["logic_error"]["f1-score"], 4),
        "variable_f1": round(report["variable_misuse"]["f1-score"], 4),
        "train_time_s": total_time,
        "history": history,
    }
    return result, final_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    train, val, test = load_data()
    vectorizer = TfidfVectorizer(
        tokenizer=python_tokenizer,
        token_pattern=None,
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    x_train = vectorizer.fit_transform(train["code"].astype(str))
    x_val = vectorizer.transform(val["code"].astype(str))
    x_test = vectorizer.transform(test["code"].astype(str))
    y_train = train["label"].values
    y_val = val["label"].values
    y_test = test["label"].values

    experiments = [
        ("SGD-SVM (TF-IDF)", "hinge"),
        ("SGD-LogReg (TF-IDF)", "log_loss"),
    ]

    results = []
    best_model = None
    best_result = None

    for name, loss in experiments:
        result, model = train_epoch_model(
            name, loss, x_train, y_train, x_val, y_val, x_test, y_test, args.epochs, args.patience
        )
        results.append(result)
        if best_result is None or result["test_macro_f1"] > best_result["test_macro_f1"]:
            best_result = result
            best_model = model

    json_path = os.path.join(LOG_DIR, "tfidf_epoch_experiment_results.json")
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    text_path = os.path.join(PROCESSED_DIR, "tfidf_epoch_experiment_results.txt")
    with open(text_path, "w", encoding="utf-8") as file:
        file.write("TF-IDF Epoch Observation Experiment\n")
        file.write("=" * 50 + "\n\n")
        file.write(f"Max epochs: {args.epochs}\n")
        file.write(f"Early stopping patience: {args.patience}\n\n")
        for result in results:
            file.write(f"Model: {result['model']}\n")
            file.write(f"  Best epoch     : {result['best_epoch']}\n")
            file.write(f"  Best val F1    : {result['best_val_f1']}\n")
            file.write(f"  Test accuracy  : {result['test_accuracy']}\n")
            file.write(f"  Test macro F1  : {result['test_macro_f1']}\n")
            file.write(f"  Train time     : {result['train_time_s']}s\n\n")

    if best_model is not None:
        joblib.dump(best_model, os.path.join(MODEL_DIR, "best_tfidf_epoch_model.pkl"))
        joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_epoch_vectorizer.pkl"))

    print(f"Saved: {json_path}")
    print(f"Saved: {text_path}")
    print(f"Best: {best_result['model']} macro_f1={best_result['test_macro_f1']}")


if __name__ == "__main__":
    main()
