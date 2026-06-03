"""
train_sklearn_classifier.py
===========================

Trains the local semantic classifier on the merged dataset.

Recommended after integrating datasets:
python train_sklearn_classifier.py --data data/guardrail_training_template.csv --output models/sklearn_safety_classifier.joblib

For a quicker demo on slower machines:
python train_sklearn_classifier.py --data data/guardrail_training_template.csv --output models/sklearn_safety_classifier.joblib --max-rows 30000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/guardrail_training_template.csv")
    parser.add_argument("--output", default="models/sklearn_safety_classifier.joblib")
    parser.add_argument("--max-rows", type=int, default=0, help="0 = use all rows; use 30000 for faster demo training")
    return parser.parse_args()


def train(data_path: str, output_path: str, max_rows: int = 0):
    df = pd.read_csv(data_path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("Dataset must contain text,label columns")

    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)

    if max_rows and len(df) > max_rows:
        parts = []
        labels = df["label"].nunique()
        per_label = max(20, max_rows // max(labels, 1))
        for _, sub in df.groupby("label"):
            parts.append(sub.sample(min(len(sub), per_label), random_state=42))
        df = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)

    X, y = df["text"], df["label"]
    stratify = y if y.value_counts().min() >= 2 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    features = FeatureUnion([
        ("word", TfidfVectorizer(lowercase=True, analyzer="word", ngram_range=(1, 2), max_features=80000, sublinear_tf=True)),
        ("char", TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5), max_features=80000, sublinear_tf=True)),
    ])

    pipeline = Pipeline([
        ("features", features),
        ("clf", SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-5, max_iter=50, tol=1e-3, class_weight="balanced", random_state=42)),
    ])

    pipeline.fit(X_train, y_train)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "labels": sorted(y.unique())}, out)

    pred = pipeline.predict(X_test)
    report = classification_report(y_test, pred, zero_division=0)
    report_path = out.parent.parent / "classifier_validation_report.txt"
    report_path.write_text(report, encoding="utf-8")

    print(f"Rows used: {len(df)}")
    print(f"Saved classifier to: {out}")
    print(f"Validation report saved to: {report_path}")
    print(report)
    return out


if __name__ == "__main__":
    args = parse_args()
    train(args.data, args.output, args.max_rows)
