"""Train a reproducible TF-IDF + Logistic Regression complaint classifier.

Expected source columns by default:
  - Consumer complaint narrative
  - Product

Example:
  python src/train_classifier.py --data data/complaints.csv
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def normalize_text(value: object) -> str:
    """Light normalization; TF-IDF handles tokenization and n-grams."""
    text = "" if pd.isna(value) else str(value)
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to a local CSV dataset")
    parser.add_argument("--text-column", default="Consumer complaint narrative")
    parser.add_argument("--label-column", default="Product")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Download an approved dataset and pass its path with --data."
        )

    df = pd.read_csv(data_path, low_memory=False)
    missing = {args.text_column, args.label_column} - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}. Available columns: {list(df.columns)}"
        )

    df = df[[args.text_column, args.label_column]].copy()
    df[args.text_column] = df[args.text_column].map(normalize_text)
    df[args.label_column] = df[args.label_column].astype("string").str.strip()
    df = df.dropna(subset=[args.label_column])
    df = df[df[args.text_column].str.len() > 0]
    df = df[df[args.label_column] != ""]

    # Exclude labels with fewer than two records so a stratified split is possible.
    counts = df[args.label_column].value_counts()
    eligible = counts[counts >= 2].index
    df = df[df[args.label_column].isin(eligible)]
    if df[args.label_column].nunique() < 2:
        raise ValueError("Need at least two target classes with two or more examples each.")

    X = df[args.text_column]
    y = df[args.label_column]
    test_count = max(int(round(len(df) * args.test_size)), y.nunique())
    if test_count >= len(df):
        raise ValueError("Dataset is too small for the requested test split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=2,
                max_features=100_000,
                sublinear_tf=True,
            )),
            ("classifier", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=args.random_state,
            )),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    output_dir.mkdir(parents=True, exist_ok=True)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    labels = sorted(y.unique())
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    metrics = {
        "dataset": str(data_path),
        "text_column": args.text_column,
        "label_column": args.label_column,
        "rows_after_cleaning": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "number_of_classes": int(y.nunique()),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        "random_state": args.random_state,
        "classification_report": report,
        "note": "Metrics are from this local run and should not be treated as production performance.",
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(output_dir / "confusion_matrix.csv")
    joblib.dump(model, output_dir / "complaint_classifier.joblib")

    print(f"Rows used: {len(df):,} | classes: {y.nunique()} | test rows: {len(X_test):,}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Macro F1: {metrics['macro_f1']:.3f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.3f}")
    print(f"Artifacts saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
