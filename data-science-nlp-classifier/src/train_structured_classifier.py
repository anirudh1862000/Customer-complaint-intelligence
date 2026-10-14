"""Train a tabular complaint-product classifier on current CFPB exports.

Example:
  python src/train_structured_classifier.py --data data/complaints.csv

The current CFPB export may not include complaint narratives. This script therefore
uses structured fields and predicts Product from other available categorical fields.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to downloaded CFPB complaints CSV")
    parser.add_argument("--target", default="Product", help="Target column (default: Product)")
    parser.add_argument(
        "--features",
        default="Issue,Sub-product,Submitted via,Company response to consumer,Timely response?",
        help="Comma-separated categorical feature columns; unavailable columns are skipped",
    )
    parser.add_argument("--output-dir", default="artifacts/structured")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    path = Path(args.data)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}. Download a permitted CFPB export first.")
    df = pd.read_csv(path, low_memory=False)
    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not found. Available columns: {list(df.columns)}")

    candidates = [c.strip() for c in args.features.split(",") if c.strip()]
    features = [c for c in candidates if c in df.columns and c != args.target]
    if not features:
        raise ValueError(
            f"None of the requested feature columns were found. Available columns: {list(df.columns)}"
        )

    frame = df[features + [args.target]].copy()
    frame[args.target] = frame[args.target].astype("string").str.strip()
    frame = frame.dropna(subset=[args.target])
    frame = frame[frame[args.target] != ""]
    for col in features:
        frame[col] = frame[col].astype("string").fillna("Unknown").str.strip()
        frame[col] = frame[col].replace("", "Unknown")

    counts = frame[args.target].value_counts()
    frame = frame[frame[args.target].isin(counts[counts >= 2].index)]
    if frame[args.target].nunique() < 2:
        raise ValueError("Need at least two target classes with two or more rows each.")
    if len(frame) < 10:
        raise ValueError("Too few usable rows; download a larger filtered export.")

    X, y = frame[features], frame[args.target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("categorical", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), features)
        ],
        remainder="drop",
    )
    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    labels = sorted(y.unique())
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = {
        "dataset": str(path),
        "target": args.target,
        "features": features,
        "rows_after_cleaning": int(len(frame)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "number_of_classes": int(y.nunique()),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        "note": "Baseline only. Review feature/target overlap and per-class errors before interpreting results.",
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(confusion_matrix(y_test, predictions, labels=labels), index=labels, columns=labels).to_csv(
        output / "confusion_matrix.csv"
    )
    joblib.dump(model, output / "complaint_product_classifier.joblib")
    print(f"Rows: {len(frame):,}; features: {features}; classes: {y.nunique()}")
    print(f"Accuracy: {metrics['accuracy']:.3f}; macro-F1: {metrics['macro_f1']:.3f}")
    print(f"Artifacts saved in {output.resolve()}")


if __name__ == "__main__":
    main()
