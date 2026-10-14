"""Run an end-to-end demo without downloading any dataset.

Creates clearly synthetic complaint-like text, trains a TF-IDF + Logistic
Regression model, evaluates it on a stratified holdout, and writes artifacts.
The generated examples are fictional and must never be described as real CFPB
records or used to estimate real-world model performance.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

SEED = 42
OUTPUT_DIR = Path("artifacts/synthetic_demo")

# The fictional templates are deliberately labeled as synthetic examples.
TEMPLATES = {
    "Credit card": [
        "My credit card statement includes an unexpected {fee} fee.",
        "I paid my credit card balance but the payment was not {status}.",
        "The card issuer changed my credit limit without a clear {reason}.",
        "A duplicate credit card charge appeared on my monthly statement.",
        "I need help disputing a credit card transaction for {item}.",
    ],
    "Mortgage": [
        "My mortgage payment was applied to the wrong {account}.",
        "The home loan servicer has not explained my escrow {topic}.",
        "I requested mortgage assistance but my application remains {status}.",
        "My monthly home loan statement shows an incorrect {fee}.",
        "The mortgage company has not sent the documents I requested.",
    ],
    "Student loan": [
        "My student loan balance did not update after a payment.",
        "The education loan servicer gave me conflicting repayment {topic}.",
        "I applied for an income driven student loan plan and need a {document}.",
        "My student debt account shows an unexpected {fee}.",
        "The student loan payment was marked {status} incorrectly.",
    ],
    "Vehicle loan": [
        "My auto loan payment was applied to the wrong {account}.",
        "The car finance company charged an unexplained {fee}.",
        "I need the payoff amount for my vehicle loan in writing.",
        "The auto lender has not corrected my loan {topic}.",
        "My car loan account shows a payment as {status}.",
    ],
    "Credit reporting": [
        "My credit report contains an account that does not belong to me.",
        "The credit bureau did not correct an inaccurate {topic}.",
        "A late payment is listed on my credit file even though I paid.",
        "I disputed an error on my credit report and have not received a {document}.",
        "My credit history still shows an account as {status}.",
    ],
    "Debt collection": [
        "A debt collector keeps contacting me about an unfamiliar {account}.",
        "The collection agency has not provided validation for the alleged debt.",
        "I asked a debt collector to stop calling but the calls continued.",
        "The collector reported an incorrect balance for my {account}.",
        "I received a collection notice for a debt I do not {status}.",
    ],
}

FILLERS = {
    "fee": ["service", "late", "processing", "annual"],
    "status": ["posted", "processed", "closed", "paid"],
    "reason": ["explanation", "notice", "reason"],
    "item": ["an online order", "a subscription", "a purchase"],
    "account": ["account", "loan account", "payment"],
    "topic": ["information", "calculation", "history", "details"],
    "document": ["written response", "statement", "confirmation"],
}


def build_dataset(rows_per_class: int = 120, seed: int = SEED) -> pd.DataFrame:
    """Create deterministic fictional examples for a reproducible demo."""
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    for label, templates in TEMPLATES.items():
        for _ in range(rows_per_class):
            template = rng.choice(templates)
            text = template
            for key, values in FILLERS.items():
                placeholder = "{" + key + "}"
                if placeholder in text:
                    text = text.replace(placeholder, rng.choice(values))
            # Add harmless phrasing variation to reduce exact duplicates.
            prefix = rng.choice(["", "Please review: ", "I am requesting help because ", "I need assistance: "])
            suffix = rng.choice(["", " Please investigate.", " I would appreciate a written update.", ""])
            rows.append({
                "complaint_narrative": prefix + text + suffix,
                "product": label,
                "data_origin": "synthetic_demo",
            })
    rng.shuffle(rows)
    return pd.DataFrame(rows)


def main() -> None:
    df = build_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        df["complaint_narrative"],
        df["product"],
        test_size=0.2,
        random_state=SEED,
        stratify=df["product"],
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2), min_df=1)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = sorted(df["product"].unique())
    metrics = {
        "dataset_type": "synthetic_fictional_examples",
        "rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "classes": labels,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        "random_seed": SEED,
        "limitation": "Synthetic data only; these metrics do not estimate performance on real complaint data.",
    }
    df.drop(columns=["data_origin"]).to_csv(OUTPUT_DIR / "synthetic_complaints.csv", index=False)
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(confusion_matrix(y_test, predictions, labels=labels), index=labels, columns=labels).to_csv(
        OUTPUT_DIR / "confusion_matrix.csv"
    )
    joblib.dump(model, OUTPUT_DIR / "synthetic_complaint_classifier.joblib")

    print("DATA ORIGIN: SYNTHETIC / FICTIONAL — NOT REAL CFPB DATA")
    print(f"Rows: {len(df):,} | classes: {len(labels)} | test rows: {len(X_test):,}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Macro F1: {metrics['macro_f1']:.3f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.3f}")
    print(f"Demo artifacts saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
