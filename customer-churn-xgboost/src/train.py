from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

TARGET = "churn"
DROP = ["customer_id", TARGET]

def build_model(frame: pd.DataFrame) -> Pipeline:
    features = frame.drop(columns=DROP, errors="ignore")
    numeric = features.select_dtypes(include=["number"]).columns.tolist()
    categorical = features.select_dtypes(exclude=["number"]).columns.tolist()
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("numeric", numeric_pipe, numeric),
                              ("categorical", categorical_pipe, categorical)])
    model = XGBClassifier(n_estimators=350, max_depth=4, learning_rate=.04,
                          subsample=.85, colsample_bytree=.85, reg_lambda=2,
                          objective="binary:logistic", eval_metric="logloss",
                          random_state=42, n_jobs=4)
    return Pipeline([("preprocess", prep), ("model", model)])

def run(input_path: str, output_dir: str) -> dict:
    df = pd.read_csv(input_path)
    required = {TARGET, "customer_id"}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset must contain {sorted(required)}")
    if df[TARGET].nunique() != 2:
        raise ValueError("Target must contain both churn classes.")
    X, y = df.drop(columns=DROP), df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.20, random_state=42, stratify=y)
    pipeline = build_model(df)
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= .5).astype(int)
    metrics = {
        "dataset_type": "synthetic_demo", "rows": int(len(df)),
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "churn_rate": round(float(y.mean()), 4), "threshold": .5,
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "average_precision": round(float(average_precision_score(y_test, probabilities)), 4),
        "confusion_matrix_labels_0_1": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "caveat": "Metrics are from generated synthetic data and are not evidence of real-world churn performance."
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame({"actual_churn": y_test.to_numpy(), "predicted_churn": predictions,
                  "churn_probability": probabilities.round(6)}).to_csv(out / "test_predictions.csv", index=False)
    model = pipeline.named_steps["model"]
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    importance = pd.DataFrame({"feature": names, "importance": model.feature_importances_})
    importance.sort_values("importance", ascending=False).to_csv(out / "feature_importance.csv", index=False)
    print(json.dumps(metrics, indent=2))
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/synthetic_customer_churn.csv")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    run(args.input, args.output_dir)
