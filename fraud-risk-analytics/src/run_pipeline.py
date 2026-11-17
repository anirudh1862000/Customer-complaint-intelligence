"""Synthetic fraud-risk analytics demonstration.

Run from this directory:
    python src/run_pipeline.py --rows 30000 --seed 42

All records and labels are synthetic. This is not a production fraud model.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

FEATURES = ["amount", "account_age_days", "transactions_24h", "amount_vs_customer_avg", "device_changed", "country_mismatch", "failed_attempts_24h", "merchant_risk_score", "transaction_hour"]

def generate_transactions(rows: int, seed: int) -> pd.DataFrame:
    if rows < 500:
        raise ValueError("rows must be at least 500")
    rng = np.random.default_rng(seed)
    account_age = rng.integers(1, 4000, rows)
    avg_amount = rng.lognormal(3.5, 0.8, rows)
    amount = rng.lognormal(3.7, 1.0, rows)
    txns_24h = rng.poisson(2.2, rows)
    device_changed = rng.binomial(1, 0.08, rows)
    country_mismatch = rng.binomial(1, 0.025, rows)
    failed_attempts = rng.poisson(0.18, rows)
    merchant_risk = rng.beta(2, 5, rows)
    hour = rng.integers(0, 24, rows)
    amount_ratio = amount / np.maximum(avg_amount, 1.0)
    # Synthetic label mechanism for a reproducible coding demonstration only.
    logit = (-5.0 + 0.65*np.clip(np.log1p(amount_ratio), 0, 3)
             + 0.55*(txns_24h >= 6) + 1.15*device_changed
             + 1.75*country_mismatch + 0.65*np.clip(failed_attempts, 0, 4)
             + 1.2*merchant_risk + 0.35*((hour <= 4) | (hour >= 23))
             - 0.00012*account_age)
    probability = 1.0 / (1.0 + np.exp(-logit))
    fraud = rng.binomial(1, probability)
    return pd.DataFrame({
        "amount": amount.round(2), "account_age_days": account_age,
        "transactions_24h": txns_24h, "amount_vs_customer_avg": amount_ratio.round(4),
        "device_changed": device_changed, "country_mismatch": country_mismatch,
        "failed_attempts_24h": failed_attempts, "merchant_risk_score": merchant_risk.round(4),
        "transaction_hour": hour, "is_fraud": fraud
    })

def evaluate(y_true, probabilities, threshold):
    pred = (probabilities >= threshold).astype(int)
    return {"threshold": float(threshold),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "flagged_rate": float(pred.mean())}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    data = generate_transactions(args.rows, args.seed)
    X, y = data[FEATURES], data["is_fraud"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=args.seed, stratify=y)
    model = RandomForestClassifier(n_estimators=250, min_samples_leaf=3,
        class_weight="balanced_subsample", random_state=args.seed, n_jobs=-1)
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = {
        "data_type": "synthetic", "rows": int(len(data)),
        "fraud_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision_pr_auc": float(average_precision_score(y_test, probabilities)),
        "default_threshold_metrics": evaluate(y_test, probabilities, 0.5),
        "note": "Synthetic benchmark only; not real-world fraud performance."
    }
    (out / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame([evaluate(y_test, probabilities, t) for t in (0.2,0.3,0.4,0.5,0.6,0.7)]).to_csv(out / "threshold_analysis.csv", index=False)
    scored = X_test.copy()
    scored["actual_label_synthetic"] = y_test.to_numpy()
    scored["risk_score"] = probabilities
    scored["flag_for_review_at_050"] = (probabilities >= 0.5).astype(int)
    scored.to_csv(out / "scored_test_transactions.csv", index=False)
    cm = confusion_matrix(y_test, (probabilities >= 0.5).astype(int), labels=[0,1])
    pd.DataFrame(cm, index=["actual_legitimate","actual_fraud"],
        columns=["predicted_legitimate","predicted_fraud"]).to_csv(out / "confusion_matrix.csv")
    print(json.dumps(metrics, indent=2))
    print(f"\\nWrote evaluation outputs to: {out.resolve()}")

if __name__ == "__main__":
    main()
