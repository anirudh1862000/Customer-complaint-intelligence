from pathlib import Path
import argparse
import numpy as np
import pandas as pd

def make_customers(n: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Create a reproducible, explicitly synthetic customer churn dataset."""
    if n < 500:
        raise ValueError("Use at least 500 rows for a meaningful demo split.")
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, n)
    monthly = np.clip(rng.normal(72, 24, n), 18, 160).round(2)
    support = rng.poisson(1.2, n)
    late = rng.poisson(0.45, n)
    usage = np.clip(rng.normal(35, 12, n), 1, 90).round(1)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.58, .27, .15])
    payment = rng.choice(["Electronic check", "Card", "Bank transfer", "Invoice"], n)
    paperless = rng.choice(["Yes", "No"], n, p=[.72, .28])
    logit = (-1.35 + 0.035 * (monthly - 70) + 0.34 * support + 0.30 * late
             - 0.035 * tenure - 1.05 * (contract == "Two year")
             - 0.48 * (contract == "One year") + 0.012 * (35 - usage)
             + 0.18 * (payment == "Electronic check"))
    probability = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, probability)
    return pd.DataFrame({
        "customer_id": [f"C{i:06d}" for i in range(n)],
        "tenure_months": tenure, "monthly_charges": monthly,
        "support_tickets_90d": support, "late_payments_90d": late,
        "monthly_usage_hours": usage, "contract_type": contract,
        "payment_method": payment, "paperless_billing": paperless, "churn": churn,
    })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/synthetic_customer_churn.csv")
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = make_customers(args.rows, args.seed)
    df.to_csv(path, index=False)
    print(f"Created {len(df):,} synthetic rows at {path}; churn rate={df.churn.mean():.3f}")
