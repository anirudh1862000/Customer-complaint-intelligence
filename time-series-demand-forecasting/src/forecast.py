"""Leakage-aware synthetic monthly demand forecasting demonstration.

Run: python src/forecast.py --periods 120 --seed 42
All data are synthetic and are generated locally for this portfolio example.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURE_LAGS = (1, 2, 3, 6, 12)

def make_series(periods: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2016-01-01", periods=periods, freq="MS")
    t = np.arange(periods)
    annual = 18 * np.sin(2 * np.pi * t / 12) + 7 * np.cos(2 * np.pi * t / 12)
    trend = 0.35 * t
    noise = rng.normal(0, 5, periods)
    demand = np.maximum(5, 100 + trend + annual + noise)
    return pd.DataFrame({"date": dates, "demand": demand})

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for lag in FEATURE_LAGS:
        out[f"lag_{lag}"] = out["demand"].shift(lag)
    out["rolling_mean_3"] = out["demand"].shift(1).rolling(3).mean()
    month = out["date"].dt.month
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    return out.dropna().reset_index(drop=True)

def metrics(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    denom = np.maximum((np.abs(np.asarray(y_true)) + np.abs(np.asarray(y_pred))) / 2, 1e-8)
    smape = float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) / denom) * 100)
    return {"MAE": mae, "RMSE": rmse, "sMAPE_percent": smape}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--periods", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    if args.periods < 60:
        raise SystemExit("--periods must be at least 60 to support a meaningful annual-lag demonstration.")
    args.output.mkdir(parents=True, exist_ok=True)

    raw = make_series(args.periods, args.seed)
    featured = make_features(raw)
    n = len(featured)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    train = featured.iloc[:train_end]
    # Validation is reserved chronologically for workflow transparency.
    test = featured.iloc[val_end:].copy()
    if len(test) == 0:
        raise SystemExit("Not enough rows after feature creation for a test period.")

    feature_cols = [c for c in featured.columns if c not in ("date", "demand")]
    model = RandomForestRegressor(
        n_estimators=400, min_samples_leaf=2, max_features=0.9,
        random_state=args.seed, n_jobs=-1
    )
    model.fit(train[feature_cols], train["demand"])
    test["rf_forecast"] = model.predict(test[feature_cols])
    test["seasonal_naive_forecast"] = test["lag_12"]
    result = {
        "data_type": "synthetic",
        "periods_generated": int(len(raw)),
        "train_rows": int(len(train)),
        "validation_rows": int(val_end - train_end),
        "test_rows": int(len(test)),
        "random_forest": metrics(test["demand"], test["rf_forecast"]),
        "seasonal_naive": metrics(test["demand"], test["seasonal_naive_forecast"]),
        "note": "Synthetic demonstration only; metrics are not real-world business performance."
    }
    raw.to_csv(args.output / "synthetic_monthly_demand.csv", index=False)
    test[["date", "demand", "rf_forecast", "seasonal_naive_forecast"]].to_csv(
        args.output / "test_forecasts.csv", index=False
    )
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
