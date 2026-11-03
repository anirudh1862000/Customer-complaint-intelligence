"""Retail customer segmentation and sales forecasting pipeline.

Dataset: UCI Online Retail II (Excel).
Usage:
    python src/run_analysis.py --input data/raw/online_retail_II.xlsx --output data/processed
"""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore", category=UserWarning)


def load_retail(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Download Online Retail II from UCI and save it here."
        )
    # UCI Online Retail II is distributed as an Excel workbook with two yearly sheets.
    xls = pd.ExcelFile(path)
    frames = [pd.read_excel(xls, sheet_name=s) for s in xls.sheet_names]
    df = pd.concat(frames, ignore_index=True)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    expected = {"invoice", "quantity", "invoicedate", "price", "customer_id"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Expected columns missing: {sorted(missing)}. Found: {list(df.columns)}")

    df["invoicedate"] = pd.to_datetime(df["invoicedate"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["customer_id"] = df["customer_id"].astype("string").str.strip()
    df["invoice"] = df["invoice"].astype("string").str.strip()

    # Invoice numbers beginning with C indicate cancellations/credit notes.
    canceled = df["invoice"].str.startswith("C", na=False)
    df = df.loc[~canceled].copy()
    df = df.dropna(subset=["invoicedate", "quantity", "price"])
    df = df.loc[(df["quantity"] > 0) & (df["price"] > 0)].copy()
    df["revenue"] = df["quantity"] * df["price"]
    return df


def create_rfm(df: pd.DataFrame) -> pd.DataFrame:
    customer_tx = df.dropna(subset=["customer_id"]).copy()
    if customer_tx.empty:
        raise ValueError("No customer IDs found. Cannot build customer segmentation.")

    snapshot = customer_tx["invoicedate"].max().normalize() + pd.Timedelta(days=1)
    rfm = customer_tx.groupby("customer_id").agg(
        last_purchase=("invoicedate", "max"),
        frequency=("invoice", "nunique"),
        monetary=("revenue", "sum"),
        transactions=("invoice", "count"),
    )
    rfm["recency_days"] = (snapshot - rfm["last_purchase"].dt.normalize()).dt.days
    rfm = rfm.reset_index()
    rfm = rfm.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["recency_days", "frequency", "monetary"]
    )
    rfm = rfm.loc[(rfm["frequency"] > 0) & (rfm["monetary"] > 0)].copy()
    return rfm


def segment_customers(rfm: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    features = rfm[["recency_days", "frequency", "monetary"]].copy()
    # Reduce heavy right-skew before scaling.
    features = np.log1p(features)
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    rows = []
    max_k = min(8, len(rfm) - 1)
    if max_k < 2:
        raise ValueError("Need at least 3 customers to compare K-Means clusters.")
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        rows.append({
            "k": k,
            "inertia": float(model.inertia_),
            "silhouette": float(silhouette_score(X, labels)) if len(set(labels)) > 1 else np.nan,
        })

    scores = pd.DataFrame(rows)
    scores.to_csv(output_dir / "cluster_selection_metrics.csv", index=False)
    # Select k by silhouette as a reproducible starting point, then inspect business meaning.
    best_k = int(scores.sort_values("silhouette", ascending=False).iloc[0]["k"])
    final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    rfm = rfm.copy()
    rfm["cluster"] = final_model.fit_predict(X)

    profile = rfm.groupby("cluster").agg(
        customers=("customer_id", "nunique"),
        avg_recency_days=("recency_days", "mean"),
        median_frequency=("frequency", "median"),
        median_monetary=("monetary", "median"),
        total_revenue=("monetary", "sum"),
    ).reset_index()
    profile.to_csv(output_dir / "cluster_profiles.csv", index=False)
    rfm.to_csv(output_dir / "customer_rfm_clusters.csv", index=False)
    print(f"Selected k={best_k} using silhouette score. Review profiles before naming segments.")
    return rfm


def forecast_monthly_sales(df: pd.DataFrame, output_dir: Path) -> None:
    monthly = (
        df.set_index("invoicedate")["revenue"]
        .resample("MS")
        .sum()
        .asfreq("MS", fill_value=0)
        .rename("revenue")
    )
    monthly.to_frame().to_csv(output_dir / "monthly_revenue.csv")
    if len(monthly) < 24:
        print("Not enough monthly history for a meaningful seasonal SARIMA evaluation.")
        return

    # Hold out the final 6 months (or about 20% for shorter series).
    holdout = min(6, max(3, int(round(len(monthly) * 0.2))))
    train, test = monthly.iloc[:-holdout], monthly.iloc[-holdout:]

    # Seasonal naive baseline: use revenue from 12 months prior where available.
    seasonal_values = []
    for idx in range(len(test)):
        test_pos = len(train) + idx
        seasonal_pos = test_pos - 12
        if seasonal_pos >= 0:
            seasonal_values.append(float(monthly.iloc[seasonal_pos]))
        else:
            seasonal_values.append(float(train.iloc[-1]))
    seasonal_pred = np.array(seasonal_values)

    model = SARIMAX(
        train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    sarima_pred = np.asarray(fitted.forecast(steps=len(test)))

    def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
        errors = actual - predicted
        nonzero = actual != 0
        mape = float(np.mean(np.abs(errors[nonzero] / actual[nonzero])) * 100) if nonzero.any() else np.nan
        return {
            "MAE": float(np.mean(np.abs(errors))),
            "RMSE": float(np.sqrt(np.mean(errors ** 2))),
            "MAPE_nonzero_actual_pct": mape,
        }

    results = pd.DataFrame([
        {"model": "Seasonal naive", **metrics(test.to_numpy(), seasonal_pred)},
        {"model": "SARIMA(1,1,1)(1,1,1,12)", **metrics(test.to_numpy(), sarima_pred)},
    ])
    results.to_csv(output_dir / "forecast_metrics.csv", index=False)

    forecast_df = pd.DataFrame({
        "month": test.index,
        "actual_revenue": test.to_numpy(),
        "seasonal_naive_forecast": seasonal_pred,
        "sarima_forecast": sarima_pred,
    })
    forecast_df.to_csv(output_dir / "forecast_holdout.csv", index=False)
    print("Forecast evaluation completed on chronological holdout:")
    print(results.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="UCI Online Retail II Excel file")
    parser.add_argument("--output", default="data/processed", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_retail(Path(args.input))
    df.to_csv(output_dir / "clean_transactions.csv", index=False)
    print(f"Cleaned transactions: {len(df):,}; unique customers: {df['customer_id'].nunique(dropna=True):,}")

    rfm = create_rfm(df)
    segment_customers(rfm, output_dir)
    forecast_monthly_sales(df, output_dir)
    print(f"Outputs saved under: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
