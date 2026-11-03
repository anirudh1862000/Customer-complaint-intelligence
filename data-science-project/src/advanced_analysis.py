"""Advanced analytics for the Retail Customer Intelligence project.

Consumes clean_transactions.csv produced by run_analysis.py and writes
cohort retention, product/country analysis, basket affinity, anomaly flags,
customer value proxies, charts, and a concise machine-readable report.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


REQUIRED = {"invoice", "stockcode", "description", "quantity", "invoicedate",
            "price", "customer_id", "country", "revenue"}


def validate_transactions(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required transaction columns: {sorted(missing)}")
    data = df.copy()
    data["invoicedate"] = pd.to_datetime(data["invoicedate"], errors="coerce")
    for col in ("quantity", "price", "revenue"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["invoicedate", "quantity", "price", "revenue"])
    data = data[(data["quantity"] > 0) & (data["price"] > 0) & (data["revenue"] > 0)]
    data["customer_id"] = data["customer_id"].astype("string").str.strip()
    data["stockcode"] = data["stockcode"].astype("string").str.strip()
    data["country"] = data["country"].fillna("Unknown").astype(str)
    data["description"] = data["description"].fillna("Unknown").astype(str).str.strip()
    return data


def build_cohort_retention(df: pd.DataFrame) -> pd.DataFrame:
    """Return monthly cohort retention as counts and percentages."""
    data = df.dropna(subset=["customer_id"]).copy()
    if data.empty:
        return pd.DataFrame(columns=["cohort_month", "period", "active_customers",
                                     "cohort_size", "retention_pct"])
    data["order_month"] = data["invoicedate"].dt.to_period("M")
    data["cohort_month"] = data.groupby("customer_id")["invoicedate"].transform("min").dt.to_period("M")
    data["period"] = (
        (data["order_month"].dt.year - data["cohort_month"].dt.year) * 12
        + data["order_month"].dt.month - data["cohort_month"].dt.month
    )
    counts = (data.groupby(["cohort_month", "period"])["customer_id"]
              .nunique().rename("active_customers").reset_index())
    sizes = (data.groupby("cohort_month")["customer_id"].nunique()
             .rename("cohort_size").reset_index())
    result = counts.merge(sizes, on="cohort_month", how="left")
    result["retention_pct"] = result["active_customers"] / result["cohort_size"] * 100
    result["cohort_month"] = result["cohort_month"].astype(str)
    return result.sort_values(["cohort_month", "period"]).reset_index(drop=True)


def build_customer_value(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate transparent historical value metrics; not predictive CLV."""
    data = df.dropna(subset=["customer_id"]).copy()
    if data.empty:
        return pd.DataFrame(columns=["customer_id", "first_purchase", "last_purchase",
                                     "active_days", "orders", "revenue", "avg_order_value",
                                     "revenue_per_active_day"])
    grouped = data.groupby("customer_id").agg(
        first_purchase=("invoicedate", "min"),
        last_purchase=("invoicedate", "max"),
        orders=("invoice", "nunique"),
        revenue=("revenue", "sum"),
    )
    grouped["active_days"] = (
        grouped["last_purchase"].dt.normalize() - grouped["first_purchase"].dt.normalize()
    ).dt.days + 1
    grouped["avg_order_value"] = grouped["revenue"] / grouped["orders"].clip(lower=1)
    grouped["revenue_per_active_day"] = grouped["revenue"] / grouped["active_days"].clip(lower=1)
    return grouped.reset_index().sort_values("revenue", ascending=False)


def product_performance(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    products = (df.groupby(["stockcode", "description"], dropna=False)
                .agg(units_sold=("quantity", "sum"),
                     revenue=("revenue", "sum"),
                     invoices=("invoice", "nunique"),
                     unique_customers=("customer_id", "nunique"))
                .reset_index())
    return products.sort_values("revenue", ascending=False).head(top_n).reset_index(drop=True)


def country_performance(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby("country")
            .agg(revenue=("revenue", "sum"),
                 orders=("invoice", "nunique"),
                 customers=("customer_id", "nunique"),
                 units=("quantity", "sum"))
            .reset_index().sort_values("revenue", ascending=False))


def basket_affinity(df: pd.DataFrame, min_support: float = 0.01,
                    top_n: int = 100) -> pd.DataFrame:
    """Simple pair co-occurrence; intentionally bounded to each invoice's top 30 items."""
    baskets = (df.groupby("invoice")["stockcode"]
               .apply(lambda x: sorted(set(x.dropna().astype(str)) )[:30]))
    n_baskets = len(baskets)
    if n_baskets == 0:
        return pd.DataFrame(columns=["item_a", "item_b", "co_invoices", "support"])
    counts: dict[tuple[str, str], int] = {}
    for items in baskets:
        for pair in combinations(items, 2):
            counts[pair] = counts.get(pair, 0) + 1
    rows = [{"item_a": a, "item_b": b, "co_invoices": count,
             "support": count / n_baskets}
            for (a, b), count in counts.items() if count / n_baskets >= min_support]
    return pd.DataFrame(rows, columns=["item_a", "item_b", "co_invoices", "support"]) \
        .sort_values(["co_invoices", "support"], ascending=False).head(top_n).reset_index(drop=True)


def detect_monthly_anomalies(df: pd.DataFrame, contamination: float = 0.08) -> pd.DataFrame:
    monthly = (df.set_index("invoicedate").resample("MS")
               .agg(revenue=("revenue", "sum"),
                    invoices=("invoice", "nunique"),
                    units=("quantity", "sum")).asfreq("MS", fill_value=0))
    if len(monthly) < 8:
        monthly["anomaly"] = 0
        monthly["anomaly_score"] = np.nan
    else:
        features = np.log1p(monthly[["revenue", "invoices", "units"]].clip(lower=0))
        model = IsolationForest(contamination=contamination, random_state=42, n_estimators=300)
        labels = model.fit_predict(features)
        monthly["anomaly"] = (labels == -1).astype(int)
        monthly["anomaly_score"] = model.decision_function(features)
    monthly.index.name = "month"
    return monthly.reset_index()


def make_charts(cohort: pd.DataFrame, products: pd.DataFrame,
                countries: pd.DataFrame, anomalies: pd.DataFrame,
                output_dir: Path) -> None:
    if not cohort.empty:
        pivot = cohort.pivot(index="cohort_month", columns="period", values="retention_pct")
        fig, ax = plt.subplots(figsize=(12, 7))
        image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
        ax.set(title="Customer cohort retention", xlabel="Months since first purchase",
               ylabel="Acquisition cohort")
        ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
        ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns)
        fig.colorbar(image, ax=ax, label="Retention (%)")
        fig.tight_layout()
        fig.savefig(output_dir / "cohort_retention.png", dpi=160)
        plt.close(fig)

    if not products.empty:
        top = products.head(15).sort_values("revenue")
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.barh(top["description"].astype(str).str.slice(0, 45), top["revenue"])
        ax.set(title="Top products by revenue", xlabel="Revenue")
        fig.tight_layout()
        fig.savefig(output_dir / "top_products.png", dpi=160)
        plt.close(fig)

    if not countries.empty:
        top = countries.head(12).sort_values("revenue")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top["country"], top["revenue"])
        ax.set(title="Top countries by revenue", xlabel="Revenue")
        fig.tight_layout()
        fig.savefig(output_dir / "country_revenue.png", dpi=160)
        plt.close(fig)

    if not anomalies.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(anomalies["month"], anomalies["revenue"], marker="o", label="Monthly revenue")
        flagged = anomalies[anomalies["anomaly"] == 1]
        ax.scatter(flagged["month"], flagged["revenue"], marker="x", s=90,
                   label="Anomaly candidate")
        ax.set(title="Monthly revenue anomaly candidates", xlabel="Month", ylabel="Revenue")
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(output_dir / "monthly_anomalies.png", dpi=160)
        plt.close(fig)


def write_summary(df: pd.DataFrame, customers: pd.DataFrame,
                  products: pd.DataFrame, countries: pd.DataFrame,
                  anomalies: pd.DataFrame, output_dir: Path) -> dict:
    summary = {
        "transactions": int(len(df)),
        "customers_with_id": int(df["customer_id"].nunique(dropna=True)),
        "invoices": int(df["invoice"].nunique()),
        "date_min": str(df["invoicedate"].min().date()) if len(df) else None,
        "date_max": str(df["invoicedate"].max().date()) if len(df) else None,
        "total_revenue": round(float(df["revenue"].sum()), 2),
        "top_product": str(products.iloc[0]["description"]) if len(products) else None,
        "top_country": str(countries.iloc[0]["country"]) if len(countries) else None,
        "anomaly_months": [str(x) for x in anomalies.loc[anomalies["anomaly"] == 1, "month"].tolist()],
        "top_customer_revenue": round(float(customers.iloc[0]["revenue"]), 2) if len(customers) else None,
        "interpretation_note": "Anomalies are statistical candidates requiring business investigation.",
        "customer_value_note": "Historical value metrics are descriptive, not predicted lifetime value.",
    }
    (output_dir / "executive_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run(input_csv: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = validate_transactions(pd.read_csv(input_csv))
    cohort = build_cohort_retention(df)
    customers = build_customer_value(df)
    products = product_performance(df)
    countries = country_performance(df)
    affinity = basket_affinity(df)
    anomalies = detect_monthly_anomalies(df)

    outputs = {
        "cohort_retention.csv": cohort,
        "customer_value_metrics.csv": customers,
        "product_performance.csv": products,
        "country_performance.csv": countries,
        "basket_affinity.csv": affinity,
        "monthly_anomalies.csv": anomalies,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, index=False)
    make_charts(cohort, products, countries, anomalies, output_dir)
    return write_summary(df, customers, products, countries, anomalies, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed/clean_transactions.csv")
    parser.add_argument("--output", default="reports")
    args = parser.parse_args()
    summary = run(Path(args.input), Path(args.output))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
