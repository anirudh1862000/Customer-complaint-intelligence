import pandas as pd

from src.advanced_analysis import (
    basket_affinity,
    build_cohort_retention,
    build_customer_value,
    country_performance,
    detect_monthly_anomalies,
    product_performance,
    validate_transactions,
)


def sample_transactions():
    rows = []
    dates = pd.date_range("2023-01-05", periods=8, freq="30D")
    for i, date in enumerate(dates):
        rows.extend([
            {"invoice": f"I{i}", "stockcode": "A", "description": "Item A",
             "quantity": 2, "invoicedate": date, "price": 5.0,
             "customer_id": "C1" if i < 5 else "C2", "country": "UK", "revenue": 10.0},
            {"invoice": f"I{i}", "stockcode": "B", "description": "Item B",
             "quantity": 1, "invoicedate": date, "price": 7.0,
             "customer_id": "C1" if i < 5 else "C2", "country": "UK", "revenue": 7.0},
        ])
    return pd.DataFrame(rows)


def test_validation_and_customer_value():
    data = validate_transactions(sample_transactions())
    values = build_customer_value(data)
    assert len(data) == 16
    assert set(values["customer_id"]) == {"C1", "C2"}
    assert (values["revenue"] > 0).all()


def test_cohort_retention_starts_at_100_percent():
    data = validate_transactions(sample_transactions())
    cohort = build_cohort_retention(data)
    first_period = cohort[cohort["period"] == 0]
    assert not first_period.empty
    assert (first_period["retention_pct"] == 100).all()


def test_product_country_and_basket_outputs():
    data = validate_transactions(sample_transactions())
    assert product_performance(data).iloc[0]["revenue"] >= 0
    assert country_performance(data).iloc[0]["country"] == "UK"
    pairs = basket_affinity(data, min_support=0.1)
    assert {"item_a", "item_b", "support"}.issubset(pairs.columns)
    assert not pairs.empty


def test_anomaly_detector_returns_monthly_rows():
    data = validate_transactions(sample_transactions())
    result = detect_monthly_anomalies(data)
    assert len(result) >= 6
    assert {"month", "revenue", "anomaly", "anomaly_score"}.issubset(result.columns)
