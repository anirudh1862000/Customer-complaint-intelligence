"""Optional read-only API for generated retail analytics outputs.

Run from data-science-project:
    uvicorn src.api:app --reload
"""
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

BASE = Path(__file__).resolve().parents[1]
PROCESSED = BASE / "data" / "processed"
REPORTS = BASE / "reports"

app = FastAPI(
    title="Retail Customer Intelligence API",
    description="Read-only endpoints over locally generated retail analytics.",
    version="1.0.0",
)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Run the analytics pipeline first; missing {path.name}")
    return pd.read_csv(path)


@app.get("/health")
def health():
    return {"status": "ok", "message": "Retail analytics API is running"}


@app.get("/summary")
def summary():
    path = REPORTS / "executive_summary.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run advanced_analysis.py first.")
    import json
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/customers/top")
def top_customers(limit: int = Query(10, ge=1, le=100)):
    frame = read_csv(REPORTS / "customer_value_metrics.csv")
    return frame.head(limit).replace({float("nan"): None}).to_dict(orient="records")


@app.get("/products/top")
def top_products(limit: int = Query(10, ge=1, le=100)):
    frame = read_csv(REPORTS / "product_performance.csv")
    return frame.head(limit).replace({float("nan"): None}).to_dict(orient="records")


@app.get("/countries")
def countries(limit: int = Query(20, ge=1, le=200)):
    frame = read_csv(REPORTS / "country_performance.csv")
    return frame.head(limit).replace({float("nan"): None}).to_dict(orient="records")


@app.get("/cohorts")
def cohorts(cohort: str | None = None, limit: int = Query(500, ge=1, le=5000)):
    frame = read_csv(REPORTS / "cohort_retention.csv")
    if cohort:
        frame = frame[frame["cohort_month"].astype(str) == cohort]
    return frame.head(limit).replace({float("nan"): None}).to_dict(orient="records")


@app.get("/anomalies")
def anomalies():
    frame = read_csv(REPORTS / "monthly_anomalies.csv")
    flagged = frame[frame["anomaly"] == 1]
    return flagged.replace({float("nan"): None}).to_dict(orient="records")
