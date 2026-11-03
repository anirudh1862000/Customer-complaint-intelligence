"""Prepare a CFPB complaint CSV export for SQL and Power BI.

Usage:
    python src/prepare_data.py --input data/raw/complaints.csv \
        --output data/processed/complaints_clean.csv

This script does not upload or share complaint-level records.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import pandas as pd


COLUMN_ALIASES = {
    "complaint_id": ["Complaint ID", "complaint_id", "ComplaintID"],
    "date_received": ["Date received", "date_received", "Date Received"],
    "product": ["Product", "product"],
    "sub_product": ["Sub-product", "Sub Product", "sub_product"],
    "issue": ["Issue", "issue"],
    "sub_issue": ["Sub-issue", "Sub Issue", "sub_issue"],
    "company": ["Company", "company"],
    "state": ["State", "state"],
    "zip_code": ["ZIP code", "Zip code", "ZIP Code", "zip_code"],
    "submitted_via": ["Submitted via", "submitted_via", "Submitted Via"],
    "date_sent_to_company": ["Date sent to company", "date_sent_to_company"],
    "company_response": ["Company response to consumer", "company_response"],
    "timely_response": ["Timely response?", "Timely response", "timely_response"],
    "consumer_disputed": ["Consumer disputed?", "consumer_disputed"],
}


def normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def find_column(columns: list[str], aliases: list[str]) -> str | None:
    normalized = {normalize_header(c): c for c in columns}
    for alias in aliases:
        match = normalized.get(normalize_header(alias))
        if match is not None:
            return match
    return None


def clean_value(series: pd.Series) -> pd.Series:
    result = series.astype("string").str.strip()
    return result.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to CFPB CSV export")
    parser.add_argument("--output", required=True, help="Path for cleaned CSV")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Could not find {input_path}. Download a CFPB CSV export and place it there."
        )

    df = pd.read_csv(input_path, low_memory=False)
    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")

    selected: dict[str, pd.Series] = {}
    missing: list[str] = []
    for canonical, aliases in COLUMN_ALIASES.items():
        source_col = find_column(list(df.columns), aliases)
        if source_col is None:
            missing.append(canonical)
            continue
        selected[canonical] = clean_value(df[source_col])

    if "complaint_id" not in selected:
        raise ValueError("Required field 'Complaint ID' was not found in the export.")
    if "date_received" not in selected:
        raise ValueError("Required field 'Date received' was not found in the export.")

    clean = pd.DataFrame(selected)
    clean["complaint_id"] = clean["complaint_id"].str.replace(r"\.0$", "", regex=True)
    clean["date_received"] = pd.to_datetime(clean["date_received"], errors="coerce")
    if "date_sent_to_company" in clean:
        clean["date_sent_to_company"] = pd.to_datetime(
            clean["date_sent_to_company"], errors="coerce"
        )

    # Standardize common boolean values without assuming every export uses the same labels.
    if "timely_response" in clean:
        clean["timely_response"] = (
            clean["timely_response"].str.lower().map(
                {"yes": True, "true": True, "no": False, "false": False}
            ).astype("boolean")
        )

    clean["received_year"] = clean["date_received"].dt.year.astype("Int64")
    clean["received_month"] = clean["date_received"].dt.to_period("M").astype("string")
    clean["received_quarter"] = clean["date_received"].dt.to_period("Q").astype("string")

    before = len(clean)
    clean = clean.drop_duplicates(subset=["complaint_id"])
    duplicates_removed = before - len(clean)
    invalid_dates = int(clean["date_received"].isna().sum())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output_path, index=False)

    print(f"Saved {len(clean):,} unique complaints to {output_path}")
    print(f"Removed duplicate complaint IDs: {duplicates_removed:,}")
    print(f"Rows with invalid/missing received dates: {invalid_dates:,}")
    if missing:
        print("Optional fields not present in this export:", ", ".join(missing))
    print("Review missing values and field definitions before reporting results.")


if __name__ == "__main__":
    main()
