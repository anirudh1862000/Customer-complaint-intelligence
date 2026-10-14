#!/usr/bin/env python3
"""Create a small, safer portfolio sample from a large CFPB complaint CSV.

The original file stays local. This script streams the source in chunks, keeps
only modeling fields, excludes complaint IDs and free-text narratives, and
uses per-class reservoir sampling so the output is bounded and reasonably
balanced. Review the output before sharing or committing it.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

TARGET_DEFAULT = "Product"
FEATURE_CANDIDATES = [
    "Issue",
    "Sub-product",
    "Submitted via",
    "Company response to consumer",
    "Timely response?",
    "Consumer disputed?",
]
EXCLUDED_NAME_FRAGMENTS = ("id", "narrative", "consumer complaint", "zip", "email", "phone", "address")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to the original CSV")
    parser.add_argument("--output", default="data/portfolio_sample.csv.gz")
    parser.add_argument("--target", default=TARGET_DEFAULT)
    parser.add_argument("--max-per-class", type=int, default=1500)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")
    if args.max_per_class < 1:
        raise SystemExit("--max-per-class must be >= 1")

    header = pd.read_csv(input_path, nrows=0)
    columns = list(header.columns)
    if args.target not in columns:
        raise SystemExit(
            f"Target column {args.target!r} not found. Available columns: {columns}"
        )

    features = [
        col for col in FEATURE_CANDIDATES
        if col in columns and col != args.target
    ]
    # Explicitly exclude identifiers and free-text/sensitive narrative fields.
    features = [
        col for col in features
        if not any(fragment in col.lower() for fragment in EXCLUDED_NAME_FRAGMENTS)
    ]
    if not features:
        raise SystemExit(
            "None of the recommended structured feature columns were found. "
            "Edit FEATURE_CANDIDATES for your CSV headers."
        )

    keep = [args.target] + features
    rng = random.Random(args.seed)
    reservoirs: dict[str, list[dict[str, object]]] = {}
    seen_by_class: dict[str, int] = {}
    rows_seen = 0

    for chunk in pd.read_csv(input_path, usecols=keep, chunksize=args.chunksize, low_memory=False):
        chunk = chunk.dropna(subset=[args.target])
        chunk[args.target] = chunk[args.target].astype(str).str.strip()
        chunk = chunk[chunk[args.target] != ""]
        rows_seen += len(chunk)

        for record in chunk.to_dict(orient="records"):
            label = str(record[args.target])
            seen = seen_by_class.get(label, 0) + 1
            seen_by_class[label] = seen
            bucket = reservoirs.setdefault(label, [])
            if len(bucket) < args.max_per_class:
                bucket.append(record)
            else:
                replacement = rng.randrange(seen)
                if replacement < args.max_per_class:
                    bucket[replacement] = record

    sampled = [row for bucket in reservoirs.values() for row in bucket]
    if not sampled:
        raise SystemExit("No usable records found after filtering.")

    result = pd.DataFrame(sampled, columns=keep)
    # Remove exact duplicates and normalize empty strings.
    result = result.drop_duplicates()
    for col in result.columns:
        if result[col].dtype == "object":
            result[col] = result[col].astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, compression="gzip")

    summary = {
        "source_file_name": input_path.name,
        "rows_scanned_after_target_filter": int(rows_seen),
        "output_rows": int(len(result)),
        "target_column": args.target,
        "feature_columns": features,
        "class_counts_in_source_seen": seen_by_class,
        "class_counts_in_sample": {str(k): int(v) for k, v in result[args.target].value_counts().items()},
        "output_file": str(output_path),
        "random_seed": args.seed,
        "max_per_class_before_deduplication": args.max_per_class,
        "privacy_note": "IDs, narratives, company names, location and contact-like fields are not included by this script.",
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\nReview the output before publishing. Do not commit the original raw export.")


if __name__ == "__main__":
    main()
