# CFPB Consumer Complaints — Business Analytics Project

A portfolio-ready analytics project using the **Consumer Financial Protection Bureau (CFPB) Consumer Complaint Database**. It demonstrates a practical workflow across Python, SQL, and Power BI for Data Analyst / BI Analyst / Product Analyst interviews.

## Business objective

Analyze complaint patterns and company response outcomes to help stakeholders understand:
- Which financial products and issues generate the most complaints?
- How do complaint volumes change over time?
- What share of companies' responses are timely?
- Which products, companies, and submission channels deserve deeper review?

> **Important interpretation:** CFPB complaints are not a representative statistical sample of all consumers or all company customers. Complaint counts should not be interpreted as company failure rates without denominator data such as customer base or market share.

## Dataset

- **Source:** [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
- **API / field reference:** [CFPB API documentation](https://cfpb.github.io/api/ccdb/)
- Download a filtered CSV export from CFPB. Keep the download local in `data/raw/`; do not commit raw complaint exports.
- The public dataset's schema changes over time. The preparation script handles common current/older column names and reports missing fields.
- **Narratives note:** CFPB stopped publishing complaint narratives in the main database in 2026. This project focuses on structured data and does not require narratives.

## Project workflow

1. Download a CFPB CSV export and place it at `data/raw/complaints.csv`.
2. Install Python dependencies.
3. Run data cleaning and feature engineering.
4. Load the cleaned output into SQLite, SQL Server, PostgreSQL, or Power BI.
5. Run the SQL analysis queries.
6. Build a Power BI report using the suggested measures and page layout.

## Quick start

From this folder:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/prepare_data.py --input data/raw/complaints.csv --output data/processed/complaints_clean.csv
```

If you have a large export, filter by date or product on CFPB before downloading. CFPB currently limits filtered CSV exports to 100,000 records; consult its release notes for current limits.

## Repository structure

```
portfolio-analyst-project/
├── data/
│   ├── raw/                 # Local CFPB CSV (not committed)
│   └── processed/           # Generated clean CSV (not committed)
├── src/
│   └── prepare_data.py
├── sql/
│   └── analysis_queries.sql
├── power-bi/
│   └── dashboard-guide.md
├── requirements.txt
└── README.md
```

## Deliverables / interview talking points

- Data cleaning, date parsing, null handling, duplicate checks, and derived metrics in Python.
- SQL aggregation using CTEs, date functions, conditional aggregation, and window functions.
- Power BI KPI cards, trend charts, product/company drilldowns, slicers, and DAX measures.
- Data quality and interpretation limitations documented rather than hidden.

## Suggested Power BI pages

1. **Executive Overview:** complaint count, timely response %, monthly trend, top products.
2. **Product & Issue Analysis:** product → sub-product → issue hierarchy and issue mix.
3. **Company Response:** timely vs. not timely, response categories, company-level comparison.
4. **Geography & Channel:** state, submitted-via channel, and complaint volume.

## Data privacy

Do not upload raw complaint records, complaint narratives, or exports to GitHub. CFPB data is public, but complaint records can still contain sensitive information. Use aggregated charts/screenshots in your portfolio and follow CFPB's terms and data-use guidance.

## Data source caveats

- Complaint publication timing is delayed; recent trends may be incomplete.
- Complaints are not a random sample and are not normalized by customer count.
- Product and issue labels can change over time.
- This repository contains code and instructions, not a redistributed dataset or fabricated findings. Run the analysis on your selected export before quoting numeric results in your resume.
