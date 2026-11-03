# Retail Customer Intelligence: Segmentation & Sales Forecasting

An end-to-end **Data Science portfolio project** using real retail transaction data. It combines customer analytics, unsupervised learning, time-series forecasting, and business recommendations.

> **Project status:** Code and notebook are uploaded. Run locally against the UCI dataset to generate results. No performance numbers are claimed until the pipeline is executed and reviewed.

## Business questions
1. Which customers are most valuable, active, or at risk of lapsing?
2. Can customers be grouped into actionable segments using RFM behavior?
3. How well can monthly revenue be forecast versus a seasonal-naive baseline?
4. What actions could marketing and commercial teams take for each segment?

## Dataset
**UCI Online Retail II** — 1,067,371 transaction records from a UK-based online retailer, Dec 2009–Dec 2011.

- Dataset page: https://archive.ics.uci.edu/dataset/502/online+retail+ii
- Direct workbook: https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx
- Citation: Chen, D. (2012). *Online Retail II*. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D
- License: CC BY 4.0. Attribute the dataset when reusing or sharing derived work.

The workbook is about 43.5 MB. **Do not commit the raw data** to GitHub.

## Project structure
```
data-science-project/
├── data/raw/                 # Dataset (gitignored)
├── data/processed/           # Generated outputs (gitignored)
├── reports/                  # Plots and reports (gitignored)
├── notebooks/
│   └── retail_customer_intelligence.ipynb
├── src/
│   └── run_analysis.py       # Reproducible pipeline
├── requirements.txt
└── README.md
```

## Setup and run (Windows PowerShell)
Run these commands from the repository root:

```powershell
cd data-science-project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
New-Item -ItemType Directory -Force data\raw, data\processed, reports
Invoke-WebRequest "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx" -OutFile "data\raw\online_retail_II.xlsx"
python src/run_analysis.py --input data/raw/online_retail_II.xlsx --output data/processed
jupyter notebook notebooks/retail_customer_intelligence.ipynb
```

## Setup and run (macOS/Linux)
```bash
cd data-science-project
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
mkdir -p data/raw data/processed reports
curl -L "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx" -o data/raw/online_retail_II.xlsx
python src/run_analysis.py --input data/raw/online_retail_II.xlsx --output data/processed
jupyter notebook notebooks/retail_customer_intelligence.ipynb
```

## Methodology
### Data preparation
- Parse timestamps and numeric fields; normalize column names.
- Remove cancellation invoices (invoice IDs beginning with `C`), nonpositive quantities/prices, and invalid dates.
- Preserve customer IDs as strings; transactions without customer IDs contribute to overall revenue but are excluded from customer-level RFM.

### Customer segmentation
- Engineer **Recency, Frequency, Monetary (RFM)** features.
- Apply log transformation and standardization to reduce skew.
- Evaluate K-Means candidates (k=2–8, limited by customer count) using silhouette score and inertia.
- Export cluster assignments, profiles, and selection metrics.
- Cluster numbers are arbitrary; name segments only after inspecting actual profile values.

### Revenue forecasting
- Aggregate revenue by month.
- Use a chronological holdout (last 3–6 months depending on history).
- Compare SARIMA with a seasonal-naive baseline.
- Report MAE, RMSE, and MAPE on nonzero actual months. Do not randomly shuffle time-series data.

## Generated outputs
- `clean_transactions.csv`
- `customer_rfm_clusters.csv`
- `cluster_profiles.csv`
- `cluster_selection_metrics.csv`
- `monthly_revenue.csv`
- `forecast_metrics.csv`
- `forecast_holdout.csv`

The notebook uses these outputs to generate EDA, cluster visualizations, and forecast plots.

## Business interpretation guide
- **High monetary + frequent + recent:** loyalty benefits and early access.
- **High monetary + recency increasing:** personalized win-back campaign.
- **Frequent + low monetary:** bundles and cross-sell.
- **Low frequency + recent first purchase:** onboarding and second-purchase campaign.
- Use forecast results to support inventory/capacity planning, not as certainty.

## Limitations
- Data is historical (2009–2011) and does not represent current demand.
- RFM clusters are descriptive, not ground truth.
- SARIMA is a baseline model; inspect residuals and compare with naive forecasts before deployment.
- No causal claims: clustering does not prove campaign impact.

## Interview walkthrough
1. Explain the business question and data-quality decisions.
2. Explain RFM and why the features are interpretable.
3. Describe cluster selection and profile-based interpretation.
4. Explain chronological validation and why the baseline matters.
5. Discuss limitations and next steps (cohort retention, CLV, campaign uplift, rolling-origin backtesting).

## Resume bullet template
Use only after you execute the project and confirm the results:

> Built a retail customer analytics pipeline in Python, engineered RFM features and segmented customers with K-Means; evaluated monthly sales forecasts against a seasonal baseline using chronological validation and MAE/RMSE.

## Dataset citation
Chen, D. (2012). *Online Retail II*. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D
