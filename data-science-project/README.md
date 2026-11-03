# Retail Customer Intelligence: Segmentation & Sales Forecasting

**Data Science portfolio project | Python · Machine Learning · Time Series**

Build a business-facing data science workflow from a real retail transactions dataset: customer segmentation with RFM features and unsupervised learning, plus monthly sales forecasting with time-aware validation.

## Why this project?

This project demonstrates more than dashboarding:
- Transaction-level data cleaning and exploratory data analysis
- Customer-level feature engineering (Recency, Frequency, Monetary value)
- Unsupervised learning with K-Means and cluster profiling
- Time-series forecasting with a seasonal naive baseline and SARIMA
- Time-based train/test evaluation and business recommendations

## Dataset

**UCI Online Retail II**: https://archive.ics.uci.edu/dataset/502/online+retail+ii

The dataset contains online retail transactions from a UK-based retailer. It includes invoice, product, quantity, transaction date, unit price, customer ID, and country fields. Download the Excel file from UCI and place it at:

`data/raw/online_retail_II.xlsx`

Do not commit the raw dataset to GitHub. See UCI's dataset page for citation and usage information.

## Project structure

```
data-science-project/
├── data/
│   ├── raw/                 # Downloaded dataset (not committed)
│   └── processed/           # Generated output (not committed)
├── notebooks/
│   └── (add EDA/modeling notebook after running pipeline)
├── src/
│   └── run_analysis.py      # Cleaning, RFM, clustering, forecasting
├── reports/
│   └── (add charts and findings after execution)
├── requirements.txt
└── README.md
```

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/run_analysis.py --input data/raw/online_retail_II.xlsx --output data/processed
```

The script writes cleaned transactions, customer RFM features, customer cluster labels, monthly sales, and forecast results to the output folder.

## Methodology

### 1. Data quality and EDA
- Remove canceled invoices and non-positive quantities/prices.
- Handle missing customer IDs separately for customer-level modeling.
- Convert transaction timestamps and derive revenue.
- Inspect country, product transaction volume, revenue, and seasonality.

### 2. Customer segmentation
- Aggregate transaction records into RFM metrics:
  - **Recency:** days since last purchase (relative to dataset snapshot date)
  - **Frequency:** number of distinct invoices
  - **Monetary:** total customer revenue
- Log-transform skewed metrics and standardize them.
- Fit K-Means for a range of cluster counts and compare inertia/silhouette scores.
- Profile clusters using original RFM values and assign interpretable labels only after inspecting actual profiles.

### 3. Sales forecasting
- Aggregate revenue by month.
- Compare a seasonal-naive forecast against SARIMA.
- Use a chronological holdout split; never randomly shuffle time series.
- Report MAE, RMSE, and MAPE (with safeguards for zero actual values).

## Business questions

- Which customers are high-value, frequent, or at risk of lapsing?
- How concentrated is revenue across customers?
- What seasonal patterns appear in monthly sales?
- How does a forecasting baseline compare with SARIMA on a held-out period?
- How could segments inform retention, campaign prioritization, or inventory planning?

## Responsible reporting

- This dataset is historical and from a single retailer; results may not generalize to other businesses.
- RFM clusters are descriptive groups, not ground-truth customer types.
- Forecast accuracy depends on the evaluation period and data quality.
- **Do not claim specific model scores or business impact until you run the pipeline and inspect the outputs.**

## Resume bullet template (fill only after running)

"Built a retail customer analytics pipeline using Python, engineered RFM features and segmented customers with K-Means; evaluated monthly sales forecasts using chronological validation and MAE/RMSE to support retention and planning decisions."

Replace this template with truthful results after you run the project.

## Dataset citation

Chen, D. (2019). *Online Retail II*. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D
