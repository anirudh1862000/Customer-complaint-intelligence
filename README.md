# Customer Complaint Intelligence

A text-analytics project exploring consumer financial-services complaints and workflows for classifying complaint narratives into product or issue categories.

> **Portfolio focus:** NLP, text preprocessing, exploratory analysis, classification baselines, and model evaluation.

## Business Problem

Manually routing customer complaints can be time-consuming and inconsistent at scale. This project investigates how complaint text and existing category labels could support an assisted triage workflow.

## Questions Explored

- How do complaint volumes vary by product, issue, and time?
- Which complaint categories are represented most heavily?
- How can complaint text be prepared for supervised classification?
- How does a traditional machine-learning baseline compare with transformer-based approaches?
- Which evaluation measures help reveal class-level errors and potential routing risks?

## Repository Projects

### 0. Data Science — Retail Customer Intelligence
- [**Retail Customer Segmentation & Sales Forecasting**](data-science-project/README.md) — RFM feature engineering, K-Means segmentation, SARIMA forecasting, and time-based model evaluation.

### 1. NLP / Data Science
- `banking-consumer-complaint-analysis.ipynb` — exploratory analysis, preprocessing, and modeling.
- [`data-science-nlp-classifier/`](data-science-nlp-classifier/README.md) — structured complaint classifier and NLP experiments.
- **Note:** CFPB stopped publishing narratives in the main database in 2026. Narrative-based work should use a properly sourced historical archive and respect data-use requirements.

### 2. Data Analyst / BI Analyst — Python + SQL + Power BI
- [**CFPB Consumer Complaints Analytics Project**](portfolio-analyst-project/README.md) — a reproducible analyst workflow with Python data cleaning, SQL business queries, Power BI dashboard design and DAX measures.

## Tools & Skills

- **Python:** Pandas, NumPy
- **SQL:** CTEs, aggregation, conditional logic, date trends, window functions
- **BI:** Power BI, DAX, KPI design, drilldowns
- **Visualization:** Plotly / Power BI
- **Machine Learning:** scikit-learn, Logistic Regression, model evaluation
- **NLP / deep learning:** PyTorch, Hugging Face Transformers, DistilBERT

## Data Source

- [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
- [CFPB API documentation and release notes](https://cfpb.github.io/api/ccdb/)

The CFPB dataset changes over time. The structured analytics project documents field handling and avoids committing raw complaint exports. Do not upload complaint-level data or personal information to GitHub.

## Responsible Use

- CFPB complaint records are not a representative statistical sample of all consumers.
- Company complaint counts should be considered alongside company size or market share.
- Automated complaint classification should support human review, not make final customer-impacting decisions.
- Report model scores only after running the code and validating results on a documented split.

---
Portfolio repository demonstrating analytics, SQL, BI reporting, and data science workflows.
