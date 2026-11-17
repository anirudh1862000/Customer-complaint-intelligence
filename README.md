# Customer Complaint Intelligence

A portfolio repository demonstrating data analytics, business intelligence, machine learning, and responsible AI workflows using documented public or synthetic datasets.

> **Portfolio focus:** Python, SQL, Power BI, customer analytics, supply chain, forecasting, NLP, and risk analytics. Each project documents its data source, evaluation approach, and limitations. No unverified business impact or model performance is claimed.

## Featured projects

### 1. Customer complaints — Data Analyst / BI Analyst
[**CFPB Consumer Complaints Analytics**](portfolio-analyst-project/README.md) — Python data preparation, SQL business queries, Power BI dashboard specification, KPI definitions, and responsible interpretation of complaint data.

### 2. Complaint classification — Data Scientist / NLP
[**Complaint Classification**](data-science-nlp-classifier/README.md) — structured classification workflow, synthetic pipeline demo, class-level evaluation outputs, and an optional historical narrative-text workflow where lawful data is available.

### 3. Retail customer intelligence — Data Scientist / Product Analyst
[**Retail Customer Segmentation & Sales Forecasting**](data-science-project/README.md) — RFM feature engineering, K-Means segmentation, cohort and customer-value analysis, monthly revenue forecasting against a seasonal baseline, and business interpretation guidance.

### 4. Fraud risk analytics — Data Scientist / Risk Analytics
[**Synthetic Fraud Risk Scoring**](fraud-risk-analytics/README.md) — reproducible transaction simulation, class-weighted Random Forest, ROC-AUC and PR-AUC evaluation, threshold trade-offs, and investigation queue outputs.

**Important:** The fraud-risk project uses synthetic data. It demonstrates the workflow only; its metrics are not evidence of real-world fraud performance or production experience.

### 5. Document RAG Question Answering — GenAI / Data Scientist
[**Document RAG Question Answering Pipeline**](rag-question-answering/README.md) — PDF/TXT/Markdown ingestion, chunking, sentence-transformer embeddings, FAISS similarity search, optional OpenAI grounded generation, source references, and a FastAPI service. Includes a fictional sample policy and a local extractive fallback when no API key is configured.

## Skills demonstrated

- **Python:** Pandas, NumPy, scikit-learn, data preparation, feature engineering
- **SQL:** CTEs, aggregations, conditional logic, date trends, window functions
- **BI:** Power BI, DAX, KPI design, dashboard specifications, drilldowns
- **Machine learning:** classification, clustering, forecasting, model comparison, error analysis
- **NLP / GenAI:** text preprocessing, TF-IDF baselines, transformer experiments, RAG concepts
- **Risk analytics:** class imbalance, precision/recall trade-offs, threshold selection, human-in-the-loop review

## Data and responsible use

- CFPB complaint records are not a representative statistical sample of all consumers. Company complaint counts should be interpreted alongside company size or market share.
- The retail transaction data and synthetic fraud data have distinct purposes and limitations; consult each project README before interpreting results.
- Do not upload raw complaint exports, personal information, or restricted datasets.
- Automated classifications and risk scores should support human review, not make final customer-impacting decisions.
- Report numeric findings only after running the relevant workflow and checking the generated outputs.

## Reproducibility

Each project README contains its own setup, data-source, run, and evaluation instructions. Generated artifacts are created locally and are not committed by default.

### 6. Customer Churn Prediction with XGBoost — Machine Learning
[**Customer Churn Prediction with XGBoost**](customer-churn-xgboost/README.md) — reproducible synthetic-data ML demo with preprocessing, stratified holdout evaluation, ROC-AUC/precision/recall/F1, confusion matrix, test predictions, and feature importance. Metrics are explicitly labelled synthetic and are not real-world business results.
