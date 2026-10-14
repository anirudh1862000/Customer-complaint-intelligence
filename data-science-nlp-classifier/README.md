# Data Scientist Portfolio: Complaint Classification

This project demonstrates a reproducible supervised-learning workflow using consumer complaint data from the U.S. Consumer Financial Protection Bureau (CFPB).



## Run the project without downloading any dataset

A dataset-free demo is included. It generates fictional complaint-like sentences, trains a TF-IDF + Logistic Regression classifier, and evaluates it using a stratified train/test split. It is useful for demonstrating that the pipeline runs end to end.

From this folder:

```bash
pip install -r requirements.txt
python src/run_synthetic_demo.py
```

The script creates `artifacts/synthetic_demo/` containing a generated CSV, `metrics.json`, `confusion_matrix.csv`, and a saved model. These files are created locally when you run the script; they are not automatically uploaded to GitHub.

**Honesty note:** the synthetic dataset is fictional and is not an official CFPB dataset. Metrics from this demo only show that the code runs; they do not represent real-world performance. For a credible final evaluation, run the existing training scripts on a suitable real dataset later.

## Start with your large CSV: create a compact portfolio sample

The raw upload can be very large. **Do not commit the full raw export to GitHub.** Use the streaming preparation script to keep only structured modeling columns, exclude complaint IDs and free-text narratives, and create a compressed, bounded sample locally.

From this directory, install dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python src/prepare_portfolio_sample.py --input /path/to/complaints.csv --output data/portfolio_sample.csv.gz --max-per-class 1500
```

The script reads the source in chunks, so it does not load the entire CSV into memory. It writes:
- `data/portfolio_sample.csv.gz` — compact gzip-compressed dataset with a maximum of 1,500 rows per Product class before duplicate removal
- `data/portfolio_sample.csv.gz.metadata.json` — row counts, columns, and class distribution

The script uses available columns from `Issue`, `Sub-product`, `Submitted via`, `Company response to consumer`, `Timely response?`, and `Consumer disputed?`, with `Product` as the target. It does not include narrative text, complaint identifiers, company names, or location fields. **Inspect the generated sample before sharing it publicly.** The raw source file remains local.

## Train a model using a current structured export

Run:

```bash
python src/train_structured_classifier.py --data data/portfolio_sample.csv.gz
```

The model predicts `Product` from the structured fields available in the sample.

If the headers differ, specify them explicitly:

```bash
python src/train_structured_classifier.py --data data/portfolio_sample.csv.gz --target Product --features "Issue,Sub-product,Submitted via"
```

## Historical dataset containing complaint narratives — NLP classification

If you have a lawful historical dataset that includes complaint narrative text, run:

```bash
python src/train_classifier.py --data data/historical_complaints.csv
```

Use `--text-column` and `--label-column` if the headers differ. Do not assume a current CFPB export contains narrative text.

## Outputs

Training scripts save local artifacts (not committed by default):
- `metrics.json` — accuracy, macro-F1, weighted-F1, and per-class precision/recall/F1
- `confusion_matrix.csv` — true versus predicted class counts
- `*.joblib` — fitted scikit-learn pipeline

## Skills demonstrated

- Chunked data processing and data quality checks
- Missing-value handling and categorical feature preparation
- Feature engineering and NLP vectorization (historical narrative mode)
- scikit-learn pipelines and supervised classification
- Class-level metrics, confusion matrices, and reproducible CLI workflows

## Critical evaluation notes

- Predicting Product from Issue/Sub-product can be an easy task because the fields are related by how the complaint form is designed. Compare against a majority-class baseline and explain this limitation in interviews.
- Review macro-F1 and per-class recall, not just accuracy. If records span time, consider a time-based holdout to test temporal generalization.
- Keep the raw export and generated model artifacts out of Git unless licensing, privacy, and repository size have been reviewed.
- This is an exploratory portfolio model, not a production decision system.
- **No model score is claimed until you run the script on the exact dataset and inspect the outputs.**

## Source

- [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
- [CFPB API documentation and release notes](https://cfpb.github.io/api/ccdb/)
