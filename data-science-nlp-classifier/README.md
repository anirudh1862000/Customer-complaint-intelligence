# Data Scientist Portfolio: Complaint Classification

This project demonstrates a reproducible supervised-learning workflow using publicly available consumer complaint data from the U.S. Consumer Financial Protection Bureau (CFPB).

## Choose the dataset mode

### A. Current CFPB export — structured classification

Recent CFPB release notes say complaint narratives were removed from the database in September 2026. For a current export, use `src/train_structured_classifier.py` to predict the `Product` category from available structured fields such as `Issue`, `Sub-product`, `Submitted via`, and response fields. The script skips feature columns that are absent from your export.

1. Visit the [official CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/).
2. Download a filtered CSV export. Keep the export at or below the current download limit and save it locally as `data/complaints.csv`.
3. Install dependencies from this folder's `requirements.txt`.
4. Run:

   ```bash
   python src/train_structured_classifier.py --data data/complaints.csv
   ```

   If the CSV has different headers, use `--target` and `--features`, for example:

   ```bash
   python src/train_structured_classifier.py --data data/complaints.csv --target Product --features "Issue,Sub-product,Submitted via"
   ```

### B. Historical dataset containing complaint narratives — NLP classification

If you have a lawful historical dataset that includes narrative text, run `src/train_classifier.py`:

```bash
python src/train_classifier.py --data data/historical_complaints.csv
```

Use the `--text-column` and `--label-column` options if its headers differ. Do not assume a current CFPB export contains narrative text.

## Outputs

The scripts save local artifacts (not committed by default):
- `metrics.json` — accuracy, macro-F1, weighted-F1, and per-class precision/recall/F1
- `confusion_matrix.csv` — true versus predicted class counts
- `*.joblib` — fitted scikit-learn pipeline

## Skills demonstrated

- Data cleaning and missing-value handling
- Feature engineering / text vectorization
- scikit-learn pipelines and categorical encoding
- Stratified train/test evaluation
- Model evaluation with class-level metrics and confusion matrices
- Reproducible command-line ML workflow

## Critical evaluation notes

- Predicting Product from Issue/Sub-product can be an easy task because the fields are related by how the complaint form is designed. Explain this limitation in interviews and compare against a simple majority-class baseline.
- Review macro-F1 and per-class recall, not just accuracy. If data spans time, consider a time-based holdout to test temporal generalization.
- Keep raw complaint records and generated model artifacts out of Git unless you have reviewed licensing, privacy, and repository size.
- This is an exploratory portfolio model, not a production decision system.
- **No model score is claimed until you run the script on the exact dataset and inspect the outputs.**

## Source

- [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
- [CFPB API documentation and release notes](https://cfpb.github.io/api/ccdb/)
