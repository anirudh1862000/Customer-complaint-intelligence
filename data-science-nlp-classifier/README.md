# Reproducible NLP Classifier

This folder adds a runnable, traditional machine-learning baseline to the complaint-intelligence portfolio project. It uses TF-IDF features and a class-weighted Logistic Regression model to predict a complaint's **Product** category from its narrative.

## Why this is relevant to Data Scientist roles

- Text preprocessing and feature engineering
- Supervised classification with scikit-learn pipelines
- Stratified holdout validation
- Accuracy, macro-F1, weighted-F1, class-level precision/recall/F1
- Confusion-matrix analysis and serialized model artifacts
- Reproducible command-line workflow

## Run locally

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   # Windows: .venv\\Scripts\\activate
   # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Obtain a permitted copy of the Consumer Financial Protection Bureau (CFPB) consumer complaints dataset. The source data is not included in this repository because it can be large and may contain sensitive narrative text. Review the source's terms and privacy guidance before use.

3. Run the model. Adjust the column names if your CSV uses different headers:

   ```bash
   python src/train_classifier.py --data data/complaints.csv
   ```

   Example with alternative headers:

   ```bash
   python src/train_classifier.py --data data/complaints.csv --text-column narrative --label-column product
   ```

4. Inspect the generated `artifacts/` directory:
   - `metrics.json` — holdout metrics and class-level report
   - `confusion_matrix.csv` — true versus predicted class counts
   - `complaint_classifier.joblib` — fitted scikit-learn pipeline

## Modeling decisions and caveats

- The script removes empty narratives and labels and excludes classes with fewer than two examples before stratified splitting.
- TF-IDF uses unigrams and bigrams; Logistic Regression uses class weights to help with imbalance.
- Macro-F1 gives each class equal weight and should be reviewed alongside per-class recall, not just accuracy.
- A single random split is a baseline, not proof of generalization. For a stronger study, compare against a majority-class baseline, tune hyperparameters on validation data, and preserve a final untouched test set.
- Complaint text can contain personal or financial information. Do not commit raw complaint records or trained artifacts derived from restricted data.
- This is decision support for human triage, not an autonomous system for customer-impacting decisions.

**No performance score is claimed here.** Run the pipeline on a verified dataset and report the resulting metrics only after checking data leakage, class balance, and error patterns.
