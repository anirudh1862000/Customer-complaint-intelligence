# Customer Churn Prediction with XGBoost

An end-to-end, reproducible machine-learning project that generates a clearly labelled synthetic customer dataset, trains an XGBoost classifier, evaluates it on a stratified holdout set, and exports predictions and feature importance.

> **Data honesty:** The included generator creates synthetic data for a portfolio demonstration. It is not a real bank, telecom, or customer dataset. Metrics from this generator must not be described as real-world business results.

## What it demonstrates

- Reproducible data generation and validation
- Stratified train/test split
- Preprocessing in an sklearn pipeline, fitted only on training rows
- Numeric imputation and categorical one-hot encoding
- XGBoost binary classification
- Accuracy, precision, recall, F1, ROC-AUC, average precision, and confusion matrix
- Customer-level holdout predictions and model feature importance

## Run locally (Python 3.10+)

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.generate_data --rows 10000 --seed 42
python -m src.train
```

Outputs are written to `outputs/metrics.json`, `outputs/test_predictions.csv`, and `outputs/feature_importance.csv`.

## Tests

```bash
pytest -q
```

## Results status

No model metrics are committed yet. Run the commands above to generate metrics from your local execution; only then should the measured results be added here or quoted on a resume.

## Limitations

- Synthetic outcomes are generated from known rules; metrics are not an estimate of real-world churn performance.
- Replace the generated CSV with a properly licensed real dataset before making business claims. Define the prediction timestamp, prevent post-outcome leakage, and use a time-based holdout where appropriate.
- The default threshold is illustrative; choose a real retention threshold using intervention capacity and the cost of false positives and false negatives.
- Feature importance is model-based, not causal explanation.
