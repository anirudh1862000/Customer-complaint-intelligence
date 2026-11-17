# Fraud Risk Analytics — Synthetic Transaction Risk Scoring

A reproducible machine-learning portfolio project demonstrating an end-to-end fraud-risk workflow: synthetic transaction generation, leakage-aware feature design, imbalanced classification, threshold analysis, and investigation-oriented outputs.

> **Data honesty:** This project generates synthetic transactions. It is a technical demonstration, not evidence of production fraud-detection experience and not a model trained on real bank/customer transactions. Synthetic metrics must not be presented as real-world fraud performance.

## Business problem

Fraud operations teams need to prioritize suspicious transactions while balancing missed fraud against false alarms. This project demonstrates how a model can produce a risk score and how the operating threshold changes precision, recall, and investigation volume.

## Features

- Reproducible synthetic data generator with a configurable random seed
- Explicitly documented, transaction-time features (amount deviation, account age, recent transaction velocity, device/country mismatch, failed attempts)
- Stratified train/test split
- Random Forest baseline with class weighting
- ROC-AUC, average precision / PR-AUC, precision, recall, F1 and confusion matrix
- Threshold comparison to illustrate operational trade-offs
- CSV outputs for scored test transactions and threshold metrics

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/run_pipeline.py --rows 30000 --seed 42
```

Outputs are written to `outputs/` (created locally and not committed):
- `model_metrics.json`
- `threshold_analysis.csv`
- `scored_test_transactions.csv`
- `confusion_matrix.csv`


## Reference run (seed 42)

A reference run of the checked-in pipeline with 30,000 generated rows and a stratified 75/25 split produced:

| Metric | Result |
|---|---:|
| Synthetic fraud prevalence | 2.66% |
| ROC-AUC | 0.681 |
| Average precision (PR-AUC) | 0.070 |
| Precision at threshold 0.20 | 11.50% |
| Recall at threshold 0.20 | 18.09% |
| Test records flagged at threshold 0.20 | 4.17% |

At threshold 0.50, recall was only 0.50%, illustrating why a default threshold can miss most minority-class examples. The lower threshold catches more synthetic positives but creates more false positives. These figures are a reproducibility check on the synthetic generator, **not evidence of real fraud-detection capability**.

The reference metrics are stored in [outputs/reference_run_metrics.json](outputs/reference_run_metrics.json). Re-run the command above to regenerate your own outputs. Minor differences may occur across library versions.

## Methodology

1. Generate a synthetic population with a low fraud prevalence.
2. Split records into train and test sets using stratification.
3. Train a class-weighted Random Forest using only generated transaction-time features.
4. Evaluate ranking quality and class-level performance.
5. Compare multiple probability thresholds instead of treating 0.50 as universally correct.
6. Export test-set scores to support a hypothetical analyst investigation queue.

## Important evaluation notes

- Because labels are generated from synthetic feature relationships, strong metrics only show that the code learns the generator's patterns.
- Real fraud detection requires representative, time-aware data, careful label maturity, entity-level leakage controls, concept-drift monitoring, and cost-sensitive validation.
- Precision and recall depend on the threshold and fraud prevalence. Accuracy alone is not an appropriate headline metric for rare events.
- A high-risk score is a prioritization signal, not proof of fraud. Human investigation is required.
- Do not claim production deployment, prevented losses, or real-world fraud detection performance from this project.

## Interview discussion

Be prepared to explain:
- Why PR-AUC and recall are useful for rare-event detection
- The trade-off between catching more fraud and creating more false positives
- Why train/test separation and feature availability at transaction time matter
- How you would replace synthetic data with a properly governed, time-stamped real dataset
- How an investigation team would use risk scores without automatically blocking customers

## Next steps for a real-data extension

Use a permitted, documented public benchmark or approved anonymized dataset; preserve chronological ordering; split by time (and, where appropriate, account/card entity); tune thresholds against explicit investigation capacity and error costs; and report confidence intervals and drift checks.
