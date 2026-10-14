# Customer Complaint Intelligence

A text-analytics project exploring consumer financial-services complaints and workflows for classifying complaint narratives into product or issue categories.

> **Portfolio focus:** NLP, text preprocessing, exploratory analysis, classification baselines, and model evaluation.

## Business Problem

Manually routing customer complaints can be time-consuming and inconsistent at scale. This project investigates how complaint text and existing category labels could support an assisted triage workflow.

## Questions Explored

- How do complaint volumes vary by product, issue, and time?
- Which complaint categories are represented most heavily?
- How can narrative text be prepared for supervised classification?
- How does a traditional machine-learning baseline compare with a transformer-based approach?
- Which evaluation measures help reveal class-level errors and potential routing risks?

## Workflow

1. **Explore the data** — inspect complaint records, dates, labels, and class distributions.
2. **Prepare text** — clean and normalize relevant fields and construct modeling subsets.
3. **Establish a baseline** — use a scikit-learn `LogisticRegression` classifier.
4. **Explore transformer modeling** — use DistilBERT-related feature extraction or fine-tuning steps present in the notebook.
5. **Evaluate** — review confusion matrices and validation metrics to understand strengths and error patterns.

## Tools & Skills

- **Python:** Pandas, NumPy
- **Visualization:** Plotly
- **Traditional ML:** scikit-learn, Logistic Regression
- **NLP / deep learning:** PyTorch, Hugging Face Transformers, DistilBERT
- **Analytics methods:** text preprocessing, class-distribution analysis, supervised classification, model evaluation

## Repository Contents

- `banking-consumer-complaint-analysis.ipynb` — analysis, preprocessing, and modeling workflow
- [`data-science-nlp-classifier/`](data-science-nlp-classifier/README.md) — reproducible machine-learning project with a structured classifier for current CFPB exports and an NLP classifier for historical datasets that contain complaint narratives

## Run the Notebook

1. Clone or download the repository.
2. Open the notebook in Jupyter or Google Colab.
3. Install the libraries required by the notebook, including `pandas`, `numpy`, `plotly`, `scikit-learn`, `torch`, and `transformers`.
4. Download or access the source data and update the notebook's dataset path.
5. Run the notebook in sequence. Transformer steps may require substantial memory and can run faster on a GPU.

The notebook references the CFPB consumer complaints data and a Kaggle path at `/kaggle/input/complaintsfull/main.csv`. Confirm dataset access and licensing before redistribution.

## Evaluation & Responsible Use

- Review class-level precision, recall, F1-score, and confusion matrices before considering operational use.
- Complaint categories can be imbalanced; aggregate accuracy alone may hide poor performance on less common categories.
- Automated predictions should support human review, not make final customer-impacting decisions without appropriate safeguards.
- Check the notebook for the actual validation setup and measured results; this README does not claim unverified scores or business impact.

## Reproducible Data Scientist Project

The [`data-science-nlp-classifier/`](data-science-nlp-classifier/README.md) folder contains two runnable baselines: a structured-field classifier for current CFPB exports and an NLP classifier for historical datasets that include complaint narratives. CFPB release notes indicate narratives were removed in September 2026, so verify the fields in your download before choosing a mode. The model has not been evaluated on your own dataset here; report performance scores only after running and validating the outputs.

## Potential Extensions

- Add a reproducible train/validation/test strategy and compare all models on the same split.
- Document class-level metrics and common misclassification patterns.
- Add a lightweight inference example with clear limitations.
- Include privacy-conscious examples and avoid committing sensitive complaint narratives.

---
Created as a portfolio project exploring NLP-assisted complaint classification.