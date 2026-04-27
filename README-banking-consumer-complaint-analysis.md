# Banking Consumer Complaint Analysis

## Project Overview
This project analyzes consumer complaints in financial services and builds NLP classification models to route complaints automatically by category.

Notebook: `banking-consumer-complaint-analysis.ipynb`

## Problem Statement
Manual complaint triage is slow and hard to scale.  
The notebook builds machine learning pipelines to classify complaint text into:
- `Product` category (Model M1)
- `Issue` category (Model M2 concept and supporting analysis)

## Dataset
Data source referenced in notebook:
- CFPB consumer complaints database
- Kaggle path used: `/kaggle/input/complaintsfull/main.csv`

Key fields include:
- `consumer complaint narrative`
- `product`, `sub-product`
- `issue`, `sub-issue`
- `date received` and related timeline fields

## Notebook Workflow
- Business background and motivation for automated complaint handling
- Data preprocessing and subset normalization
- Class distribution analysis and imbalance handling
- Exploratory analysis over time and categories
- Data preparation for text classification
- Baseline linear model (`LogisticRegression`) for complaint classification
- Transformer-based pipeline (DistilBERT feature extraction and fine-tuned modeling)
- Evaluation using confusion matrix and validation metrics

## Tech Stack
- Python
- Pandas, NumPy
- Plotly
- scikit-learn
- PyTorch
- Transformers (DistilBERT)
- Optional: `kaleido` for Plotly static export

## How to Run
1. Open `banking-consumer-complaint-analysis.ipynb`.
2. Install required dependencies (notebook includes `pip install -U kaleido`).
3. Ensure dataset path is available or update file paths for local execution.
4. Run cells in order; transformer sections may require GPU for faster execution.

## Outputs
- EDA visuals and class distribution insights
- Baseline and transformer model performance comparison
- Practical foundation for automated complaint-ticket routing
