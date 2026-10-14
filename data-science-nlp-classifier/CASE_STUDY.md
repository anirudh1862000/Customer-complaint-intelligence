# Portfolio Case Study: Consumer Complaint Intelligence

## Business context
Customer-service and compliance teams need a reliable view of complaint volume, the issues customers report, and whether responses are recorded as timely. A useful analytics workflow should make trends visible while avoiding claims the data cannot support.

## Objective
Build a reproducible pipeline that:
1. profiles the data and documents quality issues;
2. measures complaint volume and mix over time;
3. compares response timeliness across product categories;
4. identifies issue categories with notable increases;
5. optionally tests an ML classifier as a decision-support experiment.

## Data
Use the public [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/) or another authorized source. Confirm the current available fields, terms, and download limits. The source may not include narrative text in current exports, so check columns before choosing the NLP path.

**Current status:** repository code and project specifications are prepared. The user's large uploaded archive could not be processed in this session, so the work has not yet been run against that file. No real-data findings or model scores are claimed.

## Analysis plan
- Validate uniqueness and missingness for IDs, dates, product, issue, and response fields.
- Standardize dates and categorical labels.
- Quantify product mix and monthly complaint volume.
- Calculate timely response rate using only records with a known response-timeliness value.
- Identify issue categories with increasing counts, comparing equivalent time windows.
- Review class imbalance and compare any classifier with a majority-class baseline.
- Reconcile SQL output with Power BI measures.

## Deliverables in this repository
- `src/prepare_portfolio_sample.py`: chunked sampling for a large local CSV.
- `src/train_structured_classifier.py`: structured-field classification baseline.
- `src/train_classifier.py`: text classification for historical datasets that contain narrative text.
- `src/run_synthetic_demo.py`: end-to-end code demonstration using fictional examples only.
- `sql/analyst_queries.sql`: data quality and business analysis queries.
- `powerbi/dashboard_specification.md`: report pages and suggested DAX measures.

## Business questions
1. Which products account for the largest share of complaints in the selected period?
2. Are volumes increasing or decreasing, and which issues contribute most to the change?
3. How does recorded timely response rate vary by product and over time?
4. Which categories have low sample counts that make percentages unstable?
5. Can a classifier predict product category reliably enough to support human triage, and what are its most common errors?

## Caveats
- Counts are not population-adjusted rates; do not interpret them as product failure rates without denominators.
- A timely-response field is not a measure of response quality or consumer satisfaction.
- Predicting product from fields closely related to product can cause target leakage or make the task artificially easy.
- Synthetic-demo scores must never be presented as real-world results.
- Any ML prediction should support human review, not make final decisions about consumers.

## Completion checklist
- [ ] Load a permitted real dataset and record its source and download date.
- [ ] Run data-quality checks and preserve a reproducible cleaning script.
- [ ] Execute SQL queries and reconcile totals.
- [ ] Build the Power BI report from the documented measures.
- [ ] Run a baseline and model evaluation on a held-out real dataset if suitable labels exist.
- [ ] Replace this section's current-status note with verified results, limitations, and screenshots.
