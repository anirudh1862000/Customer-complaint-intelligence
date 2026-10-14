# Power BI Dashboard Specification

## Goal
Help an operations or customer-experience team understand complaint mix, volume trends, response timeliness, and emerging issue categories.

## Pages

### 1. Executive Overview
- KPI cards: total complaints, month-over-month change, timely response rate, distinct products, date coverage.
- Line chart: monthly complaint volume.
- Bar chart: complaints by product.
- Slicers: received date, product, issue, submitted via.
- Detail table: product, issue, complaint count, share of total.

### 2. Response Operations
- Timely response rate by product and month.
- Company response distribution.
- Matrix: product × response category.
- Trend: timely response rate over time.
- Show denominator counts alongside percentages to prevent small groups being overinterpreted.

### 3. Issue Monitoring
- Top issues by complaint volume.
- Month-over-month absolute change and percentage change.
- Heatmap: issue by month.
- Drill-through from product to issue and submitted channel.

### 4. Data Quality
- Total rows and distinct complaint IDs.
- Missing values for product, received date, issue, and response fields.
- Earliest/latest dates and a warning for incomplete current month.
- Duplicate ID count and unexpected categorical values.

## Suggested measures (adjust names to the imported model)

```DAX
Total Complaints =
COUNTROWS(Complaints)

Distinct Complaints =
DISTINCTCOUNT(Complaints[complaint_id])

Complaints Previous Month =
CALCULATE(
    [Total Complaints],
    DATEADD('Calendar'[Date], -1, MONTH)
)

MoM Change =
[Total Complaints] - [Complaints Previous Month]

MoM Change % =
DIVIDE([MoM Change], [Complaints Previous Month])

Timely Responses =
CALCULATE(
    [Total Complaints],
    Complaints[timely_response] = "Yes"
)

Timely Response Rate =
DIVIDE(
    [Timely Responses],
    CALCULATE(
        [Total Complaints],
        NOT ISBLANK(Complaints[timely_response])
    )
)
```

Create a proper Calendar table, relate it to `Complaints[date_received]`, and mark it as the date table. Validate source values before using the literal `"Yes"`.

## Design guidance
- Keep a clear hierarchy: headline KPIs → trends → breakdowns → detail.
- Use consistent units and display percentage measures as percentages.
- Add tooltip definitions and denominator counts.
- Use accessible contrast and avoid relying on color alone.
- Add a footer: "Complaint volume is not a direct measure of product prevalence or customer satisfaction."

## Before sharing
1. Import a documented, permitted source export.
2. Validate data types, nulls, duplicate IDs, and date range.
3. Reconcile Power BI totals to SQL counts.
4. Check measures against hand-calculated examples.
5. Capture dashboard screenshots only after real data has been loaded; do not fabricate screenshots or results.
