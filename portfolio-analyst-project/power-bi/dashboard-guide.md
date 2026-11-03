# Power BI Dashboard Guide

## Data model

Import `data/processed/complaints_clean.csv`. Keep `complaint_id` as text (not numeric) and `date_received` as Date. Create a dedicated Date table and relate it to `complaints[date_received]`.

### Date table (DAX)

```DAX
Date =
ADDCOLUMNS(
    CALENDAR(
        MIN(complaints[date_received]),
        MAX(complaints[date_received])
    ),
    "Year", YEAR([Date]),
    "Month", FORMAT([Date], "MMM"),
    "Month Number", MONTH([Date]),
    "Year-Month", FORMAT([Date], "YYYY-MM"),
    "Quarter", "Q" & FORMAT([Date], "Q")
)
```

Sort Date[Month] by Date[Month Number] and Date[Year-Month] chronologically.

## Measures (DAX)

```DAX
Complaint Count =
DISTINCTCOUNT(complaints[complaint_id])

Timely Response Count =
CALCULATE(
    [Complaint Count],
    complaints[timely_response] = TRUE()
)

Response Timeliness % =
DIVIDE(
    [Timely Response Count],
    CALCULATE(
        [Complaint Count],
        NOT ISBLANK(complaints[timely_response])
    )
)

Complaints MoM % =
VAR PreviousMonth =
    CALCULATE([Complaint Count], DATEADD('Date'[Date], -1, MONTH))
RETURN
    DIVIDE([Complaint Count] - PreviousMonth, PreviousMonth)

Company Count =
DISTINCTCOUNT(complaints[company])
```

Format Response Timeliness % and Complaints MoM % as percentages. Validate boolean conversion after import; CFPB exports may use Yes/No strings.

## Report pages

### Page 1 — Executive Summary
- KPI cards: Complaint Count, Response Timeliness %, Company Count.
- Line chart: complaint count by Date[Year-Month].
- Bar chart: top 10 Products.
- Slicers: date range, product, state, submitted via.

### Page 2 — Product & Issue
- Matrix: Product → Sub-product → Issue.
- Bar chart: complaint count by issue.
- Tooltip: complaint count, % of selected product.

### Page 3 — Company Response
- Bar chart: timely response % by company (filter to a meaningful minimum volume).
- Stacked bar: company response category by product.
- Include complaint volume beside response percentages; do not rank companies on percentage alone.

### Page 4 — Geography & Channel
- Map or filled map by state if supported in your Power BI environment.
- Bar chart: submitted-via channel.
- Include note that state counts are not adjusted for population.

## Design tips

- Use a restrained, consistent theme and clear titles.
- Add “Last data refresh” and source attribution.
- Use drill-through from product to issue and company.
- Avoid red/green status colors without accessible labels.
- Do not claim a company has a higher complaint rate unless you have customer/market-share denominators.

## Portfolio screenshots

After running the report on your own downloaded export, add screenshots to a `screenshots/` folder. Use only aggregated charts and avoid showing complaint narratives or personal details.
