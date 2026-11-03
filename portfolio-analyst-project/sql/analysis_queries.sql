/*
CFPB Consumer Complaint Analysis — SQL starter queries
Target: SQLite-compatible SQL. Adjust DATE functions / booleans for SQL Server,
PostgreSQL, or your warehouse.

Load data/processed/complaints_clean.csv into a table named complaints.
*/

-- 1) Executive KPI snapshot
SELECT
    COUNT(*) AS total_complaints,
    COUNT(DISTINCT complaint_id) AS unique_complaints,
    COUNT(DISTINCT company) AS companies_in_dataset,
    COUNT(DISTINCT product) AS product_categories,
    SUM(CASE WHEN timely_response = 1 THEN 1 ELSE 0 END) AS timely_responses,
    ROUND(
        100.0 * SUM(CASE WHEN timely_response = 1 THEN 1 ELSE 0 END)
        / NULLIF(COUNT(timely_response), 0), 2
    ) AS timely_response_pct
FROM complaints;


-- 2) Monthly complaint trend
SELECT
    strftime('%Y-%m', date_received) AS complaint_month,
    COUNT(*) AS complaint_count
FROM complaints
WHERE date_received IS NOT NULL
GROUP BY strftime('%Y-%m', date_received)
ORDER BY complaint_month;


-- 3) Top products and share of complaints
WITH product_counts AS (
    SELECT product, COUNT(*) AS complaint_count
    FROM complaints
    WHERE product IS NOT NULL
    GROUP BY product
)
SELECT
    product,
    complaint_count,
    ROUND(100.0 * complaint_count / SUM(complaint_count) OVER (), 2) AS pct_of_product_complaints,
    DENSE_RANK() OVER (ORDER BY complaint_count DESC) AS product_rank
FROM product_counts
ORDER BY complaint_count DESC;


-- 4) Most common issues within each product
WITH issue_counts AS (
    SELECT product, issue, COUNT(*) AS complaint_count
    FROM complaints
    WHERE product IS NOT NULL AND issue IS NOT NULL
    GROUP BY product, issue
)
SELECT
    product, issue, complaint_count,
    DENSE_RANK() OVER (PARTITION BY product ORDER BY complaint_count DESC) AS issue_rank
FROM issue_counts
ORDER BY product, issue_rank;


-- 5) Company response outcomes (include counts to avoid misleading percentages)
SELECT
    company,
    COUNT(*) AS complaint_count,
    SUM(CASE WHEN timely_response = 1 THEN 1 ELSE 0 END) AS timely_count,
    SUM(CASE WHEN timely_response = 0 THEN 1 ELSE 0 END) AS not_timely_count,
    ROUND(
        100.0 * SUM(CASE WHEN timely_response = 1 THEN 1 ELSE 0 END)
        / NULLIF(COUNT(timely_response), 0), 2
    ) AS timely_response_pct
FROM complaints
WHERE company IS NOT NULL
GROUP BY company
HAVING COUNT(*) >= 25
ORDER BY complaint_count DESC;


-- 6) Submission channel
SELECT
    submitted_via,
    COUNT(*) AS complaint_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS share_pct
FROM complaints
WHERE submitted_via IS NOT NULL
GROUP BY submitted_via
ORDER BY complaint_count DESC;


-- 7) State-level complaint counts (not population-normalized)
SELECT state, COUNT(*) AS complaint_count
FROM complaints
WHERE state IS NOT NULL
GROUP BY state
ORDER BY complaint_count DESC;


-- 8) Response category mix
SELECT
    company_response,
    COUNT(*) AS complaint_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS share_pct
FROM complaints
WHERE company_response IS NOT NULL
GROUP BY company_response
ORDER BY complaint_count DESC;


-- 9) Year-over-year monthly comparison
WITH monthly AS (
    SELECT
        strftime('%m', date_received) AS month_num,
        strftime('%Y', date_received) AS year_num,
        COUNT(*) AS complaint_count
    FROM complaints
    WHERE date_received IS NOT NULL
    GROUP BY year_num, month_num
)
SELECT
    month_num,
    year_num,
    complaint_count,
    LAG(complaint_count) OVER (PARTITION BY month_num ORDER BY year_num) AS prior_year_count,
    ROUND(
        100.0 * (complaint_count - LAG(complaint_count) OVER (PARTITION BY month_num ORDER BY year_num))
        / NULLIF(LAG(complaint_count) OVER (PARTITION BY month_num ORDER BY year_num), 0),
        2
    ) AS yoy_pct_change
FROM monthly
ORDER BY month_num, year_num;
