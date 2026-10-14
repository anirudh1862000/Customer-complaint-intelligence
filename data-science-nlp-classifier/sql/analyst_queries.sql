-- Customer Complaint Intelligence: analyst SQL pack
-- Dialect: PostgreSQL. Assumed staging table: complaints.
-- Map source headers to snake_case columns during ingestion.
-- This file contains analysis templates; run against real data before reporting results.

-- 1) Data quality: row count, date coverage, missing key fields
SELECT
    COUNT(*) AS total_rows,
    COUNT(*) FILTER (WHERE product IS NULL OR BTRIM(product) = '') AS missing_product,
    COUNT(*) FILTER (WHERE date_received IS NULL) AS missing_date_received,
    COUNT(DISTINCT complaint_id) AS distinct_complaint_ids,
    MIN(date_received) AS first_received_date,
    MAX(date_received) AS last_received_date
FROM complaints;

-- 2) Complaint volume by product
SELECT
    product,
    COUNT(*) AS complaints,
    ROUND(100.0 * COUNT(*) / NULLIF(SUM(COUNT(*)) OVER (), 0), 2) AS share_pct
FROM complaints
WHERE product IS NOT NULL
GROUP BY product
ORDER BY complaints DESC;

-- 3) Monthly trend and month-over-month change
WITH monthly AS (
    SELECT DATE_TRUNC('month', date_received)::date AS month,
           COUNT(*) AS complaints
    FROM complaints
    WHERE date_received IS NOT NULL
    GROUP BY 1
)
SELECT month, complaints,
       LAG(complaints) OVER (ORDER BY month) AS previous_month,
       ROUND(
         100.0 * (complaints - LAG(complaints) OVER (ORDER BY month))
         / NULLIF(LAG(complaints) OVER (ORDER BY month), 0), 2
       ) AS mom_change_pct
FROM monthly
ORDER BY month;

-- 4) Timely response rate by product
-- Verify how source values are encoded before interpreting this KPI.
SELECT product,
       COUNT(*) AS eligible_complaints,
       COUNT(*) FILTER (WHERE LOWER(BTRIM(timely_response)) IN ('yes', 'true', '1')) AS timely_responses,
       ROUND(
         100.0 * COUNT(*) FILTER (WHERE LOWER(BTRIM(timely_response)) IN ('yes', 'true', '1'))
         / NULLIF(COUNT(*) FILTER (WHERE timely_response IS NOT NULL), 0), 2
       ) AS timely_response_rate_pct
FROM complaints
GROUP BY product
ORDER BY eligible_complaints DESC;

-- 5) Company response mix (not a satisfaction measure)
SELECT product, company_response,
       COUNT(*) AS complaints,
       ROUND(100.0 * COUNT(*) / NULLIF(SUM(COUNT(*)) OVER (PARTITION BY product), 0), 2) AS within_product_pct
FROM complaints
GROUP BY product, company_response
ORDER BY product, complaints DESC;

-- 6) Issue categories with rising month-over-month volume
WITH issue_month AS (
    SELECT DATE_TRUNC('month', date_received)::date AS month,
           issue, COUNT(*) AS complaints
    FROM complaints
    WHERE date_received IS NOT NULL AND issue IS NOT NULL
    GROUP BY 1, 2
),
changes AS (
    SELECT *, LAG(complaints) OVER (PARTITION BY issue ORDER BY month) AS previous_month
    FROM issue_month
)
SELECT month, issue, complaints, previous_month,
       complaints - previous_month AS absolute_change,
       ROUND(100.0 * (complaints - previous_month) / NULLIF(previous_month, 0), 2) AS pct_change
FROM changes
WHERE previous_month IS NOT NULL
ORDER BY absolute_change DESC
LIMIT 25;

-- Interpretation guardrails:
-- * Complaint counts are not automatically rates; population/exposure denominators may be absent.
-- * Company response labels do not directly measure consumer satisfaction or case quality.
-- * Compare equivalent time windows and flag partial months.
