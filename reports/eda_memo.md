# EDA Memo

## Dataset overview

The generated data contains 200 SKUs, 731 calendar dates, 146,200 daily sales rows, and 21,200 inventory snapshots covering 2024-01-01 through 2025-12-31 in warehouse WH_001. Observed units sold total 3,427,521 and generated revenue totals Rs 6,475,936,299.24.

## Quality and cleaning

All raw tables loaded successfully with no row duplicates or invalid dates. SKU keys resolve to the 200-row master. Sales have one row per SKU/date. Units sold and inventory quantities are non-negative, units sold do not exceed latent demand, and revenue reconciles to units sold times unit price within rounding tolerance. Literal `None` calendar labels are preserved as category values during loading.

## Observations

The sales table includes promotion, holiday, season, weekday, price, and discount fields suitable for descriptive analysis. Inventory contains 565 stockout snapshots. The pipeline aggregates observed units sold to Monday-based weeks and computes backward-looking lags and rolling features.

## Limitations and assumptions

Promotion associations are descriptive, not causal. Latent demand is an audit/simulation field and is excluded from forecasting. Snapshot inventory is used as the latest available operational state; no unobserved supplier information is invented.