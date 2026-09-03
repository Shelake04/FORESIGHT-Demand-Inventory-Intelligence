# Executive Readout

## Business problem

Operations needs a six-week view of SKU demand and a transparent way to prioritize stockout and overstock actions.

## What we built

FORESIGHT validates raw data, produces weekly SKU demand, compares a seasonal-naive baseline with one feature-based model using rolling-origin WAPE, and converts forecasts plus inventory snapshots into operational actions.

## Forecasting result

Across three chronological six-week origins, seasonal-naive average WAPE was **0.118572** and HistGradientBoosting average WAPE was **0.124364**. The baseline therefore remains the selected production method. These are calculated validation results, not guarantees of future performance.

## Inventory and actions

The decision output covers all 200 SKUs and classifies each as REORDER, WATCH, MARKDOWN, or HEALTHY. Estimated rupee impact is a prioritization estimate based on list price for projected shortage or unit cost for excess stock.

## Limitations and next steps

The current baseline does not use known future promotion calendars and has only two years of history. Next steps are to validate the decision thresholds with operations, add a promotion-aware model only if future promotion plans are available, and monitor WAPE and stockout outcomes prospectively.