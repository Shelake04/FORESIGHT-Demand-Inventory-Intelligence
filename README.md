# FORESIGHT - Demand & Inventory Intelligence

FORESIGHT is an explainable demand forecasting and inventory decision-support MVP for NorthBay Living. It uses the supplied synthetic two-year, 200-SKU, one-warehouse dataset.

## Setup and run

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python generate_data.py
python run_pipeline.py
streamlit run app/streamlit_app.py
```

The raw generator is reproducible with seed 42. Raw files are not overwritten by the processing pipeline. Processed CSVs are regenerated under `data/processed/`.

## Architecture

`data/raw` -> `src/pipeline.py` -> weekly SKU features -> `src/forecast.py` -> forecasts/backtests -> `src/risk.py` -> decisions -> Streamlit dashboard.

Weeks start on Monday. The target is observed `units_sold`; latent `demand_units` is retained for audit only and is never a model feature. Lags and rolling statistics use only prior weeks. Future promotion and holiday values are treated as unknown for the current baseline forecast.

## Forecasting

The seasonal-naive baseline repeats the value from 52 weeks earlier, with a recent-history fallback where needed. The candidate is a HistGradientBoostingRegressor with recursive six-week prediction. Rolling-origin validation uses chronological six-week windows and WAPE.

Measured average across three origins: baseline WAPE `0.118572`; candidate WAPE `0.124364`. The baseline wins overall, so production outputs use SeasonalNaive52. See `data/processed/backtest_results.csv` after running the pipeline.

## Inventory decisions

Stockout risk compares inventory position (on hand plus on order) with forecast lead-time demand plus safety stock. Overstock risk compares inventory position with six-week forecast demand plus safety stock. HIGH/MEDIUM/LOW risk maps to REORDER, MARKDOWN, WATCH, or HEALTHY. Impact is a planning estimate: shortage units times list price, or excess units times unit cost; it is not an accounting figure.

## Repository

`notebooks/` contains audit, EDA, forecasting, and risk walkthroughs. `reports/` contains the EDA memo and executive readout. `tests/` contains focused contract tests. Raw data is intentionally ignored by git because the brief treats simulated client data as confidential; teammates can regenerate it.

## Limitations

The history contains only two years, the baseline does not model known future promotions, and inventory snapshots are weekly. Confidence intervals are not claimed. The candidate model is retained for comparison, but not selected because the measured average WAPE is worse than baseline.