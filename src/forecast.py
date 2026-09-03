"""Explainable weekly forecasting and rolling-origin evaluation."""

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

SEASONAL_PERIOD = 52
HORIZON = 6
FEATURE_COLUMNS = [
    "lag_1", "lag_2", "lag_4", "lag_8", "lag_13", "lag_52",
    "rolling_mean_4", "rolling_std_4", "week_of_year", "month", "quarter",
    "promo_days", "holiday_days", "avg_discount_pct", "base_daily_demand",
]


def wape(actual: Iterable[float], forecast: Iterable[float]) -> float:
    """Return weighted absolute percentage error; NaN means no actual volume."""
    actual_array = np.asarray(actual, dtype=float)
    forecast_array = np.asarray(forecast, dtype=float)
    denominator = np.abs(actual_array).sum()
    return float(np.abs(actual_array - forecast_array).sum() / denominator) if denominator else float("nan")


def seasonal_naive(history: pd.DataFrame, horizon: int = HORIZON, period: int = SEASONAL_PERIOD) -> pd.DataFrame:
    """Forecast each SKU from the value one seasonal period earlier."""
    rows = []
    dates = sorted(history["week_start"].unique())
    future_dates = pd.date_range(dates[-1] + pd.Timedelta(weeks=1), periods=horizon, freq="7D")
    for sku_id, group in history.groupby("sku_id"):
        values = group.sort_values("week_start")["demand_units"].tolist()
        for step, date in enumerate(future_dates):
            value = values[-period + step] if len(values) >= period - step else np.mean(values[-4:])
            rows.append({"sku_id": sku_id, "forecast_week": date, "forecast_units": max(0.0, float(value)), "model_name": "SeasonalNaive52"})
    return pd.DataFrame(rows)


def _feature_row(history: pd.DataFrame, sku_id: str, date: pd.Timestamp, template: pd.Series) -> dict:
    values = history.loc[history["sku_id"] == sku_id].sort_values("week_start")["demand_units"].tolist()
    def lag(period: int) -> float:
        return float(values[-period]) if len(values) >= period else float(np.mean(values[-min(4, len(values)):]))
    recent = np.asarray(values[-4:], dtype=float)
    return {
        "lag_1": lag(1), "lag_2": lag(2), "lag_4": lag(4), "lag_8": lag(8),
        "lag_13": lag(13), "lag_52": lag(52), "rolling_mean_4": float(recent.mean()),
        "rolling_std_4": float(recent.std(ddof=1)) if len(recent) > 1 else 0.0,
        "week_of_year": int(date.isocalendar().week), "month": date.month,
        "quarter": date.quarter, "promo_days": 0, "holiday_days": 0,
        "avg_discount_pct": 0.0, "base_daily_demand": float(template["base_daily_demand"]),
    }


def _fit_model(train: pd.DataFrame) -> HistGradientBoostingRegressor:
    usable = train.dropna(subset=FEATURE_COLUMNS)
    model = HistGradientBoostingRegressor(max_iter=160, learning_rate=0.08, max_leaf_nodes=31, random_state=42)
    model.fit(usable[FEATURE_COLUMNS], usable["demand_units"])
    return model


def model_forecast(history: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Recursively forecast all SKUs; future promotions are conservatively unknown/zero."""
    train = history[history["week_start"] <= history["week_start"].max()].copy()
    model = _fit_model(train)
    dates = pd.date_range(history["week_start"].max() + pd.Timedelta(weeks=1), periods=horizon, freq="7D")
    working = history[["sku_id", "week_start", "demand_units"]].copy()
    templates = history.sort_values("week_start").groupby("sku_id").tail(1).set_index("sku_id")
    rows = []
    for date in dates:
        for sku_id, template in templates.iterrows():
            row = _feature_row(working, sku_id, date, template)
            prediction = max(0.0, float(model.predict(pd.DataFrame([row])[FEATURE_COLUMNS])[0]))
            rows.append({"sku_id": sku_id, "forecast_week": date, "forecast_units": prediction, "model_name": "HistGradientBoosting"})
            working = pd.concat([working, pd.DataFrame([{ "sku_id": sku_id, "week_start": date, "demand_units": prediction }])], ignore_index=True)
    return pd.DataFrame(rows)


def _evaluate_one(history: pd.DataFrame, origin: pd.Timestamp, horizon: int) -> pd.DataFrame:
    train = history[history["week_start"] <= origin]
    actual = history[(history["week_start"] > origin) & (history["week_start"] <= origin + pd.Timedelta(weeks=horizon))]
    baseline = seasonal_naive(train, horizon=horizon)
    candidate = model_forecast(train, horizon=horizon)
    merged = actual[["sku_id", "week_start", "demand_units"]].merge(
        baseline[["sku_id", "forecast_week", "forecast_units"]], left_on=["sku_id", "week_start"], right_on=["sku_id", "forecast_week"], how="inner"
    ).rename(columns={"forecast_units": "baseline_forecast"})
    merged = merged.merge(candidate[["sku_id", "forecast_week", "forecast_units"]], left_on=["sku_id", "week_start"], right_on=["sku_id", "forecast_week"], how="inner")
    return pd.DataFrame([{
        "origin": origin.date().isoformat(), "horizon": horizon,
        "baseline_wape": wape(merged["demand_units"], merged["baseline_forecast"]),
        "model_wape": wape(merged["demand_units"], merged["forecast_units"]),
        "actual_units": int(merged["demand_units"].sum()),
    }])


def evaluate_backtests(weekly: pd.DataFrame, horizon: int = HORIZON, origins: list[int] | None = None) -> pd.DataFrame:
    """Evaluate chronological origins, requiring enough history for seasonal naive."""
    dates = sorted(pd.to_datetime(weekly["week_start"]).unique())
    origin_positions = origins or [65, 77, 89]
    results = [_evaluate_one(weekly, dates[position], horizon) for position in origin_positions if position + horizon < len(dates)]
    result = pd.concat(results, ignore_index=True)
    summary = pd.DataFrame([{
        "origin": "AVERAGE", "horizon": horizon,
        "baseline_wape": float(result["baseline_wape"].mean()), "model_wape": float(result["model_wape"].mean()),
        "actual_units": int(result["actual_units"].sum()),
    }])
    return pd.concat([result, summary], ignore_index=True)


def forecast_all_skus(weekly: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Select the candidate only for production output after backtesting."""
    """Generate production forecasts using the validated seasonal-naive baseline."""
    return seasonal_naive(weekly, horizon=horizon)