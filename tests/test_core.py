import numpy as np
import pandas as pd

from src.forecast import wape
from src.pipeline import load_raw_data, validate_raw_data, build_weekly_demand_features
from src.risk import build_decision_table


def test_raw_contract_and_weekly_grain():
    data = load_raw_data()
    summary = validate_raw_data(data)
    assert summary["sku_count"] == 200
    weekly = build_weekly_demand_features(data)
    assert weekly["sku_id"].nunique() == 200
    assert not weekly.duplicated(["sku_id", "week_start"]).any()
    assert "demand_units" in weekly.columns


def test_wape_zero_denominator():
    assert np.isnan(wape([0, 0], [1, 2]))
    assert wape([2, 4], [1, 5]) == 0.3333333333333333


def test_risk_actions_are_constrained():
    skus = pd.DataFrame([{"sku_id": "A", "category": "Decor", "demand_class": "Fast", "unit_cost": 10, "list_price": 20}])
    inventory = pd.DataFrame([{"date": "2025-12-31", "sku_id": "A", "on_hand_units": 1, "on_order_units": 0, "lead_time_days": 7, "reorder_point": 10, "safety_stock_units": 2}])
    forecasts = pd.DataFrame([{"sku_id": "A", "forecast_week": "2026-01-05", "forecast_units": 20}])
    result = build_decision_table(inventory, skus, forecasts)
    assert set(result["recommended_action"]) <= {"REORDER", "WATCH", "MARKDOWN", "HEALTHY"}