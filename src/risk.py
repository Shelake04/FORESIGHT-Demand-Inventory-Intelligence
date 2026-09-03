"""Transparent inventory risk and operational action rules."""

import numpy as np
import pandas as pd


def build_decision_table(inventory: pd.DataFrame, skus: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    """Build latest-snapshot SKU decisions from six-week forecast demand.

    Stockout impact is shortage units during lead time multiplied by list price.
    Overstock impact is excess units above six-week demand plus safety stock,
    multiplied by unit cost. Both are planning estimates, not accounting values.
    """
    latest = inventory.copy()
    latest["date"] = pd.to_datetime(latest["date"])
    latest = latest.sort_values("date").groupby("sku_id", as_index=False).tail(1)
    demand = forecasts.groupby("sku_id", as_index=False)["forecast_units"].sum().rename(columns={"forecast_units": "forecast_6_week_units"})
    sku_attributes = skus.drop(columns=["lead_time_days", "warehouse_id"], errors="ignore")
    decisions = latest.merge(sku_attributes, on="sku_id", how="left", validate="one_to_one").merge(demand, on="sku_id", how="left", validate="one_to_one")
    decisions["lead_time_demand"] = decisions["forecast_6_week_units"] * decisions["lead_time_days"] / 42
    decisions["inventory_position"] = decisions["on_hand_units"] + decisions["on_order_units"]
    decisions["stockout_risk"] = np.where(
        decisions["inventory_position"] < decisions["lead_time_demand"] + decisions["safety_stock_units"], "HIGH",
        np.where(decisions["inventory_position"] < decisions["lead_time_demand"] + 2 * decisions["safety_stock_units"], "MEDIUM", "LOW"),
    )
    coverage_target = decisions["forecast_6_week_units"] + decisions["safety_stock_units"]
    decisions["overstock_units"] = np.maximum(0, decisions["inventory_position"] - coverage_target)
    decisions["overstock_risk"] = np.where(
        decisions["overstock_units"] > 0.5 * np.maximum(decisions["forecast_6_week_units"], 1), "HIGH",
        np.where(decisions["overstock_units"] > 0, "MEDIUM", "LOW"),
    )
    decisions["recommended_action"] = np.select(
        [decisions["stockout_risk"] == "HIGH", decisions["overstock_risk"] == "HIGH", (decisions["stockout_risk"] == "MEDIUM") | (decisions["overstock_risk"] == "MEDIUM")],
        ["REORDER", "MARKDOWN", "WATCH"], default="HEALTHY",
    )
    shortage = np.maximum(0, decisions["lead_time_demand"] + decisions["safety_stock_units"] - decisions["inventory_position"])
    decisions["estimated_rupee_impact"] = np.where(
        decisions["recommended_action"] == "MARKDOWN", decisions["overstock_units"] * decisions["unit_cost"], shortage * decisions["list_price"]
    ).round(2)
    return decisions[["sku_id", "category", "demand_class", "forecast_6_week_units", "on_hand_units", "on_order_units", "lead_time_days", "reorder_point", "safety_stock_units", "stockout_risk", "overstock_risk", "recommended_action", "estimated_rupee_impact"]].sort_values("estimated_rupee_impact", ascending=False)