"""Data loading, validation, and weekly feature preparation for FORESIGHT."""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def load_raw_data(raw_dir: Path = RAW_DIR) -> Dict[str, pd.DataFrame]:
    """Load all raw tables while preserving literal ``None`` category values."""
    names = {
        "sales": "sales_daily.csv",
        "skus": "sku_master.csv",
        "calendar": "calendar.csv",
        "inventory": "inventory_snapshots.csv",
        "metadata": "generation_metadata.csv",
    }
    return {
        key: pd.read_csv(raw_dir / filename, keep_default_na=False)
        for key, filename in names.items()
    }


def validate_raw_data(data: Dict[str, pd.DataFrame]) -> dict:
    """Validate expected relationships and return a concise audit summary."""
    sales, skus = data["sales"], data["skus"]
    calendar, inventory = data["calendar"], data["inventory"]
    for frame, column in [(sales, "date"), (calendar, "date"), (inventory, "date")]:
        parsed = pd.to_datetime(frame[column], errors="coerce")
        assert parsed.notna().all(), f"Invalid dates in {column}"
    sku_ids = set(skus["sku_id"])
    assert len(skus) == 200 and skus["sku_id"].nunique() == 200
    assert set(sales["sku_id"]).issubset(sku_ids)
    assert set(inventory["sku_id"]).issubset(sku_ids)
    assert calendar["date"].is_unique
    assert not sales.duplicated(["date", "sku_id"]).any()
    assert (sales["units_sold"] >= 0).all()
    assert (sales["units_sold"] <= sales["demand_units"]).all()
    assert (inventory[["on_hand_units", "on_order_units"]] >= 0).all().all()
    assert np.isclose(
        sales["revenue"], sales["units_sold"] * sales["unit_price"], atol=0.02
    ).all()
    return {
        "sku_count": len(skus),
        "sales_rows": len(sales),
        "calendar_rows": len(calendar),
        "inventory_rows": len(inventory),
        "sales_start": str(pd.to_datetime(sales["date"]).min().date()),
        "sales_end": str(pd.to_datetime(sales["date"]).max().date()),
        "total_units_sold": int(sales["units_sold"].sum()),
        "total_revenue": float(sales["revenue"].sum()),
        "stockout_snapshots": int(inventory["stockout_flag"].sum()),
    }


def build_weekly_demand_features(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create Monday-based weekly observed demand and backward-looking features."""
    sales = data["sales"].copy()
    calendar = data["calendar"].copy()
    skus = data["skus"].copy()
    sales["date"] = pd.to_datetime(sales["date"])
    calendar["date"] = pd.to_datetime(calendar["date"])
    calendar["calendar_promo_flag"] = (calendar["promo_event"] != "None").astype(int)
    sales = sales.merge(
        calendar[["date", "calendar_promo_flag", "holiday_flag", "promo_event", "season"]],
        on="date", how="left", suffixes=("", "_calendar"),
    )
    sales["week_start"] = sales["date"] - pd.to_timedelta(sales["date"].dt.dayofweek, unit="D")
    weekly = sales.groupby(["sku_id", "week_start"], as_index=False).agg(
        demand_units=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        avg_unit_price=("unit_price", "mean"),
        avg_discount_pct=("discount_pct", "mean"),
        promo_days=("promo_flag", "sum"),
        holiday_days=("holiday_flag", "sum"),
    )
    weekly = weekly.merge(skus, on="sku_id", how="left", validate="many_to_one")
    weekly["week_of_year"] = weekly["week_start"].dt.isocalendar().week.astype(int)
    weekly["month"] = weekly["week_start"].dt.month
    weekly["quarter"] = weekly["week_start"].dt.quarter
    weekly["time_index"] = ((weekly["week_start"] - weekly["week_start"].min()).dt.days // 7).astype(int)
    grouped = weekly.sort_values(["sku_id", "week_start"]).groupby("sku_id")["demand_units"]
    for lag in (1, 2, 4, 8, 13, 52):
        weekly[f"lag_{lag}"] = grouped.shift(lag)
    shifted = grouped.shift(1)
    weekly["rolling_mean_4"] = shifted.groupby(weekly["sku_id"]).transform(
        lambda values: values.rolling(4, min_periods=2).mean()
    )
    weekly["rolling_std_4"] = shifted.groupby(weekly["sku_id"]).transform(
        lambda values: values.rolling(4, min_periods=2).std()
    ).fillna(0)
    weekly = weekly.sort_values(["week_start", "sku_id"]).reset_index(drop=True)
    return weekly


def run_pipeline(raw_dir: Path = RAW_DIR, output_path: Path = PROCESSED_DIR / "weekly_demand_features.csv") -> pd.DataFrame:
    """Validate raw data, build weekly features, and persist the result."""
    data = load_raw_data(raw_dir)
    validate_raw_data(data)
    weekly = build_weekly_demand_features(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(output_path, index=False)
    return weekly


if __name__ == "__main__":
    summary = validate_raw_data(load_raw_data())
    weekly = run_pipeline()
    print(summary)
    print(f"Created {len(weekly):,} weekly SKU rows")