
"""
FORESIGHT Synthetic Retail Dataset Generator
=============================================

Creates the four core FORESIGHT data tables:
    data/raw/sales_daily.csv
    data/raw/sku_master.csv
    data/raw/calendar.csv
    data/raw/inventory_snapshots.csv

Configuration:
    200 SKUs
    1 warehouse
    2024-01-01 through 2025-12-31
    ~146,200 daily SKU records
    weekly inventory snapshots

Install:
    pip install pandas numpy

Run:
    python generate_data.py
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd

SEED = 42
START_DATE = "2024-01-01"
END_DATE = "2025-12-31"
N_SKUS = 200
WAREHOUSE_ID = "WH_001"
SNAPSHOT_FREQ = "7D"
OUTPUT_DIR = Path("data/raw")

rng = np.random.default_rng(SEED)


def season(month):
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


def holiday(date):
    holidays = {
        (1, 1): "New Year's Day",
        (1, 26): "Republic Day",
        (8, 15): "Independence Day",
        (10, 2): "Gandhi Jayanti",
        (12, 25): "Christmas",
    }
    return holidays.get((date.month, date.day), "None")


def promotion(date):
    if date.month == 1 and date.day <= 7:
        return "New Year Sale", 1.25
    if date.month == 3 and 10 <= date.day <= 20:
        return "Spring Festival Sale", 1.20
    if date.month == 5 and 15 <= date.day <= 31:
        return "Summer Sale", 1.30
    if date.month == 8 and 10 <= date.day <= 20:
        return "Mid-Year Sale", 1.18
    if date.month == 10 and 15 <= date.day <= 31:
        return "Festive Sale", 1.45
    if date.month == 11 and 20 <= date.day <= 30:
        return "Black Friday", 1.35
    if date.month == 12 and date.day >= 15:
        return "Year End Sale", 1.28
    return "None", 1.0


def weekly_factor(day):
    return [0.94, 0.98, 1.00, 1.04, 1.10, 1.20, 1.15][day]


def annual_factor(day_of_year):
    x = 2 * math.pi * day_of_year / 365.25
    return (1 + 0.12 * math.sin(x - 0.8)) * (
        1 + 0.06 * math.cos(2 * x + 0.5)
    )


def make_sku_master():
    category_map = {
        "Furniture": ["Chair", "Table", "Sofa", "Desk"],
        "Decor": ["Wall Art", "Vase", "Cushion", "Decor Piece"],
        "Kitchen": ["Cookware", "Blender", "Storage", "Appliance"],
        "Lighting": ["Table Lamp", "Floor Lamp", "Ceiling Light", "LED Light"],
        "Storage": ["Shelf", "Cabinet", "Organizer", "Drawer"],
    }

    cats = list(category_map)
    cat_values = rng.choice(
        cats, N_SKUS, p=[0.22, 0.20, 0.23, 0.17, 0.18]
    )

    rows = []
    for i, cat in enumerate(cat_values, 1):
        price = float(np.clip(
            rng.lognormal(np.log(1800), 0.65), 300, 15000
        ))
        demand_class = rng.choice(
            ["Slow", "Medium", "Fast"], p=[0.25, 0.55, 0.20]
        )
        if demand_class == "Slow":
            base = float(rng.uniform(2, 8))
        elif demand_class == "Medium":
            base = float(rng.uniform(8, 25))
        else:
            base = float(rng.uniform(25, 60))

        rows.append({
            "sku_id": f"SKU_{i:03d}",
            "category": cat,
            "subcategory": rng.choice(category_map[cat]),
            "launch_date": (
                pd.Timestamp(START_DATE)
                - pd.Timedelta(days=int(rng.integers(0, 365)))
            ).date().isoformat(),
            "unit_cost": round(price * rng.uniform(0.45, 0.72), 2),
            "list_price": round(price, 2),
            "demand_class": demand_class,
            "base_daily_demand": round(base, 3),
            "lead_time_days": int(rng.integers(2, 11)),
            "warehouse_id": WAREHOUSE_ID,
        })
    return pd.DataFrame(rows)


def make_calendar(dates):
    rows = []
    for d in dates:
        promo, mult = promotion(d)
        hol = holiday(d)
        rows.append({
            "date": d.date().isoformat(),
            "week": int(d.isocalendar().week),
            "month": int(d.month),
            "quarter": int(d.quarter),
            "season": season(d.month),
            "day_of_week": int(d.dayofweek),
            "is_weekend": int(d.dayofweek >= 5),
            "holiday_flag": int(hol != "None"),
            "holiday_name": hol,
            "promo_event": promo,
            "promo_multiplier": mult,
        })
    return pd.DataFrame(rows)


def make_demand(skus, calendar):
    cal = calendar.copy()
    cal["date"] = pd.to_datetime(cal["date"])
    rows = []

    for s in skus.itertuples(index=False):
        promo_sensitivity = float(rng.uniform(0.75, 1.30))
        noise = float(rng.uniform(0.08, 0.25))

        for d, wd, wknd, hol, pmult, pflag, seas in zip(
            cal["date"], cal["day_of_week"], cal["is_weekend"],
            cal["holiday_flag"], cal["promo_multiplier"],
            (cal["promo_multiplier"] > 1).astype(int), cal["season"]
        ):
            days = (d - cal["date"].min()).days
            trend = 1 + 0.00015 * days
            expected = (
                s.base_daily_demand
                * weekly_factor(int(wd))
                * annual_factor(int(d.dayofyear))
                * (1 + (pmult - 1) * promo_sensitivity)
                * (1.12 if hol else 1.0)
                * trend
            )
            noisy = max(
                0.0,
                expected * rng.lognormal(-0.5 * noise**2, noise)
            )
            demand = int(rng.poisson(max(noisy, 0.01)))
            if rng.random() < 0.008:
                demand = int(demand * rng.uniform(1.5, 2.5))

            discount = float(rng.uniform(5, 25)) if pflag else 0.0
            price = s.list_price * (1 - discount / 100)

            rows.append({
                "date": d.date().isoformat(),
                "sku_id": s.sku_id,
                "demand_units": demand,
                "units_sold": demand,  # reconciled after inventory simulation
                "revenue": round(demand * price, 2),
                "unit_price": round(price, 2),
                "promo_flag": int(pflag),
                "discount_pct": round(discount, 2),
                "day_of_week": int(wd),
                "is_weekend": int(wknd),
                "holiday_flag": int(hol),
                "season": seas,
            })
    return pd.DataFrame(rows)


def make_inventory(skus, sales):
    sales = sales.copy()
    sales["date"] = pd.to_datetime(sales["date"])
    all_dates = pd.date_range(START_DATE, END_DATE, freq="D")
    snapshot_dates = set(pd.date_range(
        START_DATE, END_DATE, freq=SNAPSHOT_FREQ
    ))
    snapshot_dates.add(pd.Timestamp(END_DATE))

    rows = []
    actual_sales = {}

    for s in skus.itertuples(index=False):
        ss = sales[sales.sku_id == s.sku_id].set_index("date")
        demand_std = float(ss["demand_units"].rolling(
            28, min_periods=7
        ).std().median())
        if not np.isfinite(demand_std) or demand_std <= 0:
            demand_std = max(s.base_daily_demand * 0.25, 1)

        safety = max(
            10,
            int(math.ceil(
                1.65 * demand_std * math.sqrt(s.lead_time_days)
            ))
        )
        reorder = int(math.ceil(
            s.base_daily_demand * s.lead_time_days + safety
        ))

        on_hand = int(math.ceil(
            s.base_daily_demand * rng.uniform(8, 35) + safety
        ))
        if s.demand_class == "Slow" and rng.random() < 0.35:
            on_hand = int(on_hand * rng.uniform(1.8, 3.5))

        scheduled = {}

        for d in all_dates:
            received = int(scheduled.pop(d, 0))
            on_hand += received

            demand = int(ss.loc[d, "demand_units"])
            sold = min(demand, on_hand)
            actual_sales[(d, s.sku_id)] = sold
            on_hand -= sold

            on_order = int(sum(scheduled.values()))
            target = int(math.ceil(
                s.base_daily_demand * (s.lead_time_days + 14)
                + safety
            ))
            position = on_hand + on_order

            if position <= reorder:
                qty = max(0, target - position)
                qty = max(
                    qty,
                    max(5, int(math.ceil(s.base_daily_demand * 7)))
                )
                arrival = d + pd.Timedelta(days=s.lead_time_days)
                if arrival <= pd.Timestamp(END_DATE):
                    scheduled[arrival] = (
                        scheduled.get(arrival, 0) + int(qty)
                    )

            on_order = int(sum(scheduled.values()))

            if d in snapshot_dates:
                rows.append({
                    "date": d.date().isoformat(),
                    "sku_id": s.sku_id,
                    "warehouse_id": WAREHOUSE_ID,
                    "on_hand_units": int(on_hand),
                    "on_order_units": on_order,
                    "lead_time_days": int(s.lead_time_days),
                    "reorder_point": reorder,
                    "safety_stock_units": safety,
                    "inventory_value": round(
                        on_hand * s.unit_cost, 2
                    ),
                    "stockout_flag": int(on_hand == 0),
                })

    sales["units_sold"] = [
        actual_sales[(d, sku)]
        for d, sku in zip(sales["date"], sales["sku_id"])
    ]
    sales["revenue"] = np.round(
        sales["units_sold"] * sales["unit_price"], 2
    )

    return sales, pd.DataFrame(rows)


def validate(skus, calendar, sales, inventory):
    assert len(skus) == 200
    assert skus.sku_id.nunique() == 200
    assert len(sales) == 200 * len(calendar)
    assert sales[["date", "sku_id"]].duplicated().sum() == 0
    assert sales.units_sold.min() >= 0
    assert sales.demand_units.min() >= 0
    assert (sales.units_sold <= sales.demand_units).all()
    assert inventory.on_hand_units.min() >= 0
    assert inventory.on_order_units.min() >= 0
    assert inventory.lead_time_days.min() > 0
    assert inventory.reorder_point.min() >= 0
    print("VALIDATION PASSED")
    print(f"SKUs: {len(skus):,}")
    print(f"Calendar rows: {len(calendar):,}")
    print(f"Sales rows: {len(sales):,}")
    print(f"Inventory snapshots: {len(inventory):,}")
    print(f"Total actual units sold: {sales.units_sold.sum():,}")
    print(f"Total revenue: {sales.revenue.sum():,.2f}")
    print(f"Stockout snapshots: {inventory.stockout_flag.sum():,}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    skus = make_sku_master()
    calendar = make_calendar(dates)
    sales = make_demand(skus, calendar)
    sales, inventory = make_inventory(skus, sales)

    validate(skus, calendar, sales, inventory)

    skus.to_csv(OUTPUT_DIR / "sku_master.csv", index=False)
    calendar.to_csv(OUTPUT_DIR / "calendar.csv", index=False)
    sales.to_csv(OUTPUT_DIR / "sales_daily.csv", index=False)
    inventory.to_csv(
        OUTPUT_DIR / "inventory_snapshots.csv", index=False
    )

    pd.DataFrame([{
        "seed": SEED,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "number_of_skus": N_SKUS,
        "warehouse_id": WAREHOUSE_ID,
        "sales_rows": len(sales),
        "inventory_rows": len(inventory),
    }]).to_csv(
        OUTPUT_DIR / "generation_metadata.csv", index=False
    )

    print("\nCreated:")
    for f in sorted(OUTPUT_DIR.glob("*.csv")):
        print(" ", f)


if __name__ == "__main__":
    main()
