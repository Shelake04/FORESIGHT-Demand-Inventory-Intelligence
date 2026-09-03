"""Build processed data, forecasts, and inventory decisions."""

from pathlib import Path

from src.forecast import evaluate_backtests, forecast_all_skus
from src.pipeline import load_raw_data, run_pipeline
from src.risk import build_decision_table


def main() -> None:
    weekly = run_pipeline()
    data = load_raw_data()
    evaluation = evaluate_backtests(weekly)
    evaluation.to_csv(Path("data/processed/backtest_results.csv"), index=False)
    forecasts = forecast_all_skus(weekly, horizon=6)
    forecasts.to_csv(Path("data/processed/forecasts.csv"), index=False)
    decisions = build_decision_table(data["inventory"], data["skus"], forecasts)
    decisions.to_csv(Path("data/processed/inventory_decisions.csv"), index=False)
    print(evaluation.to_string(index=False))
    print(f"Created {len(forecasts):,} forecasts and {len(decisions):,} decisions")


if __name__ == "__main__":
    main()