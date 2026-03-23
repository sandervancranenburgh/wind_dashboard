from __future__ import annotations

import argparse
from pathlib import Path

from update_model_and_predict import save_daily_mae_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create/update daily MAE history PNG from CSV.")
    parser.add_argument(
        "--csv",
        default="next_day_wind_model/artifacts/daily_mae_history.csv",
        help="Path to daily MAE history CSV.",
    )
    parser.add_argument(
        "--png",
        default="next_day_wind_model/artifacts/daily_mae_history.png",
        help="Path to output PNG.",
    )
    parser.add_argument(
        "--last-months",
        type=int,
        default=3,
        help="Limit the plot to the most recent N months; use 0 to show all history.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    png_path = Path(args.png)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    save_daily_mae_plot(csv_path, png_path, last_months=(args.last_months or None))
    print(f"Daily MAE plot saved to: {png_path}")


if __name__ == "__main__":
    main()
