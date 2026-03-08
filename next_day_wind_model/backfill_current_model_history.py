from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data_pipeline import DatasetConfig, _apply_standardizer, _build_samples, _build_training_frame
from train_lstm import NextDayLSTM


LSTM_HIGHLIGHT_COLOR = "#d7191c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill historical day-level MAE using the current trained next-day model.",
    )
    parser.add_argument("--db", default="data/wind_data.db", help="Path to SQLite DB.")
    parser.add_argument("--site", default="valkenburgsemeer", help="Site name in DB.")
    parser.add_argument("--model", default="HARMONIE", help="Forecast model name in DB.")
    parser.add_argument(
        "--metadata",
        default="next_day_wind_model/artifacts/metadata_update.json",
        help="Metadata JSON from latest model update (used for window/target defaults).",
    )
    parser.add_argument(
        "--model-path",
        default="next_day_wind_model/artifacts/next_day_lstm_speed_residual.pt",
        help="Path to trained speed model checkpoint.",
    )
    parser.add_argument("--x-mean", default="next_day_wind_model/artifacts/x_mean_speed.npy", help="Path to X mean scaler.")
    parser.add_argument("--x-std", default="next_day_wind_model/artifacts/x_std_speed.npy", help="Path to X std scaler.")
    parser.add_argument("--y-mean", default="next_day_wind_model/artifacts/y_mean_speed.npy", help="Path to y mean scaler.")
    parser.add_argument("--y-std", default="next_day_wind_model/artifacts/y_std_speed.npy", help="Path to y std scaler.")
    parser.add_argument("--window-hours", type=int, default=None, help="Override input window size.")
    parser.add_argument("--target-hours", type=int, default=None, help="Override prediction horizon.")
    parser.add_argument("--local-timezone", default="Europe/Amsterdam", help="Timezone for daily grouping/plot.")
    parser.add_argument("--out-csv", default="next_day_wind_model/artifacts/dayahead_backfill_history.csv", help="Output CSV.")
    parser.add_argument("--out-png", default="next_day_wind_model/artifacts/dayahead_backfill_history.png", help="Output PNG.")
    return parser.parse_args()


def _load_model(path: Path) -> tuple[NextDayLSTM, dict]:
    device = torch.device("cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = NextDayLSTM(
        n_features=int(ckpt["n_features"]),
        target_hours=int(ckpt["target_hours"]),
        output_activation=str(ckpt.get("output_activation", "linear")),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def _plot_daily_mae(df: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    ax.plot(df["day_local"], df["mae_lstm"], color=LSTM_HIGHLIGHT_COLOR, linewidth=2.2, label="Super local vs measured wind")
    ax.plot(df["day_local"], df["mae_forecast"], color="gray", linewidth=1.8, label="Harmonie vs measured wind")
    ax.set_title("Backfilled Day-ahead MAE (Current Model on Past Data)")
    ax.set_xlabel("Date")
    ax.set_ylabel("MAE (kts)")
    ax.set_ylim(0.0, max(4.0, float(np.nanmax([df["mae_lstm"].max(), df["mae_forecast"].max()])) * 1.06))
    ax.margins(x=0, y=0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    avg_lstm = float(df["mae_lstm"].mean())
    avg_fc = float(df["mae_forecast"].mean())
    ax.text(
        0.985,
        1.02,
        f"Average MAE in shown period\nSuper local: {avg_lstm:.2f} kts\nHarmonie: {avg_fc:.2f} kts",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="black",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    fig.autofmt_xdate(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata)
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    window_hours = int(args.window_hours if args.window_hours is not None else metadata.get("window_hours", 72))
    target_hours = int(args.target_hours if args.target_hours is not None else metadata.get("target_hours", 24))

    cfg = DatasetConfig(
        site=args.site,
        model=args.model,
        window_hours=window_hours,
        target_hours=target_hours,
    )

    frame = _build_training_frame(Path(args.db), cfg)
    feature_cols = ["forecast_avg", "forecast_max", "forecast_dir", "month_sin", "month_cos"]
    X_raw, y_actual_raw, y_forecast_raw, timestamps = _build_samples(frame, cfg, feature_cols, "actual_avg")
    if len(X_raw) == 0:
        raise ValueError("No historical samples available for backfill.")

    x_mean = np.load(args.x_mean)
    x_std = np.load(args.x_std)
    y_mean = float(np.load(args.y_mean)[0])
    y_std = float(np.load(args.y_std)[0])
    X_scaled = _apply_standardizer(X_raw, x_mean, x_std).astype(np.float32)

    model, ckpt = _load_model(Path(args.model_path))
    target_mode = str(ckpt.get("target_mode", "residual")).strip().lower()
    constraint_eps = ckpt.get("constraint_eps", None)
    eps = float(0.1 if constraint_eps is None else constraint_eps)

    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(X_scaled).float()).cpu().numpy()
    pred_target = pred_scaled * y_std + y_mean

    if target_mode == "residual":
        pred_speed = y_forecast_raw + pred_target
    elif target_mode == "constrained_logratio":
        pred_speed = np.exp(np.log(y_forecast_raw + eps) + pred_target)
    elif target_mode == "absolute":
        pred_speed = pred_target
    else:
        raise ValueError(f"Unsupported speed target mode: {target_mode}")

    mae_forecast = np.mean(np.abs(y_forecast_raw - y_actual_raw), axis=1)
    mae_lstm = np.mean(np.abs(pred_speed - y_actual_raw), axis=1)
    avg_actual_wind_speed = np.mean(y_actual_raw, axis=1)
    avg_forecast_wind_speed = np.mean(y_forecast_raw, axis=1)
    avg_lstm_wind_speed = np.mean(pred_speed, axis=1)

    tz = ZoneInfo(args.local_timezone)
    anchor_times_utc = pd.to_datetime(timestamps, utc=True, errors="coerce")
    target_start_local = (anchor_times_utc + pd.Timedelta(hours=1)).tz_convert(tz)
    day_local = target_start_local.date.astype(str)

    sample_df = pd.DataFrame(
        {
            "anchor_time_utc": anchor_times_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "target_start_local": target_start_local.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "day_local": day_local,
            "mae_forecast": mae_forecast.astype(float),
            "mae_lstm": mae_lstm.astype(float),
            "avg_actual_wind_speed": avg_actual_wind_speed.astype(float),
            "avg_forecast_wind_speed": avg_forecast_wind_speed.astype(float),
            "avg_lstm_wind_speed": avg_lstm_wind_speed.astype(float),
            "improvement_vs_forecast": (mae_forecast - mae_lstm).astype(float),
        }
    )
    daily_df = (
        sample_df.groupby("day_local", as_index=False)[
            [
                "mae_forecast",
                "mae_lstm",
                "avg_actual_wind_speed",
                "avg_forecast_wind_speed",
                "avg_lstm_wind_speed",
                "improvement_vs_forecast",
            ]
        ]
        .mean(numeric_only=True)
        .sort_values("day_local")
    )
    daily_df["day_local"] = pd.to_datetime(daily_df["day_local"], errors="coerce")
    daily_df = daily_df.dropna(subset=["day_local"])

    out_csv = Path(args.out_csv)
    out_png = Path(args.out_png)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    daily_df.to_csv(out_csv, index=False, date_format="%Y-%m-%d")
    _plot_daily_mae(daily_df, out_png)

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_png}")
    print(f"Rows (days): {len(daily_df)}")
    print(f"Average MAE Harmonie: {daily_df['mae_forecast'].mean():.3f} kts")
    print(f"Average MAE Super local: {daily_df['mae_lstm'].mean():.3f} kts")
    print(f"Average gain: {(daily_df['mae_forecast'].mean() - daily_df['mae_lstm'].mean()):.3f} kts")
    print(f"Generated at UTC: {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    main()
