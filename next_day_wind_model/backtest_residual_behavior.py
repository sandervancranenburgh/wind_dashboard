from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_pipeline import DatasetConfig, build_all_training_arrays
from train_lstm import NextDayLSTM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chronological holdout backtest for residual speed model behavior.",
    )
    parser.add_argument("--db", default="data/wind_data.db", help="Path to SQLite DB.")
    parser.add_argument("--site", default="valkenburgsemeer", help="Site name in DB.")
    parser.add_argument("--model", default="HARMONIE", help="Forecast model name in DB.")
    parser.add_argument("--window-hours", type=int, default=72, help="Input history length for X.")
    parser.add_argument("--target-hours", type=int, default=24, help="Prediction horizon in hours for Y.")
    parser.add_argument("--holdout-fraction", type=float, default=0.3, help="Fraction of latest samples used as test.")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--out-dir", default="next_day_wind_model/artifacts", help="Output directory.")
    return parser.parse_args()


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    batch_size: int,
) -> nn.Module:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()),
        batch_size=batch_size,
        shuffle=True,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            opt.step()
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = DatasetConfig(
        site=args.site,
        model=args.model,
        window_hours=args.window_hours,
        target_hours=args.target_hours,
    )
    arrays = build_all_training_arrays(Path(args.db), cfg, target_mode="residual")
    X_all = arrays["X_all"]
    y_all = arrays["y_all"]
    y_actual_raw = arrays["y_actual_all_raw"]
    y_forecast_raw = arrays["y_forecast_all_raw"]
    timestamps = pd.to_datetime(arrays["timestamps"], utc=True)

    n = len(X_all)
    holdout = int(round(n * float(args.holdout_fraction)))
    holdout = max(24, min(holdout, n - 24))
    split = n - holdout

    X_train, y_train = X_all[:split], y_all[:split]
    X_test, y_test = X_all[split:], y_all[split:]
    y_actual_test = y_actual_raw[split:]
    y_forecast_test = y_forecast_raw[split:]
    ts_test = timestamps[split:]

    model = NextDayLSTM(n_features=X_all.shape[2], target_hours=y_all.shape[1])
    model = train_model(model, X_train, y_train, epochs=args.epochs, batch_size=args.batch_size)

    y_mean = float(arrays["y_mean"][0])
    y_std = float(arrays["y_std"][0])

    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(X_test).float()).numpy()
    pred_residual = pred_scaled * y_std + y_mean
    pred_lstm = y_forecast_test + pred_residual

    # Constant-offset baseline: train residual mean
    train_residual_raw = (y_all[:split] * y_std) + y_mean
    const_offset = float(np.mean(train_residual_raw))
    pred_const = y_forecast_test + const_offset

    mae_forecast_all = np.mean(np.abs(y_forecast_test - y_actual_test), axis=1)
    mae_lstm_all = np.mean(np.abs(pred_lstm - y_actual_test), axis=1)
    mae_const_all = np.mean(np.abs(pred_const - y_actual_test), axis=1)

    sample_df = pd.DataFrame(
        {
            "anchor_time_utc": ts_test.astype(str),
            "target_start_utc": (ts_test + pd.Timedelta(hours=1)).astype(str),
            "target_day_utc": (ts_test + pd.Timedelta(hours=1)).date.astype(str),
            "mae_forecast": mae_forecast_all,
            "mae_lstm": mae_lstm_all,
            "mae_const_offset": mae_const_all,
            "improvement_lstm_vs_forecast": mae_forecast_all - mae_lstm_all,
            "improvement_lstm_vs_const": mae_const_all - mae_lstm_all,
            "pred_residual_mean": pred_residual.mean(axis=1),
            "pred_residual_std": pred_residual.std(axis=1),
        }
    )
    sample_csv = out_dir / "backtest_speed_sample_metrics.csv"
    sample_df.to_csv(sample_csv, index=False)

    daily_df = (
        sample_df.groupby("target_day_utc", as_index=False)[
            ["mae_forecast", "mae_lstm", "mae_const_offset", "improvement_lstm_vs_forecast", "improvement_lstm_vs_const"]
        ]
        .mean()
        .sort_values("target_day_utc")
    )
    daily_csv = out_dir / "backtest_speed_daily_metrics.csv"
    daily_df.to_csv(daily_csv, index=False)

    # Plot daily MAE comparison.
    x = pd.to_datetime(daily_df["target_day_utc"])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(x, daily_df["mae_forecast"], marker="o", linewidth=1.8, label="Forecast MAE")
    ax.plot(x, daily_df["mae_lstm"], marker="o", linewidth=1.8, label="LSTM MAE")
    ax.plot(x, daily_df["mae_const_offset"], marker="o", linewidth=1.5, linestyle="--", label="Const-offset MAE")
    ax.set_title("Backtest Daily MAE (Chronological Holdout)")
    ax.set_xlabel("Target day (UTC)")
    ax.set_ylabel("MAE (kts)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.autofmt_xdate(rotation=35, ha="right")
    fig.tight_layout()
    plot_png = out_dir / "backtest_speed_daily_mae.png"
    fig.savefig(plot_png, dpi=150)
    plt.close(fig)

    print(f"Samples total: {n} | train: {split} | test: {holdout}")
    print(f"Overall test MAE forecast: {mae_forecast_all.mean():.3f} kts")
    print(f"Overall test MAE LSTM: {mae_lstm_all.mean():.3f} kts")
    print(f"Overall test MAE const-offset: {mae_const_all.mean():.3f} kts")
    print(f"LSTM - forecast MAE delta: {(mae_lstm_all.mean() - mae_forecast_all.mean()):.3f} kts")
    print(f"LSTM predicted residual std (per sample mean): {sample_df['pred_residual_std'].mean():.3f} kts")
    print(f"Share of days where LSTM beats forecast: {(daily_df['improvement_lstm_vs_forecast'] > 0).mean():.2%}")
    print(f"Saved: {sample_csv}")
    print(f"Saved: {daily_csv}")
    print(f"Saved: {plot_png}")


if __name__ == "__main__":
    main()
