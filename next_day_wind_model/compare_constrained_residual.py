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
        description="Compare non-constrained residual model vs constrained residual model on chronological holdout.",
    )
    parser.add_argument("--db", default="data/wind_data.db", help="Path to SQLite DB.")
    parser.add_argument("--site", default="valkenburgsemeer", help="Site name in DB.")
    parser.add_argument("--model", default="HARMONIE", help="Forecast model name in DB.")
    parser.add_argument("--window-hours", type=int, default=72, help="Input history length for X.")
    parser.add_argument("--target-hours", type=int, default=24, help="Prediction horizon in hours for Y.")
    parser.add_argument("--holdout-fraction", type=float, default=0.3, help="Fraction of latest samples as test.")
    parser.add_argument("--epochs", type=int, default=25, help="Max training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--eps", type=float, default=0.1, help="Stability epsilon for log-ratio constrained target.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--out-dir", default="next_day_wind_model/artifacts", help="Output folder.")
    return parser.parse_args()


def fit_target_scaler(y: np.ndarray) -> tuple[float, float]:
    mean = float(y.mean())
    std = float(y.std())
    if std == 0:
        std = 1.0
    return mean, std


def train_lstm_with_val(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    batch_size: int,
    seed: int,
) -> NextDayLSTM:
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(X_train)
    n_val = max(64, int(0.2 * n))
    n_val = min(n_val, n - 1)
    n_tr = n - n_val
    X_tr, X_val = X_train[:n_tr], X_train[n_tr:]
    y_tr, y_val = y_train[:n_tr], y_train[n_tr:]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).float()),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()),
        batch_size=batch_size,
        shuffle=False,
    )

    model = NextDayLSTM(n_features=X_train.shape[2], target_hours=y_train.shape[1], output_activation="linear")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    no_improve = 0
    patience = 8

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                loss = criterion(pred, yb)
                val_sum += float(loss.item()) * xb.size(0)
                val_n += xb.size(0)
        val_loss = val_sum / max(val_n, 1)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
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
    y_actual_raw = arrays["y_actual_all_raw"].astype(np.float32)
    y_forecast_raw = arrays["y_forecast_all_raw"].astype(np.float32)
    timestamps = pd.to_datetime(arrays["timestamps"], utc=True)

    n = len(X_all)
    holdout = int(round(n * float(args.holdout_fraction)))
    holdout = max(120, min(holdout, n - 120))
    split = n - holdout

    X_train, X_test = X_all[:split], X_all[split:]
    actual_train, actual_test = y_actual_raw[:split], y_actual_raw[split:]
    forecast_train, forecast_test = y_forecast_raw[:split], y_forecast_raw[split:]
    ts_test = timestamps[split:]

    # Variant 1: non-constrained additive residual.
    y_res_train_raw = actual_train - forecast_train
    m_res, s_res = fit_target_scaler(y_res_train_raw)
    y_res_train_scaled = (y_res_train_raw - m_res) / s_res
    model_res = train_lstm_with_val(X_train, y_res_train_scaled, args.epochs, args.batch_size, args.seed)
    with torch.no_grad():
        pred_res_scaled = model_res(torch.from_numpy(X_test).float()).numpy()
    pred_res_raw = pred_res_scaled * s_res + m_res
    pred_speed_residual = forecast_test + pred_res_raw

    # Variant 2: constrained residual via log-ratio target:
    # z = log(actual + eps) - log(forecast + eps)
    # actual_hat = exp(log(forecast + eps) + z_hat) - eps  (>= -eps; here > 0 in practice)
    eps = float(args.eps)
    y_logratio_train_raw = np.log(actual_train + eps) - np.log(forecast_train + eps)
    m_lr, s_lr = fit_target_scaler(y_logratio_train_raw)
    y_logratio_train_scaled = (y_logratio_train_raw - m_lr) / s_lr
    model_constrained = train_lstm_with_val(X_train, y_logratio_train_scaled, args.epochs, args.batch_size, args.seed + 1)
    with torch.no_grad():
        pred_lr_scaled = model_constrained(torch.from_numpy(X_test).float()).numpy()
    pred_lr_raw = pred_lr_scaled * s_lr + m_lr
    pred_speed_constrained = np.exp(np.log(forecast_test + eps) + pred_lr_raw) - eps

    mae_forecast = np.mean(np.abs(forecast_test - actual_test), axis=1)
    mae_res = np.mean(np.abs(pred_speed_residual - actual_test), axis=1)
    mae_constrained = np.mean(np.abs(pred_speed_constrained - actual_test), axis=1)

    sample_df = pd.DataFrame(
        {
            "anchor_time_utc": ts_test.astype(str),
            "target_day_utc": (ts_test + pd.Timedelta(hours=1)).date.astype(str),
            "mae_forecast": mae_forecast,
            "mae_residual_unconstrained": mae_res,
            "mae_residual_constrained": mae_constrained,
            "impr_unconstrained_vs_forecast": mae_forecast - mae_res,
            "impr_constrained_vs_forecast": mae_forecast - mae_constrained,
            "impr_unconstrained_vs_constrained": mae_constrained - mae_res,
            "min_pred_unconstrained": pred_speed_residual.min(axis=1),
            "min_pred_constrained": pred_speed_constrained.min(axis=1),
        }
    )
    sample_csv = out_dir / "compare_residual_variants_samples.csv"
    sample_df.to_csv(sample_csv, index=False)

    daily_df = (
        sample_df.groupby("target_day_utc", as_index=False)[
            [
                "mae_forecast",
                "mae_residual_unconstrained",
                "mae_residual_constrained",
                "impr_unconstrained_vs_forecast",
                "impr_constrained_vs_forecast",
                "impr_unconstrained_vs_constrained",
            ]
        ]
        .mean()
        .sort_values("target_day_utc")
    )
    daily_csv = out_dir / "compare_residual_variants_daily.csv"
    daily_df.to_csv(daily_csv, index=False)

    x = pd.to_datetime(daily_df["target_day_utc"])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(x, daily_df["mae_forecast"], marker="o", linewidth=1.6, label="Forecast")
    ax.plot(x, daily_df["mae_residual_unconstrained"], marker="o", linewidth=1.6, label="Residual (unconstrained)")
    ax.plot(x, daily_df["mae_residual_constrained"], marker="o", linewidth=1.6, label="Residual (constrained)")
    ax.set_title("Speed MAE by Day: Residual Model Variants")
    ax.set_xlabel("Target day (UTC)")
    ax.set_ylabel("MAE (kts)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.autofmt_xdate(rotation=35, ha="right")
    fig.tight_layout()
    plot_png = out_dir / "compare_residual_variants_daily_mae.png"
    fig.savefig(plot_png, dpi=150)
    plt.close(fig)

    print(f"Samples total: {n} | train: {split} | test: {holdout}")
    print(f"Forecast MAE: {mae_forecast.mean():.3f} kts")
    print(f"Residual unconstrained MAE: {mae_res.mean():.3f} kts")
    print(f"Residual constrained MAE: {mae_constrained.mean():.3f} kts")
    print(f"Share of days unconstrained beats forecast: {(daily_df['impr_unconstrained_vs_forecast'] > 0).mean():.2%}")
    print(f"Share of days constrained beats forecast: {(daily_df['impr_constrained_vs_forecast'] > 0).mean():.2%}")
    print(f"Share of days unconstrained beats constrained: {(daily_df['impr_unconstrained_vs_constrained'] > 0).mean():.2%}")
    print(f"Any negative preds (unconstrained): {bool((pred_speed_residual < 0).any())}")
    print(f"Any negative preds (constrained): {bool((pred_speed_constrained < 0).any())}")
    print(f"Saved: {sample_csv}")
    print(f"Saved: {daily_csv}")
    print(f"Saved: {plot_png}")


if __name__ == "__main__":
    main()
