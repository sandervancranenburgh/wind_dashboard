from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import TimeSeriesSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_pipeline import DatasetConfig, build_all_training_arrays
from intraday_model import (
    IntradayTrainParams,
    _fit_intraday_from_arrays,
    build_intraday_training_xy,
)
from train_lstm import NextDayLSTM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest tuned vs default hyperparameters for next-day and intraday models.",
    )
    parser.add_argument("--db", default="data/wind_data.db", help="Path to SQLite DB.")
    parser.add_argument("--site", default="valkenburgsemeer", help="Site name in DB.")
    parser.add_argument("--model", default="HARMONIE", help="Forecast model name in DB.")
    parser.add_argument("--target-hours", type=int, default=24, help="Prediction horizon in hours.")
    parser.add_argument("--n-splits", type=int, default=4, help="Number of time-series CV folds.")
    parser.add_argument(
        "--tuning-summary",
        default="next_day_wind_model/artifacts/tuning_summary.json",
        help="Path to tuning summary JSON produced by tune_hyperparameters.py",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--out-dir", default="next_day_wind_model/artifacts", help="Output directory.")
    return parser.parse_args()


def pick_torch_device() -> torch.device:
    return torch.device("cpu")


def _train_next_day_model(
    X_train: np.ndarray,
    y_train_scaled: np.ndarray,
    X_val: np.ndarray,
    y_val_scaled: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> NextDayLSTM:
    model = NextDayLSTM(
        n_features=X_train.shape[2],
        target_hours=y_train_scaled.shape[1],
        output_activation="linear",
    ).to(device)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train_scaled).float()),
        batch_size=int(batch_size),
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val_scaled).float()),
        batch_size=int(batch_size),
        shuffle=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    criterion = nn.MSELoss()

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    no_improve = 0
    patience = 8

    for _ in range(int(epochs)):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
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
                xb = xb.to(device)
                yb = yb.to(device)
                l = criterion(model(xb), yb)
                val_sum += float(l.item()) * xb.size(0)
                val_n += xb.size(0)
        v_loss = val_sum / max(val_n, 1)
        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model


def _mean_metrics(df: pd.DataFrame, model_name: str, baseline_col: str, model_col: str) -> dict:
    sub = df[df["model_name"] == model_name]
    m_model = float(sub[model_col].mean())
    m_base = float(sub[baseline_col].mean())
    return {
        "model_mae_mean": m_model,
        "baseline_mae_mean": m_base,
        "improvement_vs_baseline_kts": float(m_base - m_model),
    }


def run_next_day_backtest(args: argparse.Namespace, tuned_cfg: dict, device: torch.device) -> pd.DataFrame:
    default_cfg = {
        "window_hours": 72,
        "epochs": 30,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "speed_constraint_eps": 0.1,
    }
    configs = [
        ("default_next_day", default_cfg),
        (
            "tuned_next_day",
            {
                "window_hours": int(tuned_cfg["window_hours"]),
                "epochs": int(tuned_cfg["epochs"]),
                "batch_size": int(tuned_cfg["batch_size"]),
                "learning_rate": float(tuned_cfg["learning_rate"]),
                "speed_constraint_eps": float(tuned_cfg["speed_constraint_eps"]),
            },
        ),
    ]

    rows: list[dict] = []
    for model_name, hp in configs:
        cfg = DatasetConfig(
            site=args.site,
            model=args.model,
            window_hours=int(hp["window_hours"]),
            target_hours=int(args.target_hours),
        )
        arrays = build_all_training_arrays(Path(args.db), cfg, target_mode="residual")
        X_all = arrays["X_all"].astype(np.float32)
        actual_all = arrays["y_actual_all_raw"].astype(np.float32)
        forecast_all = arrays["y_forecast_all_raw"].astype(np.float32)
        eps = float(hp["speed_constraint_eps"])
        y_raw_all = np.log(actual_all + eps) - np.log(forecast_all + eps)
        tscv = TimeSeriesSplit(n_splits=int(args.n_splits))

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(np.arange(len(X_all))), start=1):
            X_train = X_all[train_idx]
            X_val = X_all[val_idx]
            y_train_raw = y_raw_all[train_idx]
            y_val_raw = y_raw_all[val_idx]
            actual_val = actual_all[val_idx]
            forecast_val = forecast_all[val_idx]

            y_mean = float(y_train_raw.mean())
            y_std = float(y_train_raw.std())
            if y_std == 0.0:
                y_std = 1.0
            y_train_scaled = ((y_train_raw - y_mean) / y_std).astype(np.float32)
            y_val_scaled = ((y_val_raw - y_mean) / y_std).astype(np.float32)

            model = _train_next_day_model(
                X_train=X_train,
                y_train_scaled=y_train_scaled,
                X_val=X_val,
                y_val_scaled=y_val_scaled,
                epochs=int(hp["epochs"]),
                batch_size=int(hp["batch_size"]),
                learning_rate=float(hp["learning_rate"]),
                device=device,
            )
            with torch.no_grad():
                pred_scaled = model(torch.from_numpy(X_val).float().to(device)).cpu().numpy()
            pred_logratio = pred_scaled * y_std + y_mean
            pred_speed = np.exp(np.log(forecast_val + eps) + pred_logratio) - eps

            mae_forecast = float(np.mean(np.abs(forecast_val - actual_val)))
            mae_model = float(np.mean(np.abs(pred_speed - actual_val)))
            rows.append(
                {
                    "task": "next_day",
                    "model_name": model_name,
                    "fold": int(fold_idx),
                    "n_val_samples": int(len(val_idx)),
                    "mae_forecast_kts": mae_forecast,
                    "mae_model_kts": mae_model,
                    "improvement_vs_forecast_kts": float(mae_forecast - mae_model),
                    **hp,
                }
            )
    return pd.DataFrame(rows)


def run_intraday_backtest(args: argparse.Namespace, tuned_cfg: dict, device: torch.device) -> pd.DataFrame:
    default_cfg = {
        "epochs": 50,
        "batch_size": 32,
        "hidden1": 128,
        "hidden2": 64,
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "recency_power": 1.0,
    }
    configs = [
        ("default_intraday", default_cfg),
        (
            "tuned_intraday",
            {
                "epochs": int(tuned_cfg["epochs"]),
                "batch_size": int(tuned_cfg["batch_size"]),
                "hidden1": int(tuned_cfg["hidden1"]),
                "hidden2": int(tuned_cfg["hidden2"]),
                "dropout": float(tuned_cfg["dropout"]),
                "learning_rate": float(tuned_cfg["learning_rate"]),
                "recency_power": float(tuned_cfg["recency_power"]),
            },
        ),
    ]

    cfg = DatasetConfig(
        site=args.site,
        model=args.model,
        window_hours=72,
        target_hours=int(args.target_hours),
    )
    X_all, y_all = build_intraday_training_xy(Path(args.db), cfg)
    tscv = TimeSeriesSplit(n_splits=int(args.n_splits))
    rows: list[dict] = []

    for model_name, hp in configs:
        params = IntradayTrainParams(
            epochs=int(hp["epochs"]),
            batch_size=int(hp["batch_size"]),
            validation_split=0.2,
            hidden1=int(hp["hidden1"]),
            hidden2=int(hp["hidden2"]),
            dropout=float(hp["dropout"]),
            learning_rate=float(hp["learning_rate"]),
            recency_power=float(hp["recency_power"]),
        )
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(np.arange(len(X_all))), start=1):
            X_train = X_all[train_idx]
            y_train = y_all[train_idx]
            X_val = X_all[val_idx]
            y_val = y_all[val_idx]

            bundle, _ = _fit_intraday_from_arrays(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                params=params,
                device=device,
            )
            X_val_s = (X_val - bundle.x_mean) / bundle.x_std
            with torch.no_grad():
                pred_s = bundle.model(torch.from_numpy(X_val_s).float().to(device)).cpu().numpy()
            pred_res = pred_s * bundle.y_std + bundle.y_mean

            mae_forecast = float(np.mean(np.abs(y_val)))  # baseline residual=0 (forecast itself)
            mae_model = float(np.mean(np.abs(pred_res - y_val)))
            rows.append(
                {
                    "task": "intraday",
                    "model_name": model_name,
                    "fold": int(fold_idx),
                    "n_val_samples": int(len(val_idx)),
                    "mae_forecast_kts": mae_forecast,
                    "mae_model_kts": mae_model,
                    "improvement_vs_forecast_kts": float(mae_forecast - mae_model),
                    **hp,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = pick_torch_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.tuning_summary).open("r", encoding="utf-8") as f:
        tuning_summary = json.load(f)
    tuned_next = tuning_summary["next_day"]["best"]
    tuned_intraday = tuning_summary["intraday"]["best"]

    next_day_df = run_next_day_backtest(args, tuned_next, device)
    intraday_df = run_intraday_backtest(args, tuned_intraday, device)
    all_df = pd.concat([next_day_df, intraday_df], ignore_index=True)
    all_csv = out_dir / "backtest_tuned_vs_default_folds.csv"
    all_df.to_csv(all_csv, index=False)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "db_path": str(args.db),
        "n_splits": int(args.n_splits),
        "next_day_default": _mean_metrics(next_day_df, "default_next_day", "mae_forecast_kts", "mae_model_kts"),
        "next_day_tuned": _mean_metrics(next_day_df, "tuned_next_day", "mae_forecast_kts", "mae_model_kts"),
        "intraday_default": _mean_metrics(intraday_df, "default_intraday", "mae_forecast_kts", "mae_model_kts"),
        "intraday_tuned": _mean_metrics(intraday_df, "tuned_intraday", "mae_forecast_kts", "mae_model_kts"),
        "tuned_configs": {
            "next_day": tuned_next,
            "intraday": tuned_intraday,
        },
        "fold_results_csv": str(all_csv),
    }
    summary_json = out_dir / "backtest_tuned_vs_default_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved: {all_csv}")
    print(f"Saved: {summary_json}")
    print()
    print("Next-day (MAE kts):")
    print("default:", summary["next_day_default"])
    print("tuned:  ", summary["next_day_tuned"])
    print()
    print("Intraday (MAE kts):")
    print("default:", summary["intraday_default"])
    print("tuned:  ", summary["intraday_tuned"])


if __name__ == "__main__":
    main()
