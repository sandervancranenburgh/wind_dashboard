from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
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
        description="Tune hyperparameters for next-day (constrained residual) and intraday residual models.",
    )
    parser.add_argument("--db", default="data/wind_data.db", help="Path to SQLite DB.")
    parser.add_argument("--site", default="valkenburgsemeer", help="Site name in DB.")
    parser.add_argument("--model", default="HARMONIE", help="Forecast model name in DB.")
    parser.add_argument("--target-hours", type=int, default=24, help="Prediction horizon in hours.")
    parser.add_argument("--n-splits", type=int, default=3, help="TimeSeriesSplit folds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for config sampling.")
    parser.add_argument("--max-configs-next-day", type=int, default=18, help="Max sampled configs for next-day model.")
    parser.add_argument("--max-configs-intraday", type=int, default=24, help="Max sampled configs for intraday model.")
    parser.add_argument("--out-dir", default="next_day_wind_model/artifacts", help="Output directory for tuning reports.")
    return parser.parse_args()


def pick_torch_device() -> torch.device:
    return torch.device("cpu")


def _sample_configs(grid: dict, max_configs: int, seed: int) -> list[dict]:
    all_cfg = list(ParameterGrid(grid))
    if len(all_cfg) <= max_configs:
        return all_cfg
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_cfg), size=max_configs, replace=False)
    return [all_cfg[int(i)] for i in sorted(idx.tolist())]


def _train_next_day_fold(
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
    opt = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    crit = nn.MSELoss()

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    no_improve = 0
    patience = 8

    for _ in range(int(epochs)):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        v_sum = 0.0
        v_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                l = crit(model(xb), yb)
                v_sum += float(l.item()) * xb.size(0)
                v_n += xb.size(0)
        v_loss = v_sum / max(v_n, 1)
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


def tune_next_day_model(args: argparse.Namespace, device: torch.device) -> pd.DataFrame:
    grid = {
        "window_hours": [48, 72, 96],
        "epochs": [20, 30],
        "batch_size": [16, 32],
        "learning_rate": [5e-4, 1e-3, 2e-3],
        "speed_constraint_eps": [0.05, 0.1, 0.2],
    }
    configs = _sample_configs(grid, max_configs=int(args.max_configs_next_day), seed=int(args.seed))
    rows: list[dict] = []
    for cfg_i, hp in enumerate(configs, start=1):
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

        n = len(X_all)
        if n < (args.n_splits + 1) * 10:
            continue

        eps = float(hp["speed_constraint_eps"])
        target_all = np.log(actual_all + eps) - np.log(forecast_all + eps)
        fold_mae: list[float] = []

        tscv = TimeSeriesSplit(n_splits=int(args.n_splits))
        for train_idx, val_idx in tscv.split(np.arange(n)):
            X_train = X_all[train_idx]
            X_val = X_all[val_idx]
            y_train_raw = target_all[train_idx]
            y_val_raw = target_all[val_idx]
            actual_val = actual_all[val_idx]
            forecast_val = forecast_all[val_idx]

            y_mean = float(y_train_raw.mean())
            y_std = float(y_train_raw.std())
            if y_std == 0.0:
                y_std = 1.0
            y_train_scaled = ((y_train_raw - y_mean) / y_std).astype(np.float32)
            y_val_scaled = ((y_val_raw - y_mean) / y_std).astype(np.float32)

            model = _train_next_day_fold(
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
            mae = float(np.mean(np.abs(pred_speed - actual_val)))
            fold_mae.append(mae)

        if not fold_mae:
            continue
        rows.append(
            {
                "config_id": cfg_i,
                **hp,
                "cv_mae_kts_mean": float(np.mean(fold_mae)),
                "cv_mae_kts_std": float(np.std(fold_mae)),
                "n_folds": int(len(fold_mae)),
                "n_samples": int(n),
            }
        )
        print(
            "[next_day]",
            f"cfg={cfg_i:02d}",
            f"window={hp['window_hours']}",
            f"eps={hp['speed_constraint_eps']}",
            f"cv_mae={np.mean(fold_mae):.3f}",
        )

    if not rows:
        raise RuntimeError("No next-day tuning result generated. Check dataset size.")
    return pd.DataFrame(rows).sort_values(["cv_mae_kts_mean", "cv_mae_kts_std"]).reset_index(drop=True)


def tune_intraday_model(args: argparse.Namespace, device: torch.device) -> pd.DataFrame:
    grid = {
        "epochs": [40, 60],
        "batch_size": [16, 32],
        "hidden1": [64, 128, 192],
        "hidden2": [32, 64],
        "dropout": [0.05, 0.1, 0.2],
        "learning_rate": [3e-4, 1e-3, 2e-3],
        "recency_power": [1.0, 1.5, 2.0],
    }
    configs = _sample_configs(grid, max_configs=int(args.max_configs_intraday), seed=int(args.seed) + 1)

    cfg = DatasetConfig(
        site=args.site,
        model=args.model,
        window_hours=72,
        target_hours=int(args.target_hours),
    )
    X_all, y_all = build_intraday_training_xy(Path(args.db), cfg)
    n = len(X_all)
    if n < (args.n_splits + 1) * 20:
        raise RuntimeError("Not enough intraday samples for requested TimeSeriesSplit.")

    rows: list[dict] = []
    for cfg_i, hp in enumerate(configs, start=1):
        fold_mae: list[float] = []
        tscv = TimeSeriesSplit(n_splits=int(args.n_splits))
        for train_idx, val_idx in tscv.split(np.arange(n)):
            X_train = X_all[train_idx]
            X_val = X_all[val_idx]
            y_train = y_all[train_idx]
            y_val = y_all[val_idx]
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
            mae = float(np.mean(np.abs(pred_res - y_val)))
            fold_mae.append(mae)

        rows.append(
            {
                "config_id": cfg_i,
                **hp,
                "cv_mae_kts_mean": float(np.mean(fold_mae)),
                "cv_mae_kts_std": float(np.std(fold_mae)),
                "n_folds": int(len(fold_mae)),
                "n_samples": int(n),
            }
        )
        print(
            "[intraday]",
            f"cfg={cfg_i:02d}",
            f"h1={hp['hidden1']}",
            f"h2={hp['hidden2']}",
            f"recency={hp['recency_power']}",
            f"cv_mae={np.mean(fold_mae):.3f}",
        )

    return pd.DataFrame(rows).sort_values(["cv_mae_kts_mean", "cv_mae_kts_std"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_torch_device()

    next_day_df = tune_next_day_model(args, device)
    intraday_df = tune_intraday_model(args, device)

    next_day_csv = out_dir / "tuning_next_day_results.csv"
    intraday_csv = out_dir / "tuning_intraday_results.csv"
    next_day_df.to_csv(next_day_csv, index=False)
    intraday_df.to_csv(intraday_csv, index=False)

    best_next = next_day_df.iloc[0].to_dict()
    best_intraday = intraday_df.iloc[0].to_dict()

    recommend_cmd = (
        "python3 next_day_wind_model/update_model_and_predict.py "
        f"--db {args.db} "
        f"--window-hours {int(best_next['window_hours'])} "
        f"--epochs {int(best_next['epochs'])} "
        f"--batch-size {int(best_next['batch_size'])} "
        f"--speed-constraint-eps {float(best_next['speed_constraint_eps']):.4f} "
        f"--intraday-epochs {int(best_intraday['epochs'])} "
        f"--intraday-hidden1 {int(best_intraday['hidden1'])} "
        f"--intraday-hidden2 {int(best_intraday['hidden2'])} "
        f"--intraday-dropout {float(best_intraday['dropout']):.4f} "
        f"--intraday-learning-rate {float(best_intraday['learning_rate']):.6f} "
        f"--intraday-recency-power {float(best_intraday['recency_power']):.4f}"
    )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "db_path": str(args.db),
        "site": args.site,
        "model": args.model,
        "n_splits": int(args.n_splits),
        "next_day": {
            "best": best_next,
            "results_csv": str(next_day_csv),
        },
        "intraday": {
            "best": best_intraday,
            "results_csv": str(intraday_csv),
        },
        "recommended_train_command": recommend_cmd,
    }
    summary_json = out_dir / "tuning_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved: {next_day_csv}")
    print(f"Saved: {intraday_csv}")
    print(f"Saved: {summary_json}")
    print()
    print("Best next-day config:")
    print(next_day_df.head(1).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print("Best intraday config:")
    print(intraday_df.head(1).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print("Recommended training command:")
    print(recommend_cmd)


if __name__ == "__main__":
    main()
