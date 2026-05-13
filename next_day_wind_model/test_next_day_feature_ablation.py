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

from data_pipeline import (
    DatasetConfig,
    _apply_standardizer,
    _fit_standardizer,
    build_all_training_arrays,
)
from train_lstm import TargetAwareNextDayLSTM


DEFAULT_SCHEMAS = [
    ("baseline", "speed_v2"),
    ("temperature", "speed_v2_plus_temperature"),
    ("pressure", "speed_v2_plus_pressure"),
    ("rain", "speed_v2_plus_rain"),
    ("weather_core", "speed_v2_plus_weather_core"),
    ("relative_humidity", "speed_v2_plus_rh"),
    ("clouds", "speed_v2_plus_clouds"),
    ("cloud_layers", "speed_v2_plus_cloud_layers"),
    ("radiation", "speed_v2_plus_radiation"),
    ("all_meteo", "speed_v2_plus_all_meteo"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chronological ablation test for next-day HARMONIE payload features.",
    )
    parser.add_argument("--db", default="data/wind_data_all_sites.db", help="Path to SQLite DB.")
    parser.add_argument("--site", default="valkenburgsemeer", help="Site name in DB.")
    parser.add_argument("--model", default="HARMONIE", help="Forecast model name in DB.")
    parser.add_argument("--window-hours", type=int, default=72, help="Input history/context hours.")
    parser.add_argument("--target-hours", type=int, default=24, help="Prediction horizon in hours.")
    parser.add_argument("--n-splits", type=int, default=3, help="Number of chronological CV folds.")
    parser.add_argument("--epochs", type=int, default=12, help="Maximum epochs per fold.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--patience", type=int, default=5, help="Early-stopping patience.")
    parser.add_argument("--constraint-eps", type=float, default=0.1, help="Log-ratio speed epsilon.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--repeats", type=int, default=1, help="Repeat each fold with different seeds.")
    parser.add_argument(
        "--schemas",
        default=",".join(name for name, _ in DEFAULT_SCHEMAS),
        help="Comma-separated schema names to run. Use names from DEFAULT_SCHEMAS.",
    )
    parser.add_argument(
        "--out-dir",
        default="next_day_wind_model/artifacts/feature_ablation",
        help="Directory for ablation CSV/JSON outputs.",
    )
    return parser.parse_args()


def _selected_schemas(schema_arg: str) -> list[tuple[str, str]]:
    requested = [item.strip() for item in str(schema_arg).split(",") if item.strip()]
    by_name = dict(DEFAULT_SCHEMAS)
    unknown = [name for name in requested if name not in by_name]
    if unknown:
        raise ValueError(f"Unknown schema names: {', '.join(unknown)}")
    return [(name, by_name[name]) for name in requested]


def _fit_target_scaler(y: np.ndarray) -> tuple[float, float]:
    mean = float(y.mean())
    std = float(y.std())
    if std == 0.0:
        std = 1.0
    return mean, std


def _train_fold_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    target_hours: int,
    history_hours: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    seed: int,
    device: torch.device,
) -> TargetAwareNextDayLSTM:
    torch.manual_seed(int(seed))
    model = TargetAwareNextDayLSTM(
        n_features=X_train.shape[2],
        target_hours=int(target_hours),
        history_hours=int(history_hours),
        output_activation="linear",
    ).to(device)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()),
        batch_size=int(batch_size),
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()),
        batch_size=int(batch_size),
        shuffle=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    criterion = nn.MSELoss()

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    no_improve = 0
    for _ in range(int(epochs)):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                val_loss = criterion(model(xb), yb)
                val_sum += float(val_loss.item()) * xb.size(0)
                val_n += xb.size(0)
        fold_val = val_sum / max(val_n, 1)
        if fold_val < best_val:
            best_val = fold_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= int(patience):
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


def _evaluate_schema(
    *,
    db_path: Path,
    cfg: DatasetConfig,
    schema_name: str,
    feature_schema: str,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict]:
    arrays = build_all_training_arrays(db_path, cfg, target_mode="residual", feature_schema=feature_schema)
    X_raw = arrays["X_all_raw"].astype(np.float32)
    actual_all = arrays["y_actual_all_raw"].astype(np.float32)
    forecast_all = arrays["y_forecast_all_raw"].astype(np.float32)
    y_logratio = np.log(actual_all + float(args.constraint_eps)) - np.log(forecast_all + float(args.constraint_eps))

    rows: list[dict] = []
    tscv = TimeSeriesSplit(n_splits=int(args.n_splits))
    for repeat_idx in range(int(args.repeats)):
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(np.arange(len(X_raw))), start=1):
            X_train_raw = X_raw[train_idx]
            X_val_raw = X_raw[val_idx]
            y_train_raw = y_logratio[train_idx]
            y_val_raw = y_logratio[val_idx]
            actual_val = actual_all[val_idx]
            forecast_val = forecast_all[val_idx]

            x_mean, x_std = _fit_standardizer(X_train_raw)
            y_mean, y_std = _fit_target_scaler(y_train_raw)
            X_train = _apply_standardizer(X_train_raw, x_mean, x_std).astype(np.float32)
            X_val = _apply_standardizer(X_val_raw, x_mean, x_std).astype(np.float32)
            y_train = ((y_train_raw - y_mean) / y_std).astype(np.float32)
            y_val = ((y_val_raw - y_mean) / y_std).astype(np.float32)

            seed = int(args.seed) + repeat_idx * 1000 + fold_idx * 37
            model = _train_fold_model(
                X_train,
                y_train,
                X_val,
                y_val,
                target_hours=int(args.target_hours),
                history_hours=int(args.window_hours),
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                learning_rate=float(args.learning_rate),
                patience=int(args.patience),
                seed=seed,
                device=device,
            )
            with torch.no_grad():
                pred_scaled = model(torch.from_numpy(X_val).float().to(device)).cpu().numpy()
            pred_logratio = pred_scaled * y_std + y_mean
            pred_speed = np.exp(np.log(forecast_val + float(args.constraint_eps)) + pred_logratio) - float(
                args.constraint_eps
            )
            pred_speed = np.maximum(0.0, pred_speed)

            harmonie_abs = np.abs(forecast_val - actual_val)
            model_abs = np.abs(pred_speed - actual_val)
            rows.append(
                {
                    "schema_name": schema_name,
                    "feature_schema": feature_schema,
                    "repeat": int(repeat_idx + 1),
                    "fold": int(fold_idx),
                    "n_train_samples": int(len(train_idx)),
                    "n_val_samples": int(len(val_idx)),
                    "n_features": int(X_raw.shape[2]),
                    "mae_harmonie_kts": float(harmonie_abs.mean()),
                    "mae_model_kts": float(model_abs.mean()),
                    "rmse_model_kts": float(np.sqrt(np.mean((pred_speed - actual_val) ** 2))),
                    "mae_gain_vs_harmonie_kts": float(harmonie_abs.mean() - model_abs.mean()),
                    "feature_cols": "|".join(str(col) for col in arrays["feature_cols"]),
                }
            )
    return rows


def _summarize(folds: pd.DataFrame) -> pd.DataFrame:
    summary = (
        folds.groupby(["schema_name", "feature_schema"], as_index=False)
        .agg(
            folds=("fold", "count"),
            val_samples=("n_val_samples", "sum"),
            n_features=("n_features", "first"),
            mae_harmonie_kts=("mae_harmonie_kts", "mean"),
            mae_model_kts=("mae_model_kts", "mean"),
            mae_model_std_kts=("mae_model_kts", "std"),
            mae_gain_vs_harmonie_kts=("mae_gain_vs_harmonie_kts", "mean"),
        )
        .sort_values("mae_model_kts")
        .reset_index(drop=True)
    )
    baseline = summary.loc[summary["schema_name"] == "baseline", "mae_model_kts"]
    if not baseline.empty:
        baseline_mae = float(baseline.iloc[0])
        summary["mae_delta_vs_baseline_kts"] = summary["mae_model_kts"] - baseline_mae
        summary["improvement_vs_baseline_kts"] = baseline_mae - summary["mae_model_kts"]
        summary["helps_vs_baseline"] = summary["mae_model_kts"] < baseline_mae
    return summary


def main() -> None:
    args = parse_args()
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device("cpu")
    db_path = Path(args.db)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = DatasetConfig(
        site=args.site,
        model=args.model,
        window_hours=int(args.window_hours),
        target_hours=int(args.target_hours),
    )

    rows: list[dict] = []
    for schema_name, feature_schema in _selected_schemas(args.schemas):
        print(f"Running {schema_name}: {feature_schema}", flush=True)
        rows.extend(
            _evaluate_schema(
                db_path=db_path,
                cfg=cfg,
                schema_name=schema_name,
                feature_schema=feature_schema,
                args=args,
                device=device,
            )
        )

    folds = pd.DataFrame(rows)
    summary = _summarize(folds)
    folds_csv = out_dir / "next_day_feature_ablation_folds.csv"
    summary_csv = out_dir / "next_day_feature_ablation_summary.csv"
    metadata_json = out_dir / "next_day_feature_ablation_metadata.json"
    folds.to_csv(folds_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "db_path": str(db_path),
        "site": args.site,
        "model": args.model,
        "window_hours": int(args.window_hours),
        "target_hours": int(args.target_hours),
        "n_splits": int(args.n_splits),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "patience": int(args.patience),
        "constraint_eps": float(args.constraint_eps),
        "seed": int(args.seed),
        "repeats": int(args.repeats),
        "folds_csv": str(folds_csv),
        "summary_csv": str(summary_csv),
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved: {folds_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {metadata_json}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
