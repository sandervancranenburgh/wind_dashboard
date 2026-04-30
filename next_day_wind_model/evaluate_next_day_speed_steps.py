from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_pipeline import (
    DatasetConfig,
    _apply_standardizer,
    _fit_standardizer,
    _fit_target_scaler,
    build_all_training_arrays,
)
from train_lstm import NextDayLSTM, TargetAwareNextDayLSTM
from update_model_and_predict import (
    _build_speed_calibration_context,
    _eval_start_index,
    _predict_speed_batch,
    _validation_start_index,
    apply_speed_regime_calibration,
    fit_speed_regime_calibration,
)


SECTOR8_LABELS = np.array(["N", "NE", "E", "SE", "S", "SW", "W", "NW"], dtype=object)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate cumulative next-day speed-model improvement steps on a fixed holdout.",
    )
    parser.add_argument("--db", default="data/wind_data_all_sites.db")
    parser.add_argument("--site", default="valkenburgsemeer")
    parser.add_argument("--model", default="HARMONIE")
    parser.add_argument("--window-hours", type=int, default=72)
    parser.add_argument("--target-hours", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--challenge-eval-split", type=float, default=0.15)
    parser.add_argument("--challenge-min-eval-samples", type=int, default=60)
    parser.add_argument("--speed-constraint-eps", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--out-dir", default="next_day_wind_model/artifacts/step_eval")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def inverse_standardizer(X_scaled: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return X_scaled * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)


def train_model(
    model: nn.Module,
    X_all: np.ndarray,
    y_all: np.ndarray,
    *,
    batch_size: int,
    epochs: int,
    validation_split: float,
    seed: int,
    label: str,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    split_idx = _validation_start_index(len(X_all), validation_split)
    X_train, X_val = X_all[:split_idx], X_all[split_idx:]
    y_train, y_val = y_all[:split_idx], y_all[split_idx:]
    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val_loss = float("inf")
    epochs_without_improve = 0
    epochs_ran = 0

    for epoch in range(1, epochs + 1):
        epochs_ran = epoch
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * X_batch.size(0)
            train_count += X_batch.size(0)

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_loss_sum += float(loss.item()) * X_batch.size(0)
                val_count += X_batch.size(0)
        train_loss = train_loss_sum / max(train_count, 1)
        val_loss = val_loss_sum / max(val_count, 1)
        scheduler.step(val_loss)
        print(f"[{label}] epoch={epoch:03d} train_loss={train_loss:.5f} val_loss={val_loss:.5f}", flush=True)

        if val_loss + 1e-9 < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= 8:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "best_val_loss": float(best_val_loss),
        "epochs_ran": int(epochs_ran),
        "train_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
    }


def prepare_speed_arrays(
    db_path: Path,
    cfg: DatasetConfig,
    *,
    feature_schema: str,
    eval_split: float,
    min_eval_samples: int,
    eps: float,
) -> dict:
    arrays = build_all_training_arrays(db_path, cfg, target_mode="residual", feature_schema=feature_schema)
    X_raw_all = inverse_standardizer(arrays["X_all"], arrays["x_mean"], arrays["x_std"]).astype(np.float32)
    y_raw_all = (
        np.log(arrays["y_actual_all_raw"] + eps) - np.log(arrays["y_forecast_all_raw"] + eps)
    ).astype(np.float32)
    eval_start = _eval_start_index(int(X_raw_all.shape[0]), eval_split, min_eval_samples)
    X_train_raw, X_eval_raw = X_raw_all[:eval_start], X_raw_all[eval_start:]
    y_train_raw = y_raw_all[:eval_start]
    x_mean, x_std = _fit_standardizer(X_train_raw)
    y_mean, y_std = _fit_target_scaler(y_train_raw)
    X_train = _apply_standardizer(X_train_raw, x_mean, x_std).astype(np.float32)
    X_eval = _apply_standardizer(X_eval_raw, x_mean, x_std).astype(np.float32)
    y_train = ((y_train_raw - y_mean) / y_std).astype(np.float32)

    anchor_times = pd.to_datetime(arrays["timestamps"], utc=True)
    train_calibration_context = _build_speed_calibration_context(
        anchor_dir_deg=arrays["anchor_forecast_dir_all_raw"][:eval_start],
        target_times_utc=anchor_times[:eval_start] + pd.Timedelta(hours=1),
    )
    eval_calibration_context = _build_speed_calibration_context(
        anchor_dir_deg=arrays["anchor_forecast_dir_all_raw"][eval_start:],
        target_times_utc=anchor_times[eval_start:] + pd.Timedelta(hours=1),
    )

    return {
        "arrays": arrays,
        "feature_schema": feature_schema,
        "eval_start": int(eval_start),
        "X_train": X_train,
        "X_eval": X_eval,
        "y_train": y_train,
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "y_mean": float(y_mean),
        "y_std": float(y_std),
        "actual_train": arrays["y_actual_all_raw"][:eval_start],
        "actual_eval": arrays["y_actual_all_raw"][eval_start:],
        "forecast_train": arrays["y_forecast_all_raw"][:eval_start],
        "forecast_eval": arrays["y_forecast_all_raw"][eval_start:],
        "target_times_train": arrays["target_times_all"][:eval_start],
        "target_times_eval": arrays["target_times_all"][eval_start:],
        "target_dirs_train": arrays["target_forecast_dir_all_raw"][:eval_start],
        "target_dirs_eval": arrays["target_forecast_dir_all_raw"][eval_start:],
        "target_horizon_train": arrays["target_horizon_hr_all"][:eval_start],
        "target_horizon_eval": arrays["target_horizon_hr_all"][eval_start:],
        "train_calibration_context": train_calibration_context,
        "eval_calibration_context": eval_calibration_context,
    }


def predict_speed_batch(
    model: nn.Module,
    prepared: dict,
    X_input: np.ndarray,
    forecast: np.ndarray,
    *,
    calibration: dict | None,
    calibration_context: dict | None,
    eps: float,
    device: torch.device,
) -> np.ndarray:
    return _predict_speed_batch(
        model,
        X_input,
        forecast,
        y_mean=float(prepared["y_mean"]),
        y_std=float(prepared["y_std"]),
        target_mode="constrained_logratio",
        constraint_eps=eps,
        speed_calibration=calibration,
        speed_calibration_context=calibration_context,
        device=device,
    )


def fit_current_sample_calibration(
    model: nn.Module,
    prepared: dict,
    *,
    validation_split: float,
    eps: float,
    device: torch.device,
) -> dict | None:
    start = _validation_start_index(len(prepared["X_train"]), validation_split)
    pred_cal = predict_speed_batch(
        model,
        prepared,
        prepared["X_train"][start:],
        prepared["forecast_train"][start:],
        calibration=None,
        calibration_context=None,
        eps=eps,
        device=device,
    )
    return fit_speed_regime_calibration(
        pred_speed=pred_cal,
        forecast_speed=prepared["forecast_train"][start:],
        actual_speed=prepared["actual_train"][start:],
        speed_calibration_context={
            "anchor_dir_deg": prepared["train_calibration_context"]["anchor_dir_deg"][start:],
            "target_month": prepared["train_calibration_context"]["target_month"][start:],
        },
        signal="pred_max",
    )


def target_hour_feature_matrix(
    pred: np.ndarray,
    forecast: np.ndarray,
    target_dirs: np.ndarray,
    target_times: np.ndarray,
    target_horizon: np.ndarray,
    stats: dict | None = None,
) -> tuple[np.ndarray, dict]:
    pred_flat = np.asarray(pred, dtype=np.float32).reshape(-1)
    forecast_flat = np.asarray(forecast, dtype=np.float32).reshape(-1)
    dirs = np.asarray(target_dirs, dtype=np.float32).reshape(-1)
    horizon = np.asarray(target_horizon, dtype=np.float32).reshape(-1)
    times = pd.to_datetime(np.asarray(target_times).reshape(-1), utc=True)

    if stats is None:
        pred_mean = float(np.nanmean(pred_flat))
        pred_std = float(np.nanstd(pred_flat)) or 1.0
        forecast_mean = float(np.nanmean(forecast_flat))
        forecast_std = float(np.nanstd(forecast_flat)) or 1.0
        horizon_mean = float(np.nanmean(horizon))
        horizon_std = float(np.nanstd(horizon)) or 1.0
    else:
        pred_mean = float(stats["pred_mean"])
        pred_std = float(stats["pred_std"])
        forecast_mean = float(stats["forecast_mean"])
        forecast_std = float(stats["forecast_std"])
        horizon_mean = float(stats["horizon_mean"])
        horizon_std = float(stats["horizon_std"])
    pred_norm = ((pred_flat - pred_mean) / max(pred_std, 1e-6)).astype(np.float32)
    forecast_norm = ((forecast_flat - forecast_mean) / max(forecast_std, 1e-6)).astype(np.float32)
    horizon_norm = ((horizon - horizon_mean) / max(horizon_std, 1e-6)).astype(np.float32)
    dir_rad = np.deg2rad(dirs % 360.0)
    dir_sin = np.sin(dir_rad).astype(np.float32)
    dir_cos = np.cos(dir_rad).astype(np.float32)
    hour_angle = (2.0 * np.pi * times.hour.to_numpy(dtype=np.float32)) / 24.0
    month_angle = (2.0 * np.pi * (times.month.to_numpy(dtype=np.float32) - 1.0)) / 12.0
    hour_sin = np.sin(hour_angle).astype(np.float32)
    hour_cos = np.cos(hour_angle).astype(np.float32)
    month_sin = np.sin(month_angle).astype(np.float32)
    month_cos = np.cos(month_angle).astype(np.float32)
    features = np.column_stack(
        [
            np.ones_like(pred_norm),
            pred_norm,
            forecast_norm,
            pred_norm - forecast_norm,
            dir_sin,
            dir_cos,
            hour_sin,
            hour_cos,
            month_sin,
            month_cos,
            horizon_norm,
            pred_norm * dir_sin,
            pred_norm * dir_cos,
            forecast_norm * dir_sin,
            forecast_norm * dir_cos,
            hour_sin * dir_sin,
            hour_cos * dir_cos,
        ]
    ).astype(np.float32)
    return features, {
        "pred_mean": pred_mean,
        "pred_std": pred_std,
        "forecast_mean": forecast_mean,
        "forecast_std": forecast_std,
        "horizon_mean": horizon_mean,
        "horizon_std": horizon_std,
    }


def fit_target_hour_calibration(
    pred: np.ndarray,
    forecast: np.ndarray,
    actual: np.ndarray,
    target_dirs: np.ndarray,
    target_times: np.ndarray,
    target_horizon: np.ndarray,
) -> dict:
    X, stats = target_hour_feature_matrix(pred, forecast, target_dirs, target_times, target_horizon)
    y = (np.asarray(actual, dtype=np.float32).reshape(-1) - np.asarray(pred, dtype=np.float32).reshape(-1))
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[mask].astype(np.float64)
    y = y[mask].astype(np.float64)
    if X.shape[0] < 64:
        return {"enabled": False, "reason": "too_few_rows"}

    split = max(1, min(int(round(X.shape[0] * 0.7)), X.shape[0] - 1))
    X_fit, X_val = X[:split], X[split:]
    y_fit, y_val = y[:split], y[split:]
    identity = np.eye(X.shape[1], dtype=np.float64)
    identity[0, 0] = 0.0
    best_ridge = 1.0
    best_mae = float("inf")
    for ridge in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0):
        try:
            coef = np.linalg.solve(X_fit.T @ X_fit + ridge * identity, X_fit.T @ y_fit)
        except np.linalg.LinAlgError:
            continue
        mae = float(np.mean(np.abs((X_val @ coef) - y_val)))
        if mae < best_mae:
            best_mae = mae
            best_ridge = float(ridge)

    coef = np.linalg.solve(X.T @ X + best_ridge * identity, X.T @ y)
    return {
        "enabled": True,
        "type": "target_hour_ridge_v1",
        "ridge": float(best_ridge),
        "stats": stats,
        "coefficients": [float(v) for v in coef.tolist()],
    }


def apply_target_hour_calibration(
    pred: np.ndarray,
    forecast: np.ndarray,
    target_dirs: np.ndarray,
    target_times: np.ndarray,
    target_horizon: np.ndarray,
    calibration: dict,
) -> np.ndarray:
    if not calibration or not bool(calibration.get("enabled", False)):
        return np.asarray(pred, dtype=np.float32)
    X, _ = target_hour_feature_matrix(
        pred,
        forecast,
        target_dirs,
        target_times,
        target_horizon,
        stats=calibration["stats"],
    )
    coef = np.asarray(calibration["coefficients"], dtype=np.float32)
    correction = (X @ coef).reshape(np.asarray(pred).shape)
    return np.maximum(np.asarray(pred, dtype=np.float32) + correction.astype(np.float32), 0.0)


def sector12_index(direction_deg: np.ndarray) -> np.ndarray:
    return np.floor(((np.asarray(direction_deg, dtype=np.float32) % 360.0) + 15.0) / 30.0).astype(int) % 12


def fit_sector_residual_calibration(
    pred: np.ndarray,
    actual: np.ndarray,
    target_dirs: np.ndarray,
    *,
    shrinkage_count: float = 30.0,
) -> dict:
    resid = np.asarray(actual, dtype=np.float32).reshape(-1) - np.asarray(pred, dtype=np.float32).reshape(-1)
    sectors = sector12_index(np.asarray(target_dirs).reshape(-1))
    offsets = np.zeros(12, dtype=np.float32)
    counts = np.zeros(12, dtype=np.int32)
    for sector in range(12):
        mask = sectors == sector
        counts[sector] = int(np.sum(mask))
        if counts[sector] > 0:
            shrink = counts[sector] / (counts[sector] + float(shrinkage_count))
            offsets[sector] = float(np.mean(resid[mask])) * float(shrink)
    return {
        "enabled": True,
        "type": "sector12_residual_mean_v1",
        "offsets": [float(v) for v in offsets.tolist()],
        "counts": [int(v) for v in counts.tolist()],
        "shrinkage_count": float(shrinkage_count),
    }


def apply_sector_residual_calibration(pred: np.ndarray, target_dirs: np.ndarray, calibration: dict) -> np.ndarray:
    if not calibration or not bool(calibration.get("enabled", False)):
        return np.asarray(pred, dtype=np.float32)
    offsets = np.asarray(calibration["offsets"], dtype=np.float32)
    sectors = sector12_index(target_dirs)
    correction = offsets[sectors].reshape(np.asarray(pred).shape)
    return np.maximum(np.asarray(pred, dtype=np.float32) + correction, 0.0)


def sector8_index(direction_deg: np.ndarray) -> np.ndarray:
    return np.floor(((np.asarray(direction_deg, dtype=np.float32) % 360.0) + 22.5) / 45.0).astype(int) % 8


def summarize_prediction(label: str, pred: np.ndarray, prepared: dict, previous_pred: np.ndarray | None) -> tuple[dict, list[dict]]:
    actual = np.asarray(prepared["actual_eval"], dtype=np.float32)
    forecast = np.asarray(prepared["forecast_eval"], dtype=np.float32)
    dirs = np.asarray(prepared["target_dirs_eval"], dtype=np.float32)
    mae = float(np.mean(np.abs(pred - actual)))
    bias = float(np.mean(pred - actual))
    forecast_mae = float(np.mean(np.abs(forecast - actual)))
    forecast_bias = float(np.mean(forecast - actual))
    previous_mae = None if previous_pred is None else float(np.mean(np.abs(previous_pred - actual)))
    overall = {
        "step": label,
        "n_samples": int(actual.shape[0]),
        "n_points": int(actual.size),
        "mae": mae,
        "bias_pred_minus_actual": bias,
        "forecast_mae": forecast_mae,
        "forecast_bias_pred_minus_actual": forecast_bias,
        "mae_gain_vs_forecast": forecast_mae - mae,
        "mae_gain_vs_previous_step": None if previous_mae is None else previous_mae - mae,
    }

    rows: list[dict] = []
    flat_pred = pred.reshape(-1)
    flat_actual = actual.reshape(-1)
    flat_forecast = forecast.reshape(-1)
    flat_dirs = dirs.reshape(-1)
    sector_idx = sector8_index(flat_dirs)
    prev_flat = None if previous_pred is None else previous_pred.reshape(-1)
    for idx, name in enumerate(SECTOR8_LABELS):
        mask = sector_idx == idx
        if not mask.any():
            continue
        sector_mae = float(np.mean(np.abs(flat_pred[mask] - flat_actual[mask])))
        sector_forecast_mae = float(np.mean(np.abs(flat_forecast[mask] - flat_actual[mask])))
        sector_prev_mae = None if prev_flat is None else float(np.mean(np.abs(prev_flat[mask] - flat_actual[mask])))
        rows.append(
            {
                "step": label,
                "sector": str(name),
                "n_points": int(np.sum(mask)),
                "mae": sector_mae,
                "bias_pred_minus_actual": float(np.mean(flat_pred[mask] - flat_actual[mask])),
                "forecast_mae": sector_forecast_mae,
                "forecast_bias_pred_minus_actual": float(np.mean(flat_forecast[mask] - flat_actual[mask])),
                "mae_gain_vs_forecast": sector_forecast_mae - sector_mae,
                "mae_gain_vs_previous_step": None if sector_prev_mae is None else sector_prev_mae - sector_mae,
            }
        )
    return overall, rows


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = DatasetConfig(
        site=args.site,
        model=args.model,
        window_hours=args.window_hours,
        target_hours=args.target_hours,
    )
    db_path = Path(args.db)

    prepared_v2 = prepare_speed_arrays(
        db_path,
        cfg,
        feature_schema="speed_v2",
        eval_split=args.challenge_eval_split,
        min_eval_samples=args.challenge_min_eval_samples,
        eps=args.speed_constraint_eps,
    )
    prepared_v3 = prepare_speed_arrays(
        db_path,
        cfg,
        feature_schema="speed_v3_actual_history",
        eval_split=args.challenge_eval_split,
        min_eval_samples=args.challenge_min_eval_samples,
        eps=args.speed_constraint_eps,
    )

    results: list[dict] = []
    sector_rows: list[dict] = []
    previous_pred: np.ndarray | None = None
    train_stats: dict[str, dict] = {}
    calibrations: dict[str, dict | None] = {}

    def record(label: str, pred: np.ndarray, prepared: dict) -> None:
        nonlocal previous_pred
        overall, rows = summarize_prediction(label, pred, prepared, previous_pred)
        results.append(overall)
        sector_rows.extend(rows)
        print(
            f"[{label}] MAE={overall['mae']:.4f} bias={overall['bias_pred_minus_actual']:+.4f} "
            f"gain_vs_forecast={overall['mae_gain_vs_forecast']:+.4f} "
            f"gain_vs_previous={overall['mae_gain_vs_previous_step']}",
            flush=True,
        )
        previous_pred = pred

    set_seed(args.seed)
    baseline_model = NextDayLSTM(
        n_features=prepared_v2["X_train"].shape[2],
        target_hours=args.target_hours,
        output_activation="linear",
    ).to(device)
    baseline_model, train_stats["reference_current_v2"] = train_model(
        baseline_model,
        prepared_v2["X_train"],
        prepared_v2["y_train"],
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=args.validation_split,
        seed=args.seed,
        label="reference_current_v2",
        device=device,
    )
    baseline_sample_cal = fit_current_sample_calibration(
        baseline_model,
        prepared_v2,
        validation_split=args.validation_split,
        eps=args.speed_constraint_eps,
        device=device,
    )
    calibrations["reference_current_v2_sample_calibration"] = baseline_sample_cal
    baseline_pred = predict_speed_batch(
        baseline_model,
        prepared_v2,
        prepared_v2["X_eval"],
        prepared_v2["forecast_eval"],
        calibration=baseline_sample_cal,
        calibration_context=prepared_v2["eval_calibration_context"],
        eps=args.speed_constraint_eps,
        device=device,
    )
    record("reference_current_v2_sample_cal", baseline_pred, prepared_v2)

    set_seed(args.seed + 1)
    target_model_v2 = TargetAwareNextDayLSTM(
        n_features=prepared_v2["X_train"].shape[2],
        target_hours=args.target_hours,
        history_hours=args.window_hours,
        output_activation="linear",
    ).to(device)
    target_model_v2, train_stats["step1_target_aware"] = train_model(
        target_model_v2,
        prepared_v2["X_train"],
        prepared_v2["y_train"],
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=args.validation_split,
        seed=args.seed + 1,
        label="step1_target_aware",
        device=device,
    )
    step1_sample_cal = fit_current_sample_calibration(
        target_model_v2,
        prepared_v2,
        validation_split=args.validation_split,
        eps=args.speed_constraint_eps,
        device=device,
    )
    calibrations["step1_sample_calibration"] = step1_sample_cal
    step1_pred = predict_speed_batch(
        target_model_v2,
        prepared_v2,
        prepared_v2["X_eval"],
        prepared_v2["forecast_eval"],
        calibration=step1_sample_cal,
        calibration_context=prepared_v2["eval_calibration_context"],
        eps=args.speed_constraint_eps,
        device=device,
    )
    record("step1_target_aware", step1_pred, prepared_v2)

    cal_start = _validation_start_index(len(prepared_v2["X_train"]), args.validation_split)
    step1_cal_raw = predict_speed_batch(
        target_model_v2,
        prepared_v2,
        prepared_v2["X_train"][cal_start:],
        prepared_v2["forecast_train"][cal_start:],
        calibration=None,
        calibration_context=None,
        eps=args.speed_constraint_eps,
        device=device,
    )
    target_hour_cal_v2 = fit_target_hour_calibration(
        pred=step1_cal_raw,
        forecast=prepared_v2["forecast_train"][cal_start:],
        actual=prepared_v2["actual_train"][cal_start:],
        target_dirs=prepared_v2["target_dirs_train"][cal_start:],
        target_times=prepared_v2["target_times_train"][cal_start:],
        target_horizon=prepared_v2["target_horizon_train"][cal_start:],
    )
    calibrations["step2_target_hour_calibration"] = target_hour_cal_v2
    step2_pred_raw = predict_speed_batch(
        target_model_v2,
        prepared_v2,
        prepared_v2["X_eval"],
        prepared_v2["forecast_eval"],
        calibration=None,
        calibration_context=None,
        eps=args.speed_constraint_eps,
        device=device,
    )
    step2_pred = apply_target_hour_calibration(
        step2_pred_raw,
        prepared_v2["forecast_eval"],
        prepared_v2["target_dirs_eval"],
        prepared_v2["target_times_eval"],
        prepared_v2["target_horizon_eval"],
        target_hour_cal_v2,
    )
    record("step2_target_hour_cal", step2_pred, prepared_v2)

    set_seed(args.seed + 2)
    target_model_v3 = TargetAwareNextDayLSTM(
        n_features=prepared_v3["X_train"].shape[2],
        target_hours=args.target_hours,
        history_hours=args.window_hours,
        output_activation="linear",
    ).to(device)
    target_model_v3, train_stats["step3_actual_history"] = train_model(
        target_model_v3,
        prepared_v3["X_train"],
        prepared_v3["y_train"],
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=args.validation_split,
        seed=args.seed + 2,
        label="step3_actual_history",
        device=device,
    )
    cal_start_v3 = _validation_start_index(len(prepared_v3["X_train"]), args.validation_split)
    step3_cal_raw = predict_speed_batch(
        target_model_v3,
        prepared_v3,
        prepared_v3["X_train"][cal_start_v3:],
        prepared_v3["forecast_train"][cal_start_v3:],
        calibration=None,
        calibration_context=None,
        eps=args.speed_constraint_eps,
        device=device,
    )
    target_hour_cal_v3 = fit_target_hour_calibration(
        pred=step3_cal_raw,
        forecast=prepared_v3["forecast_train"][cal_start_v3:],
        actual=prepared_v3["actual_train"][cal_start_v3:],
        target_dirs=prepared_v3["target_dirs_train"][cal_start_v3:],
        target_times=prepared_v3["target_times_train"][cal_start_v3:],
        target_horizon=prepared_v3["target_horizon_train"][cal_start_v3:],
    )
    calibrations["step3_target_hour_calibration"] = target_hour_cal_v3
    step3_raw_eval = predict_speed_batch(
        target_model_v3,
        prepared_v3,
        prepared_v3["X_eval"],
        prepared_v3["forecast_eval"],
        calibration=None,
        calibration_context=None,
        eps=args.speed_constraint_eps,
        device=device,
    )
    step3_pred = apply_target_hour_calibration(
        step3_raw_eval,
        prepared_v3["forecast_eval"],
        prepared_v3["target_dirs_eval"],
        prepared_v3["target_times_eval"],
        prepared_v3["target_horizon_eval"],
        target_hour_cal_v3,
    )
    record("step3_actual_history", step3_pred, prepared_v3)

    step3_cal_target_hour = apply_target_hour_calibration(
        step3_cal_raw,
        prepared_v3["forecast_train"][cal_start_v3:],
        prepared_v3["target_dirs_train"][cal_start_v3:],
        prepared_v3["target_times_train"][cal_start_v3:],
        prepared_v3["target_horizon_train"][cal_start_v3:],
        target_hour_cal_v3,
    )
    sector_cal = fit_sector_residual_calibration(
        step3_cal_target_hour,
        prepared_v3["actual_train"][cal_start_v3:],
        prepared_v3["target_dirs_train"][cal_start_v3:],
    )
    calibrations["step4_sector_calibration"] = sector_cal
    step4_pred = apply_sector_residual_calibration(step3_pred, prepared_v3["target_dirs_eval"], sector_cal)
    record("step4_sector_residual_cal", step4_pred, prepared_v3)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    overall_csv = out_dir / f"{stamp}_next_day_speed_step_overall.csv"
    sector_csv = out_dir / f"{stamp}_next_day_speed_step_direction_bias.csv"
    report_json = out_dir / f"{stamp}_next_day_speed_step_report.json"
    pd.DataFrame(results).to_csv(overall_csv, index=False)
    pd.DataFrame(sector_rows).to_csv(sector_csv, index=False)
    with report_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "args": vars(args),
                "feature_cols_v2": prepared_v2["arrays"]["feature_cols"],
                "feature_cols_v3": prepared_v3["arrays"]["feature_cols"],
                "train_stats": train_stats,
                "calibrations": calibrations,
                "overall_csv": str(overall_csv),
                "sector_csv": str(sector_csv),
            },
            f,
            indent=2,
        )
    print(f"Saved overall metrics to {overall_csv}")
    print(f"Saved direction-bias metrics to {sector_csv}")
    print(f"Saved report to {report_json}")


if __name__ == "__main__":
    main()
