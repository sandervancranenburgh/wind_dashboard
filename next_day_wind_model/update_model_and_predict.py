from __future__ import annotations

import argparse
import copy
import json
import shutil
import sqlite3
import subprocess
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_pipeline import (
    DatasetConfig,
    _angle_add_deg,
    _build_forecast_feature_frame,
    _apply_standardizer,
    _fit_standardizer,
    _fit_target_scaler,
    build_all_direction_training_arrays,
    build_all_training_arrays,
    build_next_day_inference_input,
)
from intraday_model import (
    IntradayBundle,
    load_intraday_model,
    predict_intraday_day_speed,
    save_intraday_model,
    train_intraday_model,
)
from train_lstm import NextDayLSTM


LSTM_HIGHLIGHT_COLOR = "#d7191c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrain residual models (speed + direction) on all data and output next-day predictions.",
    )
    parser.add_argument("--db", default="data/wind_data.db", help="Path to SQLite DB.")
    parser.add_argument("--site", default="valkenburgsemeer", help="Site name in DB.")
    parser.add_argument("--model", default="HARMONIE", help="Forecast model name in DB.")
    parser.add_argument("--window-hours", type=int, default=72, help="Input history length for X.")
    parser.add_argument("--target-hours", type=int, default=24, help="Prediction horizon in hours for Y.")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--intraday-epochs", type=int, default=50, help="Training epochs for intraday model.")
    parser.add_argument("--intraday-hidden1", type=int, default=128, help="Intraday MLP first hidden size.")
    parser.add_argument("--intraday-hidden2", type=int, default=64, help="Intraday MLP second hidden size.")
    parser.add_argument("--intraday-dropout", type=float, default=0.1, help="Intraday MLP dropout.")
    parser.add_argument("--intraday-learning-rate", type=float, default=1e-3, help="Intraday MLP Adam learning rate.")
    parser.add_argument(
        "--intraday-recency-power",
        type=float,
        default=1.0,
        help="Recency weighting strength for intraday training (higher = more recent emphasis).",
    )
    parser.add_argument(
        "--speed-constraint-eps",
        type=float,
        default=0.1,
        help="Epsilon for constrained speed log-ratio residual model.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip model training and only refresh prediction outputs using saved models/scalers.",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Chronological holdout fraction for validation (e.g. 0.2).",
    )
    parser.add_argument(
        "--challenge-eval-split",
        type=float,
        default=0.15,
        help="Chronological holdout fraction used for champion-vs-challenger promotion checks.",
    )
    parser.add_argument(
        "--challenge-min-eval-samples",
        type=int,
        default=60,
        help="Minimum number of chronological samples in challenger evaluation holdout.",
    )
    parser.add_argument(
        "--promotion-margin-pct",
        type=float,
        default=1.0,
        help="Required relative MAE improvement (percent) to promote challenger over champion.",
    )
    parser.add_argument(
        "--local-timezone",
        default="Europe/Amsterdam",
        help="Timezone used for current-day plotting (e.g. Europe/Amsterdam).",
    )
    parser.add_argument(
        "--test-now-local-hour",
        type=int,
        default=None,
        help="Testing only: override local current hour (0-23) for current-day plot logic.",
    )
    parser.add_argument(
        "--current-day-interval-minutes",
        type=int,
        default=6,
        help="Sampling interval (minutes) for current-day plot/MAE using raw observations.",
    )
    parser.add_argument(
        "--skip-data-refresh-check",
        action="store_true",
        help="Skip pre-run check that may call source_fetch.py for newer data.",
    )
    parser.add_argument(
        "--max-forecast-age-hours",
        type=float,
        default=8.0,
        help="If latest forecast run in DB is older than this, fetch script is triggered.",
    )
    parser.add_argument(
        "--expected-update-hour-utc",
        type=int,
        default=1,
        help="Expected daily forecast arrival hour in UTC (used for stale detection).",
    )
    parser.add_argument(
        "--out-dir",
        default="next_day_wind_model/artifacts",
        help="Directory where model, metadata, table and plot are saved.",
    )
    parser.add_argument(
        "--web-out-dir",
        default="next_day_wind_model/web_dashboard",
        help="Directory for static web dashboard files (HTML + latest plots/tables).",
    )
    parser.add_argument(
        "--web-refresh-seconds",
        type=int,
        default=900,
        help="Auto-refresh interval for dashboard HTML.",
    )
    parser.add_argument(
        "--git-auto-push-pages",
        action="store_true",
        help="Auto-commit and push web dashboard folder changes to GitHub.",
    )
    parser.add_argument("--git-remote", default="origin", help="Git remote name for auto-push.")
    parser.add_argument("--git-branch", default="main", help="Git branch name for auto-push.")
    return parser.parse_args()


def pick_torch_device() -> torch.device:
    return torch.device("cpu")


def _latest_forecast_run_ts_ms(db_path: Path, site: str, model: str) -> int | None:
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            """
            SELECT MAX(run_ts)
            FROM forecasts
            WHERE site = ?
              AND model = ?
            """,
            (site, model),
        ).fetchone()
    finally:
        conn.close()
    if row is None or row[0] is None:
        return None
    return int(row[0])


def _run_fetch_script(repo_root: Path, out_data_dir: Path) -> None:
    fetch_script = repo_root / "source_fetch.py"
    cmd = ["python3", str(fetch_script), str(out_data_dir)]
    print(f"Refreshing source data via: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(repo_root), check=True)


def ensure_fresh_source_data(
    db_path: Path,
    site: str,
    model: str,
    max_forecast_age_hours: float,
    expected_update_hour_utc: int,
) -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    out_data_dir = repo_root / "data"
    latest_before = _latest_forecast_run_ts_ms(db_path, site, model)
    now_utc = datetime.now(timezone.utc)

    need_refresh = False
    reason = ""
    if latest_before is None:
        need_refresh = True
        reason = "no_forecast_rows_in_db"
    else:
        latest_dt = datetime.fromtimestamp(latest_before / 1000, tz=timezone.utc)
        age_h = (now_utc - latest_dt).total_seconds() / 3600.0

        # Daily expected-slot heuristic: after expected update time, today's slot should be present.
        expected_slot = now_utc.replace(
            hour=int(expected_update_hour_utc),
            minute=0,
            second=0,
            microsecond=0,
        )
        if now_utc < expected_slot:
            expected_slot = expected_slot - timedelta(days=1)

        if age_h > float(max_forecast_age_hours):
            need_refresh = True
            reason = f"latest_forecast_age_{age_h:.2f}h_exceeds_{max_forecast_age_hours:.2f}h"
        elif latest_dt < expected_slot:
            need_refresh = True
            reason = "latest_forecast_run_before_expected_daily_slot"

    refreshed = False
    if need_refresh:
        _run_fetch_script(repo_root, out_data_dir)
        refreshed = True

    latest_after = _latest_forecast_run_ts_ms(db_path, site, model)
    return {
        "need_refresh": need_refresh,
        "refreshed": refreshed,
        "reason": reason,
        "latest_run_before_utc": (
            None
            if latest_before is None
            else datetime.fromtimestamp(latest_before / 1000, tz=timezone.utc).isoformat()
        ),
        "latest_run_after_utc": (
            None
            if latest_after is None
            else datetime.fromtimestamp(latest_after / 1000, tz=timezone.utc).isoformat()
        ),
    }


def train_with_validation(
    model: nn.Module,
    X_all: np.ndarray,
    y_all: np.ndarray,
    batch_size: int,
    epochs: int,
    validation_split: float,
    model_label: str,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    if not (0.0 < validation_split < 0.5):
        raise ValueError("--validation-split must be > 0 and < 0.5.")
    n = len(X_all)
    if n < 10:
        raise ValueError("Not enough samples to use a validation split.")

    split_idx = int(n * (1.0 - validation_split))
    split_idx = max(1, min(split_idx, n - 1))
    X_train, X_val = X_all[:split_idx], X_all[split_idx:]
    y_train, y_val = y_all[:split_idx], y_all[split_idx:]

    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    best_val_loss = float("inf")
    best_train_loss = float("inf")
    best_train_loss_eval = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improve = 0
    early_stopping_patience = 8
    epochs_ran = 0

    for epoch in range(1, epochs + 1):
        epochs_ran = epoch
        model.train()
        total_loss = 0.0
        total_count = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

            batch_size_now = X_batch.size(0)
            total_loss += float(loss.item()) * batch_size_now
            total_count += batch_size_now

        train_loss = total_loss / max(total_count, 1)

        model.eval()
        train_eval_loss_sum = 0.0
        train_eval_count = 0
        with torch.no_grad():
            for X_batch, y_batch in train_eval_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                train_eval_loss_sum += float(loss.item()) * X_batch.size(0)
                train_eval_count += X_batch.size(0)
        train_loss_eval = train_eval_loss_sum / max(train_eval_count, 1)

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
        val_loss = val_loss_sum / max(val_count, 1)
        scheduler.step(val_loss)
        print(
            f"[{model_label}] Epoch {epoch:03d} | train_loss={train_loss:.5f} | "
            f"train_loss_eval={train_loss_eval:.5f} | val_loss={val_loss:.5f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_train_loss = train_loss
            best_train_loss_eval = train_loss_eval
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_state)
    stats = {
        "best_train_loss": float(best_train_loss),
        "best_train_loss_eval": float(best_train_loss_eval),
        "best_val_loss": float(best_val_loss),
        "epochs_ran": int(epochs_ran),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
    }
    return model, stats


def _validation_start_index(n_samples: int, validation_split: float) -> int:
    if not (0.0 < validation_split < 0.5):
        raise ValueError("validation_split must be > 0 and < 0.5.")
    if n_samples < 2:
        raise ValueError("Need at least two samples for a validation split.")
    split_idx = int(n_samples * (1.0 - validation_split))
    split_idx = max(1, min(split_idx, n_samples - 1))
    return int(split_idx)


def _extract_speed_regime_signal(
    pred_speed: np.ndarray,
    forecast_speed: np.ndarray,
    signal: str,
) -> np.ndarray:
    pred_arr = np.asarray(pred_speed, dtype=np.float32)
    forecast_arr = np.asarray(forecast_speed, dtype=np.float32)
    mode = str(signal).strip().lower()
    if mode == "pred_max":
        return pred_arr.max(axis=1).astype(np.float32)
    if mode == "pred_mean":
        return pred_arr.mean(axis=1).astype(np.float32)
    if mode == "forecast_max":
        return forecast_arr.max(axis=1).astype(np.float32)
    if mode == "forecast_mean":
        return forecast_arr.mean(axis=1).astype(np.float32)
    raise ValueError(f"Unsupported speed regime signal: {signal}")


def _build_speed_calibration_context(
    anchor_dir_deg: np.ndarray | pd.Series | list[float] | float,
    target_times_utc: pd.DatetimeIndex | pd.Series | np.ndarray | list[str] | str,
) -> dict:
    anchor_dir_arr = np.asarray(anchor_dir_deg, dtype=np.float32).reshape(-1)
    if np.isscalar(target_times_utc) or isinstance(target_times_utc, str):
        target_times = pd.DatetimeIndex([pd.to_datetime(target_times_utc, utc=True)])
    else:
        target_times = pd.to_datetime(target_times_utc, utc=True)
        if isinstance(target_times, pd.Timestamp):
            target_times = pd.DatetimeIndex([target_times])
    target_month = np.asarray(target_times.month, dtype=np.int16).reshape(-1)
    if anchor_dir_arr.shape[0] != target_month.shape[0]:
        raise ValueError("anchor_dir_deg and target_times_utc must have the same length.")
    return {
        "anchor_dir_deg": anchor_dir_arr.astype(np.float32),
        "target_month": target_month.astype(np.int16),
    }


def _speed_calibration_feature_matrix(
    signal_values: np.ndarray,
    speed_calibration_context: dict | None,
    signal_mean: float | None = None,
    signal_std: float | None = None,
) -> tuple[np.ndarray, dict]:
    signal_arr = np.asarray(signal_values, dtype=np.float32).reshape(-1)
    n = signal_arr.shape[0]
    if speed_calibration_context is None:
        anchor_dir = np.zeros(n, dtype=np.float32)
        target_month = np.ones(n, dtype=np.int16)
    else:
        anchor_dir = np.asarray(speed_calibration_context.get("anchor_dir_deg"), dtype=np.float32).reshape(-1)
        target_month = np.asarray(speed_calibration_context.get("target_month"), dtype=np.int16).reshape(-1)
        if anchor_dir.shape[0] != n or target_month.shape[0] != n:
            raise ValueError("Speed calibration context length must match the number of samples.")

    sig_mean = float(signal_arr.mean()) if signal_mean is None else float(signal_mean)
    sig_std = float(signal_arr.std()) if signal_std is None else float(signal_std)
    if sig_std <= 0.0:
        sig_std = 1.0
    sig_norm = ((signal_arr - sig_mean) / sig_std).astype(np.float32)

    dir_rad = np.deg2rad(anchor_dir.astype(np.float32))
    dir_sin = np.sin(dir_rad).astype(np.float32)
    dir_cos = np.cos(dir_rad).astype(np.float32)
    month_angle = (2.0 * np.pi * (target_month.astype(np.float32) - 1.0)) / 12.0
    month_sin = np.sin(month_angle).astype(np.float32)
    month_cos = np.cos(month_angle).astype(np.float32)

    features = np.column_stack(
        [
            np.ones(n, dtype=np.float32),
            sig_norm,
            dir_sin,
            dir_cos,
            month_sin,
            month_cos,
            sig_norm * dir_sin,
            sig_norm * dir_cos,
            sig_norm * month_sin,
            sig_norm * month_cos,
        ]
    ).astype(np.float32)
    return features, {
        "signal_mean": sig_mean,
        "signal_std": sig_std,
        "feature_names": [
            "bias",
            "signal_norm",
            "dir_sin",
            "dir_cos",
            "month_sin",
            "month_cos",
            "signal_x_dir_sin",
            "signal_x_dir_cos",
            "signal_x_month_sin",
            "signal_x_month_cos",
        ],
    }


def _fit_threshold_speed_calibration(
    pred_arr: np.ndarray,
    forecast_arr: np.ndarray,
    actual_arr: np.ndarray,
    signal_values: np.ndarray,
    signal: str,
) -> dict | None:
    baseline_mae = float(np.mean(np.abs(pred_arr - actual_arr)))
    sig_min = float(np.min(signal_values))
    sig_max = float(np.max(signal_values))
    if not np.isfinite(sig_min) or not np.isfinite(sig_max) or sig_max - sig_min < 0.5:
        return None

    start_grid = np.arange(max(0.0, np.floor(sig_min * 2.0) / 2.0), np.ceil(sig_max * 2.0) / 2.0 + 0.5, 0.5)
    min_scale_grid = np.arange(0.0, 1.01, 0.05)
    best_mae = baseline_mae
    best_params: tuple[float, float, float] | None = None

    for start in start_grid:
        end_grid = np.arange(start + 0.5, np.ceil((sig_max + 2.0) * 2.0) / 2.0 + 0.5, 0.5)
        for end in end_grid:
            t = np.clip((signal_values - start) / (end - start), 0.0, 1.0).astype(np.float32)
            for min_scale in min_scale_grid:
                scale = (1.0 + (float(min_scale) - 1.0) * t).astype(np.float32)
                cand = forecast_arr + scale[:, None] * (pred_arr - forecast_arr)
                cand = np.maximum(cand, 0.0).astype(np.float32)
                mae = float(np.mean(np.abs(cand - actual_arr)))
                if mae + 1e-9 < best_mae:
                    best_mae = mae
                    best_params = (float(start), float(end), float(min_scale))

    if best_params is None:
        return None

    start, end, min_scale = best_params
    improvement_abs = baseline_mae - best_mae
    if improvement_abs <= 1e-6:
        return None
    return {
        "enabled": True,
        "type": "threshold_v1",
        "signal": str(signal),
        "start": float(start),
        "end": float(end),
        "min_scale": float(min_scale),
        "baseline_mae": float(baseline_mae),
        "calibrated_mae": float(best_mae),
        "improvement_abs": float(improvement_abs),
        "improvement_pct": float(improvement_abs / max(baseline_mae, 1e-6)),
        "n_samples": int(pred_arr.shape[0]),
    }


def _fit_contextual_speed_calibration(
    pred_arr: np.ndarray,
    forecast_arr: np.ndarray,
    actual_arr: np.ndarray,
    signal_values: np.ndarray,
    speed_calibration_context: dict | None,
    signal: str,
) -> dict | None:
    if speed_calibration_context is None:
        return None

    delta = (pred_arr - forecast_arr).astype(np.float32)
    target_delta = (actual_arr - forecast_arr).astype(np.float32)
    denom = np.sum(delta * delta, axis=1).astype(np.float32)
    informative = denom > 1e-6
    if int(np.sum(informative)) < 32:
        return None

    target_scale = np.ones(pred_arr.shape[0], dtype=np.float32)
    numer = np.sum(delta * target_delta, axis=1).astype(np.float32)
    target_scale[informative] = np.clip(numer[informative] / denom[informative], 0.0, 1.0)

    X, feature_meta = _speed_calibration_feature_matrix(signal_values, speed_calibration_context)
    mean_denom = float(np.mean(denom[informative])) if informative.any() else 1.0
    sample_w = np.sqrt(np.clip(denom / max(mean_denom, 1e-6), 0.25, 4.0)).astype(np.float32)

    best_coef: np.ndarray | None = None
    best_mae = float(np.mean(np.abs(pred_arr - actual_arr)))
    best_ridge = None
    xt = X.astype(np.float64)
    yt = target_scale.astype(np.float64)
    wt = sample_w.astype(np.float64)
    identity = np.eye(xt.shape[1], dtype=np.float64)
    identity[0, 0] = 0.0

    for ridge in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0):
        xw = xt * wt[:, None]
        yw = yt * wt
        try:
            coef = np.linalg.solve(xw.T @ xw + float(ridge) * identity, xw.T @ yw).astype(np.float32)
        except np.linalg.LinAlgError:
            continue
        scale = np.clip(xt @ coef.astype(np.float64), 0.0, 1.0).astype(np.float32)
        cand = forecast_arr + scale[:, None] * delta
        cand = np.maximum(cand, 0.0).astype(np.float32)
        mae = float(np.mean(np.abs(cand - actual_arr)))
        if mae + 1e-9 < best_mae:
            best_mae = mae
            best_coef = coef
            best_ridge = float(ridge)

    if best_coef is None:
        return None

    baseline_mae = float(np.mean(np.abs(pred_arr - actual_arr)))
    improvement_abs = baseline_mae - best_mae
    if improvement_abs <= 1e-6:
        return None
    return {
        "enabled": True,
        "type": "contextual_linear_v2",
        "signal": str(signal),
        "signal_mean": float(feature_meta["signal_mean"]),
        "signal_std": float(feature_meta["signal_std"]),
        "feature_names": feature_meta["feature_names"],
        "coefficients": [float(v) for v in best_coef.tolist()],
        "ridge": best_ridge,
        "baseline_mae": float(baseline_mae),
        "calibrated_mae": float(best_mae),
        "improvement_abs": float(improvement_abs),
        "improvement_pct": float(improvement_abs / max(baseline_mae, 1e-6)),
        "n_samples": int(pred_arr.shape[0]),
    }


def fit_speed_regime_calibration(
    pred_speed: np.ndarray,
    forecast_speed: np.ndarray,
    actual_speed: np.ndarray,
    speed_calibration_context: dict | None = None,
    signal: str = "pred_max",
) -> dict | None:
    pred_arr = np.asarray(pred_speed, dtype=np.float32)
    forecast_arr = np.asarray(forecast_speed, dtype=np.float32)
    actual_arr = np.asarray(actual_speed, dtype=np.float32)
    if pred_arr.ndim != 2 or forecast_arr.shape != pred_arr.shape or actual_arr.shape != pred_arr.shape:
        raise ValueError("Speed calibration expects (samples, horizon) arrays of identical shape.")
    if pred_arr.shape[0] < 32:
        return None

    signal_values = _extract_speed_regime_signal(pred_arr, forecast_arr, signal)
    threshold_cal = _fit_threshold_speed_calibration(pred_arr, forecast_arr, actual_arr, signal_values, signal)
    contextual_cal = _fit_contextual_speed_calibration(
        pred_arr,
        forecast_arr,
        actual_arr,
        signal_values,
        speed_calibration_context,
        signal,
    )
    candidates = [c for c in [threshold_cal, contextual_cal] if c is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda c: float(c["calibrated_mae"]))


def apply_speed_regime_calibration(
    pred_speed: np.ndarray,
    forecast_speed: np.ndarray,
    speed_calibration: dict | None,
    speed_calibration_context: dict | None = None,
) -> np.ndarray:
    if not speed_calibration or not bool(speed_calibration.get("enabled", False)):
        return np.asarray(pred_speed, dtype=np.float32)

    pred_arr = np.asarray(pred_speed, dtype=np.float32)
    forecast_arr = np.asarray(forecast_speed, dtype=np.float32)
    if pred_arr.shape != forecast_arr.shape:
        raise ValueError("pred_speed and forecast_speed must have the same shape for calibration.")

    input_was_vector = pred_arr.ndim == 1
    if input_was_vector:
        pred_arr = pred_arr[np.newaxis, :]
        forecast_arr = forecast_arr[np.newaxis, :]
    elif pred_arr.ndim != 2:
        raise ValueError("Speed calibration expects a 1D or 2D prediction array.")

    cal_type = str(speed_calibration.get("type", "threshold_v1")).strip().lower()
    signal_values = _extract_speed_regime_signal(
        pred_arr,
        forecast_arr,
        str(speed_calibration.get("signal", "pred_max")),
    )
    if cal_type == "contextual_linear_v2":
        features, _ = _speed_calibration_feature_matrix(
            signal_values,
            speed_calibration_context,
            signal_mean=float(speed_calibration.get("signal_mean", 0.0)),
            signal_std=float(speed_calibration.get("signal_std", 1.0)),
        )
        coef = np.asarray(speed_calibration.get("coefficients", []), dtype=np.float32)
        if coef.shape[0] != features.shape[1]:
            raise ValueError("Speed contextual calibration coefficients do not match feature dimensions.")
        scale = np.clip(features @ coef, 0.0, 1.0).astype(np.float32)
    else:
        start = float(speed_calibration["start"])
        end = float(speed_calibration["end"])
        min_scale = float(speed_calibration["min_scale"])
        t = np.clip((signal_values - start) / max(end - start, 1e-6), 0.0, 1.0).astype(np.float32)
        scale = (1.0 + (min_scale - 1.0) * t).astype(np.float32)
    calibrated = forecast_arr + scale[:, None] * (pred_arr - forecast_arr)
    calibrated = np.maximum(calibrated, 0.0).astype(np.float32)
    if input_was_vector:
        return calibrated[0].astype(np.float32)
    return calibrated.astype(np.float32)


def predict_speed(
    model: nn.Module,
    X_input: np.ndarray,
    forecast_speed: np.ndarray,
    y_mean: float,
    y_std: float,
    target_mode: str,
    constraint_eps: float | None,
    speed_calibration: dict | None,
    speed_calibration_context: dict | None,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(X_input).float().to(device)).cpu().numpy()[0]
    pred = pred_scaled * y_std + y_mean
    mode = str(target_mode).strip().lower()
    if mode == "residual":
        pred = pred + forecast_speed
    elif mode == "constrained_logratio":
        eps = float(0.1 if constraint_eps is None else constraint_eps)
        pred = np.exp(np.log(forecast_speed + eps) + pred)
    elif mode != "absolute":
        raise ValueError(f"Unsupported speed target mode: {target_mode}")
    pred = apply_speed_regime_calibration(pred, forecast_speed, speed_calibration, speed_calibration_context)
    return pred.astype(np.float32)


def predict_direction_residual(
    model: nn.Module,
    X_input: np.ndarray,
    forecast_dir: np.ndarray,
    y_mean: float,
    y_std: float,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(X_input).float().to(device)).cpu().numpy()[0]
    pred_residual = pred_scaled * y_std + y_mean
    pred_dir = _angle_add_deg(forecast_dir.astype(np.float32), pred_residual.astype(np.float32))
    return pred_dir.astype(np.float32)


def _inverse_standardizer(X_scaled: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return X_scaled * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)


def _predict_speed_batch(
    model: nn.Module,
    X_input: np.ndarray,
    forecast_speed: np.ndarray,
    y_mean: float,
    y_std: float,
    target_mode: str,
    constraint_eps: float | None,
    speed_calibration: dict | None,
    speed_calibration_context: dict | None,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(X_input).float().to(device)).cpu().numpy()
    pred = pred_scaled * y_std + y_mean
    mode = str(target_mode).strip().lower()
    if mode == "residual":
        pred = pred + forecast_speed
    elif mode == "constrained_logratio":
        eps = float(0.1 if constraint_eps is None else constraint_eps)
        pred = np.exp(np.log(forecast_speed + eps) + pred)
    elif mode != "absolute":
        raise ValueError(f"Unsupported speed target mode: {target_mode}")
    pred = apply_speed_regime_calibration(pred, forecast_speed, speed_calibration, speed_calibration_context)
    return pred.astype(np.float32)


def _predict_direction_batch(
    model: nn.Module,
    X_input: np.ndarray,
    forecast_dir: np.ndarray,
    y_mean: float,
    y_std: float,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(X_input).float().to(device)).cpu().numpy()
    pred_residual = pred_scaled * y_std + y_mean
    pred_dir = _angle_add_deg(forecast_dir.astype(np.float32), pred_residual.astype(np.float32))
    return pred_dir.astype(np.float32)


def _angular_mae_deg(pred: np.ndarray, actual: np.ndarray) -> float:
    diff = ((pred - actual + 180.0) % 360.0) - 180.0
    return float(np.mean(np.abs(diff)))


def _eval_start_index(n_samples: int, eval_fraction: float, min_eval_samples: int) -> int:
    eval_n = max(int(round(n_samples * float(eval_fraction))), int(min_eval_samples))
    eval_n = min(eval_n, n_samples - 2)
    if eval_n < 1:
        raise ValueError("Not enough samples to build challenger evaluation holdout.")
    return int(n_samples - eval_n)


def build_prediction_table(
    inference_input: dict,
    speed_pred: np.ndarray,
    dir_pred: np.ndarray,
) -> pd.DataFrame:
    target_times = pd.to_datetime(inference_input["target_times"], utc=True)
    forecast_speed = inference_input["forecast_next24"].astype(np.float32)
    forecast_min = inference_input["forecast_min_next24"].astype(np.float32)
    forecast_max = inference_input["forecast_max_next24"].astype(np.float32)
    forecast_min = np.where(np.isnan(forecast_min), forecast_speed, forecast_min).astype(np.float32)
    forecast_max = np.where(np.isnan(forecast_max), forecast_speed, forecast_max).astype(np.float32)
    lo = np.minimum(forecast_min, forecast_max)
    hi = np.maximum(forecast_min, forecast_max)
    forecast_dir = inference_input["forecast_dir_next24"].astype(np.float32)

    table = pd.DataFrame(
        {
            "target_time_utc": target_times,
            "forecast_wind_speed": forecast_speed,
            "forecast_wind_min": lo,
            "forecast_wind_max": hi,
            "lstm_pred_wind_speed": speed_pred.astype(np.float32),
            "forecast_wind_dir_deg": forecast_dir,
            "lstm_pred_wind_dir_deg": dir_pred.astype(np.float32),
        }
    ).assign(
        hour_utc=lambda d: d["target_time_utc"].dt.strftime("%H"),
        delta_speed_lstm_minus_forecast=lambda d: d["lstm_pred_wind_speed"] - d["forecast_wind_speed"],
        delta_dir_lstm_minus_forecast=lambda d: ((d["lstm_pred_wind_dir_deg"] - d["forecast_wind_dir_deg"] + 180) % 360)
        - 180,
    )
    return table


def _apply_speed_background(ax: plt.Axes, y_top: float, x_left: float, x_right: float) -> None:
    y_max = max(20.0, float(y_top))
    grad = np.linspace(0.0, 1.0, 256).reshape(256, 1)
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "wind_bg_blue_green",
        [
            (0.00, "#7fb6ff"),
            (0.25, "#57a6ff"),
            (0.45, "#4ecdb2"),
            (0.62, "#30c27a"),
            (1.00, "#22b36a"),
        ],
    )
    ax.imshow(
        grad,
        extent=[x_left, x_right, 0.0, y_max],
        origin="lower",
        aspect="auto",
        cmap=cmap,
        alpha=0.5,
        zorder=0,
        interpolation="bicubic",
    )


def _load_observations_raw(conn: sqlite3.Connection, site: str) -> pd.DataFrame:
    query = """
    SELECT ts, wind_speed, wind_dir, payload
    FROM observations
    WHERE site = ?
      AND ts IS NOT NULL
    ORDER BY ts
    """
    rows = conn.execute(query, (site,)).fetchall()
    if not rows:
        return pd.DataFrame(columns=["actual_avg", "actual_dir"])

    records: list[dict] = []
    for ts, wind_speed, wind_dir, payload_raw in rows:
        payload = {}
        if payload_raw:
            try:
                payload = json.loads(payload_raw)
            except json.JSONDecodeError:
                payload = {}
        avg = payload.get("AverageWind")
        if avg is None:
            avg = payload.get("WindSpeedAvg")
        if avg is None:
            avg = wind_speed
        direc = payload.get("WindDirection")
        if direc is None:
            direc = wind_dir
        try:
            avg_f = float(avg) if avg is not None else np.nan
        except (TypeError, ValueError):
            avg_f = np.nan
        try:
            dir_f = float(direc) if direc is not None else np.nan
        except (TypeError, ValueError):
            dir_f = np.nan
        records.append({"obs_ts": int(ts), "actual_avg": avg_f, "actual_dir": dir_f})

    out = pd.DataFrame.from_records(records)
    out["obs_dt"] = pd.to_datetime(out["obs_ts"], unit="ms", utc=True)
    out = out.set_index("obs_dt").sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out[["actual_avg", "actual_dir"]]


def _interp_hourly_to_dense(hourly_values: np.ndarray, hourly_index: pd.DatetimeIndex, dense_index: pd.DatetimeIndex) -> np.ndarray:
    s = pd.Series(hourly_values, index=hourly_index, dtype=float)
    dense = s.reindex(s.index.union(dense_index)).sort_index().interpolate(method="time").reindex(dense_index)
    return dense.to_numpy(dtype=np.float32)


def _smooth_series(values: np.ndarray, window: int = 3) -> np.ndarray:
    """Gentle centered rolling mean for plotting; preserves NaN gaps."""
    arr = np.asarray(values, dtype=float)
    if window <= 1:
        return arr.copy()
    s = pd.Series(arr)
    out = s.rolling(window=window, center=True, min_periods=1).mean().to_numpy(dtype=float)
    out[np.isnan(arr)] = np.nan
    return out


def _parse_iso_utc(ts: str | None) -> datetime | None:
    if not ts:
        return None
    value = ts.strip()
    if not value:
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_iso_series_utc(values: pd.Series) -> pd.Series:
    parsed = [
        _parse_iso_utc(None if pd.isna(v) else str(v))
        for v in values
    ]
    return pd.to_datetime(parsed, utc=True)


def _format_plot_meta_text(
    prediction_generated_at_utc: str,
    prediction_updated_at_utc: str | None,
    model_trained_at_utc: str | None,
    local_tz: str,
) -> str:
    pred_dt = _parse_iso_utc(prediction_generated_at_utc)
    pred_upd_dt = _parse_iso_utc(prediction_updated_at_utc)
    train_dt = _parse_iso_utc(model_trained_at_utc)
    tz = ZoneInfo(local_tz)
    pred_txt = pred_dt.astimezone(tz).strftime("%H:%M") if pred_dt is not None else "unknown"
    pred_upd_txt = pred_upd_dt.astimezone(tz).strftime("%H:%M") if pred_upd_dt is not None else "unknown"
    train_txt = (
        f"{train_dt.astimezone(tz).day} {train_dt.astimezone(tz).strftime('%B %H:%M')}"
        if train_dt is not None
        else "unknown"
    )
    return f"Last plot update: {pred_txt}\nLast prediction update: {pred_upd_txt}\nLast model training: {train_txt}"


def _format_model_id(model_trained_at_utc: str | None, local_tz: str) -> str:
    train_dt = _parse_iso_utc(model_trained_at_utc)
    if train_dt is None:
        return "unknown"
    return train_dt.astimezone(ZoneInfo(local_tz)).strftime("%Y%m%d-%H%M")


def _format_model_id_text(model_trained_at_utc: str | None, local_tz: str) -> str:
    return f"Model ID: {_format_model_id(model_trained_at_utc, local_tz)}"


def save_prediction_plot(
    table: pd.DataFrame,
    plot_path: Path,
    local_tz: str,
    prediction_generated_at_utc: str,
    prediction_updated_at_utc: str | None,
    model_trained_at_utc: str | None,
    mobile: bool = False,
) -> None:
    table = table.copy()
    table = table[(table["target_time_utc"].dt.hour >= 8) & (table["target_time_utc"].dt.hour <= 22)].reset_index(
        drop=True
    )
    if table.empty:
        raise ValueError("No rows available in 08:00-22:00 range for plotting.")

    x = np.arange(len(table))
    table["hour_label"] = table["target_time_utc"].dt.strftime("%H")
    first_dt = table["target_time_utc"].iloc[0]
    day_label = f"{first_dt.day} {first_dt.strftime('%B %Y')}"

    y_min = float(min(table["forecast_wind_min"].min(), table["forecast_wind_speed"].min(), table["lstm_pred_wind_speed"].min()))
    y_max = float(max(table["forecast_wind_max"].max(), table["forecast_wind_speed"].max(), table["lstm_pred_wind_speed"].max()))
    pad = max((y_max - y_min) * 0.08, 0.8)

    fig_size = (8.4, 8.8) if mobile else (14, 7.2)
    title_fs = 14 if mobile else None
    label_fs = 12 if mobile else None
    tick_fs = 11 if mobile else None
    legend_fs = 10 if mobile else None
    meta_fs = 10 if mobile else 9
    meta_y = 1.14 if mobile else 1.13
    fig, ax = plt.subplots(figsize=fig_size)
    _apply_speed_background(ax, y_max + pad, x_left=0.0, x_right=len(table) - 1.0)
    marker_size = 3.0
    fc_low = table["forecast_wind_min"].to_numpy(dtype=float)
    fc_high = table["forecast_wind_max"].to_numpy(dtype=float)
    fc_avg = table["forecast_wind_speed"].to_numpy(dtype=float)
    lstm_avg = table["lstm_pred_wind_speed"].to_numpy(dtype=float)
    ax.fill_between(
        x,
        fc_low,
        fc_high,
        color="gray",
        alpha=0.25,
        linewidth=0.8,
        edgecolor="gray",
        label="_nolegend_",
        zorder=1,
    )
    ax.plot(x, fc_high, color="#666666", linewidth=1.2, linestyle="--", label="Harmonie model - max speed")
    ax.plot(x, fc_avg, color="gray", linewidth=1.5, label="Harmonie model - avg speed")
    ax.plot(
        x,
        lstm_avg,
        color=LSTM_HIGHLIGHT_COLOR,
        linewidth=2.4,
        label="Super local wind prediction - avg speed",
    )
    ax.set_title(f"Next-day wind speed: {day_label}.", fontsize=title_fs)
    ax.set_xlabel("Time", fontsize=label_fs)
    ax.set_ylabel("Wind speed (kts)", fontsize=label_fs)
    ax.grid(axis="y", alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    desired_order = [
        "Super local wind prediction - avg speed",
        "Harmonie model - avg speed",
        "Harmonie model - max speed",
    ]
    order_map = {label: handle for handle, label in zip(handles, labels)}
    ordered_handles = [order_map[label] for label in desired_order if label in order_map]
    ordered_labels = [label for label in desired_order if label in order_map]
    ax.legend(
        ordered_handles,
        ordered_labels,
        loc="upper left",
        bbox_to_anchor=(0.015, 0.99),
        borderaxespad=0.0,
        fontsize=legend_fs,
    )
    ax.set_xticks(x, table["hour_label"], rotation=0)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.set_xlim(0.0, len(table) - 1.0)
    ax.set_ylim(0.0, y_max + pad)
    ax.text(
        0.015,
        meta_y,
        _format_plot_meta_text(prediction_generated_at_utc, prediction_updated_at_utc, model_trained_at_utc, local_tz),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=meta_fs,
        color="black",
        clip_on=False,
    )
    ax.text(
        0.985,
        meta_y,
        _format_model_id_text(model_trained_at_utc, local_tz),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=meta_fs,
        color="black",
        clip_on=False,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )

    # Draw wind direction arrows under x-axis.
    # Mapping: up-arrow means South wind (from South), per user preference.
    # Using x-axis transform keeps arrows below axis regardless of y-scale.
    y_base_axes = -0.115 if mobile else -0.14
    arrow_len_axes = 0.058 if mobile else 0.065
    for i, (fdir, ldir) in enumerate(zip(table["forecast_wind_dir_deg"], table["lstm_pred_wind_dir_deg"])):
        for direction_deg, color in [(fdir, "gray"), (ldir, LSTM_HIGHLIGHT_COLOR)]:
            theta = np.deg2rad((float(direction_deg) + 180.0) % 360.0)
            dx = 0.22 * np.sin(theta)
            dy = arrow_len_axes * np.cos(theta)
            ax.annotate(
                "",
                xy=(i + dx, y_base_axes + dy),
                xytext=(i, y_base_axes),
                xycoords=ax.get_xaxis_transform(),
                textcoords=ax.get_xaxis_transform(),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": color,
                    "lw": 1.6,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
                clip_on=False,
            )

    layout_top = 0.93 if mobile else 0.965
    layout_bottom = 0.055 if mobile else 0.04
    fig.tight_layout(rect=[0, layout_bottom, 1, layout_top])
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def build_current_day_table(
    db_path: Path,
    cfg: DatasetConfig,
    intraday_bundle: IntradayBundle,
    speed_model: nn.Module,
    direction_model: nn.Module,
    speed_scalers: dict,
    direction_scalers: dict,
    speed_target_mode: str,
    speed_constraint_eps: float | None,
    local_tz: str,
    latest_prior_prediction_table: pd.DataFrame | None,
    test_now_local_hour: int | None,
    current_day_interval_minutes: int,
    device: torch.device,
) -> pd.DataFrame:
    def _predict_residuals_for_targets(
        target_local_index: pd.DatetimeIndex,
    ) -> tuple[np.ndarray, np.ndarray]:
        target_n = len(target_local_index)
        if target_n == 0:
            return (
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
            )

        anchor_local_for_targets = target_local_index[0] - timedelta(hours=1)
        history_local_for_targets = pd.date_range(
            end=anchor_local_for_targets,
            periods=cfg.window_hours,
            freq="1h",
            tz=tz,
        )
        history_utc_for_targets = history_local_for_targets.tz_convert("UTC")
        target_utc_index = target_local_index.tz_convert("UTC")

        feature_cols = ["forecast_avg", "forecast_max", "forecast_dir", "month_sin", "month_cos"]
        history_frame = forecast_frame_utc.reindex(history_utc_for_targets)
        future_frame = forecast_frame_utc.reindex(target_utc_index)
        if history_frame[feature_cols].isna().any().any():
            raise ValueError("Missing forecast rows in history window for current-day inference.")
        if future_frame[["forecast_avg", "forecast_dir"]].isna().any().any():
            raise ValueError("Missing forecast rows in current-day target window.")

        x_window = history_frame[feature_cols].to_numpy(dtype=np.float32)[np.newaxis, :, :]
        x_speed = _apply_standardizer(x_window, speed_scalers["x_mean"], speed_scalers["x_std"]).astype(np.float32)
        x_dir = _apply_standardizer(x_window, direction_scalers["x_mean"], direction_scalers["x_std"]).astype(np.float32)

        speed_model.eval()
        direction_model.eval()
        with torch.no_grad():
            speed_res_scaled = speed_model(torch.from_numpy(x_speed).float().to(device)).cpu().numpy()[0]
            dir_res_scaled = direction_model(torch.from_numpy(x_dir).float().to(device)).cpu().numpy()[0]

        speed_out = speed_res_scaled * float(speed_scalers["y_std"][0]) + float(speed_scalers["y_mean"][0])
        dir_res = dir_res_scaled * float(direction_scalers["y_std"][0]) + float(direction_scalers["y_mean"][0])
        speed_out = speed_out[:target_n]
        dir_res = dir_res[:target_n]

        fc_speed = future_frame["forecast_avg"].to_numpy(dtype=np.float32)
        fc_dir = future_frame["forecast_dir"].to_numpy(dtype=np.float32)
        mode = str(speed_target_mode).strip().lower()
        if mode == "residual":
            lstm_speed = fc_speed + speed_out
        elif mode == "constrained_logratio":
            eps = float(0.1 if speed_constraint_eps is None else speed_constraint_eps)
            lstm_speed = np.exp(np.log(fc_speed + eps) + speed_out)
        elif mode == "absolute":
            lstm_speed = speed_out
        else:
            raise ValueError(f"Unsupported speed target mode: {speed_target_mode}")
        lstm_dir = _angle_add_deg(fc_dir, dir_res.astype(np.float32))
        return lstm_speed.astype(np.float32), lstm_dir.astype(np.float32)

    tz = ZoneInfo(local_tz)
    now_local = datetime.now(tz=tz)
    if test_now_local_hour is not None:
        hour = int(test_now_local_hour)
        if hour < 0 or hour > 23:
            raise ValueError("--test-now-local-hour must be between 0 and 23.")
        now_local = now_local.replace(hour=hour, minute=0, second=0, microsecond=0)
    now_hour_local = now_local.replace(minute=0, second=0, microsecond=0)
    day_start_local = now_hour_local.replace(hour=0)
    day_end_local = day_start_local + timedelta(hours=23)
    interval_min = int(current_day_interval_minutes)
    if interval_min <= 0 or 60 % interval_min != 0:
        raise ValueError("--current-day-interval-minutes must be a positive divisor of 60.")

    # Build forecast frame (UTC indexed) and raw observations, then convert to local timezone.
    forecast_frame_utc = _build_forecast_feature_frame(db_path, cfg)
    conn = sqlite3.connect(str(db_path))
    try:
        obs_raw_utc = _load_observations_raw(conn, cfg.site)
    finally:
        conn.close()

    forecast_frame_local = forecast_frame_utc.tz_convert(tz)
    obs_raw_local = obs_raw_utc.tz_convert(tz)
    obs_hourly_local = obs_raw_local.resample("1h").mean(numeric_only=True)
    previous_issue_forecast = None
    if latest_prior_prediction_table is not None and not latest_prior_prediction_table.empty:
        prior = latest_prior_prediction_table.copy()
        if "time_local" in prior.columns and "lstm_pred_wind_speed" in prior.columns:
            prior["time_local"] = pd.to_datetime(prior["time_local"], utc=True, errors="coerce").dt.tz_convert(tz)
            prior = prior.dropna(subset=["time_local"]).copy()
            if not prior.empty:
                prior = prior[prior["time_local"].dt.minute.eq(0)].copy()
                if not prior.empty:
                    previous_issue_forecast = (
                        pd.Series(
                            pd.to_numeric(prior["lstm_pred_wind_speed"], errors="coerce").to_numpy(dtype=float),
                            index=prior["time_local"],
                        )
                        .sort_index()
                        .groupby(level=0)
                        .last()
                    )

    # Remaining hours today (prediction target): next hour .. 23:00 local.
    remaining_local = pd.date_range(
        start=now_hour_local + timedelta(hours=1),
        end=day_end_local,
        freq="1h",
        tz=tz,
    )
    remaining_n = len(remaining_local)

    # Full-day context direction prediction (00..23) based on day-start anchor.
    full_hours = pd.date_range(start=day_start_local, end=day_end_local, freq="1h", tz=tz)
    _, lstm_dir_full = _predict_residuals_for_targets(full_hours)
    # Remaining-day best prediction (next hour..23) based on current anchor.
    if remaining_n > 0:
        _, lstm_dir_rem = _predict_residuals_for_targets(remaining_local)
    else:
        lstm_dir_rem = np.array([], dtype=np.float32)

    # Dedicated intraday speed model: strongly conditions on recent actuals/residuals.
    intraday_speed_full, intraday_speed_rem = predict_intraday_day_speed(
        bundle=intraday_bundle,
        forecast_frame_local=forecast_frame_local,
        actual_hourly_local=obs_hourly_local["actual_avg"] if "actual_avg" in obs_hourly_local.columns else pd.Series(dtype=float),
        previous_issue_forecast=previous_issue_forecast,
        day_start_local=day_start_local,
        day_end_local=day_end_local,
        now_hour_local=now_hour_local,
        device=device,
    )

    fc_today = forecast_frame_local.reindex(full_hours)
    fc_min = fc_today["forecast_min"].to_numpy(dtype=np.float32)
    fc_avg = fc_today["forecast_avg"].to_numpy(dtype=np.float32)
    fc_max = fc_today["forecast_max"].to_numpy(dtype=np.float32)
    fc_min = np.where(np.isnan(fc_min), fc_avg, fc_min).astype(np.float32)
    fc_max = np.where(np.isnan(fc_max), fc_avg, fc_max).astype(np.float32)
    fc_low = np.minimum(fc_min, fc_max)
    fc_high = np.maximum(fc_min, fc_max)
    dense_end = day_end_local + timedelta(minutes=(60 - interval_min))
    dense_times = pd.date_range(
        start=day_start_local,
        end=dense_end,
        freq=f"{interval_min}min",
        tz=tz,
    )

    # Forecast/LSTM hourly values -> dense timeline for plotting and MAE at higher cadence.
    fc_speed_dense = _interp_hourly_to_dense(fc_avg, full_hours, dense_times)
    fc_min_dense = _interp_hourly_to_dense(fc_low, full_hours, dense_times)
    fc_max_dense = _interp_hourly_to_dense(fc_high, full_hours, dense_times)
    fc_dir_dense = _interp_hourly_to_dense(fc_today["forecast_dir"].to_numpy(dtype=np.float32), full_hours, dense_times)
    lstm_full_dense = _interp_hourly_to_dense(intraday_speed_full.astype(np.float32), full_hours, dense_times)
    lstm_dir_full_dense = _interp_hourly_to_dense(lstm_dir_full.astype(np.float32), full_hours, dense_times)

    rem_hourly_speed = pd.Series(np.nan, index=full_hours, dtype=float)
    rem_hourly_dir = pd.Series(np.nan, index=full_hours, dtype=float)
    if remaining_n > 0:
        rem_vals = pd.Series(intraday_speed_rem, index=full_hours, dtype=float).reindex(remaining_local).to_numpy(dtype=np.float32)
        rem_hourly_speed.loc[remaining_local] = rem_vals
        rem_hourly_dir.loc[remaining_local] = lstm_dir_rem.astype(np.float32)
        # Add continuity point at the boundary for a continuous future line.
        prev_hour = remaining_local[0] - timedelta(hours=1)
        if prev_hour in rem_hourly_speed.index:
            boundary_speed = float(intraday_speed_full[np.where(full_hours == prev_hour)[0][0]])
            if previous_issue_forecast is not None:
                prior_boundary_speed = previous_issue_forecast.get(prev_hour, np.nan)
                if not pd.isna(prior_boundary_speed):
                    boundary_speed = float(prior_boundary_speed)
            rem_hourly_speed.loc[prev_hour] = boundary_speed
            rem_hourly_dir.loc[prev_hour] = float(
                lstm_dir_full[np.where(full_hours == prev_hour)[0][0]]
            )
    rem_dense_speed = (
        rem_hourly_speed.reindex(rem_hourly_speed.index.union(dense_times))
        .sort_index()
        .interpolate(method="time")
        .reindex(dense_times)
        .to_numpy(dtype=np.float32)
    )
    rem_dense_dir = (
        rem_hourly_dir.reindex(rem_hourly_dir.index.union(dense_times))
        .sort_index()
        .interpolate(method="time")
        .reindex(dense_times)
        .to_numpy(dtype=np.float32)
    )

    # Actual measurements at higher cadence (latest known values up to now).
    actual_day_raw = obs_raw_local[(obs_raw_local.index >= day_start_local) & (obs_raw_local.index <= now_local)]
    actual_speed_dense = (
        actual_day_raw["actual_avg"]
        .reindex(actual_day_raw.index.union(dense_times))
        .sort_index()
        .ffill()
        .reindex(dense_times)
        .to_numpy(dtype=np.float32)
    )
    actual_dir_dense = (
        actual_day_raw["actual_dir"]
        .reindex(actual_day_raw.index.union(dense_times))
        .sort_index()
        .ffill()
        .reindex(dense_times)
        .to_numpy(dtype=np.float32)
    )
    actual_speed_dense = np.where(dense_times <= now_local, actual_speed_dense, np.nan).astype(np.float32)
    actual_dir_dense = np.where(dense_times <= now_local, actual_dir_dense, np.nan).astype(np.float32)

    table = pd.DataFrame(
        {
            "time_local": dense_times,
            "forecast_wind_speed": fc_speed_dense,
            "forecast_wind_min": fc_min_dense,
            "forecast_wind_max": fc_max_dense,
            "forecast_wind_dir_deg": fc_dir_dense,
            "actual_wind_speed": actual_speed_dense,
            "actual_wind_dir_deg": actual_dir_dense,
            "lstm_pred_wind_speed_full": lstm_full_dense,
            "lstm_pred_wind_dir_deg_full": lstm_dir_full_dense,
            "lstm_pred_wind_speed": rem_dense_speed,
            "lstm_pred_wind_dir_deg": rem_dense_dir,
        }
    )
    future_start = now_hour_local + timedelta(hours=1)
    table["is_future"] = table["time_local"] >= future_start
    table["hour_local"] = table["time_local"].dt.strftime("%H")
    table["minute_local"] = table["time_local"].dt.minute
    return table


def save_current_day_plot(
    table: pd.DataFrame,
    plot_path: Path,
    local_tz: str,
    prediction_generated_at_utc: str,
    prediction_updated_at_utc: str | None,
    model_trained_at_utc: str | None,
    prior_prediction_tables: list[pd.DataFrame] | None = None,
    fixed_origin_mae_metrics: list[dict] | None = None,
    mobile: bool = False,
) -> None:
    table = table.copy()
    table = table[
        (table["time_local"].dt.hour >= 8)
        & ((table["time_local"].dt.hour < 22) | ((table["time_local"].dt.hour == 22) & (table["time_local"].dt.minute == 0)))
    ].reset_index(drop=True)
    if table.empty:
        raise ValueError("No rows available in 08:00-22:00 range for current-day plotting.")

    x = np.arange(len(table))
    x_lookup = {ts: idx for idx, ts in enumerate(table["time_local"])}
    first_dt = table["time_local"].iloc[0]
    day_label = f"{first_dt.day} {first_dt.strftime('%B %Y')}"

    speed_series = pd.concat(
        [
            table["forecast_wind_min"].dropna(),
            table["forecast_wind_max"].dropna(),
            table["forecast_wind_speed"].dropna(),
            table["lstm_pred_wind_speed_full"].dropna(),
            table["lstm_pred_wind_speed"].dropna(),
            table["actual_wind_speed"].dropna(),
        ]
    )
    y_min = float(speed_series.min()) if not speed_series.empty else 0.0
    y_max = float(speed_series.max()) if not speed_series.empty else 10.0
    pad = max((y_max - y_min) * 0.08, 0.8)
    y_lower = 0.0
    y_upper = y_max + pad

    fig_size = (8.4, 8.8) if mobile else (14, 7.2)
    title_fs = 14 if mobile else None
    label_fs = 12 if mobile else None
    tick_fs = 11 if mobile else None
    legend_fs = 10 if mobile else None
    meta_fs = 10 if mobile else 9
    mae_fs = 11 if mobile else 10
    meta_y = 1.16 if mobile else 1.13
    fig, ax = plt.subplots(figsize=fig_size)
    _apply_speed_background(ax, y_upper, x_left=0.0, x_right=len(table) - 1.0)
    fc_low = table["forecast_wind_min"].to_numpy(dtype=float)
    fc_high = table["forecast_wind_max"].to_numpy(dtype=float)
    fc_avg = table["forecast_wind_speed"].to_numpy(dtype=float)
    actual_avg = table["actual_wind_speed"].to_numpy(dtype=float)
    ax.fill_between(
        x,
        fc_low,
        fc_high,
        color="gray",
        alpha=0.25,
        linewidth=0.8,
        edgecolor="gray",
        label="_nolegend_",
        zorder=1,
    )
    ax.plot(x, fc_high, color="#666666", linewidth=1.2, linestyle="--", label="Harmonie model - max speed")
    ax.plot(
        x,
        fc_avg,
        color="gray",
        linewidth=1.5,
        label="Harmonie model - avg speed",
    )
    ax.plot(
        x,
        actual_avg,
        marker="o",
        markersize=2.2,
        color="magenta",
        linewidth=2.2,
        label="Wind speed - measured",
        zorder=5,
    )

    branch_lookback_by_anchor: dict[pd.Timestamp, int] = {}
    for metric in fixed_origin_mae_metrics or []:
        lookback_hours = int(metric.get("lookback_hours", 0))
        if lookback_hours not in {3, 6}:
            continue
        if not bool(metric.get("available", False)):
            continue
        snapshot_issued_at_local = metric.get("snapshot_issued_at_local")
        if not snapshot_issued_at_local:
            continue
        metric_issue_anchor = pd.to_datetime(snapshot_issued_at_local, errors="coerce")
        if pd.isna(metric_issue_anchor):
            continue
        if metric_issue_anchor.tzinfo is None:
            metric_issue_anchor = metric_issue_anchor.tz_localize(ZoneInfo(local_tz))
        else:
            metric_issue_anchor = metric_issue_anchor.tz_convert(ZoneInfo(local_tz))
        branch_lookback_by_anchor[metric_issue_anchor.floor("h")] = lookback_hours

    overlay_tables = prior_prediction_tables or []
    historical_branches: list[tuple[pd.Timestamp, pd.DataFrame]] = []
    if overlay_tables:
        for overlay_table in overlay_tables:
            overlay = overlay_table.copy()
            if "time_local" not in overlay.columns or "lstm_pred_wind_speed" not in overlay.columns:
                continue
            if "issued_at_local" not in overlay.columns:
                continue
            issued_series = overlay["issued_at_local"].dropna()
            if issued_series.empty:
                continue
            issued_at = pd.to_datetime(issued_series.iloc[0], utc=True, errors="coerce")
            if pd.isna(issued_at):
                continue
            issued_at = issued_at.tz_convert(ZoneInfo(local_tz))
            issue_anchor = issued_at.floor("h")
            overlay["time_local"] = pd.to_datetime(overlay["time_local"], utc=True, errors="coerce").dt.tz_convert(
                ZoneInfo(local_tz)
            )
            overlay = overlay.dropna(subset=["time_local"]).copy()
            if overlay.empty:
                continue
            overlay = overlay[
                (overlay["time_local"].dt.hour >= 8)
                & (
                    (overlay["time_local"].dt.hour < 22)
                    | ((overlay["time_local"].dt.hour == 22) & (overlay["time_local"].dt.minute == 0))
                )
            ].copy()
            if overlay.empty:
                continue
            if "is_future" in overlay.columns:
                is_future_mask = (
                    overlay["is_future"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .isin(["true", "1", "yes"])
                )
                overlay = overlay[
                    is_future_mask | overlay["time_local"].eq(issue_anchor)
                ].copy()
            overlay = overlay[overlay["time_local"] >= issue_anchor].copy()
            if overlay.empty:
                continue
            historical_branches.append((issue_anchor, overlay.sort_values("time_local").reset_index(drop=True)))

    mae_branch_styles = {
        3: {"color": "#0057b8", "label": "Super local - avg speed - 3 hr ago forecast"},
        6: {"color": "#008b5e", "label": "Super local - avg speed - 6 hr ago forecast"},
    }

    if historical_branches:
        current_issue_dt = _parse_iso_utc(prediction_generated_at_utc)
        if current_issue_dt is None:
            current_issue_anchor = table["time_local"].max().floor("h")
        else:
            current_issue_anchor = pd.Timestamp(current_issue_dt).tz_convert(ZoneInfo(local_tz)).floor("h")
        historical_branches.sort(key=lambda item: item[0])
        overlay_alpha_start = 0.16 if mobile else 0.14
        overlay_alpha_end = 0.42 if mobile else 0.36
        overlay_lw = 1.1 if mobile else 1.0
        active_lw = 2.2 if mobile else 2.0
        overlay_alphas = np.linspace(overlay_alpha_start, overlay_alpha_end, len(historical_branches), dtype=float)
        branch_start_markers: dict[int, tuple[float, float]] = {}
        prev_active_end_x: float | None = None
        prev_active_end_y: float | None = None
        for idx, ((issue_anchor, overlay), overlay_alpha) in enumerate(zip(historical_branches, overlay_alphas)):
            next_issue_anchor = (
                historical_branches[idx + 1][0]
                if idx + 1 < len(historical_branches)
                else current_issue_anchor
            )
            overlay_x = np.array([x_lookup.get(ts, np.nan) for ts in overlay["time_local"]], dtype=float)
            overlay_y = overlay["lstm_pred_wind_speed"].to_numpy(dtype=float)
            valid = (~np.isnan(overlay_x)) & (~np.isnan(overlay_y))
            if valid.sum() < 2:
                continue
            overlay_x = overlay_x[valid]
            overlay_y = overlay_y[valid]
            order = np.argsort(overlay_x)
            overlay_x = overlay_x[order]
            overlay_y = overlay_y[order]
            if prev_active_end_x is not None and prev_active_end_y is not None:
                if np.isclose(overlay_x[0], prev_active_end_x):
                    overlay_y[0] = prev_active_end_y
                elif overlay_x[0] > prev_active_end_x:
                    overlay_x = np.insert(overlay_x, 0, prev_active_end_x)
                    overlay_y = np.insert(overlay_y, 0, prev_active_end_y)
            branch_lookback = branch_lookback_by_anchor.get(issue_anchor)
            if branch_lookback in mae_branch_styles:
                branch_style = mae_branch_styles[branch_lookback]
                ax.plot(
                    overlay_x,
                    overlay_y,
                    color=branch_style["color"],
                    linewidth=overlay_lw + 0.3,
                    alpha=max(float(overlay_alpha), 0.6),
                    zorder=2,
                    label="_nolegend_",
                )
                if branch_lookback not in branch_start_markers:
                    branch_start_markers[branch_lookback] = (float(overlay_x[0]), float(overlay_y[0]))
            active_overlay = overlay[(overlay["time_local"] >= issue_anchor) & (overlay["time_local"] <= next_issue_anchor)].copy()
            active_x = np.array([x_lookup.get(ts, np.nan) for ts in active_overlay["time_local"]], dtype=float)
            active_y = active_overlay["lstm_pred_wind_speed"].to_numpy(dtype=float)
            active_valid = (~np.isnan(active_x)) & (~np.isnan(active_y))
            if active_valid.sum() >= 2:
                active_x = active_x[active_valid]
                active_y = active_y[active_valid]
                active_order = np.argsort(active_x)
                active_x = active_x[active_order]
                active_y = active_y[active_order]
                if prev_active_end_x is not None and prev_active_end_y is not None:
                    if np.isclose(active_x[0], prev_active_end_x):
                        active_y[0] = prev_active_end_y
                    elif active_x[0] > prev_active_end_x:
                        active_x = np.insert(active_x, 0, prev_active_end_x)
                        active_y = np.insert(active_y, 0, prev_active_end_y)
                ax.plot(
                    active_x,
                    active_y,
                    color=LSTM_HIGHLIGHT_COLOR,
                    linestyle="--",
                    linewidth=active_lw,
                    alpha=0.88,
                    zorder=2.65,
                    label="_nolegend_",
                )
                prev_active_end_x = float(active_x[-1])
                prev_active_end_y = float(active_y[-1])
        for lookback_hours, (marker_x, marker_y) in branch_start_markers.items():
            branch_style = mae_branch_styles[lookback_hours]
            ax.scatter(
                [marker_x],
                [marker_y],
                s=34 if mobile else 30,
                color=branch_style["color"],
                edgecolors="white",
                linewidths=0.8,
                zorder=3.4,
                label="_nolegend_",
            )

    # Vertical line at present hour boundary (first predicted hour).
    future_idx = np.where(table["is_future"].to_numpy(dtype=bool))[0]

    if not historical_branches:
        # Before any prior hourly updates exist, fall back to the current run's past context.
        lstm_past = table["lstm_pred_wind_speed_full"].to_numpy(dtype=float).copy()
        if len(future_idx) > 0:
            lstm_past[future_idx[0]:] = np.nan
        ax.plot(
            x,
            lstm_past,
            color=LSTM_HIGHLIGHT_COLOR,
            linestyle="--",
            linewidth=2.0,
            alpha=0.9,
            label="_nolegend_",
        )

    # LSTM future segment (solid): remaining-day best prediction from current anchor.
    lstm_future = table["lstm_pred_wind_speed"].to_numpy(dtype=float).copy()
    if len(future_idx) > 0 and future_idx[0] - 1 >= 0:
        boundary_idx = future_idx[0] - 1
        if np.isnan(lstm_future[boundary_idx]):
            lstm_future[boundary_idx] = table["lstm_pred_wind_speed_full"].to_numpy(dtype=float)[boundary_idx]
    ax.plot(
        x,
        lstm_future,
        color=LSTM_HIGHLIGHT_COLOR,
        linewidth=2.6,
        label="Super local - avg speed - hourly remaining day forecast",
        zorder=3,
    )
    ax.set_title(f"Current-day wind prediction: {day_label}", fontsize=title_fs)
    ax.set_xlabel("Time", fontsize=label_fs)
    ax.set_ylabel("Wind speed (kts)", fontsize=label_fs)
    ax.grid(axis="y", alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    desired_order = [
        "Wind speed - measured",
        "Super local - avg speed - hourly remaining day forecast",
        "Super local - avg speed - 3 hr ago forecast",
        "Super local - avg speed - 6 hr ago forecast",
        "Harmonie model - avg speed",
        "Harmonie model - max speed",
    ]
    order_map = {label: handle for handle, label in zip(handles, labels)}
    for lookback_hours, branch_style in mae_branch_styles.items():
        if lookback_hours not in branch_lookback_by_anchor.values():
            continue
        order_map[branch_style["label"]] = Line2D(
            [0],
            [0],
            color=branch_style["color"],
            linewidth=1.5 if mobile else 1.3,
            alpha=0.8,
        )
    ordered_handles = [order_map[label] for label in desired_order if label in order_map]
    ordered_labels = [label for label in desired_order if label in order_map]
    ax.legend(
        ordered_handles,
        ordered_labels,
        loc="upper left",
        bbox_to_anchor=(0.015, 0.99),
        borderaxespad=0.0,
        fontsize=legend_fs,
    )
    hour_tick_mask = table["time_local"].dt.minute.eq(0)
    tick_pos = np.where(hour_tick_mask.to_numpy())[0]
    tick_lbl = table.loc[hour_tick_mask, "time_local"].dt.strftime("%H").to_list()
    ax.set_xticks(tick_pos, tick_lbl, rotation=0)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.set_xlim(-0.05, len(table) - 1.0 + 0.02)
    ax.set_ylim(y_lower, y_upper)

    metric_map = {
        int(metric.get("lookback_hours", 0)): metric
        for metric in (fixed_origin_mae_metrics or [])
    }

    superlocal_color = "#d62728"
    measured_color = "#cc33cc"
    harmonie_color = "#666666"
    better_color = "#2ca02c"
    worse_color = "#d62728"
    neutral_color = "black"
    metric_fontfamily = "monospace"

    actual_idx = np.where(~np.isnan(actual_avg))[0]
    if len(actual_idx) > 0:
        latest_actual_idx = int(actual_idx[-1])
        latest_actual_value = float(actual_avg[latest_actual_idx])
        # Mark "now" at the latest available measured point on the dense timeline.
        ax.axvline(float(latest_actual_idx), color="gray", linestyle="--", linewidth=1.0)
        ax.annotate(
            f"{int(round(latest_actual_value))} kts",
            xy=(latest_actual_idx, latest_actual_value),
            xytext=(-8, -12) if latest_actual_idx >= len(table) - 3 else (6, -12),
            textcoords="offset points",
            ha="right" if latest_actual_idx >= len(table) - 3 else "left",
            va="top",
            fontsize=max(mae_fs - 1, 8),
            color=measured_color,
            zorder=8,
            clip_on=True,
        )

    def _metric_value_colors(metric: dict | None) -> tuple[str, str]:
        if not metric or not metric.get("available", False):
            return neutral_color, neutral_color
        mae_superlocal = float(metric["mae_superlocal"])
        mae_harmonie = float(metric["mae_harmonie"])
        if np.isclose(mae_superlocal, mae_harmonie):
            return better_color, better_color
        if mae_superlocal < mae_harmonie:
            return better_color, worse_color
        return worse_color, better_color

    def _metric_left_label(model_label: str, model_color: str, interval_color: str) -> HPacker:
        padded_model_label = model_label.ljust(len("Super local"))
        return HPacker(
            children=[
                TextArea(
                    "━━━ ",
                    textprops={
                        "fontsize": mae_fs + 1,
                        "color": interval_color,
                        "fontweight": "bold",
                        "fontfamily": metric_fontfamily,
                    },
                ),
                TextArea(
                    padded_model_label,
                    textprops={"fontsize": mae_fs, "color": model_color, "fontfamily": metric_fontfamily},
                ),
            ],
            align="center",
            pad=0,
            sep=0,
        )

    def _metric_line(
        model_label: str,
        model_color: str,
        hours: int,
        interval_color: str,
        value_text: str,
        value_color: str,
    ) -> HPacker:
        return HPacker(
            children=[
                _metric_left_label(model_label, model_color, interval_color),
                TextArea("  ", textprops={"fontsize": mae_fs, "color": "black", "fontfamily": metric_fontfamily}),
                TextArea(
                    f"{hours} hr interval: ",
                    textprops={"fontsize": mae_fs, "color": "black", "fontfamily": metric_fontfamily},
                ),
                TextArea(
                    value_text,
                    textprops={
                        "fontsize": mae_fs,
                        "color": value_color,
                        "fontweight": "bold" if value_text != "n/a" else "normal",
                        "fontfamily": metric_fontfamily,
                    },
                ),
            ],
            align="center",
            pad=0,
            sep=0,
        )

    metric_lines: list[TextArea | HPacker] = [
        TextArea(
            "Running mean absolute errors:",
            textprops={
                "fontsize": mae_fs,
                "fontweight": "bold",
                "color": "black",
                "fontfamily": metric_fontfamily,
            },
        )
    ]
    for hours in (3, 6):
        metric = metric_map.get(hours)
        interval_color = mae_branch_styles[hours]["color"]
        superlocal_value_color, harmonie_value_color = _metric_value_colors(metric)
        if metric and metric.get("available", False):
            mae_superlocal_txt = f"{float(metric['mae_superlocal']):.2f} kts"
            mae_harmonie_txt = f"{float(metric['mae_harmonie']):.2f} kts"
        else:
            mae_superlocal_txt = "n/a"
            mae_harmonie_txt = "n/a"
        metric_lines.append(
            _metric_line(
                "Super local",
                superlocal_color,
                hours,
                interval_color,
                mae_superlocal_txt,
                superlocal_value_color,
            )
        )
        metric_lines.append(
            _metric_line(
                "Harmonie",
                harmonie_color,
                hours,
                interval_color,
                mae_harmonie_txt,
                harmonie_value_color,
            )
        )

    mse_box = VPacker(
        children=metric_lines,
        align="left",
        pad=0,
        sep=1,
    )
    mse_anchored = AnchoredOffsetbox(
        loc="upper right",
        child=mse_box,
        pad=0.2,
        frameon=True,
        bbox_to_anchor=(0.992, meta_y + (0.092 if mobile else 0.076)),
        bbox_transform=ax.transAxes,
        borderpad=0.35,
    )
    mse_anchored.patch.set_facecolor("white")
    mse_anchored.patch.set_alpha(0.7)
    mse_anchored.patch.set_edgecolor("none")
    mse_anchored.set_zorder(7)
    ax.add_artist(mse_anchored)

    ax.text(
        0.015,
        meta_y,
        _format_plot_meta_text(prediction_generated_at_utc, prediction_updated_at_utc, model_trained_at_utc, local_tz),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=meta_fs,
        color="black",
        clip_on=False,
    )

    # Direction arrows below axis for forecast, LSTM (remaining where available, else full-day context), and actual.
    y_base_axes = -0.115 if mobile else -0.14
    arrow_len_axes = 0.058 if mobile else 0.065
    if len(table) >= 2:
        step_min = max(
            1,
            int(round((table["time_local"].iloc[1] - table["time_local"].iloc[0]).total_seconds() / 60.0)),
        )
    else:
        step_min = 6
    points_per_hour = max(1.0, 60.0 / step_min)
    arrow_dx_scale = 0.22 * points_per_hour
    arrow_rows = table[table["time_local"].dt.minute.eq(0)]
    for i, row in arrow_rows.iterrows():
        fdir = row["forecast_wind_dir_deg"]
        ldir = row["lstm_pred_wind_dir_deg"]
        adir = row["actual_wind_dir_deg"]
        if pd.isna(ldir):
            ldir = row["lstm_pred_wind_dir_deg_full"]
        for direction_deg, color, z in [(fdir, "gray", 3), (ldir, LSTM_HIGHLIGHT_COLOR, 4), (adir, "magenta", 6)]:
            if pd.isna(direction_deg):
                continue
            theta = np.deg2rad((float(direction_deg) + 180.0) % 360.0)
            dx = arrow_dx_scale * np.sin(theta)
            dy = arrow_len_axes * np.cos(theta)
            ax.annotate(
                "",
                xy=(i + dx, y_base_axes + dy),
                xytext=(i, y_base_axes),
                xycoords=ax.get_xaxis_transform(),
                textcoords=ax.get_xaxis_transform(),
                arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.6, "shrinkA": 0, "shrinkB": 0},
                clip_on=False,
                zorder=z,
            )

    layout_top = 0.92 if mobile else 0.965
    layout_bottom = 0.055 if mobile else 0.04
    fig.tight_layout(rect=[0, layout_bottom, 1, layout_top])
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def compute_running_mae(table: pd.DataFrame) -> tuple[float, float, int]:
    forecast_vals = table["forecast_wind_speed"].to_numpy(dtype=float)
    actual_vals = table["actual_wind_speed"].to_numpy(dtype=float)
    lstm_vals = table["lstm_pred_wind_speed_full"].to_numpy(dtype=float)
    mae_mask_fc = (~np.isnan(actual_vals)) & (~np.isnan(forecast_vals))
    mae_mask_lstm = (~np.isnan(actual_vals)) & (~np.isnan(lstm_vals))
    common_points = int(np.sum(mae_mask_fc & mae_mask_lstm))
    mae_fc = float(np.mean(np.abs(actual_vals[mae_mask_fc] - forecast_vals[mae_mask_fc]))) if mae_mask_fc.any() else float("nan")
    mae_lstm = float(np.mean(np.abs(actual_vals[mae_mask_lstm] - lstm_vals[mae_mask_lstm]))) if mae_mask_lstm.any() else float("nan")
    return mae_fc, mae_lstm, common_points


def _save_model(
    path: Path,
    model: nn.Module,
    n_features: int,
    target_hours: int,
    target_name: str,
    target_mode: str,
    output_activation: str,
    extra: dict | None = None,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "model_class": "NextDayLSTM",
        "n_features": int(n_features),
        "target_hours": int(target_hours),
        "target_name": target_name,
        "target_mode": target_mode,
        "output_activation": output_activation,
    }
    if extra:
        payload.update(extra)
    torch.save(
        payload,
        path,
    )


def _load_model(path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location=device)
    model = NextDayLSTM(
        n_features=int(ckpt["n_features"]),
        target_hours=int(ckpt["target_hours"]),
        output_activation=str(ckpt.get("output_activation", "linear")),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def _resolve_model_trained_utc(ckpt: dict, model_path: Path) -> str | None:
    ts = ckpt.get("trained_at_utc")
    if ts:
        return str(ts)
    if model_path.exists():
        return datetime.fromtimestamp(model_path.stat().st_mtime, tz=timezone.utc).isoformat()
    return None


def _resolve_now_local(local_tz: str, test_now_local_hour: int | None) -> datetime:
    tz = ZoneInfo(local_tz)
    now_local = datetime.now(tz=tz)
    if test_now_local_hour is not None:
        hour = int(test_now_local_hour)
        if hour < 0 or hour > 23:
            raise ValueError("--test-now-local-hour must be between 0 and 23.")
        now_local = now_local.replace(hour=hour, minute=0, second=0, microsecond=0)
    return now_local


def maybe_archive_current_day_plot(
    current_day_plot_path: Path,
    out_dir: Path,
    local_tz: str,
    test_now_local_hour: int | None,
) -> str | None:
    now_local = _resolve_now_local(local_tz, test_now_local_hour)
    if now_local.hour < 22:
        return None

    archive_dir = out_dir / "current_day_plot_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = now_local.strftime("%Y%m%d-%H%M%S")
    archived_path = archive_dir / f"{stamp}_current_day_predictions.png"
    shutil.copy2(current_day_plot_path, archived_path)
    return str(archived_path)


def maybe_archive_next_day_plots(
    next_day_plot_path: Path,
    next_day_plot_mobile_path: Path | None,
    out_dir: Path,
    local_tz: str,
    test_now_local_hour: int | None,
) -> dict[str, str] | None:
    now_local = _resolve_now_local(local_tz, test_now_local_hour)
    if now_local.hour < 22:
        return None

    archive_dir = out_dir / "next_day_plot_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = now_local.strftime("%Y%m%d-%H%M%S")

    archived: dict[str, str] = {}
    archived_desktop = archive_dir / f"{stamp}_next_day_predictions.png"
    shutil.copy2(next_day_plot_path, archived_desktop)
    archived["desktop"] = str(archived_desktop)

    if next_day_plot_mobile_path is not None and next_day_plot_mobile_path.exists():
        archived_mobile = archive_dir / f"{stamp}_next_day_predictions_mobile.png"
        shutil.copy2(next_day_plot_mobile_path, archived_mobile)
        archived["mobile"] = str(archived_mobile)
    return archived


def save_daily_mae_plot(
    history_csv: Path,
    plot_png: Path,
    local_tz: str = "Europe/Amsterdam",
    mobile_last_months: int | None = None,
) -> None:
    if not history_csv.exists():
        return
    hist = pd.read_csv(history_csv)
    if hist.empty:
        return

    if "evaluation_type" not in hist.columns:
        hist["evaluation_type"] = "legacy_current_day"
    hist = hist[hist["evaluation_type"] == "day_ahead_frozen"].copy()
    for c in ["avg_actual_wind_speed", "avg_forecast_wind_speed", "avg_lstm_wind_speed"]:
        if c not in hist.columns:
            hist[c] = np.nan
    if "details_csv" in hist.columns:
        missing_stats = hist[["avg_actual_wind_speed", "avg_forecast_wind_speed", "avg_lstm_wind_speed"]].isna().any(axis=1)
        for i in hist.index[missing_stats]:
            details_path = Path(str(hist.at[i, "details_csv"]))
            if not details_path.exists():
                continue
            try:
                det = pd.read_csv(details_path)
                if "actual_wind_speed" in det.columns and pd.isna(hist.at[i, "avg_actual_wind_speed"]):
                    hist.at[i, "avg_actual_wind_speed"] = float(pd.to_numeric(det["actual_wind_speed"], errors="coerce").mean())
                if "forecast_wind_speed" in det.columns and pd.isna(hist.at[i, "avg_forecast_wind_speed"]):
                    hist.at[i, "avg_forecast_wind_speed"] = float(pd.to_numeric(det["forecast_wind_speed"], errors="coerce").mean())
                if "lstm_wind_speed" in det.columns and pd.isna(hist.at[i, "avg_lstm_wind_speed"]):
                    hist.at[i, "avg_lstm_wind_speed"] = float(pd.to_numeric(det["lstm_wind_speed"], errors="coerce").mean())
            except Exception:
                pass
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist = hist.dropna(subset=["date"]).sort_values("date")

    if "mae_forecast" not in hist.columns and "mse_forecast" in hist.columns:
        hist["mae_forecast"] = hist["mse_forecast"]
    if "mae_lstm" not in hist.columns and "mse_lstm" in hist.columns:
        hist["mae_lstm"] = hist["mse_lstm"]
    hist["mae_forecast"] = pd.to_numeric(hist["mae_forecast"], errors="coerce")
    hist["mae_lstm"] = pd.to_numeric(hist["mae_lstm"], errors="coerce")
    hist["avg_actual_wind_speed"] = pd.to_numeric(hist["avg_actual_wind_speed"], errors="coerce")
    hist["avg_forecast_wind_speed"] = pd.to_numeric(hist["avg_forecast_wind_speed"], errors="coerce")
    hist["avg_lstm_wind_speed"] = pd.to_numeric(hist["avg_lstm_wind_speed"], errors="coerce")
    hist = hist.dropna(subset=["mae_forecast", "mae_lstm"])

    hist["day"] = hist["date"].dt.floor("D")
    hist_daily = hist.groupby("day", as_index=False)[
        ["mae_forecast", "mae_lstm", "avg_actual_wind_speed", "avg_forecast_wind_speed", "avg_lstm_wind_speed"]
    ].mean(numeric_only=True)

    backfill_csv = history_csv.parent / "dayahead_backfill_history.csv"
    backfill_daily = pd.DataFrame(
        columns=["day", "mae_forecast", "mae_lstm", "avg_actual_wind_speed", "avg_forecast_wind_speed", "avg_lstm_wind_speed"]
    )
    if backfill_csv.exists():
        backfill = pd.read_csv(backfill_csv)
        if "day_local" in backfill.columns:
            backfill["day"] = pd.to_datetime(backfill["day_local"], errors="coerce")
        elif "date" in backfill.columns:
            backfill["day"] = pd.to_datetime(backfill["date"], errors="coerce")
        else:
            backfill["day"] = pd.NaT
        for c in ["avg_actual_wind_speed", "avg_forecast_wind_speed", "avg_lstm_wind_speed"]:
            if c not in backfill.columns:
                backfill[c] = np.nan
        for c in ["mae_forecast", "mae_lstm", "avg_actual_wind_speed", "avg_forecast_wind_speed", "avg_lstm_wind_speed"]:
            if c in backfill.columns:
                backfill[c] = pd.to_numeric(backfill[c], errors="coerce")
        backfill = backfill.dropna(subset=["day"])
        if not backfill.empty:
            backfill_daily = backfill.groupby("day", as_index=False)[
                ["mae_forecast", "mae_lstm", "avg_actual_wind_speed", "avg_forecast_wind_speed", "avg_lstm_wind_speed"]
            ].mean(numeric_only=True)

    if hist_daily.empty and backfill_daily.empty:
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.set_title("Mean Absolute Prediction Error (Day-ahead)")
        ax.set_xlabel("Date")
        ax.set_ylabel("MAE (kts)")
        ax.set_ylim(0.0, 4.0)
        ax.grid(axis="y", alpha=0.3)
        ax.text(
            0.5,
            0.5,
            "No day-ahead history yet.\nFirst entry appears after a full target day completes.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#444",
        )
        fig.tight_layout()
        fig.savefig(plot_png, dpi=150)
        plt.close(fig)
        return

    if hist_daily.empty:
        merged = backfill_daily.set_index("day")
    elif backfill_daily.empty:
        merged = hist_daily.set_index("day")
    else:
        merged = hist_daily.set_index("day").combine_first(backfill_daily.set_index("day"))
    merged = merged.sort_index()

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(11.4, 6.8),
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.40},
    )
    now_local = datetime.now(ZoneInfo(local_tz))
    month_start_current = now_local.replace(day=1, hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    if month_start_current.month == 12:
        next_month = month_start_current.replace(year=month_start_current.year + 1, month=1)
    else:
        next_month = month_start_current.replace(month=month_start_current.month + 1)
    month_end_current = next_month - timedelta(days=1)

    first_day = merged.index.min().to_pydatetime().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if mobile_last_months is not None and int(mobile_last_months) > 0:
        months = int(mobile_last_months)
        start_current_month = month_start_current
        start_target_month = start_current_month
        for _ in range(max(0, months - 1)):
            if start_target_month.month == 1:
                start_target_month = start_target_month.replace(year=start_target_month.year - 1, month=12)
            else:
                start_target_month = start_target_month.replace(month=start_target_month.month - 1)
        x_start = max(pd.Timestamp(first_day), pd.Timestamp(start_target_month))
    else:
        september_start = datetime(first_day.year, 9, 1)
        x_start = max(pd.Timestamp(first_day), pd.Timestamp(september_start))
    x_end = pd.Timestamp(month_end_current)
    full_index = pd.date_range(start=x_start, end=x_end, freq="D")
    merged = merged.reindex(full_index)

    # Alternate monthly background shading for readability over multi-month history.
    shade_idx = 0
    cursor = pd.Timestamp(first_day)
    while cursor <= x_end:
        if cursor.month == 12:
            month_next = cursor.replace(year=cursor.year + 1, month=1, day=1)
        else:
            month_next = cursor.replace(month=cursor.month + 1, day=1)
        month_end = month_next - pd.Timedelta(days=1)
        if shade_idx % 2 == 1:
            for ax in (ax_top, ax_bottom):
                ax.axvspan(cursor, month_end + pd.Timedelta(hours=23, minutes=59), color="0.8", alpha=0.18, zorder=0)
        shade_idx += 1
        cursor = month_next

    # Top: daily average wind-speed levels.
    ax_top.plot(
        merged.index,
        merged["avg_lstm_wind_speed"],
        linewidth=1.6,
        color=LSTM_HIGHLIGHT_COLOR,
        label="Super local predicted wind speed (daily mean)",
    )
    ax_top.plot(
        merged.index,
        merged["avg_forecast_wind_speed"],
        linewidth=1.4,
        color="gray",
        label="Harmonie predicted wind speed (daily mean)",
    )
    ax_top.plot(
        merged.index,
        merged["avg_actual_wind_speed"],
        linewidth=1.4,
        color="magenta",
        label="Measured wind speed (daily mean)",
    )
    ax_top.set_title("Daily Mean Wind Speed")
    ax_top.set_ylabel("Wind speed (kts)")
    ax_top.grid(axis="y", alpha=0.3)
    ax_top.legend(loc="upper left")
    speed_top = float(
        np.nanmax(
            [
                merged["avg_lstm_wind_speed"].max(),
                merged["avg_forecast_wind_speed"].max(),
                merged["avg_actual_wind_speed"].max(),
            ]
        )
    )
    ax_top.set_ylim(0.0, max(4.0, speed_top * 1.04))
    ax_top.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter("{x:.0f}"))

    # Bottom: day-ahead MAE.
    ax_bottom.plot(
        merged.index,
        merged["mae_lstm"],
        linewidth=2.2,
        color=LSTM_HIGHLIGHT_COLOR,
        label="Day-ahead MAE super local",
    )
    ax_bottom.plot(
        merged.index,
        merged["mae_forecast"],
        linewidth=1.8,
        color="gray",
        label="Day-ahead MAE Harmonie",
    )
    ax_bottom.set_title("Day-ahead Mean Absolute Error")
    ax_bottom.set_ylabel("MAE (kts)")
    ax_bottom.grid(axis="y", alpha=0.3)
    ax_bottom.legend(loc="upper left")
    y_top_data = float(np.nanmax([merged["mae_forecast"].max(), merged["mae_lstm"].max()]))
    ax_bottom.set_ylim(0.0, max(4.0, y_top_data * 1.08))

    # Mark model-gate events where challenger was promoted.
    gate_csv = history_csv.parent / "model_gate_eval_history.csv"
    if gate_csv.exists():
        try:
            gate = pd.read_csv(gate_csv)
        except Exception:
            gate = pd.DataFrame()
        if not gate.empty:
            if "promote_speed" in gate.columns:
                promoted_mask = gate["promote_speed"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
            else:
                # Backward compatibility with older history files.
                promoted_mask = gate.get("speed_selected", pd.Series(dtype=object)).astype(str).str.strip().str.lower().eq(
                    "challenger"
                )
            gate = gate.loc[promoted_mask].copy()
            if not gate.empty:
                if "run_local_time" in gate.columns:
                    run_dt = pd.to_datetime(gate["run_local_time"], errors="coerce", utc=True)
                else:
                    run_dt = _parse_iso_series_utc(gate.get("run_utc", pd.Series(dtype=object)))
                run_dt = run_dt.dt.tz_convert(ZoneInfo(local_tz)).dt.tz_localize(None)
                gate["run_dt_local"] = run_dt
                gate = gate.dropna(subset=["run_dt_local"]).sort_values("run_dt_local")
                gate = gate[(gate["run_dt_local"] >= x_start) & (gate["run_dt_local"] <= x_end)]

                y_top = ax_bottom.get_ylim()[1]
                for _, row in gate.iterrows():
                    run_dt_local = row["run_dt_local"]
                    model_id = str(row.get("speed_model_id_challenger", "")).strip() or "unknown"
                    ax_bottom.axvline(
                        run_dt_local,
                        color="#2ca02c",
                        linestyle="--",
                        linewidth=1.1,
                        alpha=0.85,
                        zorder=1.5,
                    )
                    ax_bottom.annotate(
                        f"model {model_id}",
                        xy=(run_dt_local, y_top * 0.985),
                        xytext=(3, -2),
                        textcoords="offset points",
                        ha="left",
                        va="top",
                        fontsize=7.5,
                        color="#2ca02c",
                        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
                        zorder=3,
                    )

    for ax in (ax_top, ax_bottom):
        ax.margins(x=0, y=0)
        ax.set_xlim(x_start, x_end)
    date_locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax_bottom.xaxis.set_major_locator(date_locator)
    ax_bottom.xaxis.set_major_formatter(mdates.ConciseDateFormatter(date_locator))
    ax_bottom.tick_params(axis="x", labelrotation=20, labelsize=10, pad=6)
    ax_bottom.set_xlabel("Date")
    valid_rows = merged.dropna(subset=["mae_forecast", "mae_lstm"])
    avg_lstm = float(valid_rows["mae_lstm"].mean())
    avg_fc = float(valid_rows["mae_forecast"].mean())
    valid_speed_rows = merged.dropna(subset=["avg_lstm_wind_speed", "avg_forecast_wind_speed", "avg_actual_wind_speed"])
    avg_daily_abs_lstm = float(np.mean(np.abs(valid_speed_rows["avg_lstm_wind_speed"] - valid_speed_rows["avg_actual_wind_speed"])))
    avg_daily_abs_fc = float(np.mean(np.abs(valid_speed_rows["avg_forecast_wind_speed"] - valid_speed_rows["avg_actual_wind_speed"])))
    def _metric_line(superlocal_value: float, harmonie_value: float):
        superlocal = HPacker(
            children=[
                TextArea("Super local: ", textprops={"fontsize": 9, "color": "black"}),
                TextArea(f"{superlocal_value:.2f}", textprops={"fontsize": 9, "color": "#2ca02c", "fontweight": "bold"}),
                TextArea(" kts", textprops={"fontsize": 9, "color": "black"}),
            ],
            align="center",
            pad=0,
            sep=0,
        )
        harmonie = HPacker(
            children=[
                TextArea("Harmonie: ", textprops={"fontsize": 9, "color": "black"}),
                TextArea(f"{harmonie_value:.2f}", textprops={"fontsize": 9, "color": "black", "fontweight": "bold"}),
                TextArea(" kts", textprops={"fontsize": 9, "color": "black"}),
            ],
            align="center",
            pad=0,
            sep=0,
        )
        return superlocal, harmonie

    daily_super, daily_harm = _metric_line(avg_daily_abs_lstm, avg_daily_abs_fc)
    hourly_super, hourly_harm = _metric_line(avg_lstm, avg_fc)
    daily_box = VPacker(
        children=[
            TextArea("Daily MAE", textprops={"fontsize": 9, "fontweight": "bold", "color": "black"}),
            daily_super,
            daily_harm,
        ],
        align="left",
        pad=0,
        sep=1,
    )
    hourly_box = VPacker(
        children=[
            TextArea("Hourly MAE", textprops={"fontsize": 9, "fontweight": "bold", "color": "black"}),
            hourly_super,
            hourly_harm,
        ],
        align="left",
        pad=0,
        sep=1,
    )
    metric_box_y = 1.22
    left_anchored = AnchoredOffsetbox(
        loc="upper left",
        child=daily_box,
        pad=0.2,
        frameon=True,
        bbox_to_anchor=(0.01, metric_box_y),
        bbox_transform=ax_bottom.transAxes,
        borderpad=0.35,
    )
    left_anchored.patch.set_facecolor("white")
    left_anchored.patch.set_alpha(0.78)
    left_anchored.patch.set_edgecolor("none")
    right_anchored = AnchoredOffsetbox(
        loc="upper right",
        child=hourly_box,
        pad=0.2,
        frameon=True,
        bbox_to_anchor=(0.99, metric_box_y),
        bbox_transform=ax_bottom.transAxes,
        borderpad=0.35,
    )
    right_anchored.patch.set_facecolor("white")
    right_anchored.patch.set_alpha(0.78)
    right_anchored.patch.set_edgecolor("none")
    ax_bottom.add_artist(left_anchored)
    ax_bottom.add_artist(right_anchored)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.10)
    fig.savefig(plot_png, dpi=150)
    plt.close(fig)


def save_dayahead_snapshot(
    out_dir: Path,
    table: pd.DataFrame,
    local_tz: str,
    prediction_generated_at_utc: str,
) -> str | None:
    if table.empty:
        return None
    pred_dt = _parse_iso_utc(prediction_generated_at_utc)
    if pred_dt is None:
        pred_dt = datetime.now(timezone.utc)
    issue_local = pred_dt.astimezone(ZoneInfo(local_tz))
    target_times_utc = pd.to_datetime(table["target_time_utc"], utc=True)
    if len(target_times_utc) == 0:
        return None
    target_day_local = target_times_utc[0].astimezone(ZoneInfo(local_tz)).date()
    snapshot_dir = out_dir / "dayahead_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / (
        f"{issue_local.strftime('%Y%m%d-%H%M%S')}_target_{target_day_local.strftime('%Y%m%d')}.csv"
    )
    snap = table[
        [
            "target_time_utc",
            "hour_utc",
            "forecast_wind_speed",
            "lstm_pred_wind_speed",
            "forecast_wind_dir_deg",
            "lstm_pred_wind_dir_deg",
        ]
    ].copy()
    snap["target_time_utc"] = pd.to_datetime(snap["target_time_utc"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    snap["issue_local_time"] = issue_local.isoformat()
    snap["issue_date_local"] = issue_local.strftime("%Y-%m-%d")
    snap["target_date_local"] = target_day_local.strftime("%Y-%m-%d")
    snap.to_csv(snapshot_path, index=False)
    return str(snapshot_path)


def save_current_day_snapshot(
    out_dir: Path,
    table: pd.DataFrame,
    local_tz: str,
    prediction_generated_at_utc: str,
) -> str:
    table_local = table.copy()
    if table_local.empty:
        raise ValueError("Current-day snapshot requires a non-empty table.")
    pred_dt = _parse_iso_utc(prediction_generated_at_utc)
    if pred_dt is None:
        pred_dt = datetime.now(timezone.utc)
    pred_local = pred_dt.astimezone(ZoneInfo(local_tz))
    target_day_local = pd.to_datetime(table_local["time_local"]).dt.tz_convert(ZoneInfo(local_tz)).iloc[0].date()
    snapshot_dir = out_dir / "current_day_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / (
        f"{pred_local.strftime('%Y%m%d-%H%M%S')}_target_{target_day_local.strftime('%Y%m%d')}.csv"
    )
    snap = table_local.copy()
    snap["issued_at_utc"] = pred_dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    snap["issued_at_local"] = pred_local.strftime("%Y-%m-%dT%H:%M:%S%z")
    snap["target_day_local"] = target_day_local.isoformat()
    snap["time_local"] = pd.to_datetime(snap["time_local"]).dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    snap.to_csv(snapshot_path, index=False)
    return str(snapshot_path)


def _find_current_day_snapshot_paths(
    out_dir: Path,
    target_day_local: date,
    max_snapshots: int = 10,
) -> list[Path]:
    snapshot_dir = out_dir / "current_day_snapshots"
    if not snapshot_dir.exists():
        return []
    pattern = f"*_target_{target_day_local.strftime('%Y%m%d')}.csv"
    matches = sorted(snapshot_dir.glob(pattern))
    if not matches:
        return []
    return matches


def load_current_day_prediction_history(
    out_dir: Path,
    target_day_local: date,
    local_tz: str,
    max_snapshots: int = 10,
) -> list[pd.DataFrame]:
    snapshots_by_issue_hour: dict[pd.Timestamp, pd.DataFrame] = {}
    tz = ZoneInfo(local_tz)
    for path in _find_current_day_snapshot_paths(out_dir, target_day_local, max_snapshots=max_snapshots):
        try:
            snap = pd.read_csv(path)
        except Exception:
            continue
        if snap.empty or "time_local" not in snap.columns or "lstm_pred_wind_speed" not in snap.columns:
            continue
        snap["time_local"] = pd.to_datetime(snap["time_local"], utc=True, errors="coerce").dt.tz_convert(tz)
        if "issued_at_local" in snap.columns:
            snap["issued_at_local"] = pd.to_datetime(snap["issued_at_local"], utc=True, errors="coerce").dt.tz_convert(tz)
        elif "issued_at_utc" in snap.columns:
            snap["issued_at_local"] = pd.to_datetime(snap["issued_at_utc"], utc=True, errors="coerce").dt.tz_convert(tz)
        snap = snap.dropna(subset=["time_local"]).copy()
        if snap.empty:
            continue
        issued_series = snap.get("issued_at_local", pd.Series(dtype="datetime64[ns, UTC]")).dropna()
        if issued_series.empty:
            continue
        issue_hour = pd.to_datetime(issued_series.iloc[0]).floor("h")
        # Keep the earliest snapshot within each issue hour so repeated reruns do not
        # overwrite that hour's forecast branch.
        if issue_hour not in snapshots_by_issue_hour:
            snapshots_by_issue_hour[issue_hour] = snap.sort_values("time_local").reset_index(drop=True)
    if not snapshots_by_issue_hour:
        return []
    issue_hours = sorted(snapshots_by_issue_hour)
    max_n = max(0, int(max_snapshots))
    if max_n > 0:
        issue_hours = issue_hours[-max_n:]
    return [snapshots_by_issue_hour[issue_hour] for issue_hour in issue_hours]


def load_latest_current_day_snapshot(
    out_dir: Path,
    target_day_local: date,
    local_tz: str,
) -> pd.DataFrame | None:
    tz = ZoneInfo(local_tz)
    paths = _find_current_day_snapshot_paths(out_dir, target_day_local, max_snapshots=0)
    if not paths:
        return None
    latest_path = paths[-1]
    try:
        snap = pd.read_csv(latest_path)
    except Exception:
        return None
    if snap.empty or "time_local" not in snap.columns or "lstm_pred_wind_speed" not in snap.columns:
        return None
    snap["time_local"] = pd.to_datetime(snap["time_local"], utc=True, errors="coerce").dt.tz_convert(tz)
    if "issued_at_local" in snap.columns:
        snap["issued_at_local"] = pd.to_datetime(snap["issued_at_local"], utc=True, errors="coerce").dt.tz_convert(tz)
    elif "issued_at_utc" in snap.columns:
        snap["issued_at_local"] = pd.to_datetime(snap["issued_at_utc"], utc=True, errors="coerce").dt.tz_convert(tz)
    snap = snap.dropna(subset=["time_local"]).copy()
    if snap.empty:
        return None
    return snap.sort_values("time_local").reset_index(drop=True)


def compute_fixed_origin_current_day_mae(
    current_table: pd.DataFrame,
    prior_prediction_tables: list[pd.DataFrame] | None,
    actual_measurements: pd.DataFrame | None = None,
    lookback_hours: tuple[int, ...] = (1, 3, 6),
) -> list[dict]:
    if current_table.empty:
        return []
    history = prior_prediction_tables or []
    if not history:
        return []

    if actual_measurements is not None and not actual_measurements.empty:
        eval_actual = actual_measurements.copy()
        eval_actual = eval_actual.dropna(subset=["time_local", "actual_wind_speed"]).copy()
        eval_actual = eval_actual.sort_values("time_local").drop_duplicates(subset=["time_local"])
    else:
        eval_actual = current_table.copy()
        eval_actual = eval_actual[
            eval_actual["time_local"].dt.minute.eq(0) & eval_actual["actual_wind_speed"].notna()
        ][["time_local", "actual_wind_speed"]].drop_duplicates(subset=["time_local"])
    if eval_actual.empty:
        return []
    eval_end = eval_actual["time_local"].max()

    snapshots: list[tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]] = []
    for snap in history:
        if snap.empty or "issued_at_local" not in snap.columns:
            continue
        issued_series = snap["issued_at_local"].dropna()
        if issued_series.empty:
            continue
        issued_at = pd.to_datetime(issued_series.iloc[0])
        issue_anchor = issued_at.floor("h")
        snap_hourly = snap.copy()
        snap_hourly = snap_hourly[snap_hourly["time_local"].dt.minute.eq(0)].copy()
        if snap_hourly.empty:
            continue
        snapshots.append((issue_anchor, issued_at, snap_hourly))
    if not snapshots:
        return []
    snapshots.sort(key=lambda item: item[0])

    metrics: list[dict] = []
    for lookback in lookback_hours:
        cutoff = eval_end - pd.Timedelta(hours=int(lookback))
        eligible = [item for item in snapshots if item[0] <= cutoff]
        if not eligible:
            metrics.append(
                {
                    "lookback_hours": int(lookback),
                    "available": False,
                    "snapshot_issued_at_local": None,
                    "actual_issue_age_hours": None,
                    "point_count": 0,
                    "mae_superlocal": None,
                    "mae_harmonie": None,
                }
            )
            continue

        issue_anchor, issued_at, snap = eligible[-1]
        snap_eval = snap.copy()
        snap_eval = snap_eval[(snap_eval["time_local"] >= issue_anchor) & (snap_eval["time_local"] <= eval_end)].copy()
        snap_eval = snap_eval.dropna(subset=["time_local", "forecast_wind_speed", "lstm_pred_wind_speed"])
        actual_eval = eval_actual[(eval_actual["time_local"] > issue_anchor) & (eval_actual["time_local"] <= eval_end)].copy()
        if snap_eval.empty or actual_eval.empty:
            metrics.append(
                {
                    "lookback_hours": int(lookback),
                    "available": False,
                    "snapshot_issued_at_local": issue_anchor.isoformat(),
                    "actual_issue_age_hours": float((eval_end - issue_anchor).total_seconds() / 3600.0),
                    "point_count": 0,
                    "mae_superlocal": None,
                    "mae_harmonie": None,
                }
            )
            continue

        snap_eval = snap_eval.sort_values("time_local").drop_duplicates(subset=["time_local"]).copy()
        actual_times = actual_eval["time_local"]
        forecast_series = (
            pd.Series(pd.to_numeric(snap_eval["forecast_wind_speed"], errors="coerce").to_numpy(dtype=float), index=snap_eval["time_local"])
            .sort_index()
            .groupby(level=0)
            .last()
        )
        superlocal_series = (
            pd.Series(pd.to_numeric(snap_eval["lstm_pred_wind_speed"], errors="coerce").to_numpy(dtype=float), index=snap_eval["time_local"])
            .sort_index()
            .groupby(level=0)
            .last()
        )
        forecast_interp = (
            forecast_series.reindex(forecast_series.index.union(actual_times))
            .sort_index()
            .interpolate(method="time", limit_area="inside")
            .reindex(actual_times)
        )
        superlocal_interp = (
            superlocal_series.reindex(superlocal_series.index.union(actual_times))
            .sort_index()
            .interpolate(method="time", limit_area="inside")
            .reindex(actual_times)
        )
        merged = actual_eval.copy()
        merged["forecast_wind_speed"] = forecast_interp.to_numpy(dtype=float)
        merged["lstm_pred_wind_speed"] = superlocal_interp.to_numpy(dtype=float)
        merged = merged.dropna(subset=["forecast_wind_speed", "lstm_pred_wind_speed", "actual_wind_speed"])
        if merged.empty:
            metrics.append(
                {
                    "lookback_hours": int(lookback),
                    "available": False,
                    "snapshot_issued_at_local": issue_anchor.isoformat(),
                    "actual_issue_age_hours": float((eval_end - issue_anchor).total_seconds() / 3600.0),
                    "point_count": 0,
                    "mae_superlocal": None,
                    "mae_harmonie": None,
                }
            )
            continue

        mae_sl = float(np.mean(np.abs(merged["lstm_pred_wind_speed"] - merged["actual_wind_speed"])))
        mae_fc = float(np.mean(np.abs(merged["forecast_wind_speed"] - merged["actual_wind_speed"])))
        metrics.append(
            {
                "lookback_hours": int(lookback),
                "available": True,
                "snapshot_issued_at_local": issue_anchor.isoformat(),
                "actual_issue_age_hours": float((eval_end - issue_anchor).total_seconds() / 3600.0),
                "point_count": int(len(merged)),
                "mae_superlocal": mae_sl,
                "mae_harmonie": mae_fc,
            }
        )

    return metrics


def _find_dayahead_snapshot(out_dir: Path, target_day_local: date) -> Path | None:
    snapshot_dir = out_dir / "dayahead_snapshots"
    if not snapshot_dir.exists():
        return None
    pattern = f"*_target_{target_day_local.strftime('%Y%m%d')}.csv"
    matches = sorted(snapshot_dir.glob(pattern))
    if not matches:
        return None
    # Use earliest issued snapshot for a fair fixed day-ahead comparison.
    return matches[0]


def maybe_save_daily_mae_dayahead(
    out_dir: Path,
    db_path: Path,
    cfg: DatasetConfig,
    local_tz: str,
    test_now_local_hour: int | None,
) -> tuple[str | None, str | None]:
    now_local = _resolve_now_local(local_tz, test_now_local_hour)
    if now_local.hour < 22:
        return None, None

    target_day_local = now_local.date()
    snapshot_path = _find_dayahead_snapshot(out_dir, target_day_local)
    if snapshot_path is None or (not snapshot_path.exists()):
        return None, None

    snap = pd.read_csv(snapshot_path)
    if snap.empty or "target_time_utc" not in snap.columns:
        return None, None
    snap["target_time_utc"] = pd.to_datetime(snap["target_time_utc"], utc=True, errors="coerce")
    snap = snap.dropna(subset=["target_time_utc"]).copy()
    if snap.empty:
        return None, None

    conn = sqlite3.connect(str(db_path))
    try:
        obs_raw_utc = _load_observations_raw(conn, cfg.site)
    finally:
        conn.close()
    if obs_raw_utc.empty:
        return None, None
    obs_hourly_utc = obs_raw_utc.resample("1h").mean(numeric_only=True)
    actual = obs_hourly_utc.reindex(snap["target_time_utc"])["actual_avg"].to_numpy(dtype=float)
    forecast = pd.to_numeric(snap["forecast_wind_speed"], errors="coerce").to_numpy(dtype=float)
    lstm = pd.to_numeric(snap["lstm_pred_wind_speed"], errors="coerce").to_numpy(dtype=float)
    valid = (~np.isnan(actual)) & (~np.isnan(forecast)) & (~np.isnan(lstm))
    if not valid.any():
        return None, None

    mae_forecast = float(np.mean(np.abs(actual[valid] - forecast[valid])))
    mae_lstm = float(np.mean(np.abs(actual[valid] - lstm[valid])))
    avg_actual_speed = float(np.mean(actual[valid]))
    avg_forecast_speed = float(np.mean(forecast[valid]))
    avg_lstm_speed = float(np.mean(lstm[valid]))
    n_points = int(np.sum(valid))

    history_csv = out_dir / "daily_mae_history.csv"
    legacy_history_csv = out_dir / "daily_mse_history.csv"
    details_dir = out_dir / "daily_error_details"
    details_dir.mkdir(parents=True, exist_ok=True)
    day_stamp = target_day_local.strftime("%Y%m%d")
    details_csv = details_dir / f"{day_stamp}_dayahead_actual_forecast_lstm.csv"

    details = snap.copy()
    details["actual_wind_speed"] = actual
    details["abs_err_forecast"] = np.abs(details["actual_wind_speed"] - pd.to_numeric(details["forecast_wind_speed"], errors="coerce"))
    details["abs_err_lstm"] = np.abs(details["actual_wind_speed"] - pd.to_numeric(details["lstm_pred_wind_speed"], errors="coerce"))
    details["target_time_utc"] = pd.to_datetime(details["target_time_utc"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    details.to_csv(details_csv, index=False)

    row = pd.DataFrame(
        [
            {
                "date": target_day_local.strftime("%Y-%m-%d"),
                "run_local_time": now_local.isoformat(),
                "issue_local_time": str(snap["issue_local_time"].iloc[0]) if "issue_local_time" in snap.columns else "",
                "mae_forecast": mae_forecast,
                "mae_lstm": mae_lstm,
                "avg_actual_wind_speed": avg_actual_speed,
                "avg_forecast_wind_speed": avg_forecast_speed,
                "avg_lstm_wind_speed": avg_lstm_speed,
                "n_points": int(n_points),
                "details_csv": str(details_csv),
                "snapshot_csv": str(snapshot_path),
                "evaluation_type": "day_ahead_frozen",
            }
        ]
    )
    if history_csv.exists():
        hist = pd.read_csv(history_csv)
        if "evaluation_type" not in hist.columns:
            hist["evaluation_type"] = "legacy_current_day"
    elif legacy_history_csv.exists():
        hist = pd.read_csv(legacy_history_csv)
        if "mae_forecast" not in hist.columns and "mse_forecast" in hist.columns:
            hist["mae_forecast"] = hist["mse_forecast"]
        if "mae_lstm" not in hist.columns and "mse_lstm" in hist.columns:
            hist["mae_lstm"] = hist["mse_lstm"]
        keep_cols = [
            c
            for c in [
                "date",
                "run_local_time",
                "mae_forecast",
                "mae_lstm",
                "avg_actual_wind_speed",
                "avg_forecast_wind_speed",
                "avg_lstm_wind_speed",
                "n_points",
            ]
            if c in hist.columns
        ]
        hist = hist[keep_cols]
        hist["evaluation_type"] = "legacy_current_day"
    else:
        hist = pd.DataFrame(
            columns=[
                "date",
                "run_local_time",
                "issue_local_time",
                "mae_forecast",
                "mae_lstm",
                "avg_actual_wind_speed",
                "avg_forecast_wind_speed",
                "avg_lstm_wind_speed",
                "n_points",
                "details_csv",
                "snapshot_csv",
                "evaluation_type",
            ]
        )

    hist = hist[~((hist["date"] == row.iloc[0]["date"]) & (hist["evaluation_type"] == "day_ahead_frozen"))]
    hist = pd.concat([hist, row], ignore_index=True)

    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist = hist.dropna(subset=["date"]).sort_values("date")
    hist["date"] = hist["date"].dt.strftime("%Y-%m-%d")
    hist.to_csv(history_csv, index=False)

    history_png = out_dir / "daily_mae_history.png"
    save_daily_mae_plot(history_csv, history_png, local_tz=local_tz)
    return str(history_csv), str(history_png)


def append_model_gate_eval_history(
    history_csv: Path,
    model_selection_gate: dict,
    local_tz: str,
) -> str | None:
    if not model_selection_gate.get("enabled", False):
        return None
    if "speed_mae_challenger" not in model_selection_gate:
        return None

    now_utc = datetime.now(timezone.utc)
    now_local = now_utc.astimezone(ZoneInfo(local_tz))
    row = pd.DataFrame(
        [
            {
                "run_utc": now_utc.isoformat(),
                "run_local_time": now_local.isoformat(),
                "speed_mae_champion": model_selection_gate.get("speed_mae_champion"),
                "speed_mae_challenger": model_selection_gate.get("speed_mae_challenger"),
                "direction_mae_champion_deg": model_selection_gate.get("direction_mae_champion"),
                "direction_mae_challenger_deg": model_selection_gate.get("direction_mae_challenger"),
                "speed_selected": model_selection_gate.get("speed_selected"),
                "direction_selected": model_selection_gate.get("direction_selected"),
                "speed_eval_samples": model_selection_gate.get("speed_eval_samples"),
                "direction_eval_samples": model_selection_gate.get("direction_eval_samples"),
                "promotion_margin_pct": model_selection_gate.get("promotion_margin_pct"),
                "holdout_eval_split": model_selection_gate.get("holdout_eval_split"),
                "holdout_eval_min_samples": model_selection_gate.get("holdout_eval_min_samples"),
                "promote_speed": model_selection_gate.get("promote_speed"),
                "promote_direction": model_selection_gate.get("promote_direction"),
                "speed_model_id_champion": model_selection_gate.get("speed_model_id_champion"),
                "speed_model_id_challenger": model_selection_gate.get("speed_model_id_challenger"),
            }
        ]
    )
    if history_csv.exists():
        hist = pd.read_csv(history_csv)
    else:
        hist = pd.DataFrame(columns=row.columns)
    hist = pd.concat([hist, row], ignore_index=True)
    hist["run_utc"] = _parse_iso_series_utc(hist["run_utc"])
    hist = hist.dropna(subset=["run_utc"]).sort_values("run_utc")
    hist = hist.drop_duplicates(subset=["run_utc"], keep="last")
    hist["run_utc"] = hist["run_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    hist.to_csv(history_csv, index=False)
    return str(history_csv)


def save_model_gate_eval_history_plot(
    history_csv: Path,
    plot_png: Path,
    local_tz: str = "Europe/Amsterdam",
    eval_details_csv: Path | None = None,
) -> None:
    champion_model_id = None
    challenger_model_id = None
    if history_csv.exists():
        hist_for_id = pd.read_csv(history_csv)
        if not hist_for_id.empty:
            if "run_utc" in hist_for_id.columns:
                hist_for_id["run_utc"] = _parse_iso_series_utc(hist_for_id["run_utc"])
                hist_for_id = hist_for_id.dropna(subset=["run_utc"]).sort_values("run_utc")
            if not hist_for_id.empty:
                last_row = hist_for_id.iloc[-1]
                champion_model_id = str(last_row.get("speed_model_id_champion", "")).strip() or None
                challenger_model_id = str(last_row.get("speed_model_id_challenger", "")).strip() or None

    challenger_label = (
        f"Challenger prediction ({challenger_model_id})"
        if challenger_model_id
        else "Challenger prediction"
    )
    champion_label = (
        f"Champion prediction ({champion_model_id})"
        if champion_model_id
        else "Champion prediction"
    )

    # Preferred view: full holdout period time series with champion/challenger predictions.
    if eval_details_csv is not None and eval_details_csv.exists():
        det = pd.read_csv(eval_details_csv)
        if not det.empty and "target_time_utc" in det.columns:
            det["target_time_utc"] = pd.to_datetime(det["target_time_utc"], errors="coerce", utc=True)
            for col in ["actual_wind_speed", "champion_wind_speed", "challenger_wind_speed"]:
                det[col] = pd.to_numeric(det.get(col), errors="coerce")
            det = det.dropna(subset=["target_time_utc", "actual_wind_speed"])
            if not det.empty:
                x_local = det["target_time_utc"].dt.tz_convert(ZoneInfo(local_tz))
                mae_chall = float(np.mean(np.abs(det["challenger_wind_speed"] - det["actual_wind_speed"])))
                mae_champ = float(np.mean(np.abs(det["champion_wind_speed"] - det["actual_wind_speed"])))

                fig, ax = plt.subplots(figsize=(11.8, 5.2))
                ax.plot(x_local, det["actual_wind_speed"], color="magenta", linewidth=1.8, label="Measured wind speed")
                ax.plot(
                    x_local,
                    det["challenger_wind_speed"],
                    color=LSTM_HIGHLIGHT_COLOR,
                    linewidth=2.2,
                    label=challenger_label,
                )
                ax.plot(
                    x_local,
                    det["champion_wind_speed"],
                    color="#007A78",
                    linewidth=1.8,
                    label=champion_label,
                )
                ax.set_title("Model Gate Holdout Period (Speed)")
                ax.set_xlabel("Time")
                ax.set_ylabel("Wind speed (kts)")
                ax.grid(axis="y", alpha=0.3)
                ax.legend(loc="upper left")
                ax.margins(x=0, y=0)
                ymax = np.nanmax(
                    [
                        det["actual_wind_speed"].max(skipna=True),
                        det["challenger_wind_speed"].max(skipna=True),
                        det["champion_wind_speed"].max(skipna=True),
                        1.0,
                    ]
                )
                ax.set_ylim(0.0, max(4.0, float(ymax) * 1.08))
                date_locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
                ax.xaxis.set_major_locator(date_locator)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
                ax.tick_params(axis="x", labelrotation=20, labelsize=10)
                txt = (
                    "Hourly MAE (eval holdout)\n"
                    f"Challenger: {mae_chall:.2f} kts\n"
                    f"Champion: {mae_champ:.2f} kts"
                )
                ax.text(
                    0.985,
                    0.985,
                    txt,
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=10,
                    color="black",
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
                )
                fig.tight_layout()
                fig.savefig(plot_png, dpi=150)
                plt.close(fig)
                return

    # Fallback: run-level trend if period details are unavailable.
    if not history_csv.exists():
        return
    hist = pd.read_csv(history_csv)
    if hist.empty:
        return
    for col in ["speed_mae_champion", "speed_mae_challenger"]:
        hist[col] = pd.to_numeric(hist.get(col), errors="coerce")
    if "run_local_time" in hist.columns:
        run_dt = pd.to_datetime(hist["run_local_time"], errors="coerce")
    else:
        run_dt = _parse_iso_series_utc(hist.get("run_utc")).dt.tz_convert(ZoneInfo(local_tz))
    hist["run_dt"] = run_dt
    hist = hist.dropna(subset=["run_dt"]).sort_values("run_dt")
    if hist.empty:
        return
    fig, ax = plt.subplots(figsize=(11.4, 4.8))
    champ_mae_label = (
        f"Champion holdout MAE ({champion_model_id})"
        if champion_model_id
        else "Champion holdout MAE"
    )
    chall_mae_label = (
        f"Challenger holdout MAE ({challenger_model_id})"
        if challenger_model_id
        else "Challenger holdout MAE"
    )
    ax.plot(hist["run_dt"], hist["speed_mae_champion"], color="gray", linewidth=1.8, marker="o", markersize=3.0, label=champ_mae_label)
    ax.plot(hist["run_dt"], hist["speed_mae_challenger"], color=LSTM_HIGHLIGHT_COLOR, linewidth=2.2, marker="o", markersize=3.0, label=chall_mae_label)
    ax.set_title("Model Gate Holdout Comparison (Speed)")
    ax.set_xlabel("Run date")
    ax.set_ylabel("MAE (kts)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right")
    ax.margins(x=0, y=0)
    ymax = np.nanmax([hist["speed_mae_champion"].max(skipna=True), hist["speed_mae_challenger"].max(skipna=True), 1.0])
    ax.set_ylim(0.0, max(3.5, float(ymax) * 1.08))
    date_locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(date_locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.tick_params(axis="x", labelrotation=20, labelsize=10)
    fig.tight_layout()
    fig.savefig(plot_png, dpi=150)
    plt.close(fig)


def publish_web_dashboard(
    web_out_dir: Path,
    local_tz: str,
    web_refresh_seconds: int,
    next_day_png: Path,
    next_day_png_mobile: Path | None,
    next_day_csv: Path,
    current_day_png: Path,
    current_day_png_mobile: Path | None,
    current_day_csv: Path,
    daily_mae_png: Path | None,
    daily_mae_png_mobile: Path | None,
    daily_mae_csv: Path | None,
    gate_eval_png: Path | None,
    gate_eval_csv: Path | None,
) -> dict:
    web_out_dir.mkdir(parents=True, exist_ok=True)

    publish_pairs: list[tuple[Path | None, str]] = [
        (next_day_png, "next_day_predictions.png"),
        (next_day_png_mobile, "next_day_predictions_mobile.png"),
        (next_day_csv, "next_day_predictions.csv"),
        (current_day_png, "current_day_predictions.png"),
        (current_day_png_mobile, "current_day_predictions_mobile.png"),
        (current_day_csv, "current_day_predictions.csv"),
        (daily_mae_png, "daily_mae_history.png"),
        (daily_mae_png_mobile, "daily_mae_history_mobile.png"),
        (daily_mae_csv, "daily_mae_history.csv"),
        (gate_eval_png, "model_gate_eval_history.png"),
        (gate_eval_csv, "model_gate_eval_history.csv"),
    ]
    copied: dict[str, str] = {}
    for src, dst_name in publish_pairs:
        if src is None:
            continue
        if not src.exists():
            continue
        dst = web_out_dir / dst_name
        shutil.copy2(src, dst)
        copied[dst_name] = str(dst)

    generated_local = datetime.now(ZoneInfo(local_tz))
    generated_local_str = generated_local.strftime("%d %B %Y %H:%M:%S %Z")
    cache_bust = int(datetime.now(timezone.utc).timestamp())
    refresh = max(60, int(web_refresh_seconds))
    current_day_mobile_src = (
        f"current_day_predictions_mobile.png?v={cache_bust}"
        if "current_day_predictions_mobile.png" in copied
        else f"current_day_predictions.png?v={cache_bust}"
    )
    next_day_mobile_src = (
        f"next_day_predictions_mobile.png?v={cache_bust}"
        if "next_day_predictions_mobile.png" in copied
        else f"next_day_predictions.png?v={cache_bust}"
    )
    daily_mae_mobile_src = (
        f"daily_mae_history_mobile.png?v={cache_bust}"
        if "daily_mae_history_mobile.png" in copied
        else f"daily_mae_history.png?v={cache_bust}"
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="{refresh}">
  <title>Super local wind prediction Valkenburgse meer [under development]</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 16px; color: #111; }}
    h1 {{ margin: 0 0 8px 0; }}
    .meta {{ color: #555; margin: 0 0 16px 0; }}
    .overview {{ color: #222; margin: 0 0 16px 0; line-height: 1.5; font-size: 15px; max-width: 1200px; }}
    .overview-mobile {{ display: none; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 20px; max-width: 1400px; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 10px; background: #fff; }}
    .desc {{ margin: 2px 0 10px 0; color: #444; font-size: 16px; line-height: 1.4; }}
    img {{ width: 100%; height: auto; display: block; border-radius: 6px; }}
    @media (max-width: 768px) {{
      body {{ margin: 10px; }}
      h1 {{ font-size: 24px; margin: 0 0 6px 0; }}
      h2 {{ font-size: 20px; margin: 0 0 6px 0; }}
      .meta {{ margin: 0 0 10px 0; font-size: 14px; }}
      .overview {{ font-size: 14px; margin: 0 0 12px 0; line-height: 1.45; }}
      .overview-desktop {{ display: none; }}
      .overview-mobile {{ display: block; margin: 10px 0 0 0; }}
      .grid {{ gap: 12px; }}
      .card {{ padding: 8px; border-radius: 6px; }}
      .desc {{ font-size: 14px; margin: 2px 0 8px 0; }}
      img {{ max-height: 60vh; object-fit: contain; }}
    }}
  </style>
</head>
<body>
  <h1>Super local wind prediction Valkenburgse meer [under development]</h1>
  <p class="meta">Last updated: {generated_local_str}</p>
  <p class="overview overview-desktop">
    <strong>What is the super local forecast?</strong> This dashboard combines two local machine learning models that take large-scale wind-model predictions as input and are trained on historical forecast values with matching measured wind values at this location.
    The local models are calibrated to local data to improve prediction performance by learning systematic local deviations from the large-scale model.
    One model is dedicated to the remaining part of the current day and gives strong weight to the most recent measured wind updates.
    A second model is dedicated to next-day (day-ahead) prediction.
    Models are retrained daily, next-day/current-day prediction lines are refreshed hourly during daytime, and measured-wind updates on the current-day plot are refreshed every 6 minutes.
  </p>
  <div class="grid">
    <div class="card">
      <h2>Current-day prediction</h2>
      <p class="desc">Measured wind speed up to now, plus the latest Harmonie and super-local prediction for the remaining hours of today.</p>
      <picture>
        <source media="(max-width: 768px)" srcset="{current_day_mobile_src}">
        <img src="current_day_predictions.png?v={cache_bust}" alt="Current day prediction">
      </picture>
    </div>
    <div class="card">
      <h2>Next-day prediction</h2>
      <p class="desc">Day-ahead forecast for tomorrow: Harmonie baseline versus the super-local model for wind speed and direction.</p>
      <picture>
        <source media="(max-width: 768px)" srcset="{next_day_mobile_src}">
        <img src="next_day_predictions.png?v={cache_bust}" alt="Next day prediction">
      </picture>
    </div>
    <div class="card">
      <h2>Day-ahead historical performance</h2>
      <p class="desc">Historical model performance: top panel shows daily mean wind speed, bottom panel shows day-ahead Mean Absolute Error (MAE) for Harmonie and super-local predictions.</p>
      <picture>
        <source media="(max-width: 768px)" srcset="{daily_mae_mobile_src}">
        <img src="daily_mae_history.png?v={cache_bust}" alt="Day-ahead MAE history">
      </picture>
    </div>
  </div>
  <p class="overview overview-mobile">
    <strong>What is the super local forecast?</strong> This dashboard combines two local machine learning models that take large-scale wind-model predictions as input and are trained on historical forecast values with matching measured wind values at this location.
    The local models are calibrated to local data to improve prediction performance by learning systematic local deviations from the large-scale model.
    One model is dedicated to the remaining part of the current day and gives strong weight to the most recent measured wind updates.
    A second model is dedicated to next-day (day-ahead) prediction.
    Models are retrained daily, next-day/current-day prediction lines are refreshed hourly during daytime, and measured-wind updates on the current-day plot are refreshed every 6 minutes.
  </p>
</body>
</html>
"""
    index_path = web_out_dir / "index.html"
    index_path.write_text(html, encoding="utf-8")
    copied["index.html"] = str(index_path)
    return copied


def auto_push_dashboard_changes(
    repo_root: Path,
    web_out_dir: Path,
    remote: str,
    branch: str,
) -> dict:
    if not (repo_root / ".git").exists():
        return {"enabled": True, "pushed": False, "reason": "not_a_git_repo"}

    try:
        rel_web_dir = web_out_dir.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return {"enabled": True, "pushed": False, "reason": "web_out_dir_outside_repo"}

    rel_web_dir_s = str(rel_web_dir)
    try:
        status = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain", "--", rel_web_dir_s],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        return {"enabled": True, "pushed": False, "reason": f"git_status_failed:{exc.returncode}"}
    if status.stdout.strip() == "":
        return {"enabled": True, "pushed": False, "reason": "no_changes"}

    # Safety: if the index already contains staged files outside dashboard path, don't auto-commit.
    try:
        staged_all = subprocess.run(
            ["git", "-C", str(repo_root), "diff", "--cached", "--name-only"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        return {"enabled": True, "pushed": False, "reason": f"git_staged_check_failed:{exc.returncode}"}
    staged_paths = [p.strip() for p in staged_all.stdout.splitlines() if p.strip()]
    outside = [
        p
        for p in staged_paths
        if not (p == rel_web_dir_s or p.startswith(rel_web_dir_s + "/"))
    ]
    if outside:
        return {
            "enabled": True,
            "pushed": False,
            "reason": "staged_changes_outside_web_dir",
            "outside_count": len(outside),
            "outside_sample": outside[:5],
        }

    try:
        subprocess.run(["git", "-C", str(repo_root), "add", "--", rel_web_dir_s], check=True)
    except subprocess.CalledProcessError as exc:
        return {"enabled": True, "pushed": False, "reason": f"git_add_failed:{exc.returncode}"}

    try:
        staged = subprocess.run(
            ["git", "-C", str(repo_root), "diff", "--cached", "--name-only", "--", rel_web_dir_s],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        return {"enabled": True, "pushed": False, "reason": f"git_staged_web_check_failed:{exc.returncode}"}
    if staged.stdout.strip() == "":
        return {"enabled": True, "pushed": False, "reason": "nothing_staged"}

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    commit_msg = f"auto: update dashboard artifacts ({stamp})"
    try:
        subprocess.run(["git", "-C", str(repo_root), "commit", "-m", commit_msg, "--", rel_web_dir_s], check=True)
    except subprocess.CalledProcessError as exc:
        return {"enabled": True, "pushed": False, "reason": f"git_commit_failed:{exc.returncode}"}
    try:
        subprocess.run(["git", "-C", str(repo_root), "push", remote, branch], check=True)
    except subprocess.CalledProcessError as exc:
        return {"enabled": True, "pushed": False, "reason": f"git_push_failed:{exc.returncode}"}

    try:
        commit_hash = (
            subprocess.run(
                ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
        )
    except subprocess.CalledProcessError:
        commit_hash = "unknown"
    return {
        "enabled": True,
        "pushed": True,
        "reason": "",
        "remote": remote,
        "branch": branch,
        "commit": commit_hash,
        "path": rel_web_dir_s,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(args.db).resolve()

    cfg = DatasetConfig(
        site=args.site,
        model=args.model,
        window_hours=args.window_hours,
        target_hours=args.target_hours,
    )
    device = pick_torch_device()
    refresh_info = {
        "need_refresh": False,
        "refreshed": False,
        "reason": "skipped",
        "latest_run_before_utc": None,
        "latest_run_after_utc": None,
    }
    if not args.skip_data_refresh_check:
        refresh_info = ensure_fresh_source_data(
            db_path=db_path,
            site=args.site,
            model=args.model,
            max_forecast_age_hours=args.max_forecast_age_hours,
            expected_update_hour_utc=args.expected_update_hour_utc,
        )
        print(
            "Source freshness check:",
            json.dumps(
                {
                    "need_refresh": refresh_info["need_refresh"],
                    "refreshed": refresh_info["refreshed"],
                    "reason": refresh_info["reason"],
                    "latest_run_after_utc": refresh_info["latest_run_after_utc"],
                }
            ),
        )

    speed_model_path = out_dir / "next_day_lstm_speed_residual.pt"
    direction_model_path = out_dir / "next_day_lstm_direction_residual.pt"
    intraday_model_path = out_dir / "intraday_speed_residual.pt"
    speed_scalers_path = {
        "x_mean": out_dir / "x_mean_speed.npy",
        "x_std": out_dir / "x_std_speed.npy",
        "y_mean": out_dir / "y_mean_speed.npy",
        "y_std": out_dir / "y_std_speed.npy",
    }
    direction_scalers_path = {
        "x_mean": out_dir / "x_mean_direction.npy",
        "x_std": out_dir / "x_std_direction.npy",
        "y_mean": out_dir / "y_mean_direction.npy",
        "y_std": out_dir / "y_std_direction.npy",
    }
    intraday_hparams = {}
    model_selection_report: dict = {"enabled": False, "reason": "skip_training"}
    gate_eval_details_csv_src: Path | None = None
    speed_calibration: dict | None = None

    if args.skip_training:
        for p in [
            speed_model_path,
            direction_model_path,
            intraday_model_path,
            *speed_scalers_path.values(),
            *direction_scalers_path.values(),
        ]:
            if not p.exists():
                raise FileNotFoundError(f"Missing artifact for --skip-training mode: {p}")
        speed_model, speed_ckpt = _load_model(speed_model_path, device)
        direction_model, direction_ckpt = _load_model(direction_model_path, device)
        intraday_bundle, intraday_ckpt = load_intraday_model(intraday_model_path, device)
        speed_arrays = {k: np.load(v) for k, v in speed_scalers_path.items()}
        direction_arrays = {k: np.load(v) for k, v in direction_scalers_path.items()}
        speed_target_mode = str(speed_ckpt.get("target_mode", "residual")).strip().lower()
        direction_target_mode = str(direction_ckpt.get("target_mode", "residual")).strip().lower()
        speed_constraint_eps = speed_ckpt.get("constraint_eps", None)
        speed_calibration = speed_ckpt.get("speed_regime_calibration")
        model_last_trained_at_utc = _resolve_model_trained_utc(speed_ckpt, speed_model_path)
        speed_train_stats = None
        direction_train_stats = None
        n_samples_all_speed = None
        n_samples_all_direction = None
        feature_cols = ["forecast_avg", "forecast_max", "forecast_dir", "month_sin", "month_cos"]
        intraday_train_stats = None
        intraday_hparams = {
            "hidden1": intraday_ckpt.get("hidden1"),
            "hidden2": intraday_ckpt.get("hidden2"),
            "dropout": intraday_ckpt.get("dropout"),
            "learning_rate": intraday_ckpt.get("learning_rate"),
            "recency_power": intraday_ckpt.get("recency_power"),
        }
        model_selection_report = {"enabled": False, "reason": "skip_training"}
    else:
        if not (0.0 < float(args.challenge_eval_split) < 0.5):
            raise ValueError("--challenge-eval-split must be > 0 and < 0.5.")

        speed_target_mode = "constrained_logratio"
        direction_target_mode = "residual"
        speed_constraint_eps = float(args.speed_constraint_eps)
        promotion_margin = max(0.0, float(args.promotion_margin_pct)) / 100.0

        # Build full arrays once, then split chronologically into train/eval holdouts.
        speed_arrays_full = build_all_training_arrays(db_path, cfg, target_mode="residual")
        direction_arrays_full = build_all_direction_training_arrays(db_path, cfg)
        n_samples_all_speed = int(speed_arrays_full["X_all"].shape[0])
        n_samples_all_direction = int(direction_arrays_full["X_all"].shape[0])
        feature_cols = speed_arrays_full["feature_cols"]

        speed_X_raw_all = _inverse_standardizer(
            speed_arrays_full["X_all"], speed_arrays_full["x_mean"], speed_arrays_full["x_std"]
        ).astype(np.float32)
        direction_X_raw_all = _inverse_standardizer(
            direction_arrays_full["X_all"], direction_arrays_full["x_mean"], direction_arrays_full["x_std"]
        ).astype(np.float32)
        speed_y_raw_all = np.log(speed_arrays_full["y_actual_all_raw"] + speed_constraint_eps) - np.log(
            speed_arrays_full["y_forecast_all_raw"] + speed_constraint_eps
        )
        direction_y_raw_all = (
            direction_arrays_full["y_all"] * float(direction_arrays_full["y_std"][0])
            + float(direction_arrays_full["y_mean"][0])
        ).astype(np.float32)

        speed_eval_start = _eval_start_index(
            n_samples_all_speed, args.challenge_eval_split, args.challenge_min_eval_samples
        )
        direction_eval_start = _eval_start_index(
            n_samples_all_direction, args.challenge_eval_split, args.challenge_min_eval_samples
        )

        speed_X_train_raw, speed_X_eval_raw = speed_X_raw_all[:speed_eval_start], speed_X_raw_all[speed_eval_start:]
        speed_y_train_raw, speed_y_eval_raw = speed_y_raw_all[:speed_eval_start], speed_y_raw_all[speed_eval_start:]
        speed_actual_train = speed_arrays_full["y_actual_all_raw"][:speed_eval_start]
        speed_actual_eval = speed_arrays_full["y_actual_all_raw"][speed_eval_start:]
        speed_forecast_train = speed_arrays_full["y_forecast_all_raw"][:speed_eval_start]
        speed_forecast_eval = speed_arrays_full["y_forecast_all_raw"][speed_eval_start:]
        speed_train_anchor_times_utc = pd.to_datetime(speed_arrays_full["timestamps"][:speed_eval_start], utc=True)
        speed_eval_anchor_times_utc = pd.to_datetime(speed_arrays_full["timestamps"][speed_eval_start:], utc=True)
        speed_train_calibration_context = _build_speed_calibration_context(
            anchor_dir_deg=speed_X_train_raw[:, -1, 2],
            target_times_utc=speed_train_anchor_times_utc + pd.Timedelta(hours=1),
        )
        speed_eval_calibration_context = _build_speed_calibration_context(
            anchor_dir_deg=speed_X_eval_raw[:, -1, 2],
            target_times_utc=speed_eval_anchor_times_utc + pd.Timedelta(hours=1),
        )

        direction_X_train_raw = direction_X_raw_all[:direction_eval_start]
        direction_X_eval_raw = direction_X_raw_all[direction_eval_start:]
        direction_y_train_raw = direction_y_raw_all[:direction_eval_start]
        direction_actual_eval = direction_arrays_full["y_actual_all_raw"][direction_eval_start:]
        direction_forecast_eval = direction_arrays_full["y_forecast_all_raw"][direction_eval_start:]

        # Fit challenger scalers on the pre-evaluation training chunk only.
        speed_x_mean, speed_x_std = _fit_standardizer(speed_X_train_raw)
        speed_y_mean, speed_y_std = _fit_target_scaler(speed_y_train_raw)
        speed_X_train = _apply_standardizer(speed_X_train_raw, speed_x_mean, speed_x_std).astype(np.float32)
        speed_X_eval = _apply_standardizer(speed_X_eval_raw, speed_x_mean, speed_x_std).astype(np.float32)
        speed_y_train = ((speed_y_train_raw - speed_y_mean) / speed_y_std).astype(np.float32)
        challenger_speed_arrays = {
            "x_mean": speed_x_mean.astype(np.float32),
            "x_std": speed_x_std.astype(np.float32),
            "y_mean": np.array([speed_y_mean], dtype=np.float32),
            "y_std": np.array([speed_y_std], dtype=np.float32),
        }

        direction_x_mean, direction_x_std = _fit_standardizer(direction_X_train_raw)
        direction_y_mean, direction_y_std = _fit_target_scaler(direction_y_train_raw)
        direction_X_train = _apply_standardizer(direction_X_train_raw, direction_x_mean, direction_x_std).astype(np.float32)
        direction_X_eval = _apply_standardizer(direction_X_eval_raw, direction_x_mean, direction_x_std).astype(np.float32)
        direction_y_train = ((direction_y_train_raw - direction_y_mean) / direction_y_std).astype(np.float32)
        challenger_direction_arrays = {
            "x_mean": direction_x_mean.astype(np.float32),
            "x_std": direction_x_std.astype(np.float32),
            "y_mean": np.array([direction_y_mean], dtype=np.float32),
            "y_std": np.array([direction_y_std], dtype=np.float32),
        }

        # Train challengers from scratch.
        speed_model_challenger = NextDayLSTM(
            n_features=speed_X_train.shape[2],
            target_hours=speed_y_train.shape[1],
            output_activation="linear",
        ).to(device)
        speed_model_challenger, speed_train_stats = train_with_validation(
            model=speed_model_challenger,
            X_all=speed_X_train,
            y_all=speed_y_train,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_split=args.validation_split,
            model_label="speed",
            device=device,
        )
        speed_calibration_start = _validation_start_index(len(speed_X_train), args.validation_split)
        challenger_speed_calibration = fit_speed_regime_calibration(
            pred_speed=_predict_speed_batch(
                speed_model_challenger,
                speed_X_train[speed_calibration_start:],
                speed_forecast_train[speed_calibration_start:],
                y_mean=float(challenger_speed_arrays["y_mean"][0]),
                y_std=float(challenger_speed_arrays["y_std"][0]),
                target_mode=speed_target_mode,
                constraint_eps=speed_constraint_eps,
                speed_calibration=None,
                speed_calibration_context=None,
                device=device,
            ),
            forecast_speed=speed_forecast_train[speed_calibration_start:],
            actual_speed=speed_actual_train[speed_calibration_start:],
            speed_calibration_context={
                "anchor_dir_deg": speed_train_calibration_context["anchor_dir_deg"][speed_calibration_start:],
                "target_month": speed_train_calibration_context["target_month"][speed_calibration_start:],
            },
            signal="pred_max",
        )

        direction_model_challenger = NextDayLSTM(
            n_features=direction_X_train.shape[2],
            target_hours=direction_y_train.shape[1],
            output_activation="linear",
        ).to(device)
        direction_model_challenger, direction_train_stats = train_with_validation(
            model=direction_model_challenger,
            X_all=direction_X_train,
            y_all=direction_y_train,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_split=args.validation_split,
            model_label="direction",
            device=device,
        )

        intraday_bundle, intraday_train_stats = train_intraday_model(
            db_path=db_path,
            cfg=cfg,
            device=device,
            epochs=int(args.intraday_epochs),
            batch_size=args.batch_size,
            validation_split=args.validation_split,
            hidden1=int(args.intraday_hidden1),
            hidden2=int(args.intraday_hidden2),
            dropout=float(args.intraday_dropout),
            learning_rate=float(args.intraday_learning_rate),
            recency_power=float(args.intraday_recency_power),
        )

        champion_available = all(
            p.exists()
            for p in [
                speed_model_path,
                direction_model_path,
                *speed_scalers_path.values(),
                *direction_scalers_path.values(),
            ]
        )
        promote_speed = True
        promote_direction = True
        champion_speed_mae = None
        challenger_speed_eval_pred = _predict_speed_batch(
            speed_model_challenger,
            speed_X_eval,
            speed_forecast_eval,
            y_mean=float(challenger_speed_arrays["y_mean"][0]),
            y_std=float(challenger_speed_arrays["y_std"][0]),
            target_mode=speed_target_mode,
            constraint_eps=speed_constraint_eps,
            speed_calibration=challenger_speed_calibration,
            speed_calibration_context=speed_eval_calibration_context,
            device=device,
        )
        challenger_speed_mae = float(np.mean(np.abs(challenger_speed_eval_pred - speed_actual_eval)))
        challenger_direction_mae = _angular_mae_deg(
            _predict_direction_batch(
                direction_model_challenger,
                direction_X_eval,
                direction_forecast_eval,
                y_mean=float(challenger_direction_arrays["y_mean"][0]),
                y_std=float(challenger_direction_arrays["y_std"][0]),
                device=device,
            ),
            direction_actual_eval,
        )
        now_train_utc = datetime.now(timezone.utc).isoformat()
        challenger_speed_model_id = _format_model_id(now_train_utc, args.local_timezone)
        champion_speed_model_id = "none"
        champion_speed_ckpt = {}
        champion_direction_ckpt = {}
        champion_speed_calibration = None
        champion_speed_eval_pred = np.full_like(challenger_speed_eval_pred, np.nan, dtype=np.float32)
        if champion_available:
            champion_speed_model, champion_speed_ckpt = _load_model(speed_model_path, device)
            champion_direction_model, champion_direction_ckpt = _load_model(direction_model_path, device)
            champion_speed_arrays = {k: np.load(v) for k, v in speed_scalers_path.items()}
            champion_direction_arrays = {k: np.load(v) for k, v in direction_scalers_path.items()}
            champion_speed_mode = str(champion_speed_ckpt.get("target_mode", "residual")).strip().lower()
            champion_speed_eps = champion_speed_ckpt.get("constraint_eps", None)
            champion_speed_calibration = champion_speed_ckpt.get("speed_regime_calibration")
            champion_speed_model_id = _format_model_id(
                _resolve_model_trained_utc(champion_speed_ckpt, speed_model_path),
                args.local_timezone,
            )

            speed_X_eval_champion = _apply_standardizer(
                speed_X_eval_raw, champion_speed_arrays["x_mean"], champion_speed_arrays["x_std"]
            ).astype(np.float32)
            direction_X_eval_champion = _apply_standardizer(
                direction_X_eval_raw, champion_direction_arrays["x_mean"], champion_direction_arrays["x_std"]
            ).astype(np.float32)

            champion_speed_eval_pred = _predict_speed_batch(
                champion_speed_model,
                speed_X_eval_champion,
                speed_forecast_eval,
                y_mean=float(champion_speed_arrays["y_mean"][0]),
                y_std=float(champion_speed_arrays["y_std"][0]),
                target_mode=champion_speed_mode,
                constraint_eps=champion_speed_eps,
                speed_calibration=champion_speed_calibration,
                speed_calibration_context=speed_eval_calibration_context,
                device=device,
            )
            champion_speed_mae = float(np.mean(np.abs(champion_speed_eval_pred - speed_actual_eval)))
            champion_direction_mae = _angular_mae_deg(
                _predict_direction_batch(
                    champion_direction_model,
                    direction_X_eval_champion,
                    direction_forecast_eval,
                    y_mean=float(champion_direction_arrays["y_mean"][0]),
                    y_std=float(champion_direction_arrays["y_std"][0]),
                    device=device,
                ),
                direction_actual_eval,
            )

            promote_speed = challenger_speed_mae <= champion_speed_mae * (1.0 - promotion_margin)
            promote_direction = challenger_direction_mae <= champion_direction_mae * (1.0 - promotion_margin)
            model_selection_report = {
                "enabled": True,
                "holdout_eval_split": float(args.challenge_eval_split),
                "holdout_eval_min_samples": int(args.challenge_min_eval_samples),
                "promotion_margin_pct": float(args.promotion_margin_pct),
                "speed_eval_samples": int(len(speed_X_eval)),
                "direction_eval_samples": int(len(direction_X_eval)),
                "speed_mae_champion": float(champion_speed_mae),
                "speed_mae_challenger": float(challenger_speed_mae),
                "speed_regime_calibration_challenger": challenger_speed_calibration,
                "speed_regime_calibration_champion": champion_speed_calibration,
                "direction_mae_champion": float(champion_direction_mae),
                "direction_mae_challenger": float(challenger_direction_mae),
                "promote_speed": bool(promote_speed),
                "promote_direction": bool(promote_direction),
                "speed_model_id_champion": champion_speed_model_id,
                "speed_model_id_challenger": challenger_speed_model_id,
            }
            print(
                f"Model gate | speed MAE champion={champion_speed_mae:.4f}, challenger={challenger_speed_mae:.4f}, "
                f"promoted={promote_speed}"
            )
            print(
                f"Model gate | direction MAE champion={champion_direction_mae:.4f}, "
                f"challenger={challenger_direction_mae:.4f}, promoted={promote_direction}"
            )
        else:
            model_selection_report = {
                "enabled": True,
                "reason": "no_existing_champion",
                "speed_eval_samples": int(len(speed_X_eval)),
                "direction_eval_samples": int(len(direction_X_eval)),
                "speed_mae_challenger": float(challenger_speed_mae),
                "speed_regime_calibration_challenger": challenger_speed_calibration,
                "direction_mae_challenger": float(challenger_direction_mae),
                "promote_speed": True,
                "promote_direction": True,
                "speed_model_id_champion": champion_speed_model_id,
                "speed_model_id_challenger": challenger_speed_model_id,
            }
            print("Model gate | no existing champion found, promoting challenger models.")

        # Select production speed model.
        if promote_speed:
            speed_model = speed_model_challenger
            speed_arrays = challenger_speed_arrays
            speed_calibration = challenger_speed_calibration
            model_last_trained_at_utc = now_train_utc
            _save_model(
                speed_model_path,
                speed_model,
                n_features=speed_X_train.shape[2],
                target_hours=speed_y_train.shape[1],
                target_name="wind_speed",
                target_mode=speed_target_mode,
                output_activation="linear",
                extra={
                    "constraint_eps": speed_constraint_eps,
                    "trained_at_utc": now_train_utc,
                    "speed_regime_calibration": speed_calibration,
                },
            )
            np.save(speed_scalers_path["x_mean"], speed_arrays["x_mean"])
            np.save(speed_scalers_path["x_std"], speed_arrays["x_std"])
            np.save(speed_scalers_path["y_mean"], speed_arrays["y_mean"])
            np.save(speed_scalers_path["y_std"], speed_arrays["y_std"])
            model_selection_report["speed_selected"] = "challenger"
        else:
            speed_model = champion_speed_model
            speed_arrays = champion_speed_arrays
            speed_target_mode = str(champion_speed_ckpt.get("target_mode", "residual")).strip().lower()
            speed_constraint_eps = champion_speed_ckpt.get("constraint_eps", None)
            speed_calibration = champion_speed_calibration
            model_last_trained_at_utc = _resolve_model_trained_utc(champion_speed_ckpt, speed_model_path)
            model_selection_report["speed_selected"] = "champion"

        # Select production direction model.
        if promote_direction:
            direction_model = direction_model_challenger
            direction_arrays = challenger_direction_arrays
            _save_model(
                direction_model_path,
                direction_model,
                n_features=direction_X_train.shape[2],
                target_hours=direction_y_train.shape[1],
                target_name="wind_direction",
                target_mode=direction_target_mode,
                output_activation="linear",
                extra={"trained_at_utc": now_train_utc},
            )
            np.save(direction_scalers_path["x_mean"], direction_arrays["x_mean"])
            np.save(direction_scalers_path["x_std"], direction_arrays["x_std"])
            np.save(direction_scalers_path["y_mean"], direction_arrays["y_mean"])
            np.save(direction_scalers_path["y_std"], direction_arrays["y_std"])
            model_selection_report["direction_selected"] = "challenger"
        else:
            direction_model = champion_direction_model
            direction_arrays = champion_direction_arrays
            direction_target_mode = str(champion_direction_ckpt.get("target_mode", "residual")).strip().lower()
            model_selection_report["direction_selected"] = "champion"

        print(
            "Production selection | "
            f"speed={model_selection_report.get('speed_selected')} | "
            f"direction={model_selection_report.get('direction_selected')}"
        )

        # Save full holdout-period speed predictions for challenger vs champion comparisons.
        n_eval, horizon = challenger_speed_eval_pred.shape
        if n_eval > 0 and horizon > 0:
            details_dir = out_dir / "model_gate_eval_details"
            details_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            gate_eval_details_csv = details_dir / f"{stamp}_model_gate_eval_speed.csv"

            anchor_ns = speed_eval_anchor_times_utc.astype("int64").to_numpy()
            target_ns = (
                np.repeat(anchor_ns, horizon)
                + np.tile(np.arange(1, horizon + 1, dtype=np.int64), n_eval) * 3_600_000_000_000
            )
            raw_eval = pd.DataFrame(
                {
                    "target_time_utc": pd.to_datetime(target_ns, utc=True),
                    "actual_wind_speed": speed_actual_eval.reshape(-1).astype(float),
                    "forecast_wind_speed": speed_forecast_eval.reshape(-1).astype(float),
                    "challenger_wind_speed": challenger_speed_eval_pred.reshape(-1).astype(float),
                    "champion_wind_speed": champion_speed_eval_pred.reshape(-1).astype(float),
                }
            )
            agg_eval = raw_eval.groupby("target_time_utc", as_index=False).mean(numeric_only=True)
            agg_eval["n_overlaps"] = raw_eval.groupby("target_time_utc").size().to_numpy()
            agg_eval["abs_err_challenger"] = np.abs(agg_eval["challenger_wind_speed"] - agg_eval["actual_wind_speed"])
            agg_eval["abs_err_champion"] = np.abs(agg_eval["champion_wind_speed"] - agg_eval["actual_wind_speed"])
            agg_eval["target_time_utc"] = pd.to_datetime(agg_eval["target_time_utc"], utc=True).dt.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            agg_eval.to_csv(gate_eval_details_csv, index=False)
            gate_eval_details_csv_src = gate_eval_details_csv
            model_selection_report["speed_eval_details_csv"] = str(gate_eval_details_csv)

        save_intraday_model(
            intraday_model_path,
            intraday_bundle,
            extra={
                "trained_at_utc": now_train_utc,
                "hidden1": int(args.intraday_hidden1),
                "hidden2": int(args.intraday_hidden2),
                "dropout": float(args.intraday_dropout),
                "learning_rate": float(args.intraday_learning_rate),
                "recency_power": float(args.intraday_recency_power),
            },
        )
        intraday_hparams = {
            "hidden1": int(args.intraday_hidden1),
            "hidden2": int(args.intraday_hidden2),
            "dropout": float(args.intraday_dropout),
            "learning_rate": float(args.intraday_learning_rate),
            "recency_power": float(args.intraday_recency_power),
        }

    inference_input_speed = build_next_day_inference_input(
        db_path=db_path,
        cfg=cfg,
        x_mean=speed_arrays["x_mean"],
        x_std=speed_arrays["x_std"],
    )
    inference_input_direction = build_next_day_inference_input(
        db_path=db_path,
        cfg=cfg,
        x_mean=direction_arrays["x_mean"],
        x_std=direction_arrays["x_std"],
    )
    speed_inference_calibration_context = _build_speed_calibration_context(
        anchor_dir_deg=float(inference_input_speed["anchor_forecast_dir"]),
        target_times_utc=pd.to_datetime([inference_input_speed["target_times"][0]], utc=True),
    )

    speed_pred = predict_speed(
        model=speed_model,
        X_input=inference_input_speed["X_input"],
        forecast_speed=inference_input_speed["forecast_next24"],
        y_mean=float(speed_arrays["y_mean"][0]),
        y_std=float(speed_arrays["y_std"][0]),
        target_mode=speed_target_mode,
        constraint_eps=speed_constraint_eps,
        speed_calibration=speed_calibration,
        speed_calibration_context=speed_inference_calibration_context,
        device=device,
    )
    direction_pred = predict_direction_residual(
        model=direction_model,
        X_input=inference_input_direction["X_input"],
        forecast_dir=inference_input_direction["forecast_dir_next24"],
        y_mean=float(direction_arrays["y_mean"][0]),
        y_std=float(direction_arrays["y_std"][0]),
        device=device,
    )

    table = build_prediction_table(inference_input_speed, speed_pred, direction_pred)

    table_path = out_dir / "next_day_predictions.csv"
    table_for_csv = table[
        [
            "target_time_utc",
            "hour_utc",
            "forecast_wind_speed",
            "lstm_pred_wind_speed",
            "delta_speed_lstm_minus_forecast",
            "forecast_wind_dir_deg",
            "lstm_pred_wind_dir_deg",
            "delta_dir_lstm_minus_forecast",
        ]
    ].copy()
    table_for_csv["target_time_utc"] = table_for_csv["target_time_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    table_for_csv.to_csv(table_path, index=False)
    dayahead_snapshot_csv = None
    if not args.skip_training and args.test_now_local_hour is None:
        dayahead_snapshot_csv = save_dayahead_snapshot(
            out_dir=out_dir,
            table=table,
            local_tz=args.local_timezone,
            prediction_generated_at_utc=datetime.now(timezone.utc).isoformat(),
        )

    plot_path = out_dir / "next_day_predictions.png"
    plot_path_mobile = out_dir / "next_day_predictions_mobile.png"
    prediction_generated_at_utc = datetime.now(timezone.utc).isoformat()
    prediction_update_local = datetime.now(ZoneInfo(args.local_timezone)).replace(minute=0, second=0, microsecond=0)
    prediction_updated_at_utc = prediction_update_local.astimezone(timezone.utc).isoformat()
    save_prediction_plot(
        table,
        plot_path,
        local_tz=args.local_timezone,
        prediction_generated_at_utc=prediction_generated_at_utc,
        prediction_updated_at_utc=prediction_updated_at_utc,
        model_trained_at_utc=model_last_trained_at_utc,
    )
    save_prediction_plot(
        table,
        plot_path_mobile,
        local_tz=args.local_timezone,
        prediction_generated_at_utc=prediction_generated_at_utc,
        prediction_updated_at_utc=prediction_updated_at_utc,
        model_trained_at_utc=model_last_trained_at_utc,
        mobile=True,
    )

    is_test_mode = args.test_now_local_hour is not None
    test_suffix = f"_test_hour_{int(args.test_now_local_hour):02d}" if is_test_mode else ""
    current_day_prior_prediction_tables: list[pd.DataFrame] = []
    current_day_fixed_origin_mae: list[dict] = []
    current_day_target_local = datetime.now(ZoneInfo(args.local_timezone)).date()
    latest_prior_prediction_table = None
    if not is_test_mode:
        current_day_prior_prediction_tables = load_current_day_prediction_history(
            out_dir=out_dir,
            target_day_local=current_day_target_local,
            local_tz=args.local_timezone,
            max_snapshots=16,
        )
        latest_prior_prediction_table = load_latest_current_day_snapshot(
            out_dir=out_dir,
            target_day_local=current_day_target_local,
            local_tz=args.local_timezone,
        )

    # --- Current day plot/table: actuals up to present + prediction for remaining day ---
    current_day_table = build_current_day_table(
        db_path=db_path,
        cfg=cfg,
        intraday_bundle=intraday_bundle,
        speed_model=speed_model,
        direction_model=direction_model,
        speed_scalers={"x_mean": speed_arrays["x_mean"], "x_std": speed_arrays["x_std"], "y_mean": speed_arrays["y_mean"], "y_std": speed_arrays["y_std"]},
        direction_scalers={
            "x_mean": direction_arrays["x_mean"],
            "x_std": direction_arrays["x_std"],
            "y_mean": direction_arrays["y_mean"],
            "y_std": direction_arrays["y_std"],
        },
        speed_target_mode=speed_target_mode,
        speed_constraint_eps=speed_constraint_eps,
        local_tz=args.local_timezone,
        latest_prior_prediction_table=latest_prior_prediction_table,
        test_now_local_hour=args.test_now_local_hour,
        current_day_interval_minutes=args.current_day_interval_minutes,
        device=device,
    )

    current_day_table_path = out_dir / f"current_day_predictions{test_suffix}.csv"
    current_day_table_csv = current_day_table.copy()
    current_day_table_csv["time_local"] = current_day_table_csv["time_local"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    current_day_table_csv.to_csv(current_day_table_path, index=False)
    current_day_snapshot_csv = None
    if not is_test_mode and not current_day_table.empty:
        current_day_target_local = pd.to_datetime(current_day_table["time_local"]).dt.tz_convert(ZoneInfo(args.local_timezone)).iloc[0].date()
        conn = sqlite3.connect(str(db_path))
        try:
            actual_raw_utc = _load_observations_raw(conn, cfg.site)
        finally:
            conn.close()
        actual_measurements = actual_raw_utc.tz_convert(ZoneInfo(args.local_timezone)).reset_index().rename(
            columns={"obs_dt": "time_local", "actual_avg": "actual_wind_speed"}
        )
        actual_measurements = actual_measurements[
            actual_measurements["time_local"].dt.date.eq(current_day_target_local)
        ][["time_local", "actual_wind_speed"]].copy()
        current_day_fixed_origin_mae = compute_fixed_origin_current_day_mae(
            current_table=current_day_table,
            prior_prediction_tables=current_day_prior_prediction_tables,
            actual_measurements=actual_measurements,
            lookback_hours=(1, 3, 6),
        )

    current_day_plot_path = out_dir / f"current_day_predictions{test_suffix}.png"
    current_day_plot_mobile_path = out_dir / f"current_day_predictions{test_suffix}_mobile.png"
    save_current_day_plot(
        current_day_table,
        current_day_plot_path,
        args.local_timezone,
        prediction_generated_at_utc=prediction_generated_at_utc,
        prediction_updated_at_utc=prediction_updated_at_utc,
        model_trained_at_utc=model_last_trained_at_utc,
        prior_prediction_tables=current_day_prior_prediction_tables,
        fixed_origin_mae_metrics=current_day_fixed_origin_mae,
    )
    save_current_day_plot(
        current_day_table,
        current_day_plot_mobile_path,
        args.local_timezone,
        prediction_generated_at_utc=prediction_generated_at_utc,
        prediction_updated_at_utc=prediction_updated_at_utc,
        model_trained_at_utc=model_last_trained_at_utc,
        prior_prediction_tables=current_day_prior_prediction_tables,
        fixed_origin_mae_metrics=current_day_fixed_origin_mae,
        mobile=True,
    )
    if not is_test_mode:
        current_day_snapshot_csv = save_current_day_snapshot(
            out_dir=out_dir,
            table=current_day_table,
            local_tz=args.local_timezone,
            prediction_generated_at_utc=prediction_generated_at_utc,
        )
    archived_current_day_plot = None
    archived_next_day_plots: dict[str, str] | None = None
    daily_mae_csv = None
    daily_mae_png = None
    gate_eval_history_csv = out_dir / "model_gate_eval_history.csv"
    gate_eval_history_png = out_dir / "model_gate_eval_history.png"
    gate_eval_history_csv_src: Path | None = None
    gate_eval_history_png_src: Path | None = None
    if not args.skip_training:
        appended_gate_csv = append_model_gate_eval_history(
            gate_eval_history_csv,
            model_selection_gate=model_selection_report,
            local_tz=args.local_timezone,
        )
        if appended_gate_csv is not None:
            gate_eval_history_csv_src = Path(appended_gate_csv)
            save_model_gate_eval_history_plot(
                gate_eval_history_csv_src,
                gate_eval_history_png,
                local_tz=args.local_timezone,
                eval_details_csv=gate_eval_details_csv_src,
            )
            gate_eval_history_png_src = gate_eval_history_png
    if not is_test_mode:
        archived_next_day_plots = maybe_archive_next_day_plots(
            next_day_plot_path=plot_path,
            next_day_plot_mobile_path=plot_path_mobile,
            out_dir=out_dir,
            local_tz=args.local_timezone,
            test_now_local_hour=args.test_now_local_hour,
        )
        archived_current_day_plot = maybe_archive_current_day_plot(
            current_day_plot_path=current_day_plot_path,
            out_dir=out_dir,
            local_tz=args.local_timezone,
            test_now_local_hour=args.test_now_local_hour,
        )
        daily_mae_csv, daily_mae_png = maybe_save_daily_mae_dayahead(
            out_dir=out_dir,
            db_path=db_path,
            cfg=cfg,
            local_tz=args.local_timezone,
            test_now_local_hour=args.test_now_local_hour,
        )

    web_publish = None
    git_publish = {"enabled": bool(args.git_auto_push_pages), "pushed": False, "reason": "disabled"}
    if not is_test_mode:
        daily_mae_png_src = None if daily_mae_png is None else Path(daily_mae_png)
        daily_mae_png_mobile_src: Path | None = None
        daily_mae_csv_src = None if daily_mae_csv is None else Path(daily_mae_csv)
        if daily_mae_png_src is None:
            fallback_png = out_dir / "daily_mae_history.png"
            if fallback_png.exists():
                daily_mae_png_src = fallback_png
        if daily_mae_png_mobile_src is None:
            fallback_mobile_png = out_dir / "daily_mae_history_mobile.png"
            if fallback_mobile_png.exists():
                daily_mae_png_mobile_src = fallback_mobile_png
        if daily_mae_csv_src is None:
            fallback_csv = out_dir / "daily_mae_history.csv"
            if fallback_csv.exists():
                daily_mae_csv_src = fallback_csv
        # Always refresh daily MAE plot from CSV when available, not only after end-of-day save.
        if daily_mae_csv_src is not None:
            daily_mae_png_refresh = out_dir / "daily_mae_history.png"
            save_daily_mae_plot(daily_mae_csv_src, daily_mae_png_refresh, local_tz=args.local_timezone)
            daily_mae_png_src = daily_mae_png_refresh
            daily_mae_png_mobile_refresh = out_dir / "daily_mae_history_mobile.png"
            save_daily_mae_plot(
                daily_mae_csv_src,
                daily_mae_png_mobile_refresh,
                local_tz=args.local_timezone,
                mobile_last_months=3,
            )
            daily_mae_png_mobile_src = daily_mae_png_mobile_refresh
        if gate_eval_history_csv_src is None and gate_eval_history_csv.exists():
            gate_eval_history_csv_src = gate_eval_history_csv
        if gate_eval_details_csv_src is None:
            details_dir = out_dir / "model_gate_eval_details"
            if details_dir.exists():
                detail_files = sorted(details_dir.glob("*_model_gate_eval_speed.csv"))
                if detail_files:
                    gate_eval_details_csv_src = detail_files[-1]
        if gate_eval_history_csv_src is not None:
            save_model_gate_eval_history_plot(
                gate_eval_history_csv_src,
                gate_eval_history_png,
                local_tz=args.local_timezone,
                eval_details_csv=gate_eval_details_csv_src,
            )
            gate_eval_history_png_src = gate_eval_history_png
        web_publish = publish_web_dashboard(
            web_out_dir=Path(args.web_out_dir),
            local_tz=args.local_timezone,
            web_refresh_seconds=args.web_refresh_seconds,
            next_day_png=plot_path,
            next_day_png_mobile=plot_path_mobile,
            next_day_csv=table_path,
            current_day_png=current_day_plot_path,
            current_day_png_mobile=current_day_plot_mobile_path,
            current_day_csv=current_day_table_path,
            daily_mae_png=daily_mae_png_src,
            daily_mae_png_mobile=daily_mae_png_mobile_src,
            daily_mae_csv=daily_mae_csv_src,
            gate_eval_png=gate_eval_history_png_src,
            gate_eval_csv=gate_eval_history_csv_src,
        )
        if args.git_auto_push_pages:
            repo_root = Path(__file__).resolve().parents[1]
            git_publish = auto_push_dashboard_changes(
                repo_root=repo_root,
                web_out_dir=Path(args.web_out_dir),
                remote=args.git_remote,
                branch=args.git_branch,
            )

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "db_path": str(db_path),
        "site": args.site,
        "forecast_model": args.model,
        "skip_training": bool(args.skip_training),
        "window_hours": args.window_hours,
        "target_hours": args.target_hours,
        "n_samples_all_speed": n_samples_all_speed,
        "n_samples_all_direction": n_samples_all_direction,
        "n_train_speed": None if speed_train_stats is None else int(speed_train_stats["n_train"]),
        "n_val_speed": None if speed_train_stats is None else int(speed_train_stats["n_val"]),
        "n_train_direction": None if direction_train_stats is None else int(direction_train_stats["n_train"]),
        "n_val_direction": None if direction_train_stats is None else int(direction_train_stats["n_val"]),
        "feature_cols": feature_cols,
        "speed_model_target": speed_target_mode,
        "speed_constraint_eps": speed_constraint_eps,
        "speed_regime_calibration": speed_calibration,
        "direction_model_target": direction_target_mode,
        "best_train_loss_speed": None if speed_train_stats is None else float(speed_train_stats["best_train_loss"]),
        "best_train_loss_eval_speed": None if speed_train_stats is None else float(speed_train_stats["best_train_loss_eval"]),
        "best_val_loss_speed": None if speed_train_stats is None else float(speed_train_stats["best_val_loss"]),
        "epochs_ran_speed": None if speed_train_stats is None else int(speed_train_stats["epochs_ran"]),
        "best_train_loss_direction": None
        if direction_train_stats is None
        else float(direction_train_stats["best_train_loss"]),
        "best_train_loss_eval_direction": None
        if direction_train_stats is None
        else float(direction_train_stats["best_train_loss_eval"]),
        "best_val_loss_direction": None if direction_train_stats is None else float(direction_train_stats["best_val_loss"]),
        "epochs_ran_direction": None if direction_train_stats is None else int(direction_train_stats["epochs_ran"]),
        "device": str(device),
        "prediction_anchor_time": inference_input_speed["anchor_time"],
        "reference_observation_time": inference_input_speed["reference_observation_time"],
        "prediction_day_start": inference_input_speed["prediction_day_start"],
        "prediction_generated_at_utc": prediction_generated_at_utc,
        "prediction_updated_at_utc": prediction_updated_at_utc,
        "model_last_trained_at_utc": model_last_trained_at_utc,
        "y_scaler_mean_speed": float(speed_arrays["y_mean"][0]),
        "y_scaler_std_speed": float(speed_arrays["y_std"][0]),
        "y_scaler_mean_direction": float(direction_arrays["y_mean"][0]),
        "y_scaler_std_direction": float(direction_arrays["y_std"][0]),
        "prediction_table_csv": str(table_path),
        "dayahead_snapshot_csv": dayahead_snapshot_csv,
        "prediction_plot_png": str(plot_path),
        "next_day_plot_archived_png": None
        if archived_next_day_plots is None
        else archived_next_day_plots.get("desktop"),
        "next_day_plot_archived_mobile_png": None
        if archived_next_day_plots is None
        else archived_next_day_plots.get("mobile"),
        "current_day_table_csv": str(current_day_table_path),
        "current_day_snapshot_csv": current_day_snapshot_csv,
        "current_day_fixed_origin_mae": current_day_fixed_origin_mae,
        "current_day_plot_png": str(current_day_plot_path),
        "current_day_plot_archived_png": archived_current_day_plot,
        "daily_mae_history_csv": daily_mae_csv,
        "daily_mae_history_png": daily_mae_png,
        "model_gate_eval_history_csv": None if gate_eval_history_csv_src is None else str(gate_eval_history_csv_src),
        "model_gate_eval_history_png": None if gate_eval_history_png_src is None else str(gate_eval_history_png_src),
        "model_gate_eval_details_csv": None if gate_eval_details_csv_src is None else str(gate_eval_details_csv_src),
        "web_dashboard_dir": None if web_publish is None else str(Path(args.web_out_dir).resolve()),
        "web_dashboard_files": web_publish,
        "web_dashboard_git_publish": git_publish,
        "speed_model_path": str(speed_model_path),
        "direction_model_path": str(direction_model_path),
        "intraday_model_path": str(intraday_model_path),
        "intraday_model_class": "IntradayResidualMLP",
        "intraday_feature_count": int(len(getattr(intraday_bundle, "x_mean", []))),
        "intraday_n_train": None if intraday_train_stats is None else int(intraday_train_stats["n_train"]),
        "intraday_n_val": None if intraday_train_stats is None else int(intraday_train_stats["n_val"]),
        "intraday_best_val_loss": None if intraday_train_stats is None else float(intraday_train_stats["best_val_loss"]),
        "intraday_hidden1": intraday_hparams.get("hidden1"),
        "intraday_hidden2": intraday_hparams.get("hidden2"),
        "intraday_dropout": intraday_hparams.get("dropout"),
        "intraday_learning_rate": intraday_hparams.get("learning_rate"),
        "intraday_recency_power": intraday_hparams.get("recency_power"),
        "intraday_rollout_calibration": getattr(intraday_bundle, "rollout_calibration", None),
        "intraday_continuity_calibration": getattr(intraday_bundle, "continuity_calibration", None),
        "data_refresh": refresh_info,
        "model_selection_gate": model_selection_report,
        "challenge_eval_split": args.challenge_eval_split,
        "challenge_min_eval_samples": args.challenge_min_eval_samples,
        "promotion_margin_pct": args.promotion_margin_pct,
        "max_forecast_age_hours": args.max_forecast_age_hours,
        "expected_update_hour_utc": args.expected_update_hour_utc,
        "validation_split": args.validation_split,
        "local_timezone": args.local_timezone,
        "current_day_interval_minutes": args.current_day_interval_minutes,
        "test_now_local_hour": args.test_now_local_hour,
    }
    metadata_path = out_dir / ("metadata_update.json" if not is_test_mode else f"metadata_update{test_suffix}.json")
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Model update complete.")
    print(f"Speed model saved to: {speed_model_path}")
    print(f"Direction model saved to: {direction_model_path}")
    print(f"Intraday model saved to: {intraday_model_path}")
    print(f"Prediction table saved to: {table_path}")
    print(f"Prediction plot saved to: {plot_path}")
    print(f"Current-day table saved to: {current_day_table_path}")
    print(f"Current-day plot saved to: {current_day_plot_path}")
    if is_test_mode:
        print("Test mode active: skipped daily archive/history updates and preserved production current-day outputs.")
    if archived_next_day_plots is not None:
        if "desktop" in archived_next_day_plots:
            print(f"Next-day plot archived to: {archived_next_day_plots['desktop']}")
        if "mobile" in archived_next_day_plots:
            print(f"Next-day mobile plot archived to: {archived_next_day_plots['mobile']}")
    if archived_current_day_plot is not None:
        print(f"Current-day plot archived to: {archived_current_day_plot}")
    if daily_mae_csv is not None and daily_mae_png is not None:
        print(f"Daily MAE history saved to: {daily_mae_csv}")
        print(f"Daily MAE history plot saved to: {daily_mae_png}")
    if web_publish is not None:
        print(f"Web dashboard updated in: {Path(args.web_out_dir).resolve()}")
    if args.git_auto_push_pages:
        if git_publish.get("pushed"):
            print(
                "Web dashboard pushed to git: "
                f"{git_publish.get('remote')}/{git_publish.get('branch')} @ {git_publish.get('commit')}"
            )
        else:
            print(f"Web dashboard git push skipped: {git_publish.get('reason')}")
    print()
    print(table_for_csv.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
