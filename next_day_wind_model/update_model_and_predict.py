from __future__ import annotations

import argparse
import copy
import html
import json
import os
import shutil
import sqlite3
import subprocess
import sys
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

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from db_store import (
    init_db,
    load_next_day_realized_detail_rows,
    load_prediction_evaluation_summary,
    log_prediction_batch,
    materialize_prediction_log_evaluation,
    summarize_next_day_vs_harmonie,
    summarize_next_day_vs_harmonie_by_horizon,
    summarize_next_day_vs_harmonie_by_issued_day,
)
from data_pipeline import (
    DatasetConfig,
    _angle_add_deg,
    _apply_standardizer,
    _fit_standardizer,
    _fit_target_scaler,
    build_anchor_forecast_context,
    build_anchor_forecast_timeline,
    build_all_direction_training_arrays,
    build_all_training_arrays,
    build_next_day_inference_input,
)
from intraday_model import (
    IntradayBundle,
    align_intraday_holdout_frames,
    build_intraday_holdout_context_split,
    build_intraday_holdout_evaluation_frame,
    load_intraday_model,
    predict_intraday_day_speed,
    save_intraday_model,
    summarize_intraday_champion_vs_challenger,
    train_intraday_model,
)
from train_lstm import NextDayLSTM, TargetAwareNextDayLSTM


LSTM_HIGHLIGHT_COLOR = "#d7191c"
MODEL_GATE_CHAMPION_COLOR = "#ff7f0e"
SUPERLOCAL_FORECAST_COLOR = MODEL_GATE_CHAMPION_COLOR
SUFFICIENT_WIND_THRESHOLD_KTS = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrain residual models (speed + direction) on all data and output next-day predictions.",
    )
    parser.add_argument("--db", default="data/wind_data_all_sites.db", help="Path to SQLite DB.")
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
        "--intraday-challenge-eval-split",
        type=float,
        default=0.15,
        help="Chronological holdout fraction used for intraday champion-vs-challenger checks.",
    )
    parser.add_argument(
        "--intraday-challenge-min-eval-contexts",
        type=int,
        default=48,
        help="Minimum number of later intraday issue contexts in the promotion holdout.",
    )
    parser.add_argument(
        "--intraday-promotion-margin-pct",
        type=float,
        default=1.0,
        help="Required relative MAE improvement (percent) to promote the intraday challenger.",
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
        help="Sampling interval (minutes) for current-day forecast/prediction plot grid.",
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
        help="Directory where generated metadata, tables, plots, snapshots, and diagnostics are saved.",
    )
    parser.add_argument(
        "--model-artifact-dir",
        default=None,
        help=(
            "Directory for trained model artifacts (.pt) and required scaler files (.npy). "
            "Defaults to --out-dir for backward compatibility."
        ),
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
        "--companion-app-base-url",
        default=os.environ.get("COMPANION_APP_BASE_URL", "http://127.0.0.1:8080"),
        help="Public base URL for the Flask rider portal companion app linked from the static dashboard.",
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


def _next_day_feature_schema_from_scalers(arrays: dict) -> str:
    x_mean = np.asarray(arrays.get("x_mean"), dtype=float).reshape(-1)
    if x_mean.shape[0] == 16:
        return "speed_v3_actual_history"
    if x_mean.shape[0] == 10:
        return "speed_v2"
    return "legacy"


def _direction_feature_schema_from_scalers(arrays: dict) -> str:
    x_mean = np.asarray(arrays.get("x_mean"), dtype=float).reshape(-1)
    if x_mean.shape[0] == 6:
        return "direction_v2"
    return "legacy"


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


def _slice_speed_calibration_context(context: dict | None, row_slice) -> dict | None:
    if context is None:
        return None
    sliced: dict = {}
    for key, value in context.items():
        arr = np.asarray(value)
        if arr.ndim == 0:
            sliced[key] = value
        elif arr.shape[0] == np.asarray(context.get("anchor_dir_deg", arr)).reshape(-1).shape[0]:
            sliced[key] = arr[row_slice]
        else:
            sliced[key] = value
    return sliced


def _target_hour_speed_calibration_feature_matrix(
    pred_arr: np.ndarray,
    forecast_arr: np.ndarray,
    speed_calibration_context: dict | None,
    stats: dict | None = None,
) -> tuple[np.ndarray, dict] | None:
    if speed_calibration_context is None:
        return None
    required = ["target_forecast_dir_deg", "target_times_utc", "target_horizon_hr"]
    if any(key not in speed_calibration_context for key in required):
        return None

    pred = np.asarray(pred_arr, dtype=np.float32)
    forecast = np.asarray(forecast_arr, dtype=np.float32)
    target_dir = np.asarray(speed_calibration_context["target_forecast_dir_deg"], dtype=np.float32)
    target_horizon = np.asarray(speed_calibration_context["target_horizon_hr"], dtype=np.float32)
    target_times_raw = np.asarray(speed_calibration_context["target_times_utc"])
    if pred.shape != forecast.shape or target_dir.shape != pred.shape or target_horizon.shape != pred.shape:
        return None
    if target_times_raw.shape != pred.shape:
        return None

    pred_flat = pred.reshape(-1)
    forecast_flat = forecast.reshape(-1)
    dir_flat = target_dir.reshape(-1)
    horizon_flat = target_horizon.reshape(-1)
    target_times = pd.to_datetime(target_times_raw.reshape(-1), utc=True)

    if stats is None:
        pred_mean = float(np.nanmean(pred_flat))
        pred_std = float(np.nanstd(pred_flat))
        forecast_mean = float(np.nanmean(forecast_flat))
        forecast_std = float(np.nanstd(forecast_flat))
        horizon_mean = float(np.nanmean(horizon_flat))
        horizon_std = float(np.nanstd(horizon_flat))
    else:
        pred_mean = float(stats["pred_mean"])
        pred_std = float(stats["pred_std"])
        forecast_mean = float(stats["forecast_mean"])
        forecast_std = float(stats["forecast_std"])
        horizon_mean = float(stats["horizon_mean"])
        horizon_std = float(stats["horizon_std"])
    if pred_std <= 0.0:
        pred_std = 1.0
    if forecast_std <= 0.0:
        forecast_std = 1.0
    if horizon_std <= 0.0:
        horizon_std = 1.0

    pred_norm = ((pred_flat - pred_mean) / pred_std).astype(np.float32)
    forecast_norm = ((forecast_flat - forecast_mean) / forecast_std).astype(np.float32)
    horizon_norm = ((horizon_flat - horizon_mean) / horizon_std).astype(np.float32)
    dir_rad = np.deg2rad(dir_flat % 360.0)
    dir_sin = np.sin(dir_rad).astype(np.float32)
    dir_cos = np.cos(dir_rad).astype(np.float32)
    hour_angle = (2.0 * np.pi * target_times.hour.to_numpy(dtype=np.float32)) / 24.0
    month_angle = (2.0 * np.pi * (target_times.month.to_numpy(dtype=np.float32) - 1.0)) / 12.0
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
        "feature_names": [
            "bias",
            "pred_norm",
            "forecast_norm",
            "pred_minus_forecast_norm",
            "target_dir_sin",
            "target_dir_cos",
            "target_hour_sin",
            "target_hour_cos",
            "target_month_sin",
            "target_month_cos",
            "target_horizon_norm",
            "pred_x_target_dir_sin",
            "pred_x_target_dir_cos",
            "forecast_x_target_dir_sin",
            "forecast_x_target_dir_cos",
            "target_hour_sin_x_target_dir_sin",
            "target_hour_cos_x_target_dir_cos",
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


def _fit_target_hour_speed_calibration(
    pred_arr: np.ndarray,
    forecast_arr: np.ndarray,
    actual_arr: np.ndarray,
    speed_calibration_context: dict | None,
) -> dict | None:
    built = _target_hour_speed_calibration_feature_matrix(
        pred_arr,
        forecast_arr,
        speed_calibration_context,
    )
    if built is None:
        return None
    X, feature_meta = built
    y = (actual_arr - pred_arr).reshape(-1).astype(np.float32)
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    if int(np.sum(mask)) < 64:
        return None

    xt = X[mask].astype(np.float64)
    yt = y[mask].astype(np.float64)
    split_idx = int(round(xt.shape[0] * 0.7))
    split_idx = max(1, min(split_idx, xt.shape[0] - 1))
    x_fit, x_val = xt[:split_idx], xt[split_idx:]
    y_fit, y_val = yt[:split_idx], yt[split_idx:]
    identity = np.eye(xt.shape[1], dtype=np.float64)
    identity[0, 0] = 0.0

    best_ridge: float | None = None
    best_coef: np.ndarray | None = None
    best_val_mae = float("inf")
    for ridge in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0):
        try:
            coef = np.linalg.solve(x_fit.T @ x_fit + float(ridge) * identity, x_fit.T @ y_fit)
        except np.linalg.LinAlgError:
            continue
        val_mae = float(np.mean(np.abs((x_val @ coef) - y_val)))
        if val_mae + 1e-9 < best_val_mae:
            best_val_mae = val_mae
            best_ridge = float(ridge)
    if best_ridge is None:
        return None

    try:
        best_coef = np.linalg.solve(xt.T @ xt + best_ridge * identity, xt.T @ yt).astype(np.float32)
    except np.linalg.LinAlgError:
        return None
    correction = (X.astype(np.float32) @ best_coef).reshape(pred_arr.shape)
    calibrated = np.maximum(pred_arr + correction.astype(np.float32), 0.0)

    baseline_mae = float(np.mean(np.abs(pred_arr - actual_arr)))
    calibrated_mae = float(np.mean(np.abs(calibrated - actual_arr)))
    improvement_abs = baseline_mae - calibrated_mae
    if improvement_abs <= 1e-6:
        return None
    return {
        "enabled": True,
        "type": "target_hour_ridge_v1",
        "ridge": float(best_ridge),
        "stats": {
            "pred_mean": float(feature_meta["pred_mean"]),
            "pred_std": float(feature_meta["pred_std"]),
            "forecast_mean": float(feature_meta["forecast_mean"]),
            "forecast_std": float(feature_meta["forecast_std"]),
            "horizon_mean": float(feature_meta["horizon_mean"]),
            "horizon_std": float(feature_meta["horizon_std"]),
        },
        "feature_names": feature_meta["feature_names"],
        "coefficients": [float(v) for v in best_coef.tolist()],
        "baseline_mae": float(baseline_mae),
        "calibrated_mae": float(calibrated_mae),
        "improvement_abs": float(improvement_abs),
        "improvement_pct": float(improvement_abs / max(baseline_mae, 1e-6)),
        "n_samples": int(pred_arr.shape[0]),
        "n_points": int(pred_arr.size),
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
    target_hour_cal = _fit_target_hour_speed_calibration(
        pred_arr,
        forecast_arr,
        actual_arr,
        speed_calibration_context,
    )
    candidates = [c for c in [threshold_cal, contextual_cal, target_hour_cal] if c is not None]
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
    if cal_type == "target_hour_ridge_v1":
        built = _target_hour_speed_calibration_feature_matrix(
            pred_arr,
            forecast_arr,
            speed_calibration_context,
            stats=speed_calibration.get("stats"),
        )
        if built is None:
            return np.asarray(pred_speed, dtype=np.float32)
        features, _ = built
        coef = np.asarray(speed_calibration.get("coefficients", []), dtype=np.float32)
        if coef.shape[0] != features.shape[1]:
            raise ValueError("Speed target-hour calibration coefficients do not match feature dimensions.")
        correction = (features @ coef).reshape(pred_arr.shape).astype(np.float32)
        calibrated = np.maximum(pred_arr + correction, 0.0).astype(np.float32)
        if input_was_vector:
            return calibrated[0].astype(np.float32)
        return calibrated.astype(np.float32)

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


def _circular_mean_deg(values: np.ndarray | pd.Series | list[float]) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan")
    return float(
        np.rad2deg(
            np.arctan2(
                np.nanmean(np.sin(np.deg2rad(arr))),
                np.nanmean(np.cos(np.deg2rad(arr))),
            )
        )
        % 360.0
    )


def _eval_start_index(n_samples: int, eval_fraction: float, min_eval_samples: int) -> int:
    eval_n = max(int(round(n_samples * float(eval_fraction))), int(min_eval_samples))
    eval_n = min(eval_n, n_samples - 2)
    if eval_n < 1:
        raise ValueError("Not enough samples to build challenger evaluation holdout.")
    return int(n_samples - eval_n)


def _metric_summary_from_aligned_predictions(
    pred: np.ndarray,
    actual: np.ndarray,
) -> dict[str, float | int | None]:
    pred_arr = np.asarray(pred, dtype=float).reshape(-1)
    actual_arr = np.asarray(actual, dtype=float).reshape(-1)
    valid = (~np.isnan(pred_arr)) & (~np.isnan(actual_arr))
    count = int(np.sum(valid))
    if count <= 0:
        return {"count": 0, "mae": None, "rmse": None}
    errors = pred_arr[valid] - actual_arr[valid]
    return {
        "count": count,
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
    }


def summarize_champion_vs_challenger(
    *,
    actual: np.ndarray,
    forecast: np.ndarray,
    challenger_pred: np.ndarray,
    champion_pred: np.ndarray | None,
    promotion_margin_pct: float,
    holdout_eval_split: float,
    holdout_eval_min_samples: int,
    challenger_model_id: str,
    champion_model_id: str | None,
) -> dict:
    """
    Summarize the fair next-day promotion holdout for challenger vs champion.

    The input arrays must already be aligned on the same chronological holdout
    sample/target positions. This helper preserves that fair comparison by
    scoring champion, challenger, and Harmonie on exactly the same realised
    rows instead of recomputing from a dashboard-oriented artifact.
    """
    actual_arr = np.asarray(actual, dtype=float)
    forecast_arr = np.asarray(forecast, dtype=float)
    challenger_arr = np.asarray(challenger_pred, dtype=float)

    summary: dict[str, object] = {
        "holdout_eval_split": float(holdout_eval_split),
        "holdout_eval_min_samples": int(holdout_eval_min_samples),
        "promotion_margin_pct": float(promotion_margin_pct),
        "speed_model_id_challenger": challenger_model_id,
        "speed_model_id_champion": champion_model_id or "none",
    }

    if champion_pred is None:
        challenger_metrics = _metric_summary_from_aligned_predictions(challenger_arr, actual_arr)
        harmonie_metrics = _metric_summary_from_aligned_predictions(forecast_arr, actual_arr)
        summary.update(
            {
                "comparison_scope": "aligned_holdout_rows_no_existing_champion",
                "speed_eval_rows": int(challenger_metrics["count"]),
                "speed_mae_forecast": harmonie_metrics["mae"],
                "speed_rmse_forecast": harmonie_metrics["rmse"],
                "speed_mae_challenger": challenger_metrics["mae"],
                "speed_rmse_challenger": challenger_metrics["rmse"],
                "speed_mae_champion": None,
                "speed_rmse_champion": None,
                "speed_mae_improvement_challenger_vs_champion": None,
                "speed_rmse_improvement_challenger_vs_champion": None,
                "speed_mae_improvement_challenger_vs_harmonie": None
                if challenger_metrics["mae"] is None or harmonie_metrics["mae"] is None
                else float(harmonie_metrics["mae"] - challenger_metrics["mae"]),
                "speed_rmse_improvement_challenger_vs_harmonie": None
                if challenger_metrics["rmse"] is None or harmonie_metrics["rmse"] is None
                else float(harmonie_metrics["rmse"] - challenger_metrics["rmse"]),
                "promote_speed": True,
                "reason": "no_existing_champion",
            }
        )
        return summary

    champion_arr = np.asarray(champion_pred, dtype=float)
    common_mask = (
        (~np.isnan(actual_arr))
        & (~np.isnan(forecast_arr))
        & (~np.isnan(challenger_arr))
        & (~np.isnan(champion_arr))
    )
    if not np.any(common_mask):
        raise ValueError("Champion/challenger holdout comparison has no common realised rows.")

    actual_common = actual_arr[common_mask]
    forecast_common = forecast_arr[common_mask]
    challenger_common = challenger_arr[common_mask]
    champion_common = champion_arr[common_mask]

    harmonie_metrics = _metric_summary_from_aligned_predictions(forecast_common, actual_common)
    challenger_metrics = _metric_summary_from_aligned_predictions(challenger_common, actual_common)
    champion_metrics = _metric_summary_from_aligned_predictions(champion_common, actual_common)

    champion_mae = float(champion_metrics["mae"])
    challenger_mae = float(challenger_metrics["mae"])
    champion_rmse = float(champion_metrics["rmse"])
    challenger_rmse = float(challenger_metrics["rmse"])
    promote_speed = challenger_mae <= champion_mae * (1.0 - max(0.0, float(promotion_margin_pct)) / 100.0)
    summary.update(
        {
            "comparison_scope": "aligned_holdout_rows_common_to_champion_and_challenger",
            "speed_eval_rows": int(champion_metrics["count"]),
            "speed_mae_forecast": harmonie_metrics["mae"],
            "speed_rmse_forecast": harmonie_metrics["rmse"],
            "speed_mae_champion": champion_mae,
            "speed_rmse_champion": champion_rmse,
            "speed_mae_challenger": challenger_mae,
            "speed_rmse_challenger": challenger_rmse,
            "speed_mae_improvement_challenger_vs_champion": float(champion_mae - challenger_mae),
            "speed_rmse_improvement_challenger_vs_champion": float(champion_rmse - challenger_rmse),
            "speed_mae_improvement_challenger_vs_harmonie": None
            if harmonie_metrics["mae"] is None
            else float(harmonie_metrics["mae"] - challenger_mae),
            "speed_rmse_improvement_challenger_vs_harmonie": None
            if harmonie_metrics["rmse"] is None
            else float(harmonie_metrics["rmse"] - challenger_rmse),
            "promote_speed": bool(promote_speed),
        }
    )
    return summary


def _timestamp_ms_utc(value: pd.Timestamp | datetime | str) -> int:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.value // 1_000_000)


def _build_next_day_prediction_log_frame(
    inference_input: dict,
    speed_pred: np.ndarray,
) -> pd.DataFrame:
    """
    Build canonical hourly next-day prediction rows for durable logging.

    The emitted frame keeps the exact Harmonie run/fetched metadata that was
    selected by the vintage-aware inference helpers, so later evaluation can
    reconstruct the fair operational baseline without re-deriving it from
    collapsed latest-per-target views.
    """
    target_times_utc = pd.to_datetime(inference_input["target_times"], utc=True)
    return pd.DataFrame(
        {
            "target_time_utc": target_times_utc,
            "horizon_hr": np.asarray(inference_input["target_horizon_hr"], dtype=np.float32),
            "prediction_value": np.asarray(speed_pred, dtype=np.float32),
            "harmonie_value": np.asarray(inference_input["forecast_next24"], dtype=np.float32),
            "harmonie_run_ts": np.asarray(inference_input["target_run_ts"], dtype=np.int64),
            "harmonie_fetched_ts": np.asarray(inference_input["target_fetched_ts"], dtype=np.int64),
        }
    )


def _build_prediction_log_rows(
    prediction_frame: pd.DataFrame,
    *,
    site: str,
    model_type: str,
    model_name: str,
    model_version: str | None,
    model_artifact: str,
    issued_time_utc: datetime | pd.Timestamp | str,
    anchor_time_utc: datetime | pd.Timestamp | str,
    prediction_kind: str = "wind_speed",
    run_context: str | None = None,
    metadata: dict | None = None,
) -> list[dict]:
    if prediction_frame.empty:
        return []

    issued_ts = _timestamp_ms_utc(issued_time_utc)
    anchor_ts = _timestamp_ms_utc(anchor_time_utc)
    base_metadata = dict(metadata or {})
    rows: list[dict] = []
    for rec in prediction_frame.itertuples(index=False):
        harmonie_run_ts = None if pd.isna(rec.harmonie_run_ts) else int(rec.harmonie_run_ts)
        harmonie_fetched_ts = None if pd.isna(rec.harmonie_fetched_ts) else int(rec.harmonie_fetched_ts)
        actual_value = None
        if hasattr(rec, "actual_value") and not pd.isna(rec.actual_value):
            actual_value = float(rec.actual_value)
        rows.append(
            {
                "site": site,
                "model_type": model_type,
                "prediction_kind": prediction_kind,
                "model_name": model_name,
                "model_version": model_version,
                "model_artifact": model_artifact,
                "issued_ts": issued_ts,
                "anchor_ts": anchor_ts,
                "target_ts": _timestamp_ms_utc(rec.target_time_utc),
                "horizon_hr": None if pd.isna(rec.horizon_hr) else float(rec.horizon_hr),
                "prediction_value": None if pd.isna(rec.prediction_value) else float(rec.prediction_value),
                "harmonie_value": None if pd.isna(rec.harmonie_value) else float(rec.harmonie_value),
                "harmonie_run_ts": harmonie_run_ts,
                "harmonie_fetched_ts": harmonie_fetched_ts,
                "actual_value": actual_value,
                "run_context": run_context,
                "metadata_json": base_metadata,
            }
        )
    return rows


def _log_prediction_frame(
    db_path: Path,
    prediction_frame: pd.DataFrame,
    *,
    site: str,
    model_type: str,
    model_name: str,
    model_version: str | None,
    model_artifact: str,
    issued_time_utc: datetime | pd.Timestamp | str,
    anchor_time_utc: datetime | pd.Timestamp | str,
    prediction_kind: str = "wind_speed",
    run_context: str | None = None,
    metadata: dict | None = None,
) -> int:
    rows = _build_prediction_log_rows(
        prediction_frame,
        site=site,
        model_type=model_type,
        model_name=model_name,
        model_version=model_version,
        model_artifact=model_artifact,
        issued_time_utc=issued_time_utc,
        anchor_time_utc=anchor_time_utc,
        prediction_kind=prediction_kind,
        run_context=run_context,
        metadata=metadata,
    )
    if not rows:
        return 0

    conn = sqlite3.connect(str(db_path))
    try:
        init_db(conn)
        return log_prediction_batch(conn, rows)
    finally:
        conn.close()


def build_prediction_table(
    inference_input: dict,
    speed_pred: np.ndarray,
    dir_pred: np.ndarray,
    local_tz: str = "Europe/Amsterdam",
) -> pd.DataFrame:
    target_times = pd.to_datetime(inference_input["target_times"], utc=True)
    target_times_local = target_times.tz_convert(ZoneInfo(local_tz))
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
            "target_time_local": target_times_local,
            "forecast_wind_speed": forecast_speed,
            "forecast_wind_min": lo,
            "forecast_wind_max": hi,
            "lstm_pred_wind_speed": speed_pred.astype(np.float32),
            "forecast_wind_dir_deg": forecast_dir,
            "lstm_pred_wind_dir_deg": dir_pred.astype(np.float32),
        }
    ).assign(
        hour_utc=lambda d: d["target_time_utc"].dt.strftime("%H"),
        hour_local=lambda d: d["target_time_local"].dt.strftime("%H"),
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


def _draw_sufficient_wind_threshold(ax: plt.Axes) -> None:
    ax.axhline(
        SUFFICIENT_WIND_THRESHOLD_KTS,
        color="#1f5f5b",
        linewidth=1.8,
        alpha=0.82,
        zorder=1.15,
    )


def _load_observations_raw(conn: sqlite3.Connection, site: str) -> pd.DataFrame:
    query = """
    SELECT ts, wind_speed, wind_gust, wind_dir, payload
    FROM observations
    WHERE site = ?
      AND ts IS NOT NULL
    ORDER BY ts
    """
    rows = conn.execute(query, (site,)).fetchall()
    if not rows:
        return pd.DataFrame(columns=["actual_avg", "actual_min", "actual_max", "actual_dir"])

    def payload_value(payload: dict, *keys: str):
        for key in keys:
            if key in payload and payload.get(key) is not None:
                return payload.get(key)
        return None

    def as_float(value) -> float:
        try:
            return float(value) if value is not None else np.nan
        except (TypeError, ValueError):
            return np.nan

    records: list[dict] = []
    for ts, wind_speed, wind_gust, wind_dir, payload_raw in rows:
        payload = {}
        if payload_raw:
            try:
                payload = json.loads(payload_raw)
            except json.JSONDecodeError:
                payload = {}
        avg = payload_value(payload, "AverageWind", "WindSpeedAvg")
        if avg is None:
            avg = wind_speed
        min_wind = payload_value(payload, "MinWind", "WindSpeedMin")
        max_wind = payload_value(payload, "MaxWind", "WindSpeedMax")
        if max_wind is None:
            max_wind = wind_gust
        direc = payload_value(payload, "WindDirection")
        if direc is None:
            direc = wind_dir
        records.append(
            {
                "obs_ts": int(ts),
                "actual_avg": as_float(avg),
                "actual_min": as_float(min_wind),
                "actual_max": as_float(max_wind),
                "actual_dir": as_float(direc),
            }
        )

    out = pd.DataFrame.from_records(records)
    out["obs_dt"] = pd.to_datetime(out["obs_ts"], unit="ms", utc=True)
    out = out.set_index("obs_dt").sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out[["actual_avg", "actual_min", "actual_max", "actual_dir"]]


def _interp_hourly_to_dense(hourly_values: np.ndarray, hourly_index: pd.DatetimeIndex, dense_index: pd.DatetimeIndex) -> np.ndarray:
    s = pd.Series(hourly_values, index=hourly_index, dtype=float)
    dense = s.reindex(s.index.union(dense_index)).sort_index().interpolate(method="time").reindex(dense_index)
    return dense.to_numpy(dtype=np.float32)


def _interp_direction_hourly_to_dense(
    hourly_degrees: np.ndarray,
    hourly_index: pd.DatetimeIndex,
    dense_index: pd.DatetimeIndex,
) -> np.ndarray:
    direction = pd.Series(hourly_degrees, index=hourly_index, dtype=float).dropna()
    if direction.empty:
        return np.full(len(dense_index), np.nan, dtype=np.float32)
    rad = np.deg2rad(direction.to_numpy(dtype=float) % 360.0)
    sin_dense = (
        pd.Series(np.sin(rad), index=direction.index, dtype=float)
        .reindex(direction.index.union(dense_index))
        .sort_index()
        .interpolate(method="time")
        .reindex(dense_index)
        .to_numpy(dtype=float)
    )
    cos_dense = (
        pd.Series(np.cos(rad), index=direction.index, dtype=float)
        .reindex(direction.index.union(dense_index))
        .sort_index()
        .interpolate(method="time")
        .reindex(dense_index)
        .to_numpy(dtype=float)
    )
    dense = np.rad2deg(np.arctan2(sin_dense, cos_dense)) % 360.0
    dense[np.isnan(sin_dense) | np.isnan(cos_dense)] = np.nan
    return dense.astype(np.float32)


def _interp_series_at_times(series: pd.Series, query_index: pd.DatetimeIndex) -> np.ndarray:
    """
    Linearly interpolate a frozen forecast curve at arbitrary timestamps.

    Query timestamps outside the curve support stay NaN so this helper does not
    extrapolate beyond the issued forecast branch.
    """
    if len(query_index) == 0:
        return np.array([], dtype=float)
    curve = pd.Series(series, copy=True, dtype=float).dropna()
    if curve.empty:
        return np.full(len(query_index), np.nan, dtype=float)
    curve = curve[~curve.index.duplicated(keep="last")].sort_index()
    if len(curve) == 1:
        out = np.full(len(query_index), np.nan, dtype=float)
        same_time = query_index == curve.index[0]
        if np.any(same_time):
            out[same_time] = float(curve.iloc[0])
        return out

    x = curve.index.asi8.astype(np.float64)
    y = curve.to_numpy(dtype=float)
    q = query_index.asi8.astype(np.float64)
    out = np.full(len(query_index), np.nan, dtype=float)
    valid = (q >= x[0]) & (q <= x[-1])
    if np.any(valid):
        out[valid] = np.interp(q[valid], x, y)
    return out


def _smooth_series(values: np.ndarray, window: int = 3) -> np.ndarray:
    """Gentle centered rolling mean for plotting; preserves NaN gaps."""
    arr = np.asarray(values, dtype=float)
    if window <= 1:
        return arr.copy()
    s = pd.Series(arr)
    out = s.rolling(window=window, center=True, min_periods=1).mean().to_numpy(dtype=float)
    out[np.isnan(arr)] = np.nan
    return out


def _measured_actual_trend_values(time_local: pd.Series, actual_values: np.ndarray) -> np.ndarray:
    actual_series = pd.Series(np.asarray(actual_values, dtype=float), index=pd.DatetimeIndex(time_local))
    measured_series = actual_series.dropna().sort_index()
    out = pd.Series(np.nan, index=actual_series.index, dtype=float)
    if measured_series.empty:
        return out.to_numpy(dtype=float)

    trend = measured_series.rolling("30min", min_periods=1).mean()
    # This is a measured-data trend only, not a forecast; missing/future rows
    # remain NaN so the plotted line stops at the latest actual measurement.
    out.loc[trend.index] = trend
    return out.to_numpy(dtype=float)


def _build_current_day_plot_frame(
    dense_times: pd.DatetimeIndex,
    forecast_columns: dict[str, np.ndarray],
    actual_day_raw: pd.DataFrame,
    *,
    now_local: datetime | pd.Timestamp,
    future_start: datetime | pd.Timestamp,
) -> pd.DataFrame:
    """
    Build the current-day plotting frame without forcing observations onto the
    forecast grid.
    """
    actual_columns = {
        "actual_wind_speed": "actual_avg",
        "actual_wind_min": "actual_min",
        "actual_wind_max": "actual_max",
        "actual_wind_dir_deg": "actual_dir",
    }
    now_ts = pd.Timestamp(now_local)
    future_start_ts = pd.Timestamp(future_start)
    dense_times = pd.DatetimeIndex(dense_times).sort_values()
    if dense_times.empty:
        raise ValueError("Current-day plot requires at least one forecast grid timestamp.")

    if actual_day_raw is None or actual_day_raw.empty:
        actual_frame = pd.DataFrame(columns=actual_columns.values(), index=pd.DatetimeIndex([], tz=dense_times.tz))
    else:
        actual_frame = actual_day_raw.copy()
        actual_frame = actual_frame[actual_frame.index <= now_ts]
        actual_frame = actual_frame[~actual_frame.index.duplicated(keep="last")].sort_index()

    actual_times = pd.DatetimeIndex(actual_frame.index)
    plot_times = dense_times.union(actual_times).sort_values()
    is_forecast_grid = plot_times.isin(dense_times)
    is_actual_observation = plot_times.isin(actual_times)

    table_data: dict[str, object] = {
        "time_local": plot_times,
        "is_forecast_grid": is_forecast_grid,
        "is_actual_observation": is_actual_observation,
    }
    for col, values in forecast_columns.items():
        table_data[col] = pd.Series(np.asarray(values, dtype=np.float32), index=dense_times).reindex(plot_times).to_numpy(dtype=np.float32)
    for out_col, raw_col in actual_columns.items():
        if raw_col in actual_frame.columns:
            table_data[out_col] = (
                pd.to_numeric(actual_frame[raw_col], errors="coerce")
                .reindex(plot_times)
                .to_numpy(dtype=np.float32)
            )
        else:
            table_data[out_col] = np.full(len(plot_times), np.nan, dtype=np.float32)

    table = pd.DataFrame(table_data)
    table["is_future"] = table["time_local"] >= future_start_ts
    table["hour_local"] = table["time_local"].dt.strftime("%H")
    table["minute_local"] = table["time_local"].dt.minute
    return table


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


def _sqlite_table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _parse_harmonie_ms_utc(value: object) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc)
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def _load_latest_harmonie_metadata_time(db_path: Path, site: str) -> tuple[datetime | None, str]:
    conn = sqlite3.connect(str(db_path))
    try:
        if _sqlite_table_exists(conn, "harmonie_knmi_features"):
            row = conn.execute(
                """
                SELECT fetched_ts, run_ts
                FROM harmonie_knmi_features
                WHERE site = ?
                ORDER BY run_ts DESC, horizon_hr ASC
                LIMIT 1
                """,
                (site,),
            ).fetchone()
            if row is not None:
                fetched_dt = _parse_iso_utc(None if row[0] is None else str(row[0]))
                if fetched_dt is not None:
                    return fetched_dt, "fetched"
                run_dt = _parse_iso_utc(None if row[1] is None else str(row[1]))
                if run_dt is not None:
                    return run_dt, "run"

        if _sqlite_table_exists(conn, "knmi_forecasts_shadow"):
            row = conn.execute(
                """
                SELECT fetched_iso, fetched_ts, run_iso, run_ts
                FROM knmi_forecasts_shadow
                WHERE site = ?
                  AND model = 'HARMONIE'
                ORDER BY run_ts DESC, horizon_hr ASC
                LIMIT 1
                """,
                (site,),
            ).fetchone()
            if row is not None:
                fetched_dt = _parse_iso_utc(None if row[0] is None else str(row[0]))
                if fetched_dt is None:
                    fetched_dt = _parse_harmonie_ms_utc(row[1])
                if fetched_dt is not None:
                    return fetched_dt, "fetched"
                run_dt = _parse_iso_utc(None if row[2] is None else str(row[2]))
                if run_dt is None:
                    run_dt = _parse_harmonie_ms_utc(row[3])
                if run_dt is not None:
                    return run_dt, "run"

        if _sqlite_table_exists(conn, "prediction_log"):
            row = conn.execute(
                """
                SELECT harmonie_fetched_iso, harmonie_fetched_ts, harmonie_run_iso, harmonie_run_ts
                FROM prediction_log
                WHERE site = ?
                  AND harmonie_run_ts IS NOT NULL
                ORDER BY issued_ts DESC, target_ts ASC
                LIMIT 1
                """,
                (site,),
            ).fetchone()
            if row is not None:
                fetched_dt = _parse_iso_utc(None if row[0] is None else str(row[0]))
                if fetched_dt is None:
                    fetched_dt = _parse_harmonie_ms_utc(row[1])
                if fetched_dt is not None:
                    return fetched_dt, "fetched"
                run_dt = _parse_iso_utc(None if row[2] is None else str(row[2]))
                if run_dt is None:
                    run_dt = _parse_harmonie_ms_utc(row[3])
                if run_dt is not None:
                    return run_dt, "run"
    except sqlite3.Error:
        return None, "fetched"
    finally:
        conn.close()
    return None, "fetched"


def _format_harmonie_metadata_text(
    harmonie_time_utc: datetime | pd.Timestamp | str | None,
    harmonie_time_kind: str,
    local_tz: str,
) -> str:
    harmonie_dt = _parse_iso_utc(None if harmonie_time_utc is None else str(harmonie_time_utc))
    kind = "run" if harmonie_time_kind == "run" else "fetched"
    label = "HARMONIE run" if kind == "run" else "HARMONIE fetched"
    if harmonie_dt is None:
        return f"{label}: unknown"
    tz = ZoneInfo(local_tz)
    local_dt = harmonie_dt.astimezone(tz)
    if local_dt.date() == datetime.now(tz).date():
        harmonie_txt = local_dt.strftime("%H:%M")
    else:
        harmonie_txt = f"{local_dt.day} {local_dt.strftime('%B %H:%M')}"
    return f"{label}: {harmonie_txt} local"


def _format_plot_meta_text(
    prediction_generated_at_utc: str,
    prediction_updated_at_utc: str | None,
    model_trained_at_utc: str | None,
    local_tz: str,
    harmonie_time_utc: datetime | pd.Timestamp | str | None = None,
    harmonie_time_kind: str = "fetched",
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
    return (
        f"Last plot update: {pred_txt}\n"
        f"Last prediction update: {pred_upd_txt}\n"
        f"{_format_harmonie_metadata_text(harmonie_time_utc, harmonie_time_kind, local_tz)}\n"
        f"Champion model trained & promoted: {train_txt}"
    )


def _format_last_plot_update_text(plot_updated_at_utc: datetime | pd.Timestamp | str | None, local_tz: str) -> str:
    plot_dt = _parse_iso_utc(None if plot_updated_at_utc is None else str(plot_updated_at_utc))
    if plot_dt is None:
        return "Last plot update: unknown"
    plot_txt = plot_dt.astimezone(ZoneInfo(local_tz)).strftime("%H:%M")
    return f"Last plot update: {plot_txt}"


def _format_static_day_label(value: datetime | pd.Timestamp) -> str:
    dt = pd.Timestamp(value)
    return f"{dt.strftime('%A')} {dt.day} {dt.strftime('%B %Y')}"


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
    harmonie_time_utc: datetime | pd.Timestamp | str | None = None,
    harmonie_time_kind: str = "fetched",
    mobile: bool = False,
) -> None:
    table = table.copy()
    if "target_time_local" not in table.columns:
        table["target_time_local"] = table["target_time_utc"].dt.tz_convert(ZoneInfo(local_tz))
    else:
        table["target_time_local"] = pd.to_datetime(table["target_time_local"], utc=True).dt.tz_convert(
            ZoneInfo(local_tz)
        )
    table = table[(table["target_time_local"].dt.hour >= 8) & (table["target_time_local"].dt.hour <= 22)].reset_index(
        drop=True
    )
    if table.empty:
        raise ValueError("No rows available in local 08:00-22:00 range for plotting.")

    x = np.arange(len(table))
    table["hour_label"] = table["target_time_local"].dt.strftime("%H")
    first_dt = table["target_time_local"].iloc[0]
    day_label = _format_static_day_label(first_dt)

    forecast_core_parts = [
        table["forecast_wind_speed"].dropna(),
        table["lstm_pred_wind_speed"].dropna(),
    ]
    forecast_core_nonempty = [part for part in forecast_core_parts if len(part)]
    forecast_core_series = pd.concat(forecast_core_nonempty) if forecast_core_nonempty else pd.Series(dtype=float)
    forecast_core_max = float(forecast_core_series.max()) if not forecast_core_series.empty else 0.0
    harmonie_max_series = table["forecast_wind_max"].dropna()
    harmonie_max = float(harmonie_max_series.max()) if not harmonie_max_series.empty else 0.0
    y_upper_raw = max(
        forecast_core_max * 1.12,
        min(harmonie_max * 1.04, forecast_core_max * 1.35 + 2.0),
        SUFFICIENT_WIND_THRESHOLD_KTS + 0.8,
        10.0,
    )
    y_upper = float(np.ceil(y_upper_raw / 2.5) * 2.5)
    if not mobile:
        print(
            "Static next-day y-axis | "
            f"forecast_core_max={forecast_core_max:.2f} "
            f"harmonie_max={harmonie_max:.2f} "
            f"y_upper={y_upper:.2f} "
            f"harmonie_max_clipped={harmonie_max > y_upper}"
        )

    fig_size = (8.4, 8.8) if mobile else (14, 7.2)
    title_fs = 14 if mobile else None
    label_fs = 12 if mobile else None
    tick_fs = 11 if mobile else None
    legend_fs = 10 if mobile else None
    meta_fs = 10 if mobile else 9
    meta_y = 1.14 if mobile else 1.13
    fig, ax = plt.subplots(figsize=fig_size)
    _apply_speed_background(ax, y_upper, x_left=0.0, x_right=len(table) - 1.0)
    _draw_sufficient_wind_threshold(ax)
    marker_size = 3.0
    fc_low = table["forecast_wind_min"].to_numpy(dtype=float)
    fc_high = table["forecast_wind_max"].to_numpy(dtype=float)
    fc_avg = table["forecast_wind_speed"].to_numpy(dtype=float)
    lstm_avg = table["lstm_pred_wind_speed"].to_numpy(dtype=float)
    ax.plot(x, fc_high, color="#666666", linewidth=1.2, linestyle="--", label="Harmonie model - max speed")
    ax.plot(x, fc_avg, color="gray", linewidth=1.5, label="Harmonie model - avg speed")
    ax.plot(
        x,
        lstm_avg,
        color=SUPERLOCAL_FORECAST_COLOR,
        linewidth=2.4,
        label="Super local wind prediction - avg speed",
    )
    ax.set_title(day_label, fontsize=title_fs)
    ax.set_xlabel("Time", fontsize=label_fs, labelpad=24 if mobile else 26)
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
    legend = ax.legend(
        ordered_handles,
        ordered_labels,
        loc="upper left",
        bbox_to_anchor=(0.015, 0.99),
        borderaxespad=0.0,
        fontsize=legend_fs,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.96)
    legend.set_zorder(20)
    ax.set_xticks(x, table["hour_label"], rotation=0)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.set_xlim(0.0, len(table) - 1.0)
    ax.set_ylim(0.0, y_upper)
    ax.text(
        0.015,
        meta_y,
        _format_plot_meta_text(
            prediction_generated_at_utc,
            prediction_updated_at_utc,
            model_trained_at_utc,
            local_tz,
            harmonie_time_utc=harmonie_time_utc,
            harmonie_time_kind=harmonie_time_kind,
        ),
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
        for direction_deg, color in [(fdir, "gray"), (ldir, SUPERLOCAL_FORECAST_COLOR)]:
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return the dense plotting table plus the canonical hourly issued forecast.

    The first frame keeps the existing dense current-day plot output. The second
    frame is the evaluation-ready hourly issue record: one future target
    timestamp per row, using the exact Harmonie vintage that was available at
    the current issue hour.
    """
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

        direction_feature_schema = _direction_feature_schema_from_scalers(direction_scalers)
        context = build_anchor_forecast_context(
            db_path=db_path,
            cfg=cfg,
            anchor_time=anchor_local_for_targets.tz_convert("UTC"),
            history_times=history_utc_for_targets,
            target_times=target_utc_index,
        )
        history_frame = context["history_frame"]
        future_frame = context["target_frame"]
        if history_frame is None or future_frame is None:
            raise ValueError("Missing forecast rows for current-day anchor context.")
        if future_frame[["forecast_avg", "forecast_dir"]].isna().any().any():
            raise ValueError("Missing forecast rows in current-day target window.")

        if direction_feature_schema == "direction_v2":
            history_features = history_frame.copy()
            dir_rad = np.deg2rad(pd.to_numeric(history_features["forecast_dir"], errors="coerce").astype(float) % 360.0)
            history_features["forecast_dir_sin"] = np.sin(dir_rad)
            history_features["forecast_dir_cos"] = np.cos(dir_rad)
            feature_cols = [
                "forecast_avg",
                "forecast_max",
                "forecast_dir_sin",
                "forecast_dir_cos",
                "month_sin",
                "month_cos",
            ]
        else:
            history_features = history_frame
            feature_cols = ["forecast_avg", "forecast_max", "forecast_dir", "month_sin", "month_cos"]
        if history_features[feature_cols].isna().any().any():
            raise ValueError("Missing forecast rows in history window for current-day inference.")

        x_window = history_features[feature_cols].to_numpy(dtype=np.float32)[np.newaxis, :, :]
        x_dir = _apply_standardizer(x_window, direction_scalers["x_mean"], direction_scalers["x_std"]).astype(np.float32)

        direction_model.eval()
        with torch.no_grad():
            dir_res_scaled = direction_model(torch.from_numpy(x_dir).float().to(device)).cpu().numpy()[0]

        dir_res = dir_res_scaled * float(direction_scalers["y_std"][0]) + float(direction_scalers["y_mean"][0])
        dir_res = dir_res[:target_n]

        fc_speed = future_frame["forecast_avg"].to_numpy(dtype=np.float32)
        fc_dir = future_frame["forecast_dir"].to_numpy(dtype=np.float32)
        lstm_dir = _angle_add_deg(fc_dir, dir_res.astype(np.float32))
        return fc_speed.astype(np.float32), lstm_dir.astype(np.float32)

    tz = ZoneInfo(local_tz)
    now_local = _resolve_now_local(local_tz, test_now_local_hour)
    now_hour_local = now_local.replace(minute=0, second=0, microsecond=0)
    day_start_local = now_hour_local.replace(hour=0)
    day_end_local = day_start_local + timedelta(hours=23)
    interval_min = int(current_day_interval_minutes)
    if interval_min <= 0 or 60 % interval_min != 0:
        raise ValueError("--current-day-interval-minutes must be a positive divisor of 60.")

    # Build the forecast frame that was actually available at the current issue hour.
    full_hours = pd.date_range(start=day_start_local, end=day_end_local, freq="1h", tz=tz)
    issue_anchor_utc = pd.Timestamp(now_hour_local).tz_convert("UTC")
    forecast_frame_utc = build_anchor_forecast_timeline(
        db_path=db_path,
        cfg=cfg,
        anchor_time=issue_anchor_utc,
        timeline=full_hours.tz_convert("UTC"),
    )
    if forecast_frame_utc is None:
        raise ValueError("Missing current-issue forecast timeline for current-day prediction.")
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
    issued_hourly_predictions = pd.DataFrame(
        columns=[
            "target_time_utc",
            "horizon_hr",
            "prediction_value",
            "harmonie_value",
            "harmonie_run_ts",
            "harmonie_fetched_ts",
        ]
    )
    if remaining_n > 0:
        fc_remaining = forecast_frame_utc.reindex(remaining_local.tz_convert("UTC"))
        rem_prediction_speed = (
            pd.Series(intraday_speed_rem, index=full_hours, dtype=float)
            .reindex(remaining_local)
            .to_numpy(dtype=np.float32)
        )
        issued_hourly_predictions = pd.DataFrame(
            {
                "target_time_utc": remaining_local.tz_convert("UTC"),
                "horizon_hr": ((remaining_local.tz_convert("UTC") - issue_anchor_utc) / pd.Timedelta(hours=1)).astype(float),
                "prediction_value": rem_prediction_speed,
                "harmonie_value": fc_remaining["forecast_avg"].to_numpy(dtype=np.float32),
                "harmonie_run_ts": fc_remaining["run_ts"].to_numpy(dtype=np.int64),
                "harmonie_fetched_ts": fc_remaining["fetched_ts"].to_numpy(dtype=np.int64),
            }
        )
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
    fc_dir_dense = _interp_direction_hourly_to_dense(
        fc_today["forecast_dir"].to_numpy(dtype=np.float32),
        full_hours,
        dense_times,
    )
    lstm_full_dense = _interp_hourly_to_dense(intraday_speed_full.astype(np.float32), full_hours, dense_times)
    lstm_dir_full_dense = _interp_direction_hourly_to_dense(lstm_dir_full.astype(np.float32), full_hours, dense_times)

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
    rem_dense_dir = _interp_direction_hourly_to_dense(
        rem_hourly_dir.to_numpy(dtype=np.float32),
        rem_hourly_dir.index,
        dense_times,
    )

    # Actual measurements keep their source timestamps; forecast/prediction
    # values remain on the configured dense forecast grid.
    actual_day_raw = obs_raw_local[(obs_raw_local.index >= day_start_local) & (obs_raw_local.index <= now_local)].copy()
    future_start = now_hour_local + timedelta(hours=1)
    table = _build_current_day_plot_frame(
        dense_times,
        {
            "forecast_wind_speed": fc_speed_dense,
            "forecast_wind_min": fc_min_dense,
            "forecast_wind_max": fc_max_dense,
            "forecast_wind_dir_deg": fc_dir_dense,
            "lstm_pred_wind_speed_full": lstm_full_dense,
            "lstm_pred_wind_dir_deg_full": lstm_dir_full_dense,
            "lstm_pred_wind_speed": rem_dense_speed,
            "lstm_pred_wind_dir_deg": rem_dense_dir,
        },
        actual_day_raw,
        now_local=now_local,
        future_start=future_start,
    )
    return table, issued_hourly_predictions


def save_current_day_plot(
    table: pd.DataFrame,
    plot_path: Path,
    local_tz: str,
    prediction_generated_at_utc: str,
    prediction_updated_at_utc: str | None,
    model_trained_at_utc: str | None,
    harmonie_time_utc: datetime | pd.Timestamp | str | None = None,
    harmonie_time_kind: str = "fetched",
    prior_prediction_tables: list[pd.DataFrame] | None = None,
    live_monitoring_metric: dict | None = None,
    mobile: bool = False,
) -> None:
    def _prepare_branch_frame(
        raw_frame: pd.DataFrame,
        *,
        fallback_issue_anchor: pd.Timestamp | None = None,
    ) -> tuple[pd.Timestamp, pd.DataFrame] | None:
        if raw_frame is None or raw_frame.empty or "time_local" not in raw_frame.columns:
            return None
        frame = raw_frame.copy()
        frame["time_local"] = pd.to_datetime(frame["time_local"], utc=True, errors="coerce").dt.tz_convert(
            ZoneInfo(local_tz)
        )
        frame = frame.dropna(subset=["time_local"]).copy()
        if frame.empty:
            return None

        issued_at_local = None
        if "issued_at_local" in frame.columns:
            issued_series = pd.to_datetime(frame["issued_at_local"], utc=True, errors="coerce").dropna()
            if not issued_series.empty:
                issued_at_local = issued_series.iloc[0].tz_convert(ZoneInfo(local_tz))
        elif "issued_at_utc" in frame.columns:
            issued_series = pd.to_datetime(frame["issued_at_utc"], utc=True, errors="coerce").dropna()
            if not issued_series.empty:
                issued_at_local = issued_series.iloc[0].tz_convert(ZoneInfo(local_tz))
        if issued_at_local is None:
            issued_at_local = fallback_issue_anchor
        if issued_at_local is None:
            return None

        issue_anchor = pd.Timestamp(issued_at_local).floor("h")
        frame = frame[
            (frame["time_local"].dt.hour >= 8)
            & ((frame["time_local"].dt.hour < 22) | ((frame["time_local"].dt.hour == 22) & (frame["time_local"].dt.minute == 0)))
        ].copy()
        if frame.empty:
            return None
        if "is_future" in frame.columns:
            is_future_mask = (
                frame["is_future"]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(["true", "1", "yes"])
            )
            if bool(is_future_mask.any()):
                first_future_time = frame.loc[is_future_mask, "time_local"].min()
                # Keep the partial-hour bridge from the issue anchor up to the
                # first true future target so prior issued branches connect
                # cleanly into the next full forecast hour.
                bridge_mask = (
                    (frame["time_local"] >= issue_anchor)
                    & (frame["time_local"] <= first_future_time)
                )
                frame = frame[is_future_mask | bridge_mask].copy()
            else:
                frame = frame[frame["time_local"] >= issue_anchor].copy()
        frame = frame[frame["time_local"] >= issue_anchor].copy()
        if "is_forecast_grid" in frame.columns:
            is_forecast_grid = (
                frame["is_forecast_grid"]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(["true", "1", "yes"])
            )
            if bool(is_forecast_grid.any()):
                frame = frame[is_forecast_grid].copy()
        if frame.empty:
            return None
        frame = frame.sort_values("time_local").drop_duplicates(subset=["time_local"], keep="last").reset_index(drop=True)
        return issue_anchor, frame

    def _forecast_branch_changed(
        prev_frame: pd.DataFrame,
        curr_frame: pd.DataFrame,
        *,
        compare_cols: list[str],
        tolerance: float = 0.0,
    ) -> bool:
        prev_cols = ["time_local"] + [col for col in compare_cols if col in prev_frame.columns]
        curr_cols = ["time_local"] + [col for col in compare_cols if col in curr_frame.columns]
        common_cols = [col for col in compare_cols if col in prev_cols and col in curr_cols]
        if not common_cols:
            return False
        prev_view = prev_frame[
            prev_frame["time_local"].dt.minute.eq(0)
        ][["time_local"] + common_cols].set_index("time_local").sort_index()
        curr_view = curr_frame[
            curr_frame["time_local"].dt.minute.eq(0)
        ][["time_local"] + common_cols].set_index("time_local").sort_index()
        common_times = prev_view.index.intersection(curr_view.index)
        if len(common_times) == 0:
            return False
        prev_vals = prev_view.loc[common_times, common_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        curr_vals = curr_view.loc[common_times, common_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        diffs = np.abs(prev_vals - curr_vals)
        if np.isnan(diffs).all():
            return False
        return bool(np.nanmax(diffs) > float(tolerance))

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
    day_label = _format_static_day_label(first_dt)

    fc_low = table["forecast_wind_min"].to_numpy(dtype=float)
    fc_high = table["forecast_wind_max"].to_numpy(dtype=float)
    fc_avg = table["forecast_wind_speed"].to_numpy(dtype=float)
    actual_avg = table["actual_wind_speed"].to_numpy(dtype=float)
    actual_min = (
        pd.to_numeric(table["actual_wind_min"], errors="coerce").to_numpy(dtype=float)
        if "actual_wind_min" in table.columns
        else np.full(len(table), np.nan, dtype=float)
    )
    actual_max = (
        pd.to_numeric(table["actual_wind_max"], errors="coerce").to_numpy(dtype=float)
        if "actual_wind_max" in table.columns
        else np.full(len(table), np.nan, dtype=float)
    )
    has_actual_avg = bool(np.any(~np.isnan(actual_avg)))
    actual_trend = (
        _measured_actual_trend_values(table["time_local"], actual_avg)
        if has_actual_avg
        else np.full(len(table), np.nan, dtype=float)
    )
    has_actual_trend = bool(np.any(~np.isnan(actual_trend)))
    has_actual_minmax = bool(np.any(~np.isnan(actual_min)) or np.any(~np.isnan(actual_max)))

    current_issue_dt = _parse_iso_utc(prediction_generated_at_utc)
    if current_issue_dt is None:
        current_issue_anchor = table["time_local"].max().floor("h")
    else:
        current_issue_anchor = pd.Timestamp(current_issue_dt).tz_convert(ZoneInfo(local_tz)).floor("h")

    overlay_tables = prior_prediction_tables or []
    branch_frames_by_anchor: dict[pd.Timestamp, pd.DataFrame] = {}
    for overlay_table in overlay_tables:
        prepared = _prepare_branch_frame(overlay_table)
        if prepared is None:
            continue
        issue_anchor, overlay = prepared
        if issue_anchor not in branch_frames_by_anchor:
            branch_frames_by_anchor[issue_anchor] = overlay
    current_branch = _prepare_branch_frame(table, fallback_issue_anchor=current_issue_anchor)
    if current_branch is not None:
        branch_frames_by_anchor[current_branch[0]] = current_branch[1]
    historical_branches = sorted(branch_frames_by_anchor.items(), key=lambda item: item[0])

    harmonie_update_anchors: list[pd.Timestamp] = []
    if historical_branches:
        harmonie_update_anchors = [historical_branches[0][0]]
        prev_harmonie_frame = historical_branches[0][1]
        for issue_anchor, overlay in historical_branches[1:]:
            if _forecast_branch_changed(
                prev_harmonie_frame,
                overlay,
                compare_cols=["forecast_wind_speed", "forecast_wind_min", "forecast_wind_max"],
                tolerance=0.25,
            ):
                harmonie_update_anchors.append(issue_anchor)
            prev_harmonie_frame = overlay
    harmonie_next_update_anchor: dict[pd.Timestamp, pd.Timestamp] = {}
    if harmonie_update_anchors:
        for idx, issue_anchor in enumerate(harmonie_update_anchors[:-1]):
            harmonie_next_update_anchor[issue_anchor] = harmonie_update_anchors[idx + 1]
        harmonie_next_update_anchor[harmonie_update_anchors[-1]] = current_issue_anchor

    current_harmonie_anchor = (
        harmonie_update_anchors[-1]
        if harmonie_update_anchors
        else current_issue_anchor
    )
    current_comparison_frame = branch_frames_by_anchor.get(current_harmonie_anchor)
    if current_comparison_frame is None or current_comparison_frame.empty:
        current_comparison_frame = _prepare_branch_frame(table, fallback_issue_anchor=current_issue_anchor)[1]
        current_harmonie_anchor = current_issue_anchor

    # Keep the static dashboard scale primarily forecast-based so it stays
    # stable through the day. Measured values are only a safeguard against
    # clipping observations that exceed the forecast-based range. Harmonie
    # max-speed is a secondary uncertainty/gust-like line and is intentionally
    # excluded from the hard y-limit, so it may be clipped.
    forecast_core_parts = [
        table["forecast_wind_speed"].dropna(),
        table["lstm_pred_wind_speed"].dropna(),
        table["lstm_pred_wind_speed_full"].dropna(),
    ]
    measured_safeguard_parts = [
        table["actual_wind_speed"].dropna(),
        pd.Series(actual_min).dropna(),
        pd.Series(actual_max).dropna(),
        pd.Series(actual_trend).dropna(),
    ]
    harmonie_max_parts = [table["forecast_wind_max"].dropna()]
    for _issue_anchor, overlay in historical_branches:
        if "lstm_pred_wind_speed" in overlay.columns:
            forecast_core_parts.append(pd.to_numeric(overlay["lstm_pred_wind_speed"], errors="coerce").dropna())
        if "forecast_wind_speed" in overlay.columns:
            forecast_core_parts.append(pd.to_numeric(overlay["forecast_wind_speed"], errors="coerce").dropna())
        if "forecast_wind_max" in overlay.columns:
            harmonie_max_parts.append(pd.to_numeric(overlay["forecast_wind_max"], errors="coerce").dropna())

    forecast_core_nonempty = [part.dropna() for part in forecast_core_parts if len(part.dropna())]
    measured_safeguard_nonempty = [
        part.dropna() for part in measured_safeguard_parts if len(part.dropna())
    ]
    harmonie_max_nonempty = [part.dropna() for part in harmonie_max_parts if len(part.dropna())]
    forecast_core_series = (
        pd.concat(forecast_core_nonempty) if forecast_core_nonempty else pd.Series(dtype=float)
    )
    measured_safeguard_series = (
        pd.concat(measured_safeguard_nonempty)
        if measured_safeguard_nonempty
        else pd.Series(dtype=float)
    )
    harmonie_max_series = (
        pd.concat(harmonie_max_nonempty) if harmonie_max_nonempty else pd.Series(dtype=float)
    )
    forecast_core_max = float(forecast_core_series.max()) if not forecast_core_series.empty else 0.0
    measured_safeguard_max = (
        float(measured_safeguard_series.max()) if not measured_safeguard_series.empty else 0.0
    )
    harmonie_max_diagnostic = float(harmonie_max_series.max()) if not harmonie_max_series.empty else 0.0
    y_max_included = max(forecast_core_max, measured_safeguard_max)
    y_upper_raw = max(y_max_included * 1.10, SUFFICIENT_WIND_THRESHOLD_KTS + 0.8, 10.0)
    y_lower = 0.0
    y_upper = float(np.ceil(y_upper_raw / 2.5) * 2.5)
    if not mobile:
        print(
            "Static current-day y-axis | "
            f"forecast_core_max={forecast_core_max:.2f} "
            f"measured_safeguard_max={measured_safeguard_max:.2f} "
            f"harmonie_max={harmonie_max_diagnostic:.2f} "
            f"y_upper={y_upper:.2f} "
            f"harmonie_max_clipped={harmonie_max_diagnostic > y_upper}"
        )

    fig_size = (8.4, 8.8) if mobile else (14, 7.2)
    title_fs = 14 if mobile else None
    title_pad = 12 if mobile else 20
    label_fs = 12 if mobile else None
    tick_fs = 11 if mobile else None
    legend_fs = 10 if mobile else None
    meta_fs = 10 if mobile else 9
    mae_fs = 11 if mobile else 10
    meta_text_y = 1.19 if mobile else 1.16
    metric_box_y = 1.29 if mobile else 1.24
    fig, ax = plt.subplots(figsize=fig_size)

    def _plot_valid_line(x_values: np.ndarray, y_values: np.ndarray, **kwargs) -> None:
        valid = ~np.isnan(y_values)
        if np.any(valid):
            ax.plot(x_values[valid], y_values[valid], **kwargs)

    _apply_speed_background(ax, y_upper, x_left=0.0, x_right=len(table) - 1.0)
    _draw_sufficient_wind_threshold(ax)
    if np.any(~np.isnan(actual_min)):
        _plot_valid_line(
            x,
            actual_min,
            color="magenta",
            linestyle=(0, (1.2, 1.2)),
            linewidth=1.35 if mobile else 1.2,
            alpha=0.62,
            label="_nolegend_",
            zorder=4.6,
        )
    if np.any(~np.isnan(actual_max)):
        _plot_valid_line(
            x,
            actual_max,
            color="magenta",
            linestyle=(0, (1.2, 1.2)),
            linewidth=1.35 if mobile else 1.2,
            alpha=0.62,
            label="_nolegend_",
            zorder=4.6,
        )
    _plot_valid_line(
        x,
        actual_avg,
        marker="o",
        markersize=2.2,
        color="magenta",
        linewidth=2.2,
        label="_nolegend_",
        zorder=5,
    )
    if has_actual_trend:
        _plot_valid_line(
            x,
            actual_trend,
            color="#b000b8",
            linewidth=1.8 if mobile else 1.7,
            alpha=0.82,
            label="_nolegend_",
            zorder=5.25,
        )

    historical_superlocal_plotted = False
    historical_harmonie_plotted = False
    if historical_branches:
        overlay_alpha_start = 0.56 if mobile else 0.5
        overlay_alpha_end = 0.9 if mobile else 0.84
        superlocal_prior_lw = 2.45 if mobile else 2.2
        harmonie_prior_lw = 2.3 if mobile else 2.05
        superlocal_prior_dash = (0, (4.8, 1.8))
        harmonie_prior_dash = (0, (3.4, 1.6))
        overlay_alphas = np.linspace(overlay_alpha_start, overlay_alpha_end, len(historical_branches), dtype=float)
        for idx, ((issue_anchor, overlay), overlay_alpha) in enumerate(zip(historical_branches, overlay_alphas)):
            if issue_anchor == current_harmonie_anchor:
                continue
            next_issue_anchor = (
                historical_branches[idx + 1][0]
                if idx + 1 < len(historical_branches)
                else current_issue_anchor
            )
            active_overlay = overlay[(overlay["time_local"] >= issue_anchor) & (overlay["time_local"] <= next_issue_anchor)].copy()
            if active_overlay.empty:
                continue

            active_x = np.array([x_lookup.get(ts, np.nan) for ts in active_overlay["time_local"]], dtype=float)

            superlocal_y = pd.to_numeric(active_overlay["lstm_pred_wind_speed"], errors="coerce").to_numpy(dtype=float)
            superlocal_valid = (~np.isnan(active_x)) & (~np.isnan(superlocal_y))
            if np.any(superlocal_valid):
                sl_x = active_x[superlocal_valid]
                sl_y = superlocal_y[superlocal_valid]
                order = np.argsort(sl_x)
                sl_x = sl_x[order]
                sl_y = sl_y[order]
                if len(sl_x) >= 2:
                    ax.plot(
                        sl_x,
                        sl_y,
                        color=SUPERLOCAL_FORECAST_COLOR,
                        linestyle=superlocal_prior_dash,
                        linewidth=superlocal_prior_lw,
                        alpha=float(overlay_alpha),
                        zorder=2.35,
                        label="_nolegend_",
                    )
                historical_superlocal_plotted = True

            if issue_anchor not in harmonie_update_anchors or "forecast_wind_speed" not in overlay.columns:
                continue
            harmonie_end_anchor = harmonie_next_update_anchor.get(issue_anchor, current_issue_anchor)
            harmonie_overlay = overlay[
                (overlay["time_local"] >= issue_anchor) & (overlay["time_local"] <= harmonie_end_anchor)
            ].copy()
            if harmonie_overlay.empty:
                continue
            harmonie_x = np.array([x_lookup.get(ts, np.nan) for ts in harmonie_overlay["time_local"]], dtype=float)
            harmonie_y = pd.to_numeric(harmonie_overlay["forecast_wind_speed"], errors="coerce").to_numpy(dtype=float)
            harmonie_valid = (~np.isnan(harmonie_x)) & (~np.isnan(harmonie_y))
            if np.any(harmonie_valid):
                h_x = harmonie_x[harmonie_valid]
                h_y = harmonie_y[harmonie_valid]
                order = np.argsort(h_x)
                h_x = h_x[order]
                h_y = h_y[order]
                if len(h_x) >= 2:
                    ax.plot(
                        h_x,
                        h_y,
                        color="#8a8a8a",
                        linestyle=harmonie_prior_dash,
                        linewidth=harmonie_prior_lw,
                        alpha=max(float(overlay_alpha), 0.56),
                        zorder=2.2,
                        label="_nolegend_",
                    )
                historical_harmonie_plotted = True
            if "forecast_wind_max" in harmonie_overlay.columns:
                harmonie_max_y = pd.to_numeric(harmonie_overlay["forecast_wind_max"], errors="coerce").to_numpy(dtype=float)
                harmonie_max_valid = (~np.isnan(harmonie_x)) & (~np.isnan(harmonie_max_y))
                if np.any(harmonie_max_valid):
                    h_max_x = harmonie_x[harmonie_max_valid]
                    h_max_y = harmonie_max_y[harmonie_max_valid]
                    order = np.argsort(h_max_x)
                    h_max_x = h_max_x[order]
                    h_max_y = h_max_y[order]
                    if len(h_max_x) >= 2:
                        ax.plot(
                            h_max_x,
                            h_max_y,
                            color="#666666",
                            linestyle="--",
                            linewidth=1.1,
                            alpha=max(float(overlay_alpha), 0.5),
                            zorder=2.25,
                            label="_nolegend_",
                        )

    # The solid current comparison is anchored to the latest active Harmonie
    # update. We plot the Super local branch issued at that same anchor so the
    # solid lines represent one fair frozen head-to-head pair.
    current_branch_x = np.array([x_lookup.get(ts, np.nan) for ts in current_comparison_frame["time_local"]], dtype=float)
    harmonie_current = np.full(len(table), np.nan, dtype=float)
    harmonie_current_high = np.full(len(table), np.nan, dtype=float)
    harmonie_current_low = np.full(len(table), np.nan, dtype=float)
    superlocal_current = np.full(len(table), np.nan, dtype=float)
    current_branch_boundary_idx: int | None = x_lookup.get(current_harmonie_anchor)
    current_valid = ~np.isnan(current_branch_x)
    if np.any(current_valid):
        branch_positions = current_branch_x[current_valid].astype(int)
        harmonie_vals = pd.to_numeric(
            current_comparison_frame.loc[current_valid, "forecast_wind_speed"],
            errors="coerce",
        ).to_numpy(dtype=float)
        harmonie_high_vals = pd.to_numeric(
            current_comparison_frame.loc[current_valid, "forecast_wind_max"],
            errors="coerce",
        ).to_numpy(dtype=float)
        harmonie_low_vals = pd.to_numeric(
            current_comparison_frame.loc[current_valid, "forecast_wind_min"],
            errors="coerce",
        ).to_numpy(dtype=float)
        superlocal_vals = pd.to_numeric(
            current_comparison_frame.loc[current_valid, "lstm_pred_wind_speed"],
            errors="coerce",
        ).to_numpy(dtype=float)
        harmonie_current[branch_positions] = harmonie_vals
        harmonie_current_high[branch_positions] = np.where(np.isnan(harmonie_high_vals), harmonie_vals, harmonie_high_vals)
        harmonie_current_low[branch_positions] = np.where(np.isnan(harmonie_low_vals), harmonie_vals, harmonie_low_vals)
        superlocal_current[branch_positions] = superlocal_vals
        if current_branch_boundary_idx is None and len(branch_positions) > 0:
            current_branch_boundary_idx = int(branch_positions[0])

    _plot_valid_line(
        x,
        harmonie_current_high,
        color="#666666",
        linewidth=1.2,
        linestyle="--",
        label="_nolegend_",
        zorder=2.4,
    )
    _plot_valid_line(
        x,
        harmonie_current,
        color="#555555",
        linewidth=2.2,
        label="_nolegend_",
        zorder=2.6,
    )

    _plot_valid_line(
        x,
        superlocal_current,
        color=SUPERLOCAL_FORECAST_COLOR,
        linewidth=2.6,
        label="_nolegend_",
        zorder=3,
    )

    def _first_valid_index(values: np.ndarray) -> int | None:
        valid_idx = np.where(~np.isnan(values))[0]
        return int(valid_idx[0]) if len(valid_idx) else None

    harmonie_start_idx = _first_valid_index(harmonie_current)
    if harmonie_start_idx is not None:
        ax.scatter(
            [float(harmonie_start_idx)],
            [float(harmonie_current[harmonie_start_idx])],
            s=54 if mobile else 46,
            marker="s",
            color="#555555",
            edgecolors="white",
            linewidths=0.9,
            zorder=3.6,
            label="_nolegend_",
        )
    superlocal_start_idx = _first_valid_index(superlocal_current)
    if superlocal_start_idx is not None:
        ax.scatter(
            [float(superlocal_start_idx)],
            [float(superlocal_current[superlocal_start_idx])],
            s=58 if mobile else 50,
            marker="o",
            color=SUPERLOCAL_FORECAST_COLOR,
            edgecolors="white",
            linewidths=0.9,
            zorder=3.8,
            label="_nolegend_",
        )

    ax.set_title(day_label, fontsize=title_fs, pad=title_pad)
    ax.set_xlabel("Time", fontsize=label_fs, labelpad=0)
    ax.set_ylabel("Wind speed (kts)", fontsize=label_fs)
    ax.grid(axis="y", alpha=0.3)
    order_map = {
        "Measured wind": Line2D(
            [0],
            [0],
            color="magenta",
            linewidth=2.2,
            marker="o",
            markersize=4.2 if mobile else 3.6,
        ),
        "Super local forecast": Line2D(
            [0],
            [0],
            color=SUPERLOCAL_FORECAST_COLOR,
            linewidth=2.6,
            marker="o",
            markersize=5.0 if mobile else 4.4,
            markerfacecolor=SUPERLOCAL_FORECAST_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.7,
        ),
        "Harmonie forecast": Line2D(
            [0],
            [0],
            color="#555555",
            linewidth=2.2,
            marker="s",
            markersize=4.8 if mobile else 4.2,
            markerfacecolor="#555555",
            markeredgecolor="white",
            markeredgewidth=0.7,
        ),
    }
    desired_order = [
        "Measured wind",
        "Super local forecast",
        "Harmonie forecast",
    ]
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
    tick_pos = x[hour_tick_mask.to_numpy()]
    tick_lbl = table.loc[hour_tick_mask, "time_local"].dt.strftime("%H").to_list()
    ax.set_xticks(tick_pos, tick_lbl, rotation=0)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.set_xlim(-0.05, len(table) - 1.0 + 0.02)
    ax.set_ylim(y_lower, y_upper)

    superlocal_color = SUPERLOCAL_FORECAST_COLOR
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
        ax.axvline(float(latest_actual_idx), color="gray", linestyle="--", linewidth=1.0, zorder=0.8)
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
        if current_branch_boundary_idx is not None:
            ax.annotate(
                f"active anchor {current_harmonie_anchor.strftime('%H:%M')}",
                xy=(float(current_branch_boundary_idx), y_upper),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=max(mae_fs - 2, 8),
                color="black",
                clip_on=False,
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

    def _summary_line(
        label_text: str,
        value_text: str,
        *,
        label_color: str = "black",
        value_color: str = "black",
    ) -> HPacker:
        label_block = f"{label_text + ':':<16}"
        return HPacker(
            children=[
                TextArea(
                    label_block,
                    textprops={"fontsize": mae_fs, "color": label_color, "fontfamily": metric_fontfamily},
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

    summary_available = bool(live_monitoring_metric and live_monitoring_metric.get("available", False))
    superlocal_value_color, harmonie_value_color = _metric_value_colors(live_monitoring_metric)
    if summary_available:
        mae_superlocal_txt = f"{float(live_monitoring_metric['mae_superlocal']):.2f} kts"
        mae_harmonie_txt = f"{float(live_monitoring_metric['mae_harmonie']):.2f} kts"
        point_count_txt = f"{int(live_monitoring_metric.get('measurement_point_count', live_monitoring_metric.get('point_count', 0))):,}"
    else:
        mae_superlocal_txt = "n/a"
        mae_harmonie_txt = "n/a"
        point_count_txt = "0"

    metric_lines: list[TextArea | HPacker] = [
        TextArea(
            "Current-day cumulative MAE",
            textprops={
                "fontsize": mae_fs,
                "fontweight": "bold",
                "color": "black",
                "fontfamily": metric_fontfamily,
            },
        ),
        _summary_line("Super local", mae_superlocal_txt, label_color=superlocal_color, value_color=superlocal_value_color),
        _summary_line("Harmonie", mae_harmonie_txt, label_color=harmonie_color, value_color=harmonie_value_color),
        _summary_line("Measured points", point_count_txt),
    ]

    mse_box = VPacker(
        children=metric_lines,
        align="left",
        pad=0,
        sep=2,
    )
    mse_anchored = AnchoredOffsetbox(
        loc="upper right",
        child=mse_box,
        pad=0.2,
        frameon=True,
        bbox_to_anchor=(0.992, metric_box_y),
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
        meta_text_y,
        _format_plot_meta_text(
            prediction_generated_at_utc,
            prediction_updated_at_utc,
            model_trained_at_utc,
            local_tz,
            harmonie_time_utc=harmonie_time_utc,
            harmonie_time_kind=harmonie_time_kind,
        ),
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
        for direction_deg, color, z in [(fdir, "gray", 3), (ldir, SUPERLOCAL_FORECAST_COLOR, 4), (adir, "magenta", 6)]:
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

    layout_top = 0.78 if mobile else 0.82
    layout_bottom = 0.20 if mobile else 0.19
    xlabel_y = -0.21 if mobile else -0.23
    fig.tight_layout(rect=[0, layout_bottom, 1, layout_top])
    fig.subplots_adjust(bottom=layout_bottom, top=layout_top)
    ax.xaxis.set_label_coords(0.5, xlabel_y)
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
    model_class: str = "NextDayLSTM",
    history_hours: int | None = None,
    extra: dict | None = None,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "model_class": str(model_class),
        "n_features": int(n_features),
        "target_hours": int(target_hours),
        "target_name": target_name,
        "target_mode": target_mode,
        "output_activation": output_activation,
    }
    if history_hours is not None:
        payload["history_hours"] = int(history_hours)
    if extra:
        payload.update(extra)
    torch.save(
        payload,
        path,
    )


def _load_model(path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location=device)
    model_class = str(ckpt.get("model_class", "NextDayLSTM"))
    if model_class == "TargetAwareNextDayLSTM":
        model = TargetAwareNextDayLSTM(
            n_features=int(ckpt["n_features"]),
            target_hours=int(ckpt["target_hours"]),
            history_hours=int(ckpt.get("history_hours", 72)),
            output_activation=str(ckpt.get("output_activation", "linear")),
        ).to(device)
    else:
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


def _safe_model_trained_utc(
    model_path: Path,
    device: torch.device,
    *,
    intraday: bool = False,
) -> str | None:
    if not model_path.exists():
        return None
    try:
        if intraday:
            _, ckpt = load_intraday_model(model_path, device)
        else:
            _, ckpt = _load_model(model_path, device)
        return _resolve_model_trained_utc(ckpt, model_path)
    except Exception:
        return datetime.fromtimestamp(model_path.stat().st_mtime, tz=timezone.utc).isoformat()


def _champion_artifact_state(
    model_path: Path,
    device: torch.device,
    local_tz: str,
    *,
    intraday: bool = False,
) -> dict:
    trained_at_utc = _safe_model_trained_utc(model_path, device, intraday=intraday)
    return {
        "path": str(model_path),
        "exists": bool(model_path.exists()),
        "trained_at_utc": trained_at_utc,
        "model_id": _format_model_id(trained_at_utc, local_tz) if trained_at_utc else None,
    }


def _load_active_champion_states(
    *,
    speed_model_path: Path,
    direction_model_path: Path,
    intraday_model_path: Path,
    device: torch.device,
    local_tz: str,
) -> dict:
    return {
        "next_day_speed": _champion_artifact_state(speed_model_path, device, local_tz, intraday=False),
        "next_day_direction": _champion_artifact_state(direction_model_path, device, local_tz, intraday=False),
        "intraday_speed": _champion_artifact_state(intraday_model_path, device, local_tz, intraday=True),
    }


def _require_skip_training_artifacts(paths: list[Path], model_artifact_dir: Path) -> None:
    missing = [path for path in paths if not path.exists()]
    if not missing:
        return
    missing_lines = "\n".join(f"- {path}" for path in missing)
    raise FileNotFoundError(
        "Missing required model artifact(s) for --skip-training mode.\n"
        f"Model artifact directory searched: {model_artifact_dir}\n"
        f"Missing file(s):\n{missing_lines}\n"
        "Train models first, or pass --model-artifact-dir pointing to an existing model artifact directory."
    )


def _champion_state_refreshed(before_state: dict | None, after_state: dict | None) -> bool | None:
    if before_state is None or after_state is None:
        return None
    before_exists = bool(before_state.get("exists"))
    after_exists = bool(after_state.get("exists"))
    if not after_exists:
        return False
    if not before_exists and after_exists:
        return True
    return before_state.get("trained_at_utc") != after_state.get("trained_at_utc")


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
    last_months: int | None = 3,
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
    if hist_daily.empty:
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

    merged = hist_daily.set_index("day").sort_index()

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
    if last_months is not None and int(last_months) > 0:
        months = int(last_months)
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
    speed_candidates = np.asarray(
        [
            merged["avg_lstm_wind_speed"].max(),
            merged["avg_forecast_wind_speed"].max(),
            merged["avg_actual_wind_speed"].max(),
        ],
        dtype=float,
    )
    finite_speed_candidates = speed_candidates[np.isfinite(speed_candidates)]
    speed_top = float(np.max(finite_speed_candidates)) if finite_speed_candidates.size else 0.0
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
    mae_candidates = np.asarray([merged["mae_forecast"].max(), merged["mae_lstm"].max()], dtype=float)
    finite_mae_candidates = mae_candidates[np.isfinite(mae_candidates)]
    y_top_data = float(np.max(finite_mae_candidates)) if finite_mae_candidates.size else 0.0
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


def compute_current_day_table_mae(table: pd.DataFrame) -> dict:
    """Compute a display-only current-day MAE fallback from the generated table.

    This is used when prediction-log based completed-interval scoring is not
    available, for example in dev/test runs where logging is skipped. It does
    not write data or affect model training/prediction logic.
    """
    empty_summary = {
        "available": False,
        "point_count": 0,
        "measurement_point_count": 0,
        "completed_interval_count": 0,
        "partial_current_interval_included": True,
        "mae_superlocal": None,
        "mae_harmonie": None,
        "rmse_superlocal": None,
        "rmse_harmonie": None,
        "model_win_rate": None,
        "source": "current_day_table",
        "segments": [],
    }
    if table is None or table.empty:
        return empty_summary
    required_cols = ["actual_wind_speed", "forecast_wind_speed"]
    if any(col not in table.columns for col in required_cols):
        return empty_summary
    frame = table.copy()
    if "time_local" not in frame.columns:
        return empty_summary
    frame["time_local"] = pd.to_datetime(frame["time_local"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["time_local"]).copy()
    if frame.empty:
        return empty_summary

    actual = pd.to_numeric(frame["actual_wind_speed"], errors="coerce")
    harmonie = pd.to_numeric(frame["forecast_wind_speed"], errors="coerce")
    if "lstm_pred_wind_speed_full" in frame.columns:
        superlocal = pd.to_numeric(frame["lstm_pred_wind_speed_full"], errors="coerce")
    elif "lstm_pred_wind_speed" in frame.columns:
        superlocal = pd.to_numeric(frame["lstm_pred_wind_speed"], errors="coerce")
    else:
        return empty_summary

    measurement_mask = actual.notna()
    if "is_future" in frame.columns:
        is_future = (
            frame["is_future"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes"])
        )
        measurement_mask &= ~is_future
    if not bool(measurement_mask.any()):
        return empty_summary

    time_index = pd.DatetimeIndex(frame["time_local"])
    measurement_times = time_index[measurement_mask.to_numpy()]
    actual_values = actual[measurement_mask].to_numpy(dtype=float)
    harmonie_curve = pd.Series(harmonie.to_numpy(dtype=float), index=time_index).dropna().sort_index()
    superlocal_curve = pd.Series(superlocal.to_numpy(dtype=float), index=time_index).dropna().sort_index()
    harmonie_values = _interp_series_at_times(harmonie_curve, measurement_times)
    superlocal_values = _interp_series_at_times(superlocal_curve, measurement_times)
    valid = (~np.isnan(actual_values)) & (~np.isnan(harmonie_values)) & (~np.isnan(superlocal_values))
    if not np.any(valid):
        return empty_summary

    actual_values = actual_values[valid]
    harmonie_values = harmonie_values[valid]
    superlocal_values = superlocal_values[valid]
    model_abs = np.abs(superlocal_values - actual_values)
    harmonie_abs = np.abs(harmonie_values - actual_values)
    model_sq = np.square(superlocal_values - actual_values)
    harmonie_sq = np.square(harmonie_values - actual_values)
    finite = (
        ~np.isnan(model_abs)
        & ~np.isnan(harmonie_abs)
        & ~np.isnan(model_sq)
        & ~np.isnan(harmonie_sq)
    )
    if not np.any(finite):
        return empty_summary
    model_abs = model_abs[finite]
    harmonie_abs = harmonie_abs[finite]
    model_sq = model_sq[finite]
    harmonie_sq = harmonie_sq[finite]
    return {
        "available": True,
        "point_count": int(len(model_abs)),
        "measurement_point_count": int(len(model_abs)),
        "completed_interval_count": 1,
        "partial_current_interval_included": True,
        "mae_superlocal": float(np.mean(model_abs)),
        "mae_harmonie": float(np.mean(harmonie_abs)),
        "rmse_superlocal": float(np.sqrt(np.mean(model_sq))),
        "rmse_harmonie": float(np.sqrt(np.mean(harmonie_sq))),
        "model_win_rate": float(np.mean(model_abs < harmonie_abs)),
        "source": "current_day_table",
        "segments": [
            {
                "measurement_point_count": int(len(model_abs)),
                "point_count": int(len(model_abs)),
                "source": "current_day_table",
            }
        ],
    }


def compute_current_day_completed_interval_mae(
    db_path: Path,
    site: str,
    target_day_local: date,
    local_tz: str,
    prior_prediction_tables: list[pd.DataFrame] | None = None,
) -> dict:
    """
    Build the live current-day monitoring metric from completed Harmonie-update intervals.

    Each segment starts at the first intraday issue hour where a new Harmonie
    vintage becomes active, uses the frozen model/Harmonie forecast branch from
    that issue hour, and is scored on all realised measured wind-speed points up
    to the next Harmonie update boundary. Forecast values at measurement
    timestamps are obtained via the same linear interpolation rule for both
    Super local and Harmonie. The latest open interval is excluded.
    """
    tz = ZoneInfo(local_tz)
    day_start_local = pd.Timestamp(datetime.combine(target_day_local, datetime.min.time()), tz=tz)
    day_end_local = day_start_local + pd.Timedelta(days=1)
    day_start_ms = int(day_start_local.tz_convert("UTC").timestamp() * 1000)
    day_end_ms = int(day_end_local.tz_convert("UTC").timestamp() * 1000)

    conn = sqlite3.connect(str(db_path))
    try:
        rows = pd.read_sql_query(
            """
            WITH first_issue AS (
                SELECT
                    anchor_ts,
                    target_ts,
                    MIN(issued_ts) AS issued_ts
                FROM prediction_log
                WHERE site = ?
                  AND model_type = 'intraday'
                  AND prediction_kind = 'wind_speed'
                  AND anchor_ts >= ?
                  AND anchor_ts < ?
                  AND target_ts >= ?
                  AND target_ts < ?
                GROUP BY anchor_ts, target_ts
            )
            SELECT
                pl.anchor_ts,
                pl.issued_ts,
                pl.target_ts,
                pl.prediction_value,
                pl.harmonie_value,
                pl.harmonie_run_ts,
                pl.harmonie_fetched_ts,
                pl.actual_value,
                pl.model_abs_error,
                pl.harmonie_abs_error,
                pl.model_sq_error,
                pl.harmonie_sq_error
            FROM prediction_log AS pl
            INNER JOIN first_issue AS fi
                ON pl.anchor_ts = fi.anchor_ts
               AND pl.target_ts = fi.target_ts
               AND pl.issued_ts = fi.issued_ts
            WHERE pl.site = ?
              AND pl.model_type = 'intraday'
              AND pl.prediction_kind = 'wind_speed'
            ORDER BY pl.anchor_ts ASC, pl.target_ts ASC
            """,
            conn,
            params=[site, day_start_ms, day_end_ms, day_start_ms, day_end_ms, site],
        )
        obs_raw_utc = _load_observations_raw(conn, site)
    finally:
        conn.close()

    empty_summary = {
        "available": False,
        "point_count": 0,
        "measurement_point_count": 0,
        "completed_interval_count": 0,
        "partial_current_interval_included": False,
        "mae_superlocal": None,
        "mae_harmonie": None,
        "rmse_superlocal": None,
        "rmse_harmonie": None,
        "model_win_rate": None,
        "segments": [],
    }
    if rows.empty:
        return empty_summary

    snapshot_by_anchor: dict[pd.Timestamp, pd.DataFrame] = {}
    for snap in prior_prediction_tables or []:
        if snap.empty or "issued_at_local" not in snap.columns or "time_local" not in snap.columns:
            continue
        snap_frame = snap.copy()
        snap_frame["time_local"] = pd.to_datetime(snap_frame["time_local"], utc=True, errors="coerce")
        if "issued_at_local" in snap_frame.columns:
            snap_frame["issued_at_local"] = pd.to_datetime(snap_frame["issued_at_local"], utc=True, errors="coerce")
        elif "issued_at_utc" in snap_frame.columns:
            snap_frame["issued_at_local"] = pd.to_datetime(snap_frame["issued_at_utc"], utc=True, errors="coerce")
        snap_frame = snap_frame.dropna(subset=["time_local", "issued_at_local"]).copy()
        if snap_frame.empty:
            continue
        snap_frame["time_local"] = snap_frame["time_local"].dt.tz_convert(tz)
        snap_frame["issued_at_local"] = snap_frame["issued_at_local"].dt.tz_convert(tz)
        issue_anchor_local = pd.to_datetime(snap_frame["issued_at_local"].iloc[0]).floor("h")
        if issue_anchor_local not in snapshot_by_anchor:
            snapshot_by_anchor[issue_anchor_local] = snap_frame.sort_values("time_local").reset_index(drop=True)

    actual_points_local = obs_raw_utc.tz_convert(tz)
    actual_points_local = actual_points_local[
        (actual_points_local.index >= day_start_local)
        & (actual_points_local.index < day_end_local)
    ].copy()
    actual_points_local = actual_points_local[actual_points_local["actual_avg"].notna()].copy()
    if actual_points_local.empty:
        return empty_summary

    numeric_cols = [
        "anchor_ts",
        "issued_ts",
        "target_ts",
        "prediction_value",
        "harmonie_value",
        "harmonie_run_ts",
        "harmonie_fetched_ts",
        "actual_value",
        "model_abs_error",
        "harmonie_abs_error",
        "model_sq_error",
        "harmonie_sq_error",
    ]
    for col in numeric_cols:
        rows[col] = pd.to_numeric(rows[col], errors="coerce")
    rows = rows.dropna(subset=["anchor_ts", "target_ts"]).copy()
    if rows.empty:
        return empty_summary

    def _has_harmonie_update(prev_frame: pd.DataFrame, curr_frame: pd.DataFrame) -> bool:
        compare_cols = ["target_ts", "harmonie_run_ts", "harmonie_fetched_ts", "harmonie_value"]
        prev_view = prev_frame[compare_cols].drop_duplicates(subset=["target_ts"]).set_index("target_ts").sort_index()
        curr_view = curr_frame[compare_cols].drop_duplicates(subset=["target_ts"]).set_index("target_ts").sort_index()
        common_targets = prev_view.index.intersection(curr_view.index)
        if len(common_targets) == 0:
            return False
        return not prev_view.loc[common_targets].equals(curr_view.loc[common_targets])

    anchor_groups = [
        (int(anchor_ts), frame.sort_values("target_ts").reset_index(drop=True))
        for anchor_ts, frame in rows.groupby("anchor_ts", sort=True)
    ]
    if not anchor_groups:
        return empty_summary

    update_boundaries: list[tuple[int, pd.DataFrame]] = [anchor_groups[0]]
    previous_issue_frame = anchor_groups[0][1]
    for anchor_ts, frame in anchor_groups[1:]:
        if _has_harmonie_update(previous_issue_frame, frame):
            update_boundaries.append((anchor_ts, frame))
        previous_issue_frame = frame

    segment_frames: list[pd.DataFrame] = []
    segment_summaries: list[dict] = []
    for (segment_anchor_ts, segment_frame), (next_anchor_ts, _next_frame) in zip(update_boundaries, update_boundaries[1:]):
        segment_anchor_local = pd.to_datetime(segment_anchor_ts, unit="ms", utc=True).tz_convert(tz)
        next_anchor_local = pd.to_datetime(next_anchor_ts, unit="ms", utc=True).tz_convert(tz)
        snapshot = snapshot_by_anchor.get(segment_anchor_local)
        if snapshot is None or snapshot.empty:
            continue
        measurement_rows = actual_points_local[
            (actual_points_local.index >= segment_anchor_local)
            & (actual_points_local.index < next_anchor_local)
        ].copy()
        if measurement_rows.empty:
            continue

        curve = snapshot[
            (snapshot["time_local"] >= segment_anchor_local)
            & (snapshot["time_local"] <= next_anchor_local)
        ].copy()
        if curve.empty:
            continue
        forecast_curve = pd.Series(
            pd.to_numeric(curve["forecast_wind_speed"], errors="coerce").to_numpy(dtype=float),
            index=curve["time_local"],
        )
        superlocal_curve = pd.Series(
            pd.to_numeric(curve["lstm_pred_wind_speed"], errors="coerce").to_numpy(dtype=float),
            index=curve["time_local"],
        )
        interp_harmonie = _interp_series_at_times(forecast_curve, measurement_rows.index)
        interp_superlocal = _interp_series_at_times(superlocal_curve, measurement_rows.index)
        actual_values = pd.to_numeric(measurement_rows["actual_avg"], errors="coerce").to_numpy(dtype=float)
        valid = (~np.isnan(actual_values)) & (~np.isnan(interp_harmonie)) & (~np.isnan(interp_superlocal))
        if not np.any(valid):
            continue

        segment_rows = pd.DataFrame(
            {
                "actual_value": actual_values[valid],
                "superlocal_value": interp_superlocal[valid],
                "harmonie_value": interp_harmonie[valid],
            }
        )
        segment_rows["model_abs_error"] = np.abs(segment_rows["superlocal_value"] - segment_rows["actual_value"])
        segment_rows["harmonie_abs_error"] = np.abs(segment_rows["harmonie_value"] - segment_rows["actual_value"])
        segment_rows["model_sq_error"] = np.square(segment_rows["superlocal_value"] - segment_rows["actual_value"])
        segment_rows["harmonie_sq_error"] = np.square(segment_rows["harmonie_value"] - segment_rows["actual_value"])
        segment_frames.append(segment_rows)
        segment_summaries.append(
            {
                "segment_anchor_local": segment_anchor_local.isoformat(),
                "segment_end_local": next_anchor_local.isoformat(),
                "measurement_point_count": int(len(segment_rows)),
                "point_count": int(len(segment_rows)),
            }
        )

    if not segment_frames:
        out = empty_summary.copy()
        out["segments"] = segment_summaries
        return out

    evaluated = pd.concat(segment_frames, ignore_index=True)
    model_abs = evaluated["model_abs_error"].to_numpy(dtype=float)
    harmonie_abs = evaluated["harmonie_abs_error"].to_numpy(dtype=float)
    model_sq = evaluated["model_sq_error"].to_numpy(dtype=float)
    harmonie_sq = evaluated["harmonie_sq_error"].to_numpy(dtype=float)
    valid_mask = (
        ~np.isnan(model_abs)
        & ~np.isnan(harmonie_abs)
        & ~np.isnan(model_sq)
        & ~np.isnan(harmonie_sq)
    )
    if not np.any(valid_mask):
        out = empty_summary.copy()
        out["segments"] = segment_summaries
        return out

    model_abs = model_abs[valid_mask]
    harmonie_abs = harmonie_abs[valid_mask]
    model_sq = model_sq[valid_mask]
    harmonie_sq = harmonie_sq[valid_mask]

    return {
        "available": True,
        "point_count": int(len(model_abs)),
        "measurement_point_count": int(len(model_abs)),
        "completed_interval_count": int(len(segment_frames)),
        "partial_current_interval_included": False,
        "mae_superlocal": float(np.mean(model_abs)),
        "mae_harmonie": float(np.mean(harmonie_abs)),
        "rmse_superlocal": float(np.sqrt(np.mean(model_sq))),
        "rmse_harmonie": float(np.sqrt(np.mean(harmonie_sq))),
        "model_win_rate": float(np.mean(model_abs < harmonie_abs)),
        "interpolation_method": "linear",
        "segments": segment_summaries,
    }


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
    """
    Export canonical frozen next-day daily history from prediction_log.

    The CSV/PNG remain dashboard artefacts, but their metrics are derived from
    realised next-day prediction_log rows rather than snapshot CSV recomputation.
    """
    now_local = _resolve_now_local(local_tz, test_now_local_hour)
    zone = ZoneInfo(local_tz)
    max_complete_target_day = now_local.date() if now_local.hour >= 22 else (now_local.date() - timedelta(days=1))
    conn = sqlite3.connect(str(db_path))
    try:
        detail_rows = load_next_day_realized_detail_rows(conn, site=cfg.site, prediction_kind="wind_speed")
    finally:
        conn.close()

    history_csv = out_dir / "daily_mae_history.csv"
    details_dir = out_dir / "daily_error_details"
    details_dir.mkdir(parents=True, exist_ok=True)
    history_rows: list[dict[str, object]] = []
    if detail_rows:
        detail_df = pd.DataFrame(detail_rows)
        detail_df["issued_dt_utc"] = pd.to_datetime(detail_df["issued_ts"], unit="ms", utc=True, errors="coerce")
        detail_df["target_dt_utc"] = pd.to_datetime(detail_df["target_ts"], unit="ms", utc=True, errors="coerce")
        detail_df = detail_df.dropna(subset=["issued_dt_utc", "target_dt_utc"]).copy()
        if not detail_df.empty:
            detail_df["target_dt_local"] = detail_df["target_dt_utc"].dt.tz_convert(zone)
            detail_df["issued_dt_local"] = detail_df["issued_dt_utc"].dt.tz_convert(zone)
            detail_df["target_day_local"] = detail_df["target_dt_local"].dt.strftime("%Y-%m-%d")
            detail_df["target_day_local_date"] = detail_df["target_dt_local"].dt.date
            detail_df = detail_df[detail_df["target_day_local_date"] <= max_complete_target_day].copy()

            for target_day_local, day_df in detail_df.groupby("target_day_local", sort=True):
                day_df = day_df.sort_values("target_dt_utc").reset_index(drop=True)
                day_stamp = str(target_day_local).replace("-", "")
                details_csv = details_dir / f"{day_stamp}_dayahead_actual_forecast_lstm.csv"
                snapshot_path = _find_dayahead_snapshot(out_dir, datetime.strptime(target_day_local, "%Y-%m-%d").date())

                details = pd.DataFrame(
                    {
                        "target_time_utc": day_df["target_dt_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "hour_utc": day_df["target_dt_utc"].dt.strftime("%H"),
                        "forecast_wind_speed": day_df["harmonie_value"],
                        "lstm_pred_wind_speed": day_df["prediction_value"],
                        "actual_wind_speed": day_df["actual_value"],
                        "abs_err_forecast": day_df["harmonie_abs_error"],
                        "abs_err_lstm": day_df["model_abs_error"],
                        "issue_local_time": day_df["issued_dt_local"].iloc[0].isoformat(),
                        "issue_date_local": day_df["issued_dt_local"].iloc[0].strftime("%Y-%m-%d"),
                        "target_date_local": target_day_local,
                    }
                )
                details.to_csv(details_csv, index=False)

                model_sq = pd.to_numeric(day_df["model_sq_error"], errors="coerce")
                harmonie_sq = pd.to_numeric(day_df["harmonie_sq_error"], errors="coerce")
                mae_forecast = float(pd.to_numeric(day_df["harmonie_abs_error"], errors="coerce").mean())
                mae_lstm = float(pd.to_numeric(day_df["model_abs_error"], errors="coerce").mean())
                rmse_forecast = float(np.sqrt(harmonie_sq.mean()))
                rmse_lstm = float(np.sqrt(model_sq.mean()))

                history_rows.append(
                    {
                        "date": target_day_local,
                        "run_local_time": now_local.isoformat(),
                        "issue_local_time": day_df["issued_dt_local"].iloc[0].isoformat(),
                        "issued_day_utc": day_df["issued_dt_utc"].iloc[0].strftime("%Y-%m-%d"),
                        "mae_forecast": mae_forecast,
                        "mae_lstm": mae_lstm,
                        "rmse_forecast": rmse_forecast,
                        "rmse_lstm": rmse_lstm,
                        "mae_improvement_lstm_vs_forecast": mae_forecast - mae_lstm,
                        "rmse_improvement_lstm_vs_forecast": rmse_forecast - rmse_lstm,
                        "avg_actual_wind_speed": float(pd.to_numeric(day_df["actual_value"], errors="coerce").mean()),
                        "avg_forecast_wind_speed": float(pd.to_numeric(day_df["harmonie_value"], errors="coerce").mean()),
                        "avg_lstm_wind_speed": float(pd.to_numeric(day_df["prediction_value"], errors="coerce").mean()),
                        "n_points": int(len(day_df)),
                        "details_csv": str(details_csv),
                        "snapshot_csv": "" if snapshot_path is None else str(snapshot_path),
                        "evaluation_type": "day_ahead_frozen",
                        "data_source": "prediction_log",
                    }
                )

    hist = pd.DataFrame(
        history_rows,
        columns=[
            "date",
            "run_local_time",
            "issue_local_time",
            "issued_day_utc",
            "mae_forecast",
            "mae_lstm",
            "rmse_forecast",
            "rmse_lstm",
            "mae_improvement_lstm_vs_forecast",
            "rmse_improvement_lstm_vs_forecast",
            "avg_actual_wind_speed",
            "avg_forecast_wind_speed",
            "avg_lstm_wind_speed",
            "n_points",
            "details_csv",
            "snapshot_csv",
            "evaluation_type",
            "data_source",
        ],
    )
    if not hist.empty:
        hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
        hist = hist.dropna(subset=["date"]).sort_values("date")
        hist["date"] = hist["date"].dt.strftime("%Y-%m-%d")
    hist.to_csv(history_csv, index=False)

    history_png = out_dir / "daily_mae_history.png"
    save_daily_mae_plot(history_csv, history_png, local_tz=local_tz, last_months=3)
    return str(history_csv), str(history_png)


def append_model_gate_eval_history(
    history_csv: Path,
    model_selection_gate: dict,
    local_tz: str,
) -> str | None:
    """Append reporting history derived from the canonical promotion summary."""
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
                "speed_mae_forecast": model_selection_gate.get("speed_mae_forecast"),
                "speed_rmse_forecast": model_selection_gate.get("speed_rmse_forecast"),
                "run_local_time": now_local.isoformat(),
                "speed_mae_champion": model_selection_gate.get("speed_mae_champion"),
                "speed_rmse_champion": model_selection_gate.get("speed_rmse_champion"),
                "speed_mae_challenger": model_selection_gate.get("speed_mae_challenger"),
                "speed_rmse_challenger": model_selection_gate.get("speed_rmse_challenger"),
                "speed_mae_improvement_challenger_vs_champion": model_selection_gate.get(
                    "speed_mae_improvement_challenger_vs_champion"
                ),
                "speed_rmse_improvement_challenger_vs_champion": model_selection_gate.get(
                    "speed_rmse_improvement_challenger_vs_champion"
                ),
                "speed_mae_improvement_challenger_vs_harmonie": model_selection_gate.get(
                    "speed_mae_improvement_challenger_vs_harmonie"
                ),
                "speed_rmse_improvement_challenger_vs_harmonie": model_selection_gate.get(
                    "speed_rmse_improvement_challenger_vs_harmonie"
                ),
                "direction_mae_champion_deg": model_selection_gate.get("direction_mae_champion"),
                "direction_mae_challenger_deg": model_selection_gate.get("direction_mae_challenger"),
                "speed_selected": model_selection_gate.get("speed_selected"),
                "direction_selected": model_selection_gate.get("direction_selected"),
                "speed_eval_samples": model_selection_gate.get("speed_eval_samples"),
                "speed_eval_rows": model_selection_gate.get("speed_eval_rows"),
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
    db_path: Path | None = None,
    site: str | None = None,
) -> None:
    def _model_id_date_label(model_id: str | None) -> str | None:
        value = str(model_id or "").strip()
        if not value:
            return None
        if len(value) >= 8 and value[:8].isdigit():
            return f"{value[:4]}-{value[4:6]}-{value[6:8]}"
        return value

    def _repair_legacy_microsecond_detail_times(
        times: pd.Series,
        reference_utc: pd.Timestamp,
    ) -> pd.Series:
        """Repair legacy detail timestamps written with mixed microsecond/nanosecond units."""
        valid_times = times.dropna()
        if valid_times.empty or pd.isna(reference_utc):
            return times

        raw_us = valid_times.astype("int64").to_numpy(dtype=np.int64)
        int64_info = np.iinfo(np.int64)
        if raw_us.min() < int64_info.min // 1000 or raw_us.max() > int64_info.max // 1000:
            return times

        # Pandas 3 exposes these parsed UTC datetimes as epoch microseconds.
        # First recover the corrupted numeric timestamp in ns-like units.
        bad_ns = raw_us * 1000
        hour_ns = 3_600_000_000_000
        lower_ns = (reference_utc - pd.Timedelta(days=45)).value
        upper_ns = (reference_utc + pd.Timedelta(days=2)).value
        missing = int64_info.min
        repaired_ns = np.full(raw_us.shape, missing, dtype=np.int64)

        for horizon_hour in range(1, 25):
            horizon_ns = horizon_hour * hour_ns

            # Legacy bug: bad_ns = anchor_epoch_microseconds + horizon_nanoseconds.
            # Direct inverse: anchor_us = bad_ns - horizon_ns; good_ns = anchor_us * 1000 + horizon_ns.
            anchor_us = bad_ns - horizon_ns
            can_scale = (
                (anchor_us >= (int64_info.min - horizon_ns) // 1000)
                & (anchor_us <= (int64_info.max - horizon_ns) // 1000)
            )
            candidate_ns = np.full(raw_us.shape, missing, dtype=np.int64)
            candidate_ns[can_scale] = anchor_us[can_scale] * 1000 + horizon_ns
            mask = (
                (repaired_ns == missing)
                & can_scale
                & (candidate_ns >= lower_ns)
                & (candidate_ns <= upper_ns)
            )
            repaired_ns[mask] = candidate_ns[mask]

        repaired_mask = np.not_equal(repaired_ns, missing)
        if not repaired_mask.any():
            return times

        repaired = pd.Series(pd.NaT, index=valid_times.index, dtype="datetime64[ns, UTC]")
        repaired.loc[repaired_mask] = pd.to_datetime(
            repaired_ns[repaired_mask],
            errors="coerce",
            utc=True,
        )
        repaired_valid = repaired.dropna()
        if repaired_valid.empty or len(repaired_valid) != int(repaired_mask.sum()):
            return times
        if repaired_valid.min() < reference_utc - pd.Timedelta(days=45):
            return times
        if repaired_valid.max() > reference_utc + pd.Timedelta(days=2):
            return times
        if not repaired_valid.sort_values().is_monotonic_increasing:
            return times

        out = times.copy()
        out.loc[valid_times.index] = repaired
        return out

    challenger_color = "#1f77b4"
    champion_color = MODEL_GATE_CHAMPION_COLOR
    harmonie_color = "gray"
    plot_meta_text = _format_last_plot_update_text(datetime.now(timezone.utc).isoformat(), local_tz)
    champion_model_id = None
    challenger_model_id = None
    latest_history_run_utc = None
    harmonie_label = "Harmonie prediction"
    if history_csv.exists():
        hist_for_id = pd.read_csv(history_csv)
        if not hist_for_id.empty:
            if "run_utc" in hist_for_id.columns:
                hist_for_id["run_utc"] = _parse_iso_series_utc(hist_for_id["run_utc"])
                hist_for_id = hist_for_id.dropna(subset=["run_utc"]).sort_values("run_utc")
            if not hist_for_id.empty:
                latest_history_run_utc = hist_for_id["run_utc"].iloc[-1]
                last_row = hist_for_id.iloc[-1]
                champion_model_id = str(last_row.get("speed_model_id_champion", "")).strip() or None
                challenger_model_id = str(last_row.get("speed_model_id_challenger", "")).strip() or None

    challenger_label = (
        f"Challenger ({_model_id_date_label(challenger_model_id)})"
        if challenger_model_id
        else "Challenger"
    )
    champion_label = (
        f"Champion ({_model_id_date_label(champion_model_id)})"
        if champion_model_id
        else "Champion"
    )

    # Preferred view: full holdout period time series with champion/challenger predictions.
    if eval_details_csv is not None and eval_details_csv.exists():
        det = pd.read_csv(eval_details_csv)
        if not det.empty and "target_time_utc" in det.columns:
            det["target_time_utc"] = pd.to_datetime(det["target_time_utc"], errors="coerce", utc=True)
            for col in ["actual_wind_speed", "forecast_wind_speed", "champion_wind_speed", "challenger_wind_speed"]:
                det[col] = pd.to_numeric(det.get(col), errors="coerce")
            if "forecast_wind_dir_deg" in det.columns:
                det["forecast_wind_dir_deg"] = pd.to_numeric(det["forecast_wind_dir_deg"], errors="coerce")
            direction_col = "forecast_wind_dir_deg" if "forecast_wind_dir_deg" in det.columns else None
            direction_label = "6-hour Harmonie wind direction"
            if direction_col is None and db_path is not None and site:
                try:
                    conn = sqlite3.connect(str(db_path))
                    try:
                        obs_raw = _load_observations_raw(conn, site)
                    finally:
                        conn.close()
                    if not obs_raw.empty and "actual_dir" in obs_raw.columns:
                        hourly_dir = obs_raw["actual_dir"].dropna().resample("1h").agg(_circular_mean_deg).dropna()
                        if not hourly_dir.empty:
                            det["actual_wind_dir_deg"] = det["target_time_utc"].dt.floor("h").map(hourly_dir)
                            direction_col = "actual_wind_dir_deg"
                            direction_label = "6-hour measured wind direction"
                except Exception:
                    direction_col = None
            det = det.dropna(subset=["target_time_utc", "actual_wind_speed"])
            if latest_history_run_utc is not None and not det.empty:
                latest_detail_target_utc = det["target_time_utc"].dropna().max()
                if (
                    pd.notna(latest_detail_target_utc)
                    and latest_detail_target_utc < latest_history_run_utc - pd.Timedelta(days=28)
                ):
                    repaired_target_time_utc = _repair_legacy_microsecond_detail_times(
                        det["target_time_utc"],
                        latest_history_run_utc,
                    )
                    repaired_latest = repaired_target_time_utc.dropna().max()
                    if (
                        pd.notna(repaired_latest)
                        and repaired_latest >= latest_history_run_utc - pd.Timedelta(days=28)
                    ):
                        det = det.copy()
                        det["target_time_utc"] = repaired_target_time_utc
                    else:
                        det = pd.DataFrame()
            if not det.empty:
                det = det.sort_values("target_time_utc").reset_index(drop=True)
                display_window_days = 14
                display_window_weeks = max(1, int(round(display_window_days / 7)))
                holdout_weeks = max(
                    1,
                    int(
                        round(
                            (
                                (det["target_time_utc"].max() - det["target_time_utc"].min())
                                / pd.Timedelta(days=7)
                            )
                        )
                    ),
                )
                full_forecast_abs_err = np.abs(det["forecast_wind_speed"] - det["actual_wind_speed"])
                full_champ_abs_err = np.abs(det["champion_wind_speed"] - det["actual_wind_speed"])
                full_chall_abs_err = np.abs(det["challenger_wind_speed"] - det["actual_wind_speed"])
                mae_forecast = float(np.nanmean(full_forecast_abs_err))
                mae_chall = float(np.nanmean(full_chall_abs_err))
                mae_champ = float(np.nanmean(full_champ_abs_err))

                x_end_utc = det["target_time_utc"].dropna().max()
                if pd.isna(x_end_utc):
                    return
                cutoff_utc = x_end_utc - pd.Timedelta(days=display_window_days)
                det_view = det[det["target_time_utc"] >= cutoff_utc].copy()
                if det_view.empty:
                    return
                x_end_local = x_end_utc.tz_convert(ZoneInfo(local_tz))
                x_start_local = x_end_local - pd.Timedelta(days=display_window_days)
                det_view["forecast_abs_err"] = np.abs(det_view["forecast_wind_speed"] - det_view["actual_wind_speed"])
                det_view["champion_abs_err"] = np.abs(det_view["champion_wind_speed"] - det_view["actual_wind_speed"])
                det_view["challenger_abs_err"] = np.abs(det_view["challenger_wind_speed"] - det_view["actual_wind_speed"])

                plot_cols = [
                    "actual_wind_speed",
                    "forecast_wind_speed",
                    "challenger_wind_speed",
                    "champion_wind_speed",
                    "forecast_abs_err",
                    "challenger_abs_err",
                    "champion_abs_err",
                ]
                det_plot = (
                    det_view.set_index("target_time_utc")
                    .sort_index()[plot_cols]
                    .resample("1h")
                    .mean(numeric_only=True)
                    .dropna(how="all")
                    .reset_index()
                )
                det_plot = det_plot.dropna(subset=["target_time_utc", "actual_wind_speed"])
                if det_plot.empty:
                    return
                x_local = det_plot["target_time_utc"].dt.tz_convert(ZoneInfo(local_tz))

                fig, (ax_top, ax_bottom) = plt.subplots(
                    2,
                    1,
                    sharex=True,
                    figsize=(11.4, 6.8),
                    gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.40},
                )

                ax_top.plot(x_local, det_plot["actual_wind_speed"], color="magenta", linewidth=1.5, label="_nolegend_")
                ax_top.plot(
                    x_local,
                    det_plot["forecast_wind_speed"],
                    color=harmonie_color,
                    linewidth=1.4,
                    label="_nolegend_",
                )
                ax_top.plot(
                    x_local,
                    det_plot["challenger_wind_speed"],
                    color=challenger_color,
                    linewidth=1.5,
                    label="_nolegend_",
                )
                ax_top.plot(
                    x_local,
                    det_plot["champion_wind_speed"],
                    color=champion_color,
                    linewidth=1.5,
                    label="_nolegend_",
                )
                ax_top.set_title("Next-day model selection: wind speed")
                ax_top.set_ylabel("Wind speed (kts)")
                ax_top.grid(axis="y", alpha=0.3)
                ax_top.margins(x=0, y=0)
                ax_top.set_xlim(x_start_local, x_end_local)
                ax_top.text(
                    0.015,
                    1.10,
                    plot_meta_text,
                    transform=ax_top.transAxes,
                    ha="left",
                    va="top",
                    fontsize=9,
                    color="black",
                    clip_on=False,
                )
                ymax = np.nanmax(
                    [
                        det_plot["actual_wind_speed"].max(skipna=True),
                        det_plot["forecast_wind_speed"].max(skipna=True),
                        det_plot["challenger_wind_speed"].max(skipna=True),
                        det_plot["champion_wind_speed"].max(skipna=True),
                        1.0,
                    ]
                )
                _apply_speed_background(
                    ax_top,
                    float(ymax) * 1.08,
                    x_left=mdates.date2num(x_start_local.to_pydatetime()),
                    x_right=mdates.date2num(x_end_local.to_pydatetime()),
                )
                ax_top.set_ylim(0.0, max(4.0, float(ymax) * 1.08))

                top_legend_labels = [
                    "Measurement",
                    "Harmonie",
                    challenger_label,
                    champion_label,
                ]

                def _split_top_label(label: str) -> tuple[str, str]:
                    if " (" in label and label.endswith(")"):
                        name, suffix = label.split(" (", 1)
                        return name, f"({suffix}"
                    return label, ""

                split_top_labels = [_split_top_label(label) for label in top_legend_labels]
                top_name_width = max(len(name) for name, _ in split_top_labels)
                top_date_width = max(len(date_txt) for _, date_txt in split_top_labels)

                def _top_legend_entry(text: str, color: str) -> HPacker:
                    label_name, label_date = _split_top_label(text)
                    padded_name = label_name.ljust(top_name_width)
                    padded_date = label_date.ljust(top_date_width)
                    return HPacker(
                        children=[
                            TextArea(
                                "━━ ",
                                textprops={"fontsize": 9, "color": color, "fontweight": "bold", "fontfamily": "monospace"},
                            ),
                            TextArea(
                                f"{padded_name} ",
                                textprops={"fontsize": 9, "color": "black", "fontfamily": "monospace"},
                            ),
                            TextArea(
                                padded_date,
                                textprops={"fontsize": 9, "color": "black", "fontfamily": "monospace"},
                            ),
                        ],
                        align="center",
                        pad=0,
                        sep=0,
                    )

                top_legend_box = VPacker(
                    children=[
                        TextArea(
                            "Wind speed: measurement & prediction",
                            textprops={"fontsize": 9, "color": "black", "fontweight": "bold"},
                        ),
                        _top_legend_entry("Measurement", "magenta"),
                        _top_legend_entry("Harmonie", harmonie_color),
                        _top_legend_entry(challenger_label, challenger_color),
                        _top_legend_entry(champion_label, champion_color),
                    ],
                    align="left",
                    pad=0,
                    sep=1,
                )
                top_legend = AnchoredOffsetbox(
                    loc="upper left",
                    child=top_legend_box,
                    pad=0.2,
                    frameon=True,
                    bbox_to_anchor=(0.01, 0.995),
                    bbox_transform=ax_top.transAxes,
                    borderpad=0.35,
                )
                top_legend.patch.set_facecolor("white")
                top_legend.patch.set_alpha(0.78)
                top_legend.patch.set_edgecolor("none")
                ax_top.add_artist(top_legend)

                ax_bottom.plot(
                    x_local,
                    det_plot["forecast_abs_err"],
                    color=harmonie_color,
                    linewidth=1.3,
                    label=f"Harmonie ({mae_forecast:.2f} kts)",
                )
                ax_bottom.plot(
                    x_local,
                    det_plot["challenger_abs_err"],
                    color=challenger_color,
                    linewidth=1.4,
                    label=f"Challenger ({mae_chall:.2f} kts)",
                )
                ax_bottom.plot(
                    x_local,
                    det_plot["champion_abs_err"],
                    color=champion_color,
                    linewidth=1.4,
                    label=f"Champion ({mae_champ:.2f} kts)",
                )
                ax_bottom.axhline(
                    mae_forecast,
                    color=harmonie_color,
                    linestyle="--",
                    linewidth=1.1,
                    alpha=0.9,
                    label="_nolegend_",
                )
                ax_bottom.axhline(
                    mae_chall,
                    color=challenger_color,
                    linestyle="--",
                    linewidth=1.1,
                    alpha=0.9,
                    label="_nolegend_",
                )
                ax_bottom.axhline(
                    mae_champ,
                    color=champion_color,
                    linestyle="--",
                    linewidth=1.1,
                    alpha=0.9,
                    label="_nolegend_",
                )
                ax_bottom.set_title(
                    f"Hourly mean absolute error\nShowing last {display_window_weeks} weeks; model selection based on {holdout_weeks} weeks of hold-out data"
                )
                ax_bottom.set_ylabel("Absolute error (kts)")
                ax_bottom.set_xlabel("Time")
                ax_bottom.grid(axis="y", alpha=0.3)
                ax_bottom.margins(x=0, y=0)
                ax_bottom.set_xlim(x_start_local, x_end_local)
                mae_top = np.nanmax(
                    [
                        det_plot["forecast_abs_err"].max(skipna=True),
                        det_plot["challenger_abs_err"].max(skipna=True),
                        det_plot["champion_abs_err"].max(skipna=True),
                        mae_forecast,
                        mae_chall,
                        mae_champ,
                        1.0,
                    ]
                )
                ax_bottom.set_ylim(0.0, max(3.5, float(mae_top) * 1.08))
                date_locator = mdates.DayLocator(interval=3)
                ax_bottom.xaxis.set_major_locator(date_locator)
                ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter("%d %b", tz=ZoneInfo(local_tz)))
                ax_bottom.tick_params(axis="x", labelrotation=20, labelsize=10, pad=6)

                if direction_col is not None and direction_col in det_view.columns and det_view[direction_col].notna().any():
                    direction_bins = (
                        det_view.dropna(subset=[direction_col])
                        .set_index("target_time_utc")
                        .sort_index()[direction_col]
                        .resample("6h")
                        .agg(_circular_mean_deg)
                        .dropna()
                    )
                    y_base_axes = -0.21
                    arrow_len_axes = 0.062
                    for ts_utc, direction_deg in direction_bins.items():
                        ts_local = ts_utc.tz_convert(ZoneInfo(local_tz)) + pd.Timedelta(hours=3)
                        if ts_local < x_start_local or ts_local > x_end_local:
                            continue
                        theta = np.deg2rad((float(direction_deg) + 180.0) % 360.0)
                        dx_days = 0.065 * np.sin(theta)
                        dy = arrow_len_axes * np.cos(theta)
                        ax_bottom.annotate(
                            "",
                            xy=(mdates.date2num(ts_local.to_pydatetime()) + dx_days, y_base_axes + dy),
                            xytext=(mdates.date2num(ts_local.to_pydatetime()), y_base_axes),
                            xycoords=ax_bottom.get_xaxis_transform(),
                            textcoords=ax_bottom.get_xaxis_transform(),
                            arrowprops={
                                "arrowstyle": "-|>",
                                "color": "#4d4d4d",
                                "lw": 1.0,
                                "shrinkA": 0,
                                "shrinkB": 0,
                            },
                            clip_on=False,
                            zorder=5,
                        )
                    ax_bottom.text(
                        0.0,
                        -0.31,
                        direction_label,
                        transform=ax_bottom.transAxes,
                        ha="left",
                        va="top",
                        fontsize=8.5,
                        color="#4d4d4d",
                        clip_on=False,
                    )

                bottom_legend_labels = [
                    ("Harmonie", mae_forecast, harmonie_color),
                    (challenger_label, mae_chall, challenger_color),
                    (champion_label, mae_champ, champion_color),
                ]
                def _split_legend_label(label: str) -> tuple[str, str]:
                    if " (" in label and label.endswith(")"):
                        name, suffix = label.split(" (", 1)
                        return name, f"({suffix}"
                    return label, ""

                split_bottom_labels = [(_split_legend_label(label), value, color) for label, value, color in bottom_legend_labels]
                legend_name_width = max(len(name) for (name, _), _, _ in split_bottom_labels)
                legend_date_width = max(len(date_txt) for (_, date_txt), _, _ in split_bottom_labels)

                def _mae_legend_entry(model_label: str, value: float, color: str) -> HPacker:
                    model_name, model_date = _split_legend_label(model_label)
                    padded_name = model_name.ljust(legend_name_width)
                    padded_date = model_date.ljust(legend_date_width)
                    return HPacker(
                        children=[
                            TextArea(
                                "━━ ",
                                textprops={"fontsize": 9, "color": color, "fontweight": "bold", "fontfamily": "monospace"},
                            ),
                            TextArea(
                                f"{padded_name} ",
                                textprops={"fontsize": 9, "color": "black", "fontfamily": "monospace"},
                            ),
                            TextArea(
                                f"{padded_date} ",
                                textprops={"fontsize": 9, "color": "black", "fontfamily": "monospace"},
                            ),
                            TextArea(
                                f"{value:>4.2f} kts",
                                textprops={"fontsize": 9, "color": color, "fontweight": "bold", "fontfamily": "monospace"},
                            ),
                        ],
                        align="center",
                        pad=0,
                        sep=0,
                    )

                bottom_legend_box = VPacker(
                    children=[
                        TextArea(
                            "Mean absolute error (hourly)",
                            textprops={"fontsize": 9, "color": "black", "fontweight": "bold"},
                        ),
                        *[_mae_legend_entry(label, value, color) for label, value, color in bottom_legend_labels],
                    ],
                    align="left",
                    pad=0,
                    sep=1,
                )
                bottom_legend = AnchoredOffsetbox(
                    loc="upper right",
                    child=bottom_legend_box,
                    pad=0.2,
                    frameon=True,
                    bbox_to_anchor=(0.99, 0.995),
                    bbox_transform=ax_bottom.transAxes,
                    borderpad=0.35,
                )
                bottom_legend.patch.set_facecolor("white")
                bottom_legend.patch.set_alpha(0.78)
                bottom_legend.patch.set_edgecolor("none")
                ax_bottom.add_artist(bottom_legend)

                for ax in (ax_top, ax_bottom):
                    ax.set_xlim(x_start_local, x_end_local)
                fig.subplots_adjust(left=0.07, right=0.98, top=0.89, bottom=0.19)
                fig.savefig(plot_png, dpi=150)
                plt.close(fig)
                return

    # Fallback: run-level trend if period details are unavailable.
    if not history_csv.exists():
        return
    hist = pd.read_csv(history_csv)
    if hist.empty:
        return
    for col in ["speed_mae_forecast", "speed_mae_champion", "speed_mae_challenger", "speed_eval_samples"]:
        hist[col] = pd.to_numeric(hist.get(col), errors="coerce")
    if "run_local_time" in hist.columns:
        run_dt = pd.to_datetime(hist["run_local_time"], errors="coerce", utc=True).dt.tz_convert(ZoneInfo(local_tz))
    else:
        run_dt = _parse_iso_series_utc(hist.get("run_utc")).dt.tz_convert(ZoneInfo(local_tz))
    hist["run_dt"] = run_dt
    hist = hist.dropna(subset=["run_dt"]).sort_values("run_dt")
    if hist.empty:
        return
    display_window_days = 14
    x_end_local = hist["run_dt"].dropna().max()
    if pd.isna(x_end_local):
        return
    x_start_local = x_end_local - pd.Timedelta(days=display_window_days)
    hist_view = hist[hist["run_dt"] >= x_start_local].copy()
    if hist_view.empty:
        return
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(11.4, 6.8),
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.40},
    )
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
    forecast_mae_label = "Harmonie holdout MAE"
    promoted_speed = hist.get("promote_speed", pd.Series(False, index=hist.index)).astype(str).str.strip().str.lower().isin(
        ["true", "1", "yes"]
    )
    ax_top.plot(
        hist_view["run_dt"],
        hist_view["speed_eval_samples"],
        color="#444444",
        linewidth=1.4,
        marker="o",
        markersize=3.0,
        label="Holdout samples",
    )
    if promoted_speed.any():
        promoted_view = promoted_speed.reindex(hist_view.index, fill_value=False)
        ax_top.scatter(
            hist_view.loc[promoted_view, "run_dt"],
            hist_view.loc[promoted_view, "speed_eval_samples"],
            color="#2ca02c",
            s=34,
            zorder=3,
            label="Promotion run",
        )
    ax_top.set_title("Next-day model selection: evaluation coverage")
    ax_top.set_ylabel("Samples")
    ax_top.grid(axis="y", alpha=0.3)
    ax_top.legend(loc="upper left", fontsize=9)
    ax_top.margins(x=0, y=0)
    ax_top.text(
        0.015,
        1.10,
        plot_meta_text,
        transform=ax_top.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="black",
        clip_on=False,
    )
    samples_top = np.nanmax([hist_view["speed_eval_samples"].max(skipna=True), 1.0])
    ax_top.set_ylim(0.0, max(10.0, float(samples_top) * 1.08))

    ax_bottom.plot(
        hist_view["run_dt"],
        hist_view["speed_mae_champion"],
        color=champion_color,
        linewidth=1.4,
        marker="o",
        markersize=3.0,
        label=champ_mae_label,
    )
    ax_bottom.plot(
        hist_view["run_dt"],
        hist_view["speed_mae_forecast"],
        color=harmonie_color,
        linewidth=1.3,
        marker="o",
        markersize=3.0,
        label=forecast_mae_label,
    )
    ax_bottom.plot(
        hist_view["run_dt"],
        hist_view["speed_mae_challenger"],
        color=challenger_color,
        linewidth=1.4,
        marker="o",
        markersize=3.0,
        label=chall_mae_label,
    )
    ax_bottom.set_title("Next-day model selection: mean absolute error")
    ax_bottom.set_xlabel("Run date")
    ax_bottom.set_ylabel("MAE (kts)")
    ax_bottom.grid(axis="y", alpha=0.3)
    ax_bottom.legend(loc="upper right", fontsize=9)
    ax_bottom.margins(x=0, y=0)
    ymax = np.nanmax(
        [
            hist_view["speed_mae_forecast"].max(skipna=True),
            hist_view["speed_mae_champion"].max(skipna=True),
            hist_view["speed_mae_challenger"].max(skipna=True),
            1.0,
        ]
    )
    ax_bottom.set_ylim(0.0, max(3.5, float(ymax) * 1.08))
    date_locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax_bottom.xaxis.set_major_locator(date_locator)
    ax_bottom.xaxis.set_major_formatter(mdates.ConciseDateFormatter(date_locator))
    ax_bottom.tick_params(axis="x", labelrotation=20, labelsize=10, pad=6)
    for ax in (ax_top, ax_bottom):
        ax.set_xlim(x_start_local, x_end_local)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.89, bottom=0.10)
    fig.savefig(plot_png, dpi=150)
    plt.close(fig)


def _direction_performance_summary_text(direction_csv: Path | None) -> str:
    if direction_csv is None or not direction_csv.exists():
        return (
            "The spider diagram compares mean absolute error by forecast wind direction. "
            "Lower values are better."
        )
    try:
        df = pd.read_csv(direction_csv)
    except Exception:
        return (
            "The spider diagram compares mean absolute error by forecast wind direction. "
            "Lower values are better."
        )
    required = {"sector", "forecast_mae", "champion_mae"}
    if df.empty or not required.issubset(df.columns):
        return (
            "The spider diagram compares mean absolute error by forecast wind direction. "
            "Lower values are better."
        )
    for col in ["forecast_mae", "champion_mae"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["gain"] = df["forecast_mae"] - df["champion_mae"]
    df = df.dropna(subset=["sector", "gain"])
    if df.empty:
        return (
            "The spider diagram compares mean absolute error by forecast wind direction. "
            "Lower values are better."
        )

    strong = df[df["gain"] >= 0.25].sort_values("gain", ascending=False)
    weak = df[df["gain"] < 0.10].sort_values("gain")
    parts = [
        "The spider diagram compares mean absolute error by forecast wind direction; lower values are better."
    ]
    if not strong.empty:
        sectors = ", ".join(str(v) for v in strong["sector"].head(4).tolist())
        parts.append(f"The next-day champion improves most for {sectors} winds.")
    if not weak.empty:
        sectors = ", ".join(str(v) for v in weak["sector"].head(4).tolist())
        parts.append(f"The advantage is smallest for {sectors} winds.")
    else:
        parts.append("Across all shown wind sectors the champion is better than Harmonie.")
    return " ".join(parts)


def save_wind_direction_performance_spider_plot(direction_csv: Path, plot_png: Path) -> None:
    if not direction_csv.exists():
        return
    df = pd.read_csv(direction_csv)
    required = {"sector", "forecast_mae", "champion_mae"}
    if df.empty or not required.issubset(df.columns):
        return
    order = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    df = df.copy()
    for col in ["forecast_mae", "champion_mae"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "n_points" in df.columns:
        df["n_points"] = pd.to_numeric(df["n_points"], errors="coerce")
    df["sector"] = df["sector"].astype(str)
    df = df.set_index("sector").reindex(order).dropna(subset=["forecast_mae", "champion_mae"])
    if len(df) < 3:
        return

    labels = df.index.to_list()
    harmonie = df["forecast_mae"].to_numpy(dtype=float)
    champion = df["champion_mae"].to_numpy(dtype=float)
    weights = df["n_points"].to_numpy(dtype=float) if "n_points" in df.columns else np.ones(len(df), dtype=float)
    if np.isnan(weights).any() or float(np.nansum(weights)) <= 0.0:
        weights = np.ones(len(df), dtype=float)
    harmonie_mae = float(np.average(harmonie, weights=weights))
    champion_mae = float(np.average(champion, weights=weights))
    angles = np.linspace(0.0, 2.0 * np.pi, len(labels), endpoint=False)
    angles_closed = np.r_[angles, angles[0]]
    harmonie_closed = np.r_[harmonie, harmonie[0]]
    champion_closed = np.r_[champion, champion[0]]

    fig = plt.figure(figsize=(7.4, 7.0))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.plot(
        angles_closed,
        harmonie_closed,
        color="#777777",
        linewidth=2.0,
        marker="o",
        markersize=4,
        label=f"Harmonie ({harmonie_mae:.2f} kts)",
    )
    ax.plot(
        angles_closed,
        champion_closed,
        color="#f28e2b",
        linewidth=2.3,
        marker="o",
        markersize=4,
        label=f"Super local champion model next-day ({champion_mae:.2f} kts)",
    )
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0.0, 3.5)
    ax.set_rlabel_position(225)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(color="#d7d7d7", linewidth=0.8)
    ax.spines["polar"].set_color("#cfcfcf")
    ax.set_title("MAE for next-day models by forecast wind direction", pad=22, fontsize=14, fontweight="bold")
    ax.text(
        0.5,
        -0.08,
        "Lower radial value means lower mean absolute error (kts).",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color="#444444",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1, frameon=False, fontsize=10)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.90, bottom=0.18)
    fig.savefig(plot_png, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def save_current_day_direction_performance_csv(
    db_path: Path,
    cfg: DatasetConfig,
    direction_csv: Path,
) -> Path | None:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = pd.read_sql_query(
            """
            WITH first_issue AS (
                SELECT
                    anchor_ts,
                    target_ts,
                    MIN(issued_ts) AS issued_ts
                FROM prediction_log
                WHERE site = ?
                  AND model_type = 'intraday'
                  AND prediction_kind = 'wind_speed'
                  AND actual_value IS NOT NULL
                  AND prediction_value IS NOT NULL
                  AND harmonie_value IS NOT NULL
                  AND harmonie_run_ts IS NOT NULL
                GROUP BY anchor_ts, target_ts
            )
            SELECT
                pl.anchor_ts,
                pl.issued_ts,
                pl.target_ts,
                pl.horizon_hr,
                pl.prediction_value,
                pl.harmonie_value,
                pl.actual_value,
                fc.wind_dir AS forecast_wind_dir_deg
            FROM prediction_log AS pl
            INNER JOIN first_issue AS fi
                ON pl.anchor_ts = fi.anchor_ts
               AND pl.target_ts = fi.target_ts
               AND pl.issued_ts = fi.issued_ts
            INNER JOIN forecasts AS fc
                ON fc.site = pl.site
               AND fc.model = ?
               AND fc.run_ts = pl.harmonie_run_ts
               AND fc.target_ts = pl.target_ts
            WHERE pl.site = ?
              AND pl.model_type = 'intraday'
              AND pl.prediction_kind = 'wind_speed'
              AND pl.actual_value IS NOT NULL
              AND fc.wind_dir IS NOT NULL
            ORDER BY pl.target_ts ASC
            """,
            conn,
            params=[cfg.site, cfg.model, cfg.site],
        )
    finally:
        conn.close()

    columns = [
        "sector",
        "dir_min_deg",
        "dir_max_deg",
        "n_points",
        "forecast_mae",
        "superlocal_mae",
        "mae_gain_vs_harmonie",
        "forecast_bias_pred_minus_actual",
        "superlocal_bias_pred_minus_actual",
        "bias_adjustment_vs_harmonie",
    ]
    direction_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows.empty:
        pd.DataFrame(columns=columns).to_csv(direction_csv, index=False)
        return None

    numeric_cols = ["prediction_value", "harmonie_value", "actual_value", "forecast_wind_dir_deg"]
    for col in numeric_cols:
        rows[col] = pd.to_numeric(rows[col], errors="coerce")
    rows = rows.dropna(subset=numeric_cols).copy()
    if rows.empty:
        pd.DataFrame(columns=columns).to_csv(direction_csv, index=False)
        return None

    sector_order = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    sector_labels = np.array(sector_order)
    sector_idx = (np.floor(((rows["forecast_wind_dir_deg"].to_numpy(dtype=float) % 360.0) + 22.5) / 45.0).astype(int) % 8)
    rows["sector"] = sector_labels[sector_idx]
    rows["model_abs_error"] = np.abs(rows["prediction_value"] - rows["actual_value"])
    rows["harmonie_abs_error"] = np.abs(rows["harmonie_value"] - rows["actual_value"])
    rows["model_error"] = rows["prediction_value"] - rows["actual_value"]
    rows["harmonie_error"] = rows["harmonie_value"] - rows["actual_value"]

    summary_rows = []
    for idx, sector in enumerate(sector_order):
        group = rows[rows["sector"] == sector]
        if group.empty:
            summary_rows.append(
                {
                    "sector": sector,
                    "dir_min_deg": float((idx * 45 - 22.5) % 360.0),
                    "dir_max_deg": float((idx * 45 + 22.5) % 360.0),
                    "n_points": 0,
                    "forecast_mae": np.nan,
                    "superlocal_mae": np.nan,
                    "mae_gain_vs_harmonie": np.nan,
                    "forecast_bias_pred_minus_actual": np.nan,
                    "superlocal_bias_pred_minus_actual": np.nan,
                    "bias_adjustment_vs_harmonie": np.nan,
                }
            )
            continue
        forecast_mae = float(group["harmonie_abs_error"].mean())
        superlocal_mae = float(group["model_abs_error"].mean())
        forecast_bias = float(group["harmonie_error"].mean())
        superlocal_bias = float(group["model_error"].mean())
        summary_rows.append(
            {
                "sector": sector,
                "dir_min_deg": float((idx * 45 - 22.5) % 360.0),
                "dir_max_deg": float((idx * 45 + 22.5) % 360.0),
                "n_points": int(len(group)),
                "forecast_mae": forecast_mae,
                "superlocal_mae": superlocal_mae,
                "mae_gain_vs_harmonie": forecast_mae - superlocal_mae,
                "forecast_bias_pred_minus_actual": forecast_bias,
                "superlocal_bias_pred_minus_actual": superlocal_bias,
                "bias_adjustment_vs_harmonie": superlocal_bias - forecast_bias,
            }
        )

    pd.DataFrame(summary_rows, columns=columns).to_csv(direction_csv, index=False)
    return direction_csv


def _bias_phrase(value: float) -> str:
    if not np.isfinite(value):
        return "has no stable bias estimate"
    if abs(value) < 0.15:
        return "is close to unbiased"
    direction = "overestimates" if value > 0 else "underestimates"
    return f"{direction} by {abs(value):.1f} kts"


def _current_day_direction_performance_summary_text(direction_csv: Path | None) -> str:
    if direction_csv is None or not direction_csv.exists():
        return (
            "The spider diagram compares realised current-day forecast error by forecast wind direction. "
            "Lower values are better."
        )
    try:
        df = pd.read_csv(direction_csv)
    except Exception:
        return (
            "The spider diagram compares realised current-day forecast error by forecast wind direction. "
            "Lower values are better."
        )
    required = {
        "sector",
        "forecast_mae",
        "superlocal_mae",
        "mae_gain_vs_harmonie",
        "superlocal_bias_pred_minus_actual",
    }
    if df.empty or not required.issubset(df.columns):
        return (
            "The spider diagram compares realised current-day forecast error by forecast wind direction. "
            "Lower values are better."
        )
    for col in ["forecast_mae", "superlocal_mae", "mae_gain_vs_harmonie", "superlocal_bias_pred_minus_actual"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["sector", "mae_gain_vs_harmonie"]).copy()
    if df.empty:
        return (
            "The spider diagram compares realised current-day forecast error by forecast wind direction. "
            "Lower values are better."
        )

    strong = df[df["mae_gain_vs_harmonie"] >= 0.25].sort_values("mae_gain_vs_harmonie", ascending=False)
    weak = df[df["mae_gain_vs_harmonie"] < 0.10].sort_values("mae_gain_vs_harmonie")
    parts = [
        "The spider diagram compares realised current-day forecast error by forecast wind direction; lower values are better."
    ]
    if not strong.empty:
        best = strong.iloc[0]
        sectors = ", ".join(str(v) for v in strong["sector"].head(4).tolist())
        parts.append(
            f"Super local improves most for {sectors} winds; the largest gain is {float(best['mae_gain_vs_harmonie']):.1f} kts for {best['sector']}."
        )
        parts.append(
            f"For that sector the super-local model {_bias_phrase(float(best['superlocal_bias_pred_minus_actual']))} on average."
        )
    if not weak.empty:
        sectors = ", ".join(str(v) for v in weak["sector"].head(4).tolist())
        parts.append(f"The advantage is smallest, or negative, for {sectors} winds.")
    else:
        parts.append("Across all shown wind sectors the super-local current-day model is better than Harmonie.")
    return " ".join(parts)


def save_current_day_direction_performance_spider_plot(direction_csv: Path, plot_png: Path) -> None:
    if not direction_csv.exists():
        return
    df = pd.read_csv(direction_csv)
    required = {"sector", "forecast_mae", "superlocal_mae"}
    if df.empty or not required.issubset(df.columns):
        return
    order = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    df = df.copy()
    for col in ["forecast_mae", "superlocal_mae"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "n_points" in df.columns:
        df["n_points"] = pd.to_numeric(df["n_points"], errors="coerce")
    df["sector"] = df["sector"].astype(str)
    df = df.set_index("sector").reindex(order).dropna(subset=["forecast_mae", "superlocal_mae"])
    if len(df) < 3:
        return

    labels = df.index.to_list()
    harmonie = df["forecast_mae"].to_numpy(dtype=float)
    superlocal = df["superlocal_mae"].to_numpy(dtype=float)
    weights = df["n_points"].to_numpy(dtype=float) if "n_points" in df.columns else np.ones(len(df), dtype=float)
    if np.isnan(weights).any() or float(np.nansum(weights)) <= 0.0:
        weights = np.ones(len(df), dtype=float)
    harmonie_mae = float(np.average(harmonie, weights=weights))
    superlocal_mae = float(np.average(superlocal, weights=weights))
    angles = np.linspace(0.0, 2.0 * np.pi, len(labels), endpoint=False)
    angles_closed = np.r_[angles, angles[0]]
    harmonie_closed = np.r_[harmonie, harmonie[0]]
    superlocal_closed = np.r_[superlocal, superlocal[0]]

    fig = plt.figure(figsize=(7.4, 7.0))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.plot(
        angles_closed,
        harmonie_closed,
        color="#777777",
        linewidth=2.0,
        marker="o",
        markersize=4,
        label=f"Harmonie ({harmonie_mae:.2f} kts)",
    )
    ax.plot(
        angles_closed,
        superlocal_closed,
        color="#f28e2b",
        linewidth=2.3,
        marker="o",
        markersize=4,
        label=f"Super local champion model current-day ({superlocal_mae:.2f} kts)",
    )
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0.0, 3.5)
    ax.set_rlabel_position(225)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(color="#d7d7d7", linewidth=0.8)
    ax.spines["polar"].set_color("#cfcfcf")
    ax.set_title("MAE for current-day models by forecast wind direction", pad=22, fontsize=14, fontweight="bold")
    ax.text(
        0.5,
        -0.08,
        "Lower radial value means lower mean absolute error (kts).",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color="#444444",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1, frameon=False, fontsize=10)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.90, bottom=0.18)
    plot_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_png, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def _json_ready_scalar(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _frame_to_json_records(frame: pd.DataFrame) -> list[dict]:
    records: list[dict] = []
    for row in frame.to_dict(orient="records"):
        clean: dict = {}
        for key, value in row.items():
            clean[key] = _json_ready_scalar(value)
        records.append(clean)
    return records


def _write_interactive_plot_assets(
    web_out_dir: Path,
    local_tz: str,
    current_day_csv: Path,
    next_day_csv: Path,
    current_day_prior_prediction_tables: list[pd.DataFrame] | None = None,
    prediction_generated_at_utc: str | None = None,
    prediction_updated_at_utc: str | None = None,
    model_trained_at_utc: str | None = None,
    harmonie_time_utc: datetime | pd.Timestamp | str | None = None,
    harmonie_time_kind: str = "fetched",
) -> dict:
    assets: dict[str, str] = {}

    script_src = REPO_ROOT / "next_day_wind_model" / "web_dashboard" / "dashboard_interactive.js"
    script_dst = web_out_dir / "dashboard_interactive.js"
    if script_src.exists():
        try:
            if script_src.resolve() != script_dst.resolve():
                shutil.copy2(script_src, script_dst)
        except FileNotFoundError:
            shutil.copy2(script_src, script_dst)
    if script_dst.exists():
        assets["dashboard_interactive_js"] = script_dst.name

    if current_day_csv.exists():
        frame = pd.read_csv(current_day_csv)
        if "time_local" in frame.columns:
            frame["time_local"] = pd.to_datetime(frame["time_local"], errors="coerce", utc=True)
            frame = frame.dropna(subset=["time_local"]).copy()
            frame["time_local"] = frame["time_local"].dt.tz_convert(ZoneInfo(local_tz)).dt.strftime("%Y-%m-%dT%H:%M:%S%z")
            if "is_future" in frame.columns:
                frame["is_future"] = (
                    frame["is_future"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .isin(["true", "1", "yes"])
                )

            active_anchor_local = None
            current_issue_anchor_local = None
            pred_dt = _parse_iso_utc(prediction_generated_at_utc)
            if pred_dt is not None:
                current_issue_anchor_local = (
                    pd.Timestamp(pred_dt).tz_convert(ZoneInfo(local_tz)).floor("h").strftime("%Y-%m-%dT%H:%M:%S%z")
                )
            if "is_future" in frame.columns:
                future_rows = frame[frame["is_future"]]
                if not future_rows.empty:
                    first_future = pd.to_datetime(future_rows["time_local"], errors="coerce", utc=True).dropna().min()
                    if pd.notna(first_future):
                        active_anchor_local = (
                            first_future.tz_convert(ZoneInfo(local_tz)) - pd.Timedelta(hours=1)
                        ).strftime("%Y-%m-%dT%H:%M:%S%z")
            if active_anchor_local is None:
                active_anchor_local = current_issue_anchor_local
            plot_meta_text = _format_plot_meta_text(
                prediction_generated_at_utc or "",
                prediction_updated_at_utc,
                model_trained_at_utc,
                local_tz,
                harmonie_time_utc=harmonie_time_utc,
                harmonie_time_kind=harmonie_time_kind,
            )

            prior_payload_rows: list[dict] = []
            for idx, prior_frame_raw in enumerate(current_day_prior_prediction_tables or []):
                if prior_frame_raw is None or prior_frame_raw.empty or "time_local" not in prior_frame_raw.columns:
                    continue
                prior_frame = prior_frame_raw.copy()
                prior_frame["time_local"] = pd.to_datetime(prior_frame["time_local"], errors="coerce", utc=True)
                prior_frame = prior_frame.dropna(subset=["time_local"]).copy()
                if prior_frame.empty:
                    continue
                prior_frame["time_local"] = prior_frame["time_local"].dt.tz_convert(ZoneInfo(local_tz)).dt.strftime("%Y-%m-%dT%H:%M:%S%z")
                if "issued_at_local" in prior_frame.columns:
                    prior_frame["issued_at_local"] = (
                        pd.to_datetime(prior_frame["issued_at_local"], errors="coerce", utc=True)
                        .dt.tz_convert(ZoneInfo(local_tz))
                        .dt.strftime("%Y-%m-%dT%H:%M:%S%z")
                    )
                elif "issued_at_utc" in prior_frame.columns:
                    prior_frame["issued_at_local"] = (
                        pd.to_datetime(prior_frame["issued_at_utc"], errors="coerce", utc=True)
                        .dt.tz_convert(ZoneInfo(local_tz))
                        .dt.strftime("%Y-%m-%dT%H:%M:%S%z")
                    )
                prior_frame["snapshot_index"] = idx
                prior_payload_rows.extend(_frame_to_json_records(prior_frame))

            payload = {
                "plot_kind": "current_day",
                "timezone": local_tz,
                "metadata": {
                    "active_anchor_local": active_anchor_local,
                    "current_issue_anchor_local": current_issue_anchor_local,
                    "plot_meta_text": plot_meta_text,
                },
                "rows": _frame_to_json_records(frame),
                "prior_rows": prior_payload_rows,
            }
            json_path = web_out_dir / "current_day_interactive_data.json"
            json_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
            assets["current_day_json"] = json_path.name

    if next_day_csv.exists():
        frame = pd.read_csv(next_day_csv)
        if "target_time_local" in frame.columns:
            frame["target_time_local"] = pd.to_datetime(frame["target_time_local"], errors="coerce", utc=True)
            frame = frame.dropna(subset=["target_time_local"]).copy()
            frame["target_time_local"] = frame["target_time_local"].dt.tz_convert(ZoneInfo(local_tz)).dt.strftime("%Y-%m-%dT%H:%M:%S%z")
            payload = {
                "plot_kind": "next_day",
                "timezone": local_tz,
                "metadata": {},
                "rows": _frame_to_json_records(frame),
            }
            json_path = web_out_dir / "next_day_interactive_data.json"
            json_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
            assets["next_day_json"] = json_path.name

    return assets


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
    direction_spider_png: Path | None,
    direction_spider_csv: Path | None,
    current_day_direction_spider_png: Path | None,
    current_day_direction_spider_csv: Path | None,
    current_day_prior_prediction_tables: list[pd.DataFrame] | None = None,
    prediction_generated_at_utc: str | None = None,
    prediction_updated_at_utc: str | None = None,
    model_trained_at_utc: str | None = None,
    harmonie_time_utc: datetime | pd.Timestamp | str | None = None,
    harmonie_time_kind: str = "fetched",
    companion_app_base_url: str | None = None,
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
        (direction_spider_png, "model_gate_direction_spider.png"),
        (direction_spider_csv, "model_gate_speed_by_direction.csv"),
        (current_day_direction_spider_png, "current_day_direction_spider.png"),
        (current_day_direction_spider_csv, "current_day_speed_by_direction.csv"),
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

    for existing_name in ["model_gate_eval_history.png", "model_gate_eval_history.csv"]:
        existing_path = web_out_dir / existing_name
        if existing_name not in copied and existing_path.exists():
            copied[existing_name] = str(existing_path)

    interactive_assets = _write_interactive_plot_assets(
        web_out_dir=web_out_dir,
        local_tz=local_tz,
        current_day_csv=current_day_csv,
        next_day_csv=next_day_csv,
        current_day_prior_prediction_tables=current_day_prior_prediction_tables,
        prediction_generated_at_utc=prediction_generated_at_utc,
        prediction_updated_at_utc=prediction_updated_at_utc,
        model_trained_at_utc=model_trained_at_utc,
        harmonie_time_utc=harmonie_time_utc,
        harmonie_time_kind=harmonie_time_kind,
    )
    if "dashboard_interactive_js" in interactive_assets:
        copied[interactive_assets["dashboard_interactive_js"]] = str(
            web_out_dir / interactive_assets["dashboard_interactive_js"]
        )
    if "current_day_json" in interactive_assets:
        copied[interactive_assets["current_day_json"]] = str(web_out_dir / interactive_assets["current_day_json"])
    if "next_day_json" in interactive_assets:
        copied[interactive_assets["next_day_json"]] = str(web_out_dir / interactive_assets["next_day_json"])

    generated_local = datetime.now(ZoneInfo(local_tz))
    generated_local_str = generated_local.strftime("%d %B %Y %H:%M:%S %Z")
    cache_bust = int(datetime.now(timezone.utc).timestamp())
    refresh = max(60, int(web_refresh_seconds))
    companion_base = (companion_app_base_url or "").rstrip("/")
    companion_links = ""
    if companion_base:
        companion_url = html.escape(companion_base)
        companion_links = f"""
    <nav class="dashboard-actions" aria-label="Rider portal">
      <a class="button primary" href="{companion_url}/">Rider portal</a>
      <a class="button" href="{companion_url}/?login=1&amp;next=%2Fexperience%2Fnew">New submission</a>
      <a class="button" href="{companion_url}/?login=1&amp;next=%2Fexperiences">My sessions</a>
    </nav>"""
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
    interactive_js_src = (
        f"dashboard_interactive.js?v={cache_bust}"
        if "dashboard_interactive.js" in copied
        else ""
    )
    current_day_json_src = (
        f"current_day_interactive_data.json?v={cache_bust}"
        if "current_day_interactive_data.json" in copied
        else ""
    )
    next_day_json_src = (
        f"next_day_interactive_data.json?v={cache_bust}"
        if "next_day_interactive_data.json" in copied
        else ""
    )
    gate_eval_card = ""
    if "model_gate_eval_history.png" in copied:
        gate_eval_src = f"model_gate_eval_history.png?v={cache_bust}"
        gate_eval_card = f"""
    <div class="card">
      <h2>Model-gate evaluation history</h2>
      <p class="desc">Top panel shows the holdout wind-speed comparison used by the model gate, including the aligned Harmonie baseline from the same holdout forecast inputs. Bottom panel shows the corresponding MAE comparison for Harmonie, challenger, and champion.</p>
      <img src="{gate_eval_src}" alt="Model gate evaluation history">
    </div>"""
    direction_spider_row = ""
    if "model_gate_direction_spider.png" in copied:
        direction_spider_src = f"model_gate_direction_spider.png?v={cache_bust}"
        direction_text = _direction_performance_summary_text(direction_spider_csv)
        direction_spider_row = f"""
      <div class="direction-card">
      <div class="direction-copy">
        <h3>Next-day performance by wind direction</h3>
        <p class="desc">{direction_text}</p>
      </div>
      <div class="direction-plot">
        <img src="{direction_spider_src}" alt="Next-day prediction performance by wind direction">
      </div>
    </div>"""
    current_day_direction_spider_row = ""
    if "current_day_direction_spider.png" in copied:
        current_day_direction_spider_src = f"current_day_direction_spider.png?v={cache_bust}"
        current_day_direction_text = _current_day_direction_performance_summary_text(
            current_day_direction_spider_csv
        )
        current_day_direction_spider_row = f"""
      <div class="direction-card">
      <div class="direction-copy">
        <h3>Current-day performance by wind direction</h3>
        <p class="desc">{current_day_direction_text}</p>
      </div>
      <div class="direction-plot">
        <img src="{current_day_direction_spider_src}" alt="Current-day prediction performance by wind direction">
      </div>
    </div>"""
    performance_section = ""
    if current_day_direction_spider_row or direction_spider_row:
        performance_section = f"""
    <section class="card performance-section">
      <h2 class="section-title">How much better are the super local forecasts?</h2>
{current_day_direction_spider_row}
{direction_spider_row}
    </section>"""
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="{refresh}">
  <title>Super local wind prediction Valkenburgse meer [under development]</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 16px; color: #111; }}
    h1 {{ margin: 0 0 8px 0; }}
    .page-header {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 18px; flex-wrap: wrap; }}
    .dashboard-actions {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; justify-content: flex-end; }}
    .button {{ display: inline-flex; align-items: center; justify-content: center; border: 1px solid #999; border-radius: 6px; padding: 8px 12px; color: #111; background: #f7f7f7; text-decoration: none; font-size: 15px; }}
    .button.primary {{ border-color: #135f86; background: #135f86; color: #fff; }}
    .meta {{ color: #555; margin: 0 0 16px 0; }}
    .overview {{ color: #222; margin: 0 0 16px 0; line-height: 1.5; font-size: 15px; max-width: 1200px; }}
    .overview-mobile {{ display: none; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 20px; max-width: 1400px; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 10px; background: #fff; }}
    .desc {{ margin: 2px 0 10px 0; color: #444; font-size: 16px; line-height: 1.4; }}
    .section-title {{ margin: 0 0 18px 0; font-size: 24px; line-height: 1.2; color: #111; }}
    .performance-section h3 {{ margin: 0 0 8px 0; font-size: 19px; }}
    .direction-card {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); align-items: start; gap: 18px; }}
    .direction-card + .direction-card {{ border-top: 1px solid #e6e6e6; margin-top: 18px; padding-top: 18px; }}
    .direction-copy {{ padding: 4px 8px 4px 2px; }}
    .direction-plot img {{ max-width: 620px; margin: 0 auto; }}
    img {{ width: 100%; height: auto; display: block; border-radius: 6px; }}
    .interactive-block {{ display: none; margin: 8px 0 8px 0; }}
    .interactive-block.is-ready {{ display: block; }}
    .interactive-block.is-ready + .interactive-fallback-note {{ display: none; }}
    .interactive-plot {{ width: 100%; min-height: 520px; }}
    .interactive-controls {{ display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin: 4px 0 6px 0; }}
    .interactive-control-label {{ font-size: 13px; color: #444; font-weight: 600; }}
    .interactive-control {{ border: 1px solid #aaa; border-radius: 4px; background: #fff; color: #111; font-size: 13px; padding: 5px 8px; }}
    .interactive-control.is-active {{ border-color: #555; background: #e9e9e9; font-weight: 600; }}
    .interactive-point-details {{ margin-top: 6px; border: 1px solid #ddd; border-radius: 6px; background: #f7f7f7; color: #222; font-size: 12px; line-height: 1.35; padding: 6px 8px; min-height: 38px; max-height: 56px; overflow: auto; }}
    .js-plotly-plot .modebar {{ opacity: 0.3; transition: opacity 0.15s ease-in-out; }}
    .js-plotly-plot:hover .modebar {{ opacity: 0.9; }}
    .interactive-fallback-note {{ margin: 4px 0 8px 0; color: #666; font-size: 13px; }}
    @media (max-width: 768px) {{
      body {{ margin: 10px; }}
      .page-header {{ display: block; }}
      .dashboard-actions {{ justify-content: flex-start; margin: 8px 0 10px 0; }}
      h1 {{ font-size: 24px; margin: 0 0 6px 0; }}
      h2 {{ font-size: 20px; margin: 0 0 6px 0; }}
      .section-title {{ font-size: 21px; margin: 0 0 12px 0; }}
      .performance-section h3 {{ font-size: 17px; margin: 0 0 6px 0; }}
      .meta {{ margin: 0 0 10px 0; font-size: 14px; }}
      .overview {{ font-size: 14px; margin: 0 0 12px 0; line-height: 1.45; }}
      .overview-desktop {{ display: none; }}
      .overview-mobile {{ display: block; margin: 10px 0 0 0; }}
      .grid {{ gap: 12px; }}
      .card {{ padding: 8px; border-radius: 6px; }}
      .desc {{ font-size: 14px; margin: 2px 0 8px 0; }}
      .direction-card {{ display: block; }}
      .direction-copy {{ padding: 0; }}
      .direction-plot img {{ max-width: none; margin: 0; }}
      img {{ max-height: 60vh; object-fit: contain; }}
      .interactive-plot {{ min-height: 390px; }}
    }}
  </style>
</head>
<body>
  <header class="page-header">
    <div>
      <h1>Super local wind prediction Valkenburgse meer [under development]</h1>
      <p class="meta">Last updated: {generated_local_str}</p>
    </div>
{companion_links}
  </header>
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
            <div
                class="interactive-block"
                data-interactive-wind-block="true"
                data-plot-id="current-day-interactive-plot"
                data-controls-id="current-day-interactive-controls"
                data-details-id="current-day-interactive-details"
                data-fallback-id="current-day-fallback"
                data-json-url="{current_day_json_src}"
            >
                <div class="interactive-controls" id="current-day-interactive-controls"></div>
                <div class="interactive-plot" id="current-day-interactive-plot" aria-label="Interactive current-day wind plot"></div>
                <div class="interactive-point-details" id="current-day-interactive-details">Click a plotted point to view exact values.</div>
            </div>
            <picture id="current-day-fallback">
        <source media="(max-width: 768px)" srcset="{current_day_mobile_src}">
        <img src="current_day_predictions.png?v={cache_bust}" alt="Current day prediction">
      </picture>
    </div>
    <div class="card">
      <h2>Next-day prediction</h2>
      <p class="desc">Day-ahead forecast for tomorrow: Harmonie baseline versus the super-local model for wind speed and direction.</p>
            <div
                class="interactive-block"
                data-interactive-wind-block="true"
                data-plot-id="next-day-interactive-plot"
                data-controls-id="next-day-interactive-controls"
                data-details-id="next-day-interactive-details"
                data-fallback-id="next-day-fallback"
                data-json-url="{next_day_json_src}"
            >
                <div class="interactive-controls" id="next-day-interactive-controls"></div>
                <div class="interactive-plot" id="next-day-interactive-plot" aria-label="Interactive next-day wind plot"></div>
                <div class="interactive-point-details" id="next-day-interactive-details">Click a plotted point to view exact values.</div>
            </div>
            <picture id="next-day-fallback">
        <source media="(max-width: 768px)" srcset="{next_day_mobile_src}">
        <img src="next_day_predictions.png?v={cache_bust}" alt="Next day prediction">
      </picture>
    </div>
{performance_section}
{gate_eval_card}
  </div>
  <p class="overview overview-mobile">
    <strong>What is the super local forecast?</strong> This dashboard combines two local machine learning models that take large-scale wind-model predictions as input and are trained on historical forecast values with matching measured wind values at this location.
    The local models are calibrated to local data to improve prediction performance by learning systematic local deviations from the large-scale model.
    One model is dedicated to the remaining part of the current day and gives strong weight to the most recent measured wind updates.
    A second model is dedicated to next-day (day-ahead) prediction.
    Models are retrained daily, next-day/current-day prediction lines are refreshed hourly during daytime, and measured-wind updates on the current-day plot are refreshed every 6 minutes.
  </p>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js" defer></script>
    {f'<script src="{interactive_js_src}" defer></script>' if interactive_js_src else ''}
    <script>
        window.addEventListener("DOMContentLoaded", function () {{
            if (window.WindDashboardInteractive && typeof window.WindDashboardInteractive.initInteractiveWindDashboard === "function") {{
                window.WindDashboardInteractive.initInteractiveWindDashboard();
            }}
        }});
    </script>
</body>
</html>
"""
    index_path = web_out_dir / "index.html"
    index_path.write_text(html_doc, encoding="utf-8")
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
        subprocess.run(["git", "-C", str(repo_root), "add", "-A", "-f", "--", rel_web_dir_s], check=True)
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
        return {"enabled": True, "pushed": False, "reason": "no_changes"}

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
    model_artifact_dir = Path(args.model_artifact_dir) if args.model_artifact_dir else out_dir
    model_artifact_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(args.db).resolve()

    print(f"Output artifact directory: {out_dir.resolve()}")
    print(f"Model artifact directory: {model_artifact_dir.resolve()}")
    if args.model_artifact_dir is None:
        print("Model artifact directory defaults to output artifact directory for backward compatibility.")

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

    speed_model_path = model_artifact_dir / "next_day_lstm_speed_residual.pt"
    direction_model_path = model_artifact_dir / "next_day_lstm_direction_residual.pt"
    intraday_model_path = model_artifact_dir / "intraday_speed_residual.pt"
    intraday_challenger_model_path = model_artifact_dir / "intraday_speed_residual_challenger.pt"
    speed_scalers_path = {
        "x_mean": model_artifact_dir / "x_mean_speed.npy",
        "x_std": model_artifact_dir / "x_std_speed.npy",
        "y_mean": model_artifact_dir / "y_mean_speed.npy",
        "y_std": model_artifact_dir / "y_std_speed.npy",
    }
    direction_scalers_path = {
        "x_mean": model_artifact_dir / "x_mean_direction.npy",
        "x_std": model_artifact_dir / "x_std_direction.npy",
        "y_mean": model_artifact_dir / "y_mean_direction.npy",
        "y_std": model_artifact_dir / "y_std_direction.npy",
    }
    intraday_hparams = {}
    intraday_model_last_trained_at_utc: str | None = None
    model_selection_report: dict = {"enabled": False, "reason": "skip_training"}
    intraday_model_selection_report: dict = {"enabled": False, "reason": "skip_training"}
    gate_eval_details_csv_src: Path | None = None
    gate_eval_direction_csv_src: Path | None = None
    gate_eval_direction_spider_png_src: Path | None = None
    current_day_direction_csv_src: Path | None = None
    current_day_direction_spider_png_src: Path | None = None
    intraday_gate_eval_details_csv_src: Path | None = None
    speed_calibration: dict | None = None
    pre_run_champion_states = _load_active_champion_states(
        speed_model_path=speed_model_path,
        direction_model_path=direction_model_path,
        intraday_model_path=intraday_model_path,
        device=device,
        local_tz=args.local_timezone,
    )

    if args.skip_training:
        _require_skip_training_artifacts(
            [
                speed_model_path,
                direction_model_path,
                intraday_model_path,
                *speed_scalers_path.values(),
                *direction_scalers_path.values(),
            ],
            model_artifact_dir=model_artifact_dir,
        )
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
        intraday_model_last_trained_at_utc = _resolve_model_trained_utc(intraday_ckpt, intraday_model_path)
        speed_train_stats = None
        direction_train_stats = None
        n_samples_all_speed = None
        n_samples_all_direction = None
        speed_feature_schema = _next_day_feature_schema_from_scalers(speed_arrays)
        if speed_feature_schema == "speed_v3_actual_history":
            feature_cols = [
                "forecast_avg",
                "forecast_max",
                "forecast_dir_sin",
                "forecast_dir_cos",
                "hour_sin",
                "hour_cos",
                "month_sin",
                "month_cos",
                "horizon_hr",
                "is_target",
                "history_actual_avg",
                "history_actual_max",
                "history_actual_dir_sin",
                "history_actual_dir_cos",
                "history_avg_residual",
                "history_max_residual",
            ]
        elif speed_feature_schema == "speed_v2":
            feature_cols = [
                "forecast_avg",
                "forecast_max",
                "forecast_dir_sin",
                "forecast_dir_cos",
                "hour_sin",
                "hour_cos",
                "month_sin",
                "month_cos",
                "horizon_hr",
                "is_target",
            ]
        else:
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
        intraday_model_selection_report = {"enabled": False, "reason": "skip_training"}
    else:
        if not (0.0 < float(args.challenge_eval_split) < 0.5):
            raise ValueError("--challenge-eval-split must be > 0 and < 0.5.")
        if not (0.0 < float(args.intraday_challenge_eval_split) < 0.5):
            raise ValueError("--intraday-challenge-eval-split must be > 0 and < 0.5.")

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
        speed_forecast_dir_eval = speed_arrays_full["target_forecast_dir_all_raw"][speed_eval_start:]
        speed_anchor_dir_train = speed_arrays_full["anchor_forecast_dir_all_raw"][:speed_eval_start]
        speed_anchor_dir_eval = speed_arrays_full["anchor_forecast_dir_all_raw"][speed_eval_start:]
        speed_train_anchor_times_utc = pd.to_datetime(speed_arrays_full["timestamps"][:speed_eval_start], utc=True)
        speed_eval_anchor_times_utc = pd.to_datetime(speed_arrays_full["timestamps"][speed_eval_start:], utc=True)
        speed_train_calibration_context = _build_speed_calibration_context(
            anchor_dir_deg=speed_anchor_dir_train,
            target_times_utc=speed_train_anchor_times_utc + pd.Timedelta(hours=1),
        )
        speed_train_calibration_context.update(
            {
                "target_forecast_dir_deg": speed_arrays_full["target_forecast_dir_all_raw"][:speed_eval_start],
                "target_times_utc": speed_arrays_full["target_times_all"][:speed_eval_start],
                "target_horizon_hr": speed_arrays_full["target_horizon_hr_all"][:speed_eval_start],
            }
        )
        speed_eval_calibration_context = _build_speed_calibration_context(
            anchor_dir_deg=speed_anchor_dir_eval,
            target_times_utc=speed_eval_anchor_times_utc + pd.Timedelta(hours=1),
        )
        speed_eval_calibration_context.update(
            {
                "target_forecast_dir_deg": speed_arrays_full["target_forecast_dir_all_raw"][speed_eval_start:],
                "target_times_utc": speed_arrays_full["target_times_all"][speed_eval_start:],
                "target_horizon_hr": speed_arrays_full["target_horizon_hr_all"][speed_eval_start:],
            }
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
        speed_model_challenger = TargetAwareNextDayLSTM(
            n_features=speed_X_train.shape[2],
            target_hours=speed_y_train.shape[1],
            history_hours=cfg.window_hours,
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
            speed_calibration_context=_slice_speed_calibration_context(
                speed_train_calibration_context,
                slice(speed_calibration_start, None),
            ),
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

        intraday_train_contexts, intraday_eval_contexts = build_intraday_holdout_context_split(
            db_path=db_path,
            cfg=cfg,
            holdout_eval_split=float(args.intraday_challenge_eval_split),
            holdout_min_contexts=int(args.intraday_challenge_min_eval_contexts),
        )
        intraday_bundle_challenger, intraday_train_stats = train_intraday_model(
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
            contexts=intraday_train_contexts,
        )
        intraday_train_stats["holdout_contexts"] = int(len(intraday_eval_contexts))

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
        speed_promotion_summary: dict | None = None
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

            champion_speed_feature_schema = _next_day_feature_schema_from_scalers(champion_speed_arrays)
            if champion_speed_feature_schema == str(speed_arrays_full.get("feature_schema", "speed_v2")):
                speed_X_eval_champion_raw = speed_X_eval_raw
                speed_eval_calibration_context_champion = speed_eval_calibration_context
            else:
                champion_speed_arrays_full = build_all_training_arrays(
                    db_path,
                    cfg,
                    target_mode="residual",
                    feature_schema=champion_speed_feature_schema,
                )
                champion_eval_start = _eval_start_index(
                    int(champion_speed_arrays_full["X_all"].shape[0]),
                    args.challenge_eval_split,
                    args.challenge_min_eval_samples,
                )
                speed_X_eval_champion_raw = _inverse_standardizer(
                    champion_speed_arrays_full["X_all"][champion_eval_start:],
                    champion_speed_arrays_full["x_mean"],
                    champion_speed_arrays_full["x_std"],
                ).astype(np.float32)
                champion_eval_anchor_times_utc = pd.to_datetime(
                    champion_speed_arrays_full["timestamps"][champion_eval_start:],
                    utc=True,
                )
                speed_eval_calibration_context_champion = _build_speed_calibration_context(
                    anchor_dir_deg=champion_speed_arrays_full["anchor_forecast_dir_all_raw"][champion_eval_start:],
                    target_times_utc=champion_eval_anchor_times_utc + pd.Timedelta(hours=1),
                )
                speed_eval_calibration_context_champion.update(
                    {
                        "target_forecast_dir_deg": champion_speed_arrays_full["target_forecast_dir_all_raw"][
                            champion_eval_start:
                        ],
                        "target_times_utc": champion_speed_arrays_full["target_times_all"][champion_eval_start:],
                        "target_horizon_hr": champion_speed_arrays_full["target_horizon_hr_all"][champion_eval_start:],
                    }
                )

            speed_X_eval_champion = _apply_standardizer(
                speed_X_eval_champion_raw, champion_speed_arrays["x_mean"], champion_speed_arrays["x_std"]
            ).astype(np.float32)
            champion_direction_feature_schema = _direction_feature_schema_from_scalers(champion_direction_arrays)
            if champion_direction_feature_schema == str(direction_arrays_full.get("feature_schema", "direction_v2")):
                direction_X_eval_champion_raw = direction_X_eval_raw
                direction_forecast_eval_champion = direction_forecast_eval
                direction_actual_eval_champion = direction_actual_eval
            else:
                champion_direction_arrays_full = build_all_direction_training_arrays(
                    db_path,
                    cfg,
                    feature_schema=champion_direction_feature_schema,
                )
                champion_direction_eval_start = _eval_start_index(
                    int(champion_direction_arrays_full["X_all"].shape[0]),
                    args.challenge_eval_split,
                    args.challenge_min_eval_samples,
                )
                direction_X_eval_champion_raw = _inverse_standardizer(
                    champion_direction_arrays_full["X_all"][champion_direction_eval_start:],
                    champion_direction_arrays_full["x_mean"],
                    champion_direction_arrays_full["x_std"],
                ).astype(np.float32)
                direction_forecast_eval_champion = champion_direction_arrays_full["y_forecast_all_raw"][
                    champion_direction_eval_start:
                ]
                direction_actual_eval_champion = champion_direction_arrays_full["y_actual_all_raw"][
                    champion_direction_eval_start:
                ]
            direction_X_eval_champion = _apply_standardizer(
                direction_X_eval_champion_raw,
                champion_direction_arrays["x_mean"],
                champion_direction_arrays["x_std"],
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
                speed_calibration_context=speed_eval_calibration_context_champion,
                device=device,
            )
            champion_direction_mae = _angular_mae_deg(
                _predict_direction_batch(
                    champion_direction_model,
                    direction_X_eval_champion,
                    direction_forecast_eval_champion,
                    y_mean=float(champion_direction_arrays["y_mean"][0]),
                    y_std=float(champion_direction_arrays["y_std"][0]),
                    device=device,
                ),
                direction_actual_eval_champion,
            )

            speed_promotion_summary = summarize_champion_vs_challenger(
                actual=speed_actual_eval,
                forecast=speed_forecast_eval,
                challenger_pred=challenger_speed_eval_pred,
                champion_pred=champion_speed_eval_pred,
                promotion_margin_pct=float(args.promotion_margin_pct),
                holdout_eval_split=float(args.challenge_eval_split),
                holdout_eval_min_samples=int(args.challenge_min_eval_samples),
                challenger_model_id=challenger_speed_model_id,
                champion_model_id=champion_speed_model_id,
            )
            challenger_speed_mae = float(speed_promotion_summary["speed_mae_challenger"])
            champion_speed_mae = float(speed_promotion_summary["speed_mae_champion"])
            forecast_speed_mae = float(speed_promotion_summary["speed_mae_forecast"])
            promote_speed = bool(speed_promotion_summary["promote_speed"])
            promote_direction = challenger_direction_mae <= champion_direction_mae * (1.0 - promotion_margin)
            model_selection_report = {
                "enabled": True,
                "holdout_eval_split": float(args.challenge_eval_split),
                "holdout_eval_min_samples": int(args.challenge_min_eval_samples),
                "promotion_margin_pct": float(args.promotion_margin_pct),
                "speed_eval_samples": int(len(speed_X_eval)),
                "direction_eval_samples": int(len(direction_X_eval)),
                "speed_eval_rows": int(speed_promotion_summary["speed_eval_rows"]),
                "speed_mae_forecast": float(speed_promotion_summary["speed_mae_forecast"]),
                "speed_rmse_forecast": float(speed_promotion_summary["speed_rmse_forecast"]),
                "speed_mae_champion": float(speed_promotion_summary["speed_mae_champion"]),
                "speed_rmse_champion": float(speed_promotion_summary["speed_rmse_champion"]),
                "speed_mae_challenger": float(speed_promotion_summary["speed_mae_challenger"]),
                "speed_rmse_challenger": float(speed_promotion_summary["speed_rmse_challenger"]),
                "speed_mae_improvement_challenger_vs_champion": float(
                    speed_promotion_summary["speed_mae_improvement_challenger_vs_champion"]
                ),
                "speed_rmse_improvement_challenger_vs_champion": float(
                    speed_promotion_summary["speed_rmse_improvement_challenger_vs_champion"]
                ),
                "speed_mae_improvement_challenger_vs_harmonie": float(
                    speed_promotion_summary["speed_mae_improvement_challenger_vs_harmonie"]
                ),
                "speed_rmse_improvement_challenger_vs_harmonie": float(
                    speed_promotion_summary["speed_rmse_improvement_challenger_vs_harmonie"]
                ),
                "speed_regime_calibration_challenger": challenger_speed_calibration,
                "speed_regime_calibration_champion": champion_speed_calibration,
                "direction_mae_champion": float(champion_direction_mae),
                "direction_mae_challenger": float(challenger_direction_mae),
                "promote_speed": bool(promote_speed),
                "promote_direction": bool(promote_direction),
                "speed_model_id_champion": champion_speed_model_id,
                "speed_model_id_challenger": challenger_speed_model_id,
                "speed_promotion_summary": speed_promotion_summary,
            }
            print(
                "Model gate | "
                f"speed MAE forecast={speed_promotion_summary['speed_mae_forecast']:.4f}, "
                f"champion={speed_promotion_summary['speed_mae_champion']:.4f}, "
                f"challenger={speed_promotion_summary['speed_mae_challenger']:.4f}, "
                f"RMSE champion={speed_promotion_summary['speed_rmse_champion']:.4f}, "
                f"challenger={speed_promotion_summary['speed_rmse_challenger']:.4f}, "
                f"promoted={promote_speed}"
            )
            print(
                f"Model gate | direction MAE champion={champion_direction_mae:.4f}, "
                f"challenger={challenger_direction_mae:.4f}, promoted={promote_direction}"
            )
        else:
            speed_promotion_summary = summarize_champion_vs_challenger(
                actual=speed_actual_eval,
                forecast=speed_forecast_eval,
                challenger_pred=challenger_speed_eval_pred,
                champion_pred=None,
                promotion_margin_pct=float(args.promotion_margin_pct),
                holdout_eval_split=float(args.challenge_eval_split),
                holdout_eval_min_samples=int(args.challenge_min_eval_samples),
                challenger_model_id=challenger_speed_model_id,
                champion_model_id=champion_speed_model_id,
            )
            model_selection_report = {
                "enabled": True,
                "reason": "no_existing_champion",
                "holdout_eval_split": float(args.challenge_eval_split),
                "holdout_eval_min_samples": int(args.challenge_min_eval_samples),
                "promotion_margin_pct": float(args.promotion_margin_pct),
                "speed_eval_samples": int(len(speed_X_eval)),
                "direction_eval_samples": int(len(direction_X_eval)),
                "speed_eval_rows": int(speed_promotion_summary["speed_eval_rows"]),
                "speed_mae_forecast": float(speed_promotion_summary["speed_mae_forecast"]),
                "speed_rmse_forecast": float(speed_promotion_summary["speed_rmse_forecast"]),
                "speed_mae_challenger": float(speed_promotion_summary["speed_mae_challenger"]),
                "speed_rmse_challenger": float(speed_promotion_summary["speed_rmse_challenger"]),
                "speed_mae_improvement_challenger_vs_harmonie": float(
                    speed_promotion_summary["speed_mae_improvement_challenger_vs_harmonie"]
                ),
                "speed_rmse_improvement_challenger_vs_harmonie": float(
                    speed_promotion_summary["speed_rmse_improvement_challenger_vs_harmonie"]
                ),
                "speed_regime_calibration_challenger": challenger_speed_calibration,
                "direction_mae_challenger": float(challenger_direction_mae),
                "promote_speed": True,
                "promote_direction": True,
                "speed_model_id_champion": champion_speed_model_id,
                "speed_model_id_challenger": challenger_speed_model_id,
                "speed_promotion_summary": speed_promotion_summary,
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
                model_class="TargetAwareNextDayLSTM",
                history_hours=cfg.window_hours,
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
                extra={
                    "trained_at_utc": now_train_utc,
                    "feature_schema": str(direction_arrays_full.get("feature_schema", "direction_v2")),
                },
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

            target_times_utc = pd.to_datetime(
                np.asarray(speed_arrays_full["target_times_all"][speed_eval_start:]).reshape(-1),
                errors="coerce",
                utc=True,
            )
            raw_eval = pd.DataFrame(
                {
                    "target_time_utc": target_times_utc,
                    "actual_wind_speed": speed_actual_eval.reshape(-1).astype(float),
                    "forecast_wind_speed": speed_forecast_eval.reshape(-1).astype(float),
                    "forecast_wind_dir_deg": speed_forecast_dir_eval.reshape(-1).astype(float),
                    "challenger_wind_speed": challenger_speed_eval_pred.reshape(-1).astype(float),
                    "champion_wind_speed": champion_speed_eval_pred.reshape(-1).astype(float),
                }
            )
            raw_eval = raw_eval.dropna(subset=["target_time_utc"])
            agg_eval = raw_eval.groupby("target_time_utc", as_index=False).mean(numeric_only=True)
            direction_by_target = raw_eval.groupby("target_time_utc")["forecast_wind_dir_deg"].agg(
                _circular_mean_deg
            )
            agg_eval["forecast_wind_dir_deg"] = agg_eval["target_time_utc"].map(direction_by_target)
            agg_eval["n_overlaps"] = raw_eval.groupby("target_time_utc").size().to_numpy()
            agg_eval["abs_err_forecast"] = np.abs(agg_eval["forecast_wind_speed"] - agg_eval["actual_wind_speed"])
            agg_eval["abs_err_challenger"] = np.abs(agg_eval["challenger_wind_speed"] - agg_eval["actual_wind_speed"])
            agg_eval["abs_err_champion"] = np.abs(agg_eval["champion_wind_speed"] - agg_eval["actual_wind_speed"])
            sector_labels = np.array(["N", "NE", "E", "SE", "S", "SW", "W", "NW"], dtype=object)
            sector_idx = (
                np.floor(((agg_eval["forecast_wind_dir_deg"].to_numpy(dtype=float) % 360.0) + 22.5) / 45.0)
                .astype(int)
                % 8
            )
            agg_eval["forecast_wind_dir_sector"] = sector_labels[sector_idx]
            direction_rows: list[dict] = []
            for sector_name, sector_frame in agg_eval.groupby("forecast_wind_dir_sector", sort=False):
                actual_vals = sector_frame["actual_wind_speed"].to_numpy(dtype=float)
                forecast_vals = sector_frame["forecast_wind_speed"].to_numpy(dtype=float)
                challenger_vals = sector_frame["challenger_wind_speed"].to_numpy(dtype=float)
                champion_vals = sector_frame["champion_wind_speed"].to_numpy(dtype=float)
                direction_rows.append(
                    {
                        "sector": str(sector_name),
                        "n_points": int(len(sector_frame)),
                        "forecast_mae": float(np.mean(np.abs(forecast_vals - actual_vals))),
                        "forecast_bias_pred_minus_actual": float(np.mean(forecast_vals - actual_vals)),
                        "challenger_mae": float(np.mean(np.abs(challenger_vals - actual_vals))),
                        "challenger_bias_pred_minus_actual": float(np.mean(challenger_vals - actual_vals)),
                        "champion_mae": float(np.mean(np.abs(champion_vals - actual_vals))),
                        "champion_bias_pred_minus_actual": float(np.mean(champion_vals - actual_vals)),
                        "champion_mae_gain_vs_forecast": float(
                            np.mean(np.abs(forecast_vals - actual_vals))
                            - np.mean(np.abs(champion_vals - actual_vals))
                        ),
                        "challenger_mae_gain_vs_forecast": float(
                            np.mean(np.abs(forecast_vals - actual_vals))
                            - np.mean(np.abs(challenger_vals - actual_vals))
                        ),
                        "challenger_mae_gain_vs_champion": float(
                            np.mean(np.abs(champion_vals - actual_vals))
                            - np.mean(np.abs(challenger_vals - actual_vals))
                        ),
                    }
                )
            gate_eval_direction_csv = details_dir / f"{stamp}_model_gate_eval_speed_by_direction.csv"
            gate_eval_direction_stable_csv = out_dir / "model_gate_eval_speed_by_direction.csv"
            direction_frame = pd.DataFrame(direction_rows)
            direction_frame.to_csv(gate_eval_direction_csv, index=False)
            direction_frame.to_csv(gate_eval_direction_stable_csv, index=False)
            agg_eval["target_time_utc"] = pd.to_datetime(agg_eval["target_time_utc"], utc=True).dt.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            agg_eval.to_csv(gate_eval_details_csv, index=False)
            gate_eval_details_csv_src = gate_eval_details_csv
            gate_eval_direction_csv_src = gate_eval_direction_stable_csv
            model_selection_report["speed_eval_details_csv"] = str(gate_eval_details_csv)
            model_selection_report["speed_eval_direction_csv"] = str(gate_eval_direction_csv)
            model_selection_report["speed_eval_direction_stable_csv"] = str(gate_eval_direction_stable_csv)

        intraday_challenger_extra = {
            "trained_at_utc": now_train_utc,
            "hidden1": int(args.intraday_hidden1),
            "hidden2": int(args.intraday_hidden2),
            "dropout": float(args.intraday_dropout),
            "learning_rate": float(args.intraday_learning_rate),
            "recency_power": float(args.intraday_recency_power),
            "model_role": "challenger",
            "holdout_eval_split": float(args.intraday_challenge_eval_split),
            "holdout_eval_min_contexts": int(args.intraday_challenge_min_eval_contexts),
            "promotion_margin_pct": float(args.intraday_promotion_margin_pct),
        }
        save_intraday_model(
            intraday_challenger_model_path,
            intraday_bundle_challenger,
            extra=intraday_challenger_extra,
        )

        intraday_challenger_model_id = _format_model_id(now_train_utc, args.local_timezone)
        intraday_champion_bundle: IntradayBundle | None = None
        intraday_champion_ckpt: dict = {}
        intraday_champion_model_id = "none"
        intraday_champion_available = intraday_model_path.exists()
        intraday_challenger_eval_frame = build_intraday_holdout_evaluation_frame(
            bundle=intraday_bundle_challenger,
            contexts=intraday_eval_contexts,
            device=device,
        )
        intraday_champion_eval_frame: pd.DataFrame | None = None
        if intraday_champion_available:
            intraday_champion_bundle, intraday_champion_ckpt = load_intraday_model(intraday_model_path, device)
            intraday_champion_model_id = _format_model_id(
                _resolve_model_trained_utc(intraday_champion_ckpt, intraday_model_path),
                args.local_timezone,
            )
            intraday_champion_eval_frame = build_intraday_holdout_evaluation_frame(
                bundle=intraday_champion_bundle,
                contexts=intraday_eval_contexts,
                device=device,
            )

        intraday_promotion_summary = summarize_intraday_champion_vs_challenger(
            challenger_eval_frame=intraday_challenger_eval_frame,
            champion_eval_frame=intraday_champion_eval_frame,
            promotion_margin_pct=float(args.intraday_promotion_margin_pct),
            holdout_eval_split=float(args.intraday_challenge_eval_split),
            holdout_eval_min_contexts=int(args.intraday_challenge_min_eval_contexts),
            challenger_model_id=intraday_challenger_model_id,
            champion_model_id=intraday_champion_model_id,
        )
        promote_intraday = bool(intraday_promotion_summary["promote_intraday"])
        intraday_model_selection_report = {
            "enabled": True,
            "holdout_eval_split": float(args.intraday_challenge_eval_split),
            "holdout_eval_min_contexts": int(args.intraday_challenge_min_eval_contexts),
            "promotion_margin_pct": float(args.intraday_promotion_margin_pct),
            "eval_training_contexts": int(len(intraday_train_contexts)),
            "eval_holdout_contexts": int(intraday_promotion_summary["intraday_eval_contexts"]),
            "eval_holdout_rows": int(intraday_promotion_summary["intraday_eval_rows"]),
            "intraday_model_id_champion": intraday_champion_model_id,
            "intraday_model_id_challenger": intraday_challenger_model_id,
            "mae_harmonie": intraday_promotion_summary.get("intraday_mae_harmonie"),
            "rmse_harmonie": intraday_promotion_summary.get("intraday_rmse_harmonie"),
            "mae_champion": intraday_promotion_summary.get("intraday_mae_champion"),
            "rmse_champion": intraday_promotion_summary.get("intraday_rmse_champion"),
            "mae_challenger": intraday_promotion_summary.get("intraday_mae_challenger"),
            "rmse_challenger": intraday_promotion_summary.get("intraday_rmse_challenger"),
            "mae_improvement_challenger_vs_champion": intraday_promotion_summary.get(
                "intraday_mae_improvement_challenger_vs_champion"
            ),
            "rmse_improvement_challenger_vs_champion": intraday_promotion_summary.get(
                "intraday_rmse_improvement_challenger_vs_champion"
            ),
            "mae_improvement_challenger_vs_harmonie": intraday_promotion_summary.get(
                "intraday_mae_improvement_challenger_vs_harmonie"
            ),
            "rmse_improvement_challenger_vs_harmonie": intraday_promotion_summary.get(
                "intraday_rmse_improvement_challenger_vs_harmonie"
            ),
            "promote_intraday": promote_intraday,
            "reason": intraday_promotion_summary.get("reason", "evaluated"),
            "promotion_summary": intraday_promotion_summary,
        }
        intraday_aligned_eval = align_intraday_holdout_frames(
            challenger_eval_frame=intraday_challenger_eval_frame,
            champion_eval_frame=intraday_champion_eval_frame,
        )
        if not intraday_aligned_eval.empty:
            intraday_details_dir = out_dir / "intraday_model_gate_eval_details"
            intraday_details_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            intraday_gate_eval_details_csv = intraday_details_dir / f"{stamp}_intraday_model_gate_eval_speed.csv"
            intraday_aligned_eval = intraday_aligned_eval.copy()
            intraday_aligned_eval["abs_err_harmonie"] = np.abs(
                intraday_aligned_eval["harmonie_value"] - intraday_aligned_eval["actual_value"]
            )
            intraday_aligned_eval["abs_err_challenger"] = np.abs(
                intraday_aligned_eval["challenger_prediction_value"] - intraday_aligned_eval["actual_value"]
            )
            intraday_aligned_eval["abs_err_champion"] = np.abs(
                intraday_aligned_eval["champion_prediction_value"] - intraday_aligned_eval["actual_value"]
            )
            intraday_aligned_eval["anchor_time_utc"] = pd.to_datetime(
                intraday_aligned_eval["anchor_time_utc"], utc=True
            ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            intraday_aligned_eval["target_time_utc"] = pd.to_datetime(
                intraday_aligned_eval["target_time_utc"], utc=True
            ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            intraday_aligned_eval.to_csv(intraday_gate_eval_details_csv, index=False)
            intraday_gate_eval_details_csv_src = intraday_gate_eval_details_csv
            intraday_model_selection_report["intraday_eval_details_csv"] = str(intraday_gate_eval_details_csv)

        if promote_intraday:
            intraday_bundle = intraday_bundle_challenger
            save_intraday_model(
                intraday_model_path,
                intraday_bundle,
                extra={
                    **intraday_challenger_extra,
                    "model_role": "champion",
                    "promoted_from": str(intraday_challenger_model_path.name),
                },
            )
            intraday_model_last_trained_at_utc = now_train_utc
            intraday_hparams = {
                "hidden1": int(args.intraday_hidden1),
                "hidden2": int(args.intraday_hidden2),
                "dropout": float(args.intraday_dropout),
                "learning_rate": float(args.intraday_learning_rate),
                "recency_power": float(args.intraday_recency_power),
            }
            intraday_model_selection_report["selected"] = "challenger"
        else:
            if intraday_champion_bundle is None:
                raise ValueError("Intraday promotion decided to keep champion, but no champion artifact was loaded.")
            intraday_bundle = intraday_champion_bundle
            intraday_model_last_trained_at_utc = _resolve_model_trained_utc(intraday_champion_ckpt, intraday_model_path)
            intraday_hparams = {
                "hidden1": intraday_champion_ckpt.get("hidden1"),
                "hidden2": intraday_champion_ckpt.get("hidden2"),
                "dropout": intraday_champion_ckpt.get("dropout"),
                "learning_rate": intraday_champion_ckpt.get("learning_rate"),
                "recency_power": intraday_champion_ckpt.get("recency_power"),
            }
            intraday_model_selection_report["selected"] = "champion"

        intraday_champion_mae_text = (
            "none"
            if intraday_promotion_summary.get("intraday_mae_champion") is None
            else f"{float(intraday_promotion_summary['intraday_mae_champion']):.4f}"
        )
        print(
            "Intraday gate | "
            f"Harmonie MAE={float(intraday_promotion_summary['intraday_mae_harmonie']):.4f}, "
            f"champion={intraday_champion_mae_text}, "
            f"challenger={float(intraday_promotion_summary['intraday_mae_challenger']):.4f}, "
            f"promoted={promote_intraday}"
        )

    inference_input_speed = build_next_day_inference_input(
        db_path=db_path,
        cfg=cfg,
        x_mean=speed_arrays["x_mean"],
        x_std=speed_arrays["x_std"],
        feature_schema=_next_day_feature_schema_from_scalers(speed_arrays),
        local_tz=args.local_timezone,
    )
    inference_input_direction = build_next_day_inference_input(
        db_path=db_path,
        cfg=cfg,
        x_mean=direction_arrays["x_mean"],
        x_std=direction_arrays["x_std"],
        feature_schema=_direction_feature_schema_from_scalers(direction_arrays),
        local_tz=args.local_timezone,
    )
    speed_inference_calibration_context = _build_speed_calibration_context(
        anchor_dir_deg=float(inference_input_speed["anchor_forecast_dir"]),
        target_times_utc=pd.to_datetime([inference_input_speed["target_times"][0]], utc=True),
    )
    speed_inference_calibration_context.update(
        {
            "target_forecast_dir_deg": np.asarray(inference_input_speed["forecast_dir_next24"], dtype=np.float32)[
                np.newaxis, :
            ],
            "target_times_utc": np.asarray(inference_input_speed["target_times"], dtype=object)[np.newaxis, :],
            "target_horizon_hr": np.asarray(inference_input_speed["target_horizon_hr"], dtype=np.float32)[
                np.newaxis, :
            ],
        }
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

    table = build_prediction_table(inference_input_speed, speed_pred, direction_pred, local_tz=args.local_timezone)
    next_day_prediction_log_frame = _build_next_day_prediction_log_frame(inference_input_speed, speed_pred)
    is_test_mode = args.test_now_local_hour is not None
    prediction_generated_at_dt = datetime.now(timezone.utc)
    prediction_generated_at_utc = prediction_generated_at_dt.isoformat()
    prediction_update_local = prediction_generated_at_dt.astimezone(ZoneInfo(args.local_timezone)).replace(
        minute=0,
        second=0,
        microsecond=0,
    )
    prediction_updated_at_utc = prediction_update_local.astimezone(timezone.utc).isoformat()
    harmonie_time_utc, harmonie_time_kind = _load_latest_harmonie_metadata_time(db_path, cfg.site)
    next_day_prediction_log_rows = 0
    current_day_prediction_log_rows = 0
    prediction_evaluation_rows_materialized = 0
    prediction_evaluation_summary: list[dict] = []
    next_day_vs_harmonie_summary: dict = {}
    next_day_vs_harmonie_by_issued_day: list[dict] = []
    next_day_vs_harmonie_by_horizon: list[dict] = []

    table_path = out_dir / "next_day_predictions.csv"
    table_for_csv = table[
        [
            "target_time_utc",
            "target_time_local",
            "hour_utc",
            "hour_local",
            "forecast_wind_speed",
            "forecast_wind_min",
            "forecast_wind_max",
            "lstm_pred_wind_speed",
            "delta_speed_lstm_minus_forecast",
            "forecast_wind_dir_deg",
            "lstm_pred_wind_dir_deg",
            "delta_dir_lstm_minus_forecast",
        ]
    ].copy()
    table_for_csv["target_time_utc"] = table_for_csv["target_time_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    table_for_csv["target_time_local"] = table_for_csv["target_time_local"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    table_for_csv.to_csv(table_path, index=False)
    dayahead_snapshot_csv = None
    if not args.skip_training and args.test_now_local_hour is None:
        dayahead_snapshot_csv = save_dayahead_snapshot(
            out_dir=out_dir,
            table=table,
            local_tz=args.local_timezone,
            prediction_generated_at_utc=prediction_generated_at_utc,
        )
    if not is_test_mode:
        next_day_prediction_log_rows = _log_prediction_frame(
            db_path=db_path,
            prediction_frame=next_day_prediction_log_frame,
            site=cfg.site,
            model_type="next_day",
            model_name=type(speed_model).__name__,
            model_version=model_last_trained_at_utc,
            model_artifact=speed_model_path.name,
            issued_time_utc=prediction_generated_at_dt,
            anchor_time_utc=inference_input_speed["anchor_time"],
            prediction_kind="wind_speed",
            run_context=str(inference_input_speed["prediction_day_start"]),
            metadata={
                "forecast_model": args.model,
                "reference_observation_time": str(inference_input_speed["reference_observation_time"]),
                "source": "update_model_and_predict",
            },
        )

    plot_path = out_dir / "next_day_predictions.png"
    plot_path_mobile = out_dir / "next_day_predictions_mobile.png"
    save_prediction_plot(
        table,
        plot_path,
        local_tz=args.local_timezone,
        prediction_generated_at_utc=prediction_generated_at_utc,
        prediction_updated_at_utc=prediction_updated_at_utc,
        model_trained_at_utc=model_last_trained_at_utc,
        harmonie_time_utc=harmonie_time_utc,
        harmonie_time_kind=harmonie_time_kind,
    )
    save_prediction_plot(
        table,
        plot_path_mobile,
        local_tz=args.local_timezone,
        prediction_generated_at_utc=prediction_generated_at_utc,
        prediction_updated_at_utc=prediction_updated_at_utc,
        model_trained_at_utc=model_last_trained_at_utc,
        harmonie_time_utc=harmonie_time_utc,
        harmonie_time_kind=harmonie_time_kind,
        mobile=True,
    )

    test_suffix = f"_test_hour_{int(args.test_now_local_hour):02d}" if is_test_mode else ""
    current_day_prior_prediction_tables: list[pd.DataFrame] = []
    current_day_live_monitoring_metric: dict = {
        "available": False,
        "point_count": 0,
        "completed_interval_count": 0,
        "partial_current_interval_included": False,
        "mae_superlocal": None,
        "mae_harmonie": None,
        "rmse_superlocal": None,
        "rmse_harmonie": None,
        "model_win_rate": None,
        "segments": [],
    }
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

    current_day_issue_anchor_local = _resolve_now_local(args.local_timezone, args.test_now_local_hour).replace(
        minute=0,
        second=0,
        microsecond=0,
    )
    current_day_issue_anchor_utc = pd.Timestamp(current_day_issue_anchor_local).tz_convert("UTC")

    # --- Current day plot/table: actuals up to present + prediction for remaining day ---
    current_day_table, current_day_prediction_log_frame = build_current_day_table(
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
    if not is_test_mode:
        current_day_prediction_log_rows = _log_prediction_frame(
            db_path=db_path,
            prediction_frame=current_day_prediction_log_frame,
            site=cfg.site,
            model_type="intraday",
            model_name=type(intraday_bundle.model).__name__,
            model_version=intraday_model_last_trained_at_utc,
            model_artifact=intraday_model_path.name,
            issued_time_utc=prediction_generated_at_dt,
            anchor_time_utc=current_day_issue_anchor_utc,
            prediction_kind="wind_speed",
            run_context=str(current_day_issue_anchor_local.date().isoformat()),
            metadata={
                "forecast_model": args.model,
                "local_timezone": args.local_timezone,
                "source": "update_model_and_predict",
            },
        )
        conn = sqlite3.connect(str(db_path))
        try:
            init_db(conn)
            prediction_evaluation_rows_materialized = materialize_prediction_log_evaluation(
                conn,
                site=cfg.site,
                prediction_kind="wind_speed",
            )
            prediction_evaluation_summary = load_prediction_evaluation_summary(
                conn,
                site=cfg.site,
                prediction_kind="wind_speed",
            )
            next_day_vs_harmonie_summary = summarize_next_day_vs_harmonie(
                conn,
                site=cfg.site,
                prediction_kind="wind_speed",
            )
            next_day_vs_harmonie_by_issued_day = summarize_next_day_vs_harmonie_by_issued_day(
                conn,
                site=cfg.site,
                prediction_kind="wind_speed",
            )
            next_day_vs_harmonie_by_horizon = summarize_next_day_vs_harmonie_by_horizon(
                conn,
                site=cfg.site,
                prediction_kind="wind_speed",
            )
        finally:
            conn.close()
    current_day_snapshot_csv = None
    if not is_test_mode and not current_day_table.empty:
        current_day_target_local = pd.to_datetime(current_day_table["time_local"]).dt.tz_convert(ZoneInfo(args.local_timezone)).iloc[0].date()
        current_day_live_monitoring_metric = compute_current_day_completed_interval_mae(
            db_path=db_path,
            site=cfg.site,
            target_day_local=current_day_target_local,
            local_tz=args.local_timezone,
            prior_prediction_tables=current_day_prior_prediction_tables,
        )
        if not current_day_live_monitoring_metric.get("available", False):
            current_day_live_monitoring_metric = compute_current_day_table_mae(current_day_table)
        current_day_direction_csv = out_dir / "current_day_eval_speed_by_direction.csv"
        current_day_direction_spider_png = out_dir / "current_day_direction_spider.png"
        if save_current_day_direction_performance_csv(
            db_path=db_path,
            cfg=cfg,
            direction_csv=current_day_direction_csv,
        ):
            current_day_direction_csv_src = current_day_direction_csv
            save_current_day_direction_performance_spider_plot(
                current_day_direction_csv,
                current_day_direction_spider_png,
            )
            if current_day_direction_spider_png.exists():
                current_day_direction_spider_png_src = current_day_direction_spider_png

    current_day_plot_path = out_dir / f"current_day_predictions{test_suffix}.png"
    current_day_plot_mobile_path = out_dir / f"current_day_predictions{test_suffix}_mobile.png"
    save_current_day_plot(
        current_day_table,
        current_day_plot_path,
        args.local_timezone,
        prediction_generated_at_utc=prediction_generated_at_utc,
        prediction_updated_at_utc=prediction_updated_at_utc,
        model_trained_at_utc=model_last_trained_at_utc,
        harmonie_time_utc=harmonie_time_utc,
        harmonie_time_kind=harmonie_time_kind,
        prior_prediction_tables=current_day_prior_prediction_tables,
        live_monitoring_metric=current_day_live_monitoring_metric,
    )
    save_current_day_plot(
        current_day_table,
        current_day_plot_mobile_path,
        args.local_timezone,
        prediction_generated_at_utc=prediction_generated_at_utc,
        prediction_updated_at_utc=prediction_updated_at_utc,
        model_trained_at_utc=model_last_trained_at_utc,
        harmonie_time_utc=harmonie_time_utc,
        harmonie_time_kind=harmonie_time_kind,
        prior_prediction_tables=current_day_prior_prediction_tables,
        live_monitoring_metric=current_day_live_monitoring_metric,
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
                db_path=db_path,
                site=cfg.site,
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
            save_daily_mae_plot(
                daily_mae_csv_src,
                daily_mae_png_refresh,
                local_tz=args.local_timezone,
                last_months=3,
            )
            daily_mae_png_src = daily_mae_png_refresh
            daily_mae_png_mobile_refresh = out_dir / "daily_mae_history_mobile.png"
            save_daily_mae_plot(
                daily_mae_csv_src,
                daily_mae_png_mobile_refresh,
                local_tz=args.local_timezone,
                last_months=3,
            )
            daily_mae_png_mobile_src = daily_mae_png_mobile_refresh
        if gate_eval_history_csv_src is None and gate_eval_history_csv.exists():
            gate_eval_history_csv_src = gate_eval_history_csv
        if gate_eval_history_csv_src is None:
            artifact_history_csv = model_artifact_dir / "model_gate_eval_history.csv"
            if artifact_history_csv.exists():
                gate_eval_history_csv_src = artifact_history_csv
        if gate_eval_history_png_src is None and gate_eval_history_png.exists():
            gate_eval_history_png_src = gate_eval_history_png
        if gate_eval_details_csv_src is None:
            details_dir = out_dir / "model_gate_eval_details"
            if details_dir.exists():
                detail_files = sorted(details_dir.glob("*_model_gate_eval_speed.csv"))
                if detail_files:
                    gate_eval_details_csv_src = detail_files[-1]
        if gate_eval_details_csv_src is None:
            artifact_details_dir = model_artifact_dir / "model_gate_eval_details"
            if artifact_details_dir.exists():
                detail_files = sorted(artifact_details_dir.glob("*_model_gate_eval_speed.csv"))
                if detail_files:
                    gate_eval_details_csv_src = detail_files[-1]
        if gate_eval_direction_csv_src is None:
            stable_direction_csv = out_dir / "model_gate_eval_speed_by_direction.csv"
            if stable_direction_csv.exists():
                gate_eval_direction_csv_src = stable_direction_csv
            else:
                details_dir = out_dir / "model_gate_eval_details"
                if details_dir.exists():
                    direction_files = sorted(details_dir.glob("*_model_gate_eval_speed_by_direction.csv"))
                    if direction_files:
                        gate_eval_direction_csv_src = direction_files[-1]
        if gate_eval_direction_csv_src is not None:
            gate_eval_direction_spider_png = out_dir / "model_gate_direction_spider.png"
            save_wind_direction_performance_spider_plot(
                gate_eval_direction_csv_src,
                gate_eval_direction_spider_png,
            )
            if gate_eval_direction_spider_png.exists():
                gate_eval_direction_spider_png_src = gate_eval_direction_spider_png
        if gate_eval_history_csv_src is not None:
            save_model_gate_eval_history_plot(
                gate_eval_history_csv_src,
                gate_eval_history_png,
                local_tz=args.local_timezone,
                eval_details_csv=gate_eval_details_csv_src,
                db_path=db_path,
                site=cfg.site,
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
            direction_spider_png=gate_eval_direction_spider_png_src,
            direction_spider_csv=gate_eval_direction_csv_src,
            current_day_direction_spider_png=current_day_direction_spider_png_src,
            current_day_direction_spider_csv=current_day_direction_csv_src,
            current_day_prior_prediction_tables=current_day_prior_prediction_tables,
            prediction_generated_at_utc=prediction_generated_at_utc,
            prediction_updated_at_utc=prediction_updated_at_utc,
            model_trained_at_utc=model_last_trained_at_utc,
            harmonie_time_utc=harmonie_time_utc,
            harmonie_time_kind=harmonie_time_kind,
            companion_app_base_url=args.companion_app_base_url,
        )
        if args.git_auto_push_pages:
            repo_root = Path(__file__).resolve().parents[1]
            git_publish = auto_push_dashboard_changes(
                repo_root=repo_root,
                web_out_dir=Path(args.web_out_dir),
                remote=args.git_remote,
                branch=args.git_branch,
            )

    post_run_champion_states = _load_active_champion_states(
        speed_model_path=speed_model_path,
        direction_model_path=direction_model_path,
        intraday_model_path=intraday_model_path,
        device=device,
        local_tz=args.local_timezone,
    )
    champion_refresh_summary = {
        "requested": not bool(args.skip_training),
        "completed": not bool(args.skip_training),
        "reused_entrypoint": "update_model_and_predict.py (training enabled)",
        "gate_logic_enforced": {
            "next_day": not bool(args.skip_training),
            "intraday": not bool(args.skip_training),
        },
        "pre_run": pre_run_champion_states,
        "post_run": post_run_champion_states,
        "active_next_day_speed_model_id": post_run_champion_states["next_day_speed"].get("model_id"),
        "active_next_day_direction_model_id": post_run_champion_states["next_day_direction"].get("model_id"),
        "active_intraday_model_id": post_run_champion_states["intraday_speed"].get("model_id"),
        "next_day_speed_selected": model_selection_report.get("speed_selected"),
        "next_day_direction_selected": model_selection_report.get("direction_selected"),
        "intraday_selected": intraday_model_selection_report.get("selected"),
        "next_day_speed_refreshed_this_run": _champion_state_refreshed(
            pre_run_champion_states.get("next_day_speed"),
            post_run_champion_states.get("next_day_speed"),
        ),
        "next_day_direction_refreshed_this_run": _champion_state_refreshed(
            pre_run_champion_states.get("next_day_direction"),
            post_run_champion_states.get("next_day_direction"),
        ),
        "intraday_refreshed_this_run": _champion_state_refreshed(
            pre_run_champion_states.get("intraday_speed"),
            post_run_champion_states.get("intraday_speed"),
        ),
    }

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "db_path": str(db_path),
        "output_artifact_dir": str(out_dir.resolve()),
        "model_artifact_dir": str(model_artifact_dir.resolve()),
        "model_artifact_dir_defaulted_to_out_dir": args.model_artifact_dir is None,
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
        "intraday_model_last_trained_at_utc": intraday_model_last_trained_at_utc,
        "next_day_prediction_log_rows": int(next_day_prediction_log_rows),
        "current_day_prediction_log_rows": int(current_day_prediction_log_rows),
        "prediction_evaluation_rows_materialized": int(prediction_evaluation_rows_materialized),
        "prediction_evaluation_summary": prediction_evaluation_summary,
        "next_day_vs_harmonie_summary": next_day_vs_harmonie_summary,
        "next_day_vs_harmonie_by_issued_day": next_day_vs_harmonie_by_issued_day,
        "next_day_vs_harmonie_by_horizon": next_day_vs_harmonie_by_horizon,
        "model_promotion_speed_summary": model_selection_report.get("speed_promotion_summary"),
        "intraday_model_promotion_summary": intraday_model_selection_report.get("promotion_summary"),
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
        "current_day_live_monitoring_metric": current_day_live_monitoring_metric,
        "current_day_plot_png": str(current_day_plot_path),
        "current_day_plot_archived_png": archived_current_day_plot,
        "daily_mae_history_csv": daily_mae_csv,
        "daily_mae_history_png": daily_mae_png,
        "model_gate_eval_history_csv": None if gate_eval_history_csv_src is None else str(gate_eval_history_csv_src),
        "model_gate_eval_history_png": None if gate_eval_history_png_src is None else str(gate_eval_history_png_src),
        "model_gate_eval_details_csv": None if gate_eval_details_csv_src is None else str(gate_eval_details_csv_src),
        "model_gate_direction_csv": None if gate_eval_direction_csv_src is None else str(gate_eval_direction_csv_src),
        "model_gate_direction_spider_png": (
            None if gate_eval_direction_spider_png_src is None else str(gate_eval_direction_spider_png_src)
        ),
        "current_day_direction_csv": None if current_day_direction_csv_src is None else str(current_day_direction_csv_src),
        "current_day_direction_spider_png": (
            None if current_day_direction_spider_png_src is None else str(current_day_direction_spider_png_src)
        ),
        "web_dashboard_dir": None if web_publish is None else str(Path(args.web_out_dir).resolve()),
        "web_dashboard_files": web_publish,
        "web_dashboard_git_publish": git_publish,
        "speed_model_path": str(speed_model_path),
        "direction_model_path": str(direction_model_path),
        "intraday_model_path": str(intraday_model_path),
        "intraday_challenger_model_path": str(intraday_challenger_model_path),
        "intraday_model_class": "IntradayResidualMLP",
        "intraday_feature_count": int(len(getattr(intraday_bundle, "x_mean", []))),
        "intraday_n_train": None if intraday_train_stats is None else int(intraday_train_stats["n_train"]),
        "intraday_n_val": None if intraday_train_stats is None else int(intraday_train_stats["n_val"]),
        "intraday_best_val_loss": None if intraday_train_stats is None else float(intraday_train_stats["best_val_loss"]),
        "intraday_holdout_contexts": None
        if intraday_train_stats is None
        else int(intraday_train_stats.get("holdout_contexts", 0)),
        "intraday_hidden1": intraday_hparams.get("hidden1"),
        "intraday_hidden2": intraday_hparams.get("hidden2"),
        "intraday_dropout": intraday_hparams.get("dropout"),
        "intraday_learning_rate": intraday_hparams.get("learning_rate"),
        "intraday_recency_power": intraday_hparams.get("recency_power"),
        "intraday_rollout_calibration": getattr(intraday_bundle, "rollout_calibration", None),
        "intraday_continuity_calibration": getattr(intraday_bundle, "continuity_calibration", None),
        "intraday_model_gate_eval_details_csv": None
        if intraday_gate_eval_details_csv_src is None
        else str(intraday_gate_eval_details_csv_src),
        "intraday_model_selection_gate": intraday_model_selection_report,
        "champion_refresh_summary": champion_refresh_summary,
        "data_refresh": refresh_info,
        "model_selection_gate": model_selection_report,
        "challenge_eval_split": args.challenge_eval_split,
        "challenge_min_eval_samples": args.challenge_min_eval_samples,
        "promotion_margin_pct": args.promotion_margin_pct,
        "intraday_challenge_eval_split": args.intraday_challenge_eval_split,
        "intraday_challenge_min_eval_contexts": args.intraday_challenge_min_eval_contexts,
        "intraday_promotion_margin_pct": args.intraday_promotion_margin_pct,
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
    if intraday_challenger_model_path.exists():
        print(f"Intraday challenger saved to: {intraday_challenger_model_path}")
    print(f"Prediction table saved to: {table_path}")
    print(f"Prediction plot saved to: {plot_path}")
    print(f"Current-day table saved to: {current_day_table_path}")
    print(f"Current-day plot saved to: {current_day_plot_path}")
    if not is_test_mode:
        print(f"Prediction evaluation rows materialized in SQLite: {prediction_evaluation_rows_materialized}")
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
    if gate_eval_direction_csv_src is not None:
        print(f"Direction performance summary saved to: {gate_eval_direction_csv_src}")
    if gate_eval_direction_spider_png_src is not None:
        print(f"Direction performance spider plot saved to: {gate_eval_direction_spider_png_src}")
    if current_day_direction_csv_src is not None:
        print(f"Current-day direction performance summary saved to: {current_day_direction_csv_src}")
    if current_day_direction_spider_png_src is not None:
        print(f"Current-day direction performance spider plot saved to: {current_day_direction_spider_png_src}")
    if intraday_model_selection_report.get("enabled"):
        print(
            "Intraday selection | "
            f"selected={intraday_model_selection_report.get('selected')} | "
            f"promoted={intraday_model_selection_report.get('promote_intraday')}"
        )
    print(
        "Champion refresh | "
        f"next-day speed refreshed={champion_refresh_summary.get('next_day_speed_refreshed_this_run')} "
        f"selected={champion_refresh_summary.get('next_day_speed_selected')} | "
        f"next-day direction refreshed={champion_refresh_summary.get('next_day_direction_refreshed_this_run')} "
        f"selected={champion_refresh_summary.get('next_day_direction_selected')} | "
        f"intraday refreshed={champion_refresh_summary.get('intraday_refreshed_this_run')} "
        f"selected={champion_refresh_summary.get('intraday_selected')}"
    )
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
