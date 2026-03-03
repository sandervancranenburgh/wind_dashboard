from __future__ import annotations

import argparse
import copy
import json
import shutil
import sqlite3
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_pipeline import (
    DatasetConfig,
    _angle_add_deg,
    _build_forecast_feature_frame,
    _load_observations,
    _apply_standardizer,
    build_all_direction_training_arrays,
    build_all_training_arrays,
    build_next_day_inference_input,
)
from train_lstm import NextDayLSTM


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


def predict_speed_residual(
    model: nn.Module,
    X_input: np.ndarray,
    forecast_speed: np.ndarray,
    y_mean: float,
    y_std: float,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(X_input).float().to(device)).cpu().numpy()[0]
    pred_residual = pred_scaled * y_std + y_mean
    return pred_residual + forecast_speed


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


def build_prediction_table(
    inference_input: dict,
    speed_pred: np.ndarray,
    dir_pred: np.ndarray,
) -> pd.DataFrame:
    target_times = pd.to_datetime(inference_input["target_times"], utc=True)
    forecast_speed = inference_input["forecast_next24"].astype(np.float32)
    forecast_dir = inference_input["forecast_dir_next24"].astype(np.float32)

    table = pd.DataFrame(
        {
            "target_time_utc": target_times,
            "forecast_wind_speed": forecast_speed,
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


def save_prediction_plot(table: pd.DataFrame, plot_path: Path) -> None:
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

    y_min = float(min(table["forecast_wind_speed"].min(), table["lstm_pred_wind_speed"].min()))
    y_max = float(max(table["forecast_wind_speed"].max(), table["lstm_pred_wind_speed"].max()))
    pad = max((y_max - y_min) * 0.08, 0.8)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x, table["forecast_wind_speed"], marker="o", label="Forecast speed")
    ax.plot(x, table["lstm_pred_wind_speed"], marker="o", label="LSTM speed (residual)")
    ax.set_title(f"Next-Day Wind Speed ({day_label}, UTC)")
    ax.set_xlabel("Hour (UTC)")
    ax.set_ylabel("Wind speed (kts)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    ax.set_xticks(x, table["hour_label"], rotation=0)
    ax.set_ylim(y_min - pad, y_max + pad)

    # Draw wind direction arrows under x-axis.
    # Mapping: up-arrow means South wind (from South), per user preference.
    # Using x-axis transform keeps arrows below axis regardless of y-scale.
    y_base_axes = -0.20
    arrow_len_axes = 0.075
    for i, (fdir, ldir) in enumerate(zip(table["forecast_wind_dir_deg"], table["lstm_pred_wind_dir_deg"])):
        for direction_deg, color in [(fdir, "tab:blue"), (ldir, "tab:orange")]:
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

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def build_current_day_table(
    db_path: Path,
    cfg: DatasetConfig,
    speed_model: nn.Module,
    direction_model: nn.Module,
    speed_scalers: dict,
    direction_scalers: dict,
    local_tz: str,
    test_now_local_hour: int | None,
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

        speed_res = speed_res_scaled * float(speed_scalers["y_std"][0]) + float(speed_scalers["y_mean"][0])
        dir_res = dir_res_scaled * float(direction_scalers["y_std"][0]) + float(direction_scalers["y_mean"][0])
        speed_res = speed_res[:target_n]
        dir_res = dir_res[:target_n]

        fc_speed = future_frame["forecast_avg"].to_numpy(dtype=np.float32)
        fc_dir = future_frame["forecast_dir"].to_numpy(dtype=np.float32)
        lstm_speed = fc_speed + speed_res
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

    # Build forecast frame (UTC indexed) and observations (UTC indexed), then convert to local timezone.
    forecast_frame_utc = _build_forecast_feature_frame(db_path, cfg)
    conn = sqlite3.connect(str(db_path))
    try:
        obs_hourly_utc = _load_observations(conn, cfg.site)
    finally:
        conn.close()

    forecast_frame_local = forecast_frame_utc.tz_convert(tz)
    obs_hourly_local = obs_hourly_utc.tz_convert(tz)

    # Actuals for today up to current hour.
    actual_today = obs_hourly_local[
        (obs_hourly_local.index >= day_start_local) & (obs_hourly_local.index <= now_hour_local)
    ]["actual_avg"]
    actual_dir_today = obs_hourly_local[
        (obs_hourly_local.index >= day_start_local) & (obs_hourly_local.index <= now_hour_local)
    ]["actual_dir"]

    # Remaining hours today (prediction target): next hour .. 23:00 local.
    remaining_local = pd.date_range(
        start=now_hour_local + timedelta(hours=1),
        end=day_end_local,
        freq="1h",
        tz=tz,
    )
    remaining_n = len(remaining_local)

    # If day is effectively finished, return actual-only table for today.
    if remaining_n == 0:
        full_hours = pd.date_range(start=day_start_local, end=day_end_local, freq="1h", tz=tz)
        fc_today = forecast_frame_local.reindex(full_hours)["forecast_avg"]
        table = pd.DataFrame(
            {
                "time_local": full_hours,
                "forecast_wind_speed": fc_today.to_numpy(dtype=np.float32),
                "lstm_pred_wind_speed": np.full(len(full_hours), np.nan, dtype=np.float32),
                "actual_wind_speed": actual_today.reindex(full_hours).to_numpy(dtype=np.float32),
                "forecast_wind_dir_deg": forecast_frame_local.reindex(full_hours)["forecast_dir"].to_numpy(dtype=np.float32),
                "lstm_pred_wind_dir_deg": np.full(len(full_hours), np.nan, dtype=np.float32),
                "actual_wind_dir_deg": actual_dir_today.reindex(full_hours).to_numpy(dtype=np.float32),
            }
        )
        table["hour_local"] = table["time_local"].dt.strftime("%H")
        return table

    # Full-day context prediction (00..23) based on day-start anchor.
    full_hours = pd.date_range(start=day_start_local, end=day_end_local, freq="1h", tz=tz)
    lstm_speed_full, lstm_dir_full = _predict_residuals_for_targets(full_hours)
    # Remaining-day best prediction (next hour..23) based on current anchor.
    lstm_speed_rem, lstm_dir_rem = _predict_residuals_for_targets(remaining_local)

    fc_today = forecast_frame_local.reindex(full_hours)
    table = pd.DataFrame(
        {
            "time_local": full_hours,
            "forecast_wind_speed": fc_today["forecast_avg"].to_numpy(dtype=np.float32),
            "forecast_wind_dir_deg": fc_today["forecast_dir"].to_numpy(dtype=np.float32),
            "actual_wind_speed": actual_today.reindex(full_hours).to_numpy(dtype=np.float32),
            "actual_wind_dir_deg": actual_dir_today.reindex(full_hours).to_numpy(dtype=np.float32),
            "lstm_pred_wind_speed_full": lstm_speed_full.astype(np.float32),
            "lstm_pred_wind_dir_deg_full": lstm_dir_full.astype(np.float32),
            "lstm_pred_wind_speed": np.full(len(full_hours), np.nan, dtype=np.float32),
            "lstm_pred_wind_dir_deg": np.full(len(full_hours), np.nan, dtype=np.float32),
        }
    )
    rem_mask = table["time_local"].isin(remaining_local)
    table.loc[rem_mask, "lstm_pred_wind_speed"] = lstm_speed_rem.astype(np.float32)
    table.loc[rem_mask, "lstm_pred_wind_dir_deg"] = lstm_dir_rem.astype(np.float32)
    table["hour_local"] = table["time_local"].dt.strftime("%H")
    return table


def save_current_day_plot(table: pd.DataFrame, plot_path: Path, local_tz: str) -> None:
    table = table.copy()
    table = table[(table["time_local"].dt.hour >= 8) & (table["time_local"].dt.hour <= 22)].reset_index(drop=True)
    if table.empty:
        raise ValueError("No rows available in 08:00-22:00 range for current-day plotting.")

    x = np.arange(len(table))
    first_dt = table["time_local"].iloc[0]
    day_label = f"{first_dt.day} {first_dt.strftime('%B %Y')}"

    speed_series = pd.concat(
        [
            table["forecast_wind_speed"].dropna(),
            table["lstm_pred_wind_speed_full"].dropna(),
            table["lstm_pred_wind_speed"].dropna(),
            table["actual_wind_speed"].dropna(),
        ]
    )
    y_min = float(speed_series.min()) if not speed_series.empty else 0.0
    y_max = float(speed_series.max()) if not speed_series.empty else 10.0
    pad = max((y_max - y_min) * 0.08, 0.8)
    y_lower = min(y_min - pad, -0.5)
    y_upper = y_max + pad

    fig, ax = plt.subplots(figsize=(14, 6))
    marker_size = 5.5
    ax.plot(
        x,
        table["forecast_wind_speed"],
        marker="o",
        markersize=marker_size,
        color="tab:blue",
        linewidth=2.0,
        label="Forecast speed",
    )
    ax.plot(
        x,
        table["actual_wind_speed"].to_numpy(dtype=float),
        marker="o",
        markersize=marker_size,
        color="magenta",
        linewidth=2.2,
        label="Actual speed (up to now)",
        zorder=5,
    )

    # Vertical line at present hour boundary (first predicted hour).
    future_idx = np.where(~np.isnan(table["lstm_pred_wind_speed"].to_numpy(dtype=float)))[0]

    # LSTM past segment (dashed): full-day context up to current time.
    lstm_past = table["lstm_pred_wind_speed_full"].to_numpy(dtype=float).copy()
    if len(future_idx) > 0:
        lstm_past[future_idx[0]:] = np.nan
    ax.plot(
        x,
        lstm_past,
        color="tab:orange",
        linestyle="--",
        linewidth=1.6,
        alpha=0.9,
        label="_nolegend_",
    )

    # LSTM future segment (solid): remaining-day best prediction from current anchor.
    lstm_future = table["lstm_pred_wind_speed"].to_numpy(dtype=float).copy()
    if len(future_idx) > 0 and future_idx[0] - 1 >= 0:
        lstm_future[future_idx[0] - 1] = table["lstm_pred_wind_speed_full"].to_numpy(dtype=float)[future_idx[0] - 1]
    ax.plot(
        x,
        lstm_future,
        marker="o",
        markersize=marker_size,
        color="tab:orange",
        linewidth=2.0,
        label="LSTM speed",
        zorder=3,
    )
    ax.set_title(f"Current-Day Wind ({day_label}, {local_tz})")
    ax.set_xlabel(f"Hour ({local_tz})")
    ax.set_ylabel("Wind speed (kts)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(0.015, 0.99))
    ax.set_xticks(x, table["hour_local"], rotation=0)
    ax.set_ylim(y_lower, y_upper)

    if len(future_idx) > 0:
        ax.axvline(future_idx[0] - 0.5, color="gray", linestyle="--", linewidth=1.0)

    mae_fc, mae_lstm, _ = compute_running_mae(table)
    mse_text = (
        f"Running MAE up to now\n"
        f"Forecast vs actual: {mae_fc:.2f} kts\n"
        f"LSTM vs actual: {mae_lstm:.2f} kts"
    )
    ax.text(
        0.015,
        0.03,
        mse_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="black",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        zorder=7,
    )

    # Direction arrows below axis for forecast, LSTM (remaining where available, else full-day context), and actual.
    y_base_axes = -0.20
    arrow_len_axes = 0.075
    for i, row in table.iterrows():
        fdir = row["forecast_wind_dir_deg"]
        ldir = row["lstm_pred_wind_dir_deg"]
        adir = row["actual_wind_dir_deg"]
        if pd.isna(ldir):
            ldir = row["lstm_pred_wind_dir_deg_full"]
        for direction_deg, color, z in [(fdir, "tab:blue", 3), (ldir, "tab:orange", 4), (adir, "magenta", 6)]:
            if pd.isna(direction_deg):
                continue
            theta = np.deg2rad((float(direction_deg) + 180.0) % 360.0)
            dx = 0.22 * np.sin(theta)
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

    fig.tight_layout(rect=[0, 0.08, 1, 1])
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


def _save_model(path: Path, model: nn.Module, n_features: int, target_hours: int, target_name: str) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_class": "NextDayLSTM",
            "n_features": int(n_features),
            "target_hours": int(target_hours),
            "target_name": target_name,
            "target_mode": "residual",
        },
        path,
    )


def _load_model(path: Path, device: torch.device) -> nn.Module:
    ckpt = torch.load(path, map_location=device)
    model = NextDayLSTM(n_features=int(ckpt["n_features"]), target_hours=int(ckpt["target_hours"])).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


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


def save_daily_mae_plot(history_csv: Path, plot_png: Path) -> None:
    if not history_csv.exists():
        return
    hist = pd.read_csv(history_csv)
    if hist.empty:
        return

    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist = hist.dropna(subset=["date"]).sort_values("date")
    if hist.empty:
        return

    if "mae_forecast" not in hist.columns and "mse_forecast" in hist.columns:
        hist["mae_forecast"] = hist["mse_forecast"]
    if "mae_lstm" not in hist.columns and "mse_lstm" in hist.columns:
        hist["mae_lstm"] = hist["mse_lstm"]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(hist["date"], hist["mae_forecast"], marker="o", linewidth=1.8, label="Forecast MAE")
    ax.plot(hist["date"], hist["mae_lstm"], marker="o", linewidth=1.8, label="LSTM MAE")
    ax.set_title("Daily End-of-Day MAE")
    ax.set_xlabel("Date")
    ax.set_ylabel("MAE (kts)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.autofmt_xdate(rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(plot_png, dpi=150)
    plt.close(fig)


def maybe_save_daily_mae(
    out_dir: Path,
    local_tz: str,
    test_now_local_hour: int | None,
    mae_forecast: float,
    mae_lstm: float,
    n_points: int,
    current_day_table: pd.DataFrame,
) -> tuple[str | None, str | None]:
    now_local = _resolve_now_local(local_tz, test_now_local_hour)
    if now_local.hour < 22:
        return None, None

    history_csv = out_dir / "daily_mae_history.csv"
    legacy_history_csv = out_dir / "daily_mse_history.csv"
    details_dir = out_dir / "daily_error_details"
    details_dir.mkdir(parents=True, exist_ok=True)
    day_stamp = now_local.strftime("%Y%m%d")
    details_csv = details_dir / f"{day_stamp}_actual_forecast_lstm.csv"

    details = current_day_table.copy()
    details = details[
        [
            "time_local",
            "hour_local",
            "forecast_wind_speed",
            "lstm_pred_wind_speed_full",
            "actual_wind_speed",
        ]
    ].rename(
        columns={
            "lstm_pred_wind_speed_full": "lstm_wind_speed",
        }
    )
    details["abs_err_forecast"] = np.abs(details["actual_wind_speed"] - details["forecast_wind_speed"])
    details["abs_err_lstm"] = np.abs(details["actual_wind_speed"] - details["lstm_wind_speed"])
    details["time_local"] = pd.to_datetime(details["time_local"]).dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    details.to_csv(details_csv, index=False)

    row = pd.DataFrame(
        [
            {
                "date": now_local.strftime("%Y-%m-%d"),
                "run_local_time": now_local.isoformat(),
                "mae_forecast": mae_forecast,
                "mae_lstm": mae_lstm,
                "n_points": int(n_points),
                "details_csv": str(details_csv),
            }
        ]
    )
    if history_csv.exists():
        hist = pd.read_csv(history_csv)
    elif legacy_history_csv.exists():
        hist = pd.read_csv(legacy_history_csv)
        if "mae_forecast" not in hist.columns and "mse_forecast" in hist.columns:
            hist["mae_forecast"] = hist["mse_forecast"]
        if "mae_lstm" not in hist.columns and "mse_lstm" in hist.columns:
            hist["mae_lstm"] = hist["mse_lstm"]
        keep_cols = [c for c in ["date", "run_local_time", "mae_forecast", "mae_lstm", "n_points"] if c in hist.columns]
        hist = hist[keep_cols]
    else:
        hist = pd.DataFrame(columns=["date", "run_local_time", "mae_forecast", "mae_lstm", "n_points", "details_csv"])

    hist = hist[hist["date"] != row.iloc[0]["date"]]
    hist = pd.concat([hist, row], ignore_index=True)

    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist = hist.dropna(subset=["date"]).sort_values("date")
    hist["date"] = hist["date"].dt.strftime("%Y-%m-%d")
    hist.to_csv(history_csv, index=False)

    history_png = out_dir / "daily_mae_history.png"
    save_daily_mae_plot(history_csv, history_png)
    return str(history_csv), str(history_png)


def publish_web_dashboard(
    web_out_dir: Path,
    local_tz: str,
    web_refresh_seconds: int,
    next_day_png: Path,
    next_day_csv: Path,
    current_day_png: Path,
    current_day_csv: Path,
    daily_mae_png: Path | None,
    daily_mae_csv: Path | None,
) -> dict:
    web_out_dir.mkdir(parents=True, exist_ok=True)

    publish_pairs: list[tuple[Path | None, str]] = [
        (next_day_png, "next_day_predictions.png"),
        (next_day_csv, "next_day_predictions.csv"),
        (current_day_png, "current_day_predictions.png"),
        (current_day_csv, "current_day_predictions.csv"),
        (daily_mae_png, "daily_mae_history.png"),
        (daily_mae_csv, "daily_mae_history.csv"),
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
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="{refresh}">
  <title>Super local wind prediction Valkenburgse meer</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 16px; color: #111; }}
    h1 {{ margin: 0 0 8px 0; }}
    .meta {{ color: #555; margin: 0 0 16px 0; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 20px; max-width: 1400px; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 10px; background: #fff; }}
    img {{ width: 100%; height: auto; display: block; border-radius: 6px; }}
    a {{ color: #0a58ca; text-decoration: none; margin-right: 12px; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <h1>Super local wind prediction Valkenburgse meer</h1>
  <p class="meta">Last updated: {generated_local_str}</p>
  <p>
    <a href="next_day_predictions.csv">Next-day table CSV</a>
    <a href="current_day_predictions.csv">Current-day table CSV</a>
    <a href="daily_mae_history.csv">Daily MAE history CSV</a>
  </p>
  <div class="grid">
    <div class="card">
      <h2>Current Day Prediction</h2>
      <img src="current_day_predictions.png?v={cache_bust}" alt="Current day prediction">
    </div>
    <div class="card">
      <h2>Next Day Prediction</h2>
      <img src="next_day_predictions.png?v={cache_bust}" alt="Next day prediction">
    </div>
    <div class="card">
      <h2>Daily MAE History</h2>
      <img src="daily_mae_history.png?v={cache_bust}" alt="Daily MAE history">
    </div>
  </div>
</body>
</html>
"""
    index_path = web_out_dir / "index.html"
    index_path.write_text(html, encoding="utf-8")
    copied["index.html"] = str(index_path)
    return copied


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

    if args.skip_training:
        for p in [speed_model_path, direction_model_path, *speed_scalers_path.values(), *direction_scalers_path.values()]:
            if not p.exists():
                raise FileNotFoundError(f"Missing artifact for --skip-training mode: {p}")
        speed_model = _load_model(speed_model_path, device)
        direction_model = _load_model(direction_model_path, device)
        speed_arrays = {k: np.load(v) for k, v in speed_scalers_path.items()}
        direction_arrays = {k: np.load(v) for k, v in direction_scalers_path.items()}
        speed_train_stats = None
        direction_train_stats = None
        n_samples_all_speed = None
        n_samples_all_direction = None
        feature_cols = ["forecast_avg", "forecast_max", "forecast_dir", "month_sin", "month_cos"]
    else:
        # --- Train residual speed model ---
        speed_arrays = build_all_training_arrays(db_path, cfg, target_mode="residual")
        speed_model = NextDayLSTM(
            n_features=speed_arrays["X_all"].shape[2],
            target_hours=speed_arrays["y_all"].shape[1],
        ).to(device)
        speed_model, speed_train_stats = train_with_validation(
            model=speed_model,
            X_all=speed_arrays["X_all"],
            y_all=speed_arrays["y_all"],
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_split=args.validation_split,
            model_label="speed",
            device=device,
        )

        # --- Train residual direction model ---
        direction_arrays = build_all_direction_training_arrays(db_path, cfg)
        direction_model = NextDayLSTM(
            n_features=direction_arrays["X_all"].shape[2],
            target_hours=direction_arrays["y_all"].shape[1],
        ).to(device)
        direction_model, direction_train_stats = train_with_validation(
            model=direction_model,
            X_all=direction_arrays["X_all"],
            y_all=direction_arrays["y_all"],
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_split=args.validation_split,
            model_label="direction",
            device=device,
        )

        _save_model(
            speed_model_path,
            speed_model,
            n_features=speed_arrays["X_all"].shape[2],
            target_hours=speed_arrays["y_all"].shape[1],
            target_name="wind_speed",
        )
        _save_model(
            direction_model_path,
            direction_model,
            n_features=direction_arrays["X_all"].shape[2],
            target_hours=direction_arrays["y_all"].shape[1],
            target_name="wind_direction",
        )

        np.save(speed_scalers_path["x_mean"], speed_arrays["x_mean"])
        np.save(speed_scalers_path["x_std"], speed_arrays["x_std"])
        np.save(speed_scalers_path["y_mean"], speed_arrays["y_mean"])
        np.save(speed_scalers_path["y_std"], speed_arrays["y_std"])
        np.save(direction_scalers_path["x_mean"], direction_arrays["x_mean"])
        np.save(direction_scalers_path["x_std"], direction_arrays["x_std"])
        np.save(direction_scalers_path["y_mean"], direction_arrays["y_mean"])
        np.save(direction_scalers_path["y_std"], direction_arrays["y_std"])
        n_samples_all_speed = int(speed_arrays["X_all"].shape[0])
        n_samples_all_direction = int(direction_arrays["X_all"].shape[0])
        feature_cols = speed_arrays["feature_cols"]

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

    speed_pred = predict_speed_residual(
        model=speed_model,
        X_input=inference_input_speed["X_input"],
        forecast_speed=inference_input_speed["forecast_next24"],
        y_mean=float(speed_arrays["y_mean"][0]),
        y_std=float(speed_arrays["y_std"][0]),
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

    plot_path = out_dir / "next_day_predictions.png"
    save_prediction_plot(table, plot_path)

    # --- Current day plot/table: actuals up to present + prediction for remaining day ---
    current_day_table = build_current_day_table(
        db_path=db_path,
        cfg=cfg,
        speed_model=speed_model,
        direction_model=direction_model,
        speed_scalers={"x_mean": speed_arrays["x_mean"], "x_std": speed_arrays["x_std"], "y_mean": speed_arrays["y_mean"], "y_std": speed_arrays["y_std"]},
        direction_scalers={
            "x_mean": direction_arrays["x_mean"],
            "x_std": direction_arrays["x_std"],
            "y_mean": direction_arrays["y_mean"],
            "y_std": direction_arrays["y_std"],
        },
        local_tz=args.local_timezone,
        test_now_local_hour=args.test_now_local_hour,
        device=device,
    )
    is_test_mode = args.test_now_local_hour is not None
    test_suffix = f"_test_hour_{int(args.test_now_local_hour):02d}" if is_test_mode else ""

    current_day_table_path = out_dir / f"current_day_predictions{test_suffix}.csv"
    current_day_table_csv = current_day_table.copy()
    current_day_table_csv["time_local"] = current_day_table_csv["time_local"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    current_day_table_csv.to_csv(current_day_table_path, index=False)

    current_day_plot_path = out_dir / f"current_day_predictions{test_suffix}.png"
    save_current_day_plot(current_day_table, current_day_plot_path, args.local_timezone)
    mae_forecast, mae_lstm, mae_n_points = compute_running_mae(current_day_table)
    archived_current_day_plot = None
    daily_mae_csv = None
    daily_mae_png = None
    if not is_test_mode:
        archived_current_day_plot = maybe_archive_current_day_plot(
            current_day_plot_path=current_day_plot_path,
            out_dir=out_dir,
            local_tz=args.local_timezone,
            test_now_local_hour=args.test_now_local_hour,
        )
        daily_mae_csv, daily_mae_png = maybe_save_daily_mae(
        out_dir=out_dir,
        local_tz=args.local_timezone,
        test_now_local_hour=args.test_now_local_hour,
        mae_forecast=mae_forecast,
        mae_lstm=mae_lstm,
        n_points=mae_n_points,
            current_day_table=current_day_table,
        )

    web_publish = None
    if not is_test_mode:
        daily_mae_png_src = None if daily_mae_png is None else Path(daily_mae_png)
        daily_mae_csv_src = None if daily_mae_csv is None else Path(daily_mae_csv)
        if daily_mae_png_src is None:
            fallback_png = out_dir / "daily_mae_history.png"
            if fallback_png.exists():
                daily_mae_png_src = fallback_png
        if daily_mae_csv_src is None:
            fallback_csv = out_dir / "daily_mae_history.csv"
            if fallback_csv.exists():
                daily_mae_csv_src = fallback_csv
        web_publish = publish_web_dashboard(
            web_out_dir=Path(args.web_out_dir),
            local_tz=args.local_timezone,
            web_refresh_seconds=args.web_refresh_seconds,
            next_day_png=plot_path,
            next_day_csv=table_path,
            current_day_png=current_day_plot_path,
            current_day_csv=current_day_table_path,
            daily_mae_png=daily_mae_png_src,
            daily_mae_csv=daily_mae_csv_src,
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
        "speed_model_target": "residual",
        "direction_model_target": "residual",
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
        "y_scaler_mean_speed": float(speed_arrays["y_mean"][0]),
        "y_scaler_std_speed": float(speed_arrays["y_std"][0]),
        "y_scaler_mean_direction": float(direction_arrays["y_mean"][0]),
        "y_scaler_std_direction": float(direction_arrays["y_std"][0]),
        "prediction_table_csv": str(table_path),
        "prediction_plot_png": str(plot_path),
        "current_day_table_csv": str(current_day_table_path),
        "current_day_plot_png": str(current_day_plot_path),
        "current_day_plot_archived_png": archived_current_day_plot,
        "daily_mae_history_csv": daily_mae_csv,
        "daily_mae_history_png": daily_mae_png,
        "web_dashboard_dir": None if web_publish is None else str(Path(args.web_out_dir).resolve()),
        "web_dashboard_files": web_publish,
        "speed_model_path": str(speed_model_path),
        "direction_model_path": str(direction_model_path),
        "data_refresh": refresh_info,
        "max_forecast_age_hours": args.max_forecast_age_hours,
        "expected_update_hour_utc": args.expected_update_hour_utc,
        "validation_split": args.validation_split,
        "local_timezone": args.local_timezone,
        "test_now_local_hour": args.test_now_local_hour,
    }
    metadata_path = out_dir / ("metadata_update.json" if not is_test_mode else f"metadata_update{test_suffix}.json")
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Model update complete.")
    print(f"Speed model saved to: {speed_model_path}")
    print(f"Direction model saved to: {direction_model_path}")
    print(f"Prediction table saved to: {table_path}")
    print(f"Prediction plot saved to: {plot_path}")
    print(f"Current-day table saved to: {current_day_table_path}")
    print(f"Current-day plot saved to: {current_day_plot_path}")
    if is_test_mode:
        print("Test mode active: skipped daily archive/history updates and preserved production current-day outputs.")
    if archived_current_day_plot is not None:
        print(f"Current-day plot archived to: {archived_current_day_plot}")
    if daily_mae_csv is not None and daily_mae_png is not None:
        print(f"Daily MAE history saved to: {daily_mae_csv}")
        print(f"Daily MAE history plot saved to: {daily_mae_png}")
    if web_publish is not None:
        print(f"Web dashboard updated in: {Path(args.web_out_dir).resolve()}")
    print()
    print(table_for_csv.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
