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
import matplotlib.dates as mdates
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
    build_all_direction_training_arrays,
    build_all_training_arrays,
    build_next_day_inference_input,
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


def predict_speed(
    model: nn.Module,
    X_input: np.ndarray,
    forecast_speed: np.ndarray,
    y_mean: float,
    y_std: float,
    target_mode: str,
    constraint_eps: float | None,
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


def save_prediction_plot(
    table: pd.DataFrame,
    plot_path: Path,
    local_tz: str,
    prediction_generated_at_utc: str,
    prediction_updated_at_utc: str | None,
    model_trained_at_utc: str | None,
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

    fig, ax = plt.subplots(figsize=(14, 7.2))
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
    ax.set_title(f"Next-Day Wind Speed ({day_label}, UTC)")
    ax.set_xlabel("Hour (UTC)")
    ax.set_ylabel("Wind speed (kts)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(0.015, 0.99), borderaxespad=0.0)
    ax.set_xticks(x, table["hour_label"], rotation=0)
    ax.set_xlim(0.0, len(table) - 1.0)
    ax.set_ylim(0.0, y_max + pad)
    ax.text(
        0.015,
        1.13,
        _format_plot_meta_text(prediction_generated_at_utc, prediction_updated_at_utc, model_trained_at_utc, local_tz),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="black",
        clip_on=False,
    )

    # Draw wind direction arrows under x-axis.
    # Mapping: up-arrow means South wind (from South), per user preference.
    # Using x-axis transform keeps arrows below axis regardless of y-scale.
    y_base_axes = -0.14
    arrow_len_axes = 0.065
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

    fig.tight_layout(rect=[0, 0.04, 1, 0.965])
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def build_current_day_table(
    db_path: Path,
    cfg: DatasetConfig,
    speed_model: nn.Module,
    direction_model: nn.Module,
    speed_scalers: dict,
    direction_scalers: dict,
    speed_target_mode: str,
    speed_constraint_eps: float | None,
    local_tz: str,
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

    # Remaining hours today (prediction target): next hour .. 23:00 local.
    remaining_local = pd.date_range(
        start=now_hour_local + timedelta(hours=1),
        end=day_end_local,
        freq="1h",
        tz=tz,
    )
    remaining_n = len(remaining_local)

    # Full-day context prediction (00..23) based on day-start anchor.
    full_hours = pd.date_range(start=day_start_local, end=day_end_local, freq="1h", tz=tz)
    lstm_speed_full, lstm_dir_full = _predict_residuals_for_targets(full_hours)
    # Remaining-day best prediction (next hour..23) based on current anchor.
    if remaining_n > 0:
        lstm_speed_rem, lstm_dir_rem = _predict_residuals_for_targets(remaining_local)
    else:
        lstm_speed_rem = np.array([], dtype=np.float32)
        lstm_dir_rem = np.array([], dtype=np.float32)

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
    lstm_full_dense = _interp_hourly_to_dense(lstm_speed_full.astype(np.float32), full_hours, dense_times)
    lstm_dir_full_dense = _interp_hourly_to_dense(lstm_dir_full.astype(np.float32), full_hours, dense_times)

    rem_hourly_speed = pd.Series(np.nan, index=full_hours, dtype=float)
    rem_hourly_dir = pd.Series(np.nan, index=full_hours, dtype=float)
    if remaining_n > 0:
        rem_hourly_speed.loc[remaining_local] = lstm_speed_rem.astype(np.float32)
        rem_hourly_dir.loc[remaining_local] = lstm_dir_rem.astype(np.float32)
        # Add continuity point at the boundary for a continuous future line.
        prev_hour = remaining_local[0] - timedelta(hours=1)
        if prev_hour in rem_hourly_speed.index:
            rem_hourly_speed.loc[prev_hour] = float(
                lstm_speed_full[np.where(full_hours == prev_hour)[0][0]]
            )
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
) -> None:
    table = table.copy()
    table = table[(table["time_local"].dt.hour >= 8) & (table["time_local"].dt.hour <= 22)].reset_index(drop=True)
    if table.empty:
        raise ValueError("No rows available in 08:00-22:00 range for current-day plotting.")

    x = np.arange(len(table))
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

    fig, ax = plt.subplots(figsize=(14, 7.2))
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

    # Vertical line at present hour boundary (first predicted hour).
    future_idx = np.where(table["is_future"].to_numpy(dtype=bool))[0]

    # LSTM past segment (dashed): full-day context up to current time.
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
        lstm_future[future_idx[0] - 1] = table["lstm_pred_wind_speed_full"].to_numpy(dtype=float)[future_idx[0] - 1]
    ax.plot(
        x,
        lstm_future,
        color=LSTM_HIGHLIGHT_COLOR,
        linewidth=2.6,
        label="Super local wind prediction - avg speed",
        zorder=3,
    )
    ax.set_title(f"Current-day wind prediction: {day_label}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Wind speed (kts)")
    ax.grid(axis="y", alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    desired_order = [
        "Super local wind prediction - avg speed",
        "Wind speed - measured",
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
    )
    hour_tick_mask = table["time_local"].dt.minute.eq(0)
    tick_pos = np.where(hour_tick_mask.to_numpy())[0]
    tick_lbl = table.loc[hour_tick_mask, "time_local"].dt.strftime("%H").to_list()
    ax.set_xticks(tick_pos, tick_lbl, rotation=0)
    ax.set_xlim(0.0, len(table) - 1.0)
    ax.set_ylim(y_lower, y_upper)

    actual_idx = np.where(~np.isnan(actual_avg))[0]
    if len(actual_idx) > 0:
        # Mark "now" at the latest available measured point on the dense timeline.
        ax.axvline(float(actual_idx[-1]), color="gray", linestyle="--", linewidth=1.0)

    mae_fc, mae_lstm, _ = compute_running_mae(table)
    mse_text = (
        f"Running mean absolute error\n"
        f"Super local vs measured wind: {mae_lstm:.2f} kts\n"
        f"Harmonie vs measured wind: {mae_fc:.2f} kts"
    )
    ax.text(
        0.985,
        1.13,
        mse_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="black",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        zorder=7,
        clip_on=False,
    )
    ax.text(
        0.015,
        1.13,
        _format_plot_meta_text(prediction_generated_at_utc, prediction_updated_at_utc, model_trained_at_utc, local_tz),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="black",
        clip_on=False,
    )

    # Direction arrows below axis for forecast, LSTM (remaining where available, else full-day context), and actual.
    y_base_axes = -0.14
    arrow_len_axes = 0.065
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

    fig.tight_layout(rect=[0, 0.04, 1, 0.965])
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


def save_daily_mae_plot(history_csv: Path, plot_png: Path, local_tz: str = "Europe/Amsterdam") -> None:
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
    hist["mae_forecast"] = pd.to_numeric(hist["mae_forecast"], errors="coerce")
    hist["mae_lstm"] = pd.to_numeric(hist["mae_lstm"], errors="coerce")
    hist = hist.dropna(subset=["mae_forecast", "mae_lstm"])
    if hist.empty:
        return

    # If multiple runs exist for a day, plot the average daily MAE.
    hist["day"] = hist["date"].dt.floor("D")
    hist_daily = hist.groupby("day", as_index=False)[["mae_forecast", "mae_lstm"]].mean(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    now_local = datetime.now(ZoneInfo(local_tz))
    month_start_current = now_local.replace(day=1, hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    if month_start_current.month == 12:
        next_month = month_start_current.replace(year=month_start_current.year + 1, month=1)
    else:
        next_month = month_start_current.replace(month=month_start_current.month + 1)
    month_end_current = next_month - timedelta(days=1)

    first_day = hist_daily["day"].min().to_pydatetime().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    x_start = pd.Timestamp(first_day)
    x_end = pd.Timestamp(month_end_current)

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
            ax.axvspan(cursor, month_end + pd.Timedelta(hours=23, minutes=59), color="0.8", alpha=0.18, zorder=0)
        shade_idx += 1
        cursor = month_next

    if not hist_daily.empty:
        ax.plot(
            hist_daily["day"],
            hist_daily["mae_lstm"],
            linewidth=2.2,
            color=LSTM_HIGHLIGHT_COLOR,
            label="Super local vs measured wind",
        )
        ax.plot(
            hist_daily["day"],
            hist_daily["mae_forecast"],
            linewidth=1.8,
            color="gray",
            label="Harmonie vs measured wind",
        )
    ax.set_title("Daily End-of-Day MAE")
    ax.set_xlabel("Date")
    ax.set_ylabel("MAE (kts)")
    ax.grid(axis="y", alpha=0.3)
    if not hist_daily.empty:
        ax.legend(loc="upper left")
    y_top_data = float(np.nanmax([hist_daily["mae_forecast"].max(), hist_daily["mae_lstm"].max()])) if not hist_daily.empty else 0.0
    y_top = max(4.0, y_top_data * 1.06)
    ax.set_ylim(0.0, y_top)
    ax.margins(x=0, y=0)
    ax.set_xlim(x_start, x_end)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    if hist_daily.empty:
        avg_text = "Average MAE in shown period\nSuper local: n/a\nHarmonie: n/a"
    else:
        avg_lstm = float(hist_daily["mae_lstm"].mean())
        avg_fc = float(hist_daily["mae_forecast"].mean())
        avg_text = (
            "Average MAE in shown period\n"
            f"Super local: {avg_lstm:.2f} kts\n"
            f"Harmonie: {avg_fc:.2f} kts"
        )
    ax.text(
        0.985,
        1.02,
        avg_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="black",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    ax.set_title("Mean Average Prediction Error")
    fig.autofmt_xdate(rotation=25, ha="right")
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
    save_daily_mae_plot(history_csv, history_png, local_tz=local_tz)
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
        speed_model, speed_ckpt = _load_model(speed_model_path, device)
        direction_model, direction_ckpt = _load_model(direction_model_path, device)
        speed_arrays = {k: np.load(v) for k, v in speed_scalers_path.items()}
        direction_arrays = {k: np.load(v) for k, v in direction_scalers_path.items()}
        speed_target_mode = str(speed_ckpt.get("target_mode", "residual")).strip().lower()
        direction_target_mode = str(direction_ckpt.get("target_mode", "residual")).strip().lower()
        speed_constraint_eps = speed_ckpt.get("constraint_eps", None)
        model_last_trained_at_utc = _resolve_model_trained_utc(speed_ckpt, speed_model_path)
        speed_train_stats = None
        direction_train_stats = None
        n_samples_all_speed = None
        n_samples_all_direction = None
        feature_cols = ["forecast_avg", "forecast_max", "forecast_dir", "month_sin", "month_cos"]
    else:
        # --- Train constrained residual speed model (positive by construction) ---
        speed_arrays = build_all_training_arrays(db_path, cfg, target_mode="residual")
        speed_constraint_eps = float(args.speed_constraint_eps)
        speed_y_raw = np.log(speed_arrays["y_actual_all_raw"] + speed_constraint_eps) - np.log(
            speed_arrays["y_forecast_all_raw"] + speed_constraint_eps
        )
        y_mean = float(speed_y_raw.mean())
        y_std = float(speed_y_raw.std())
        if y_std == 0.0:
            y_std = 1.0
        speed_arrays["y_mean"] = np.array([y_mean], dtype=np.float32)
        speed_arrays["y_std"] = np.array([y_std], dtype=np.float32)
        speed_y_scaled = (speed_y_raw - y_mean) / y_std
        speed_model = NextDayLSTM(
            n_features=speed_arrays["X_all"].shape[2],
            target_hours=speed_y_scaled.shape[1],
            output_activation="linear",
        ).to(device)
        speed_model, speed_train_stats = train_with_validation(
            model=speed_model,
            X_all=speed_arrays["X_all"],
            y_all=speed_y_scaled.astype(np.float32),
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_split=args.validation_split,
            model_label="speed",
            device=device,
        )
        speed_target_mode = "constrained_logratio"

        # --- Train residual direction model ---
        direction_arrays = build_all_direction_training_arrays(db_path, cfg)
        direction_model = NextDayLSTM(
            n_features=direction_arrays["X_all"].shape[2],
            target_hours=direction_arrays["y_all"].shape[1],
            output_activation="linear",
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
            target_hours=speed_y_scaled.shape[1],
            target_name="wind_speed",
            target_mode=speed_target_mode,
            output_activation="linear",
            extra={"constraint_eps": speed_constraint_eps, "trained_at_utc": datetime.now(timezone.utc).isoformat()},
        )
        _save_model(
            direction_model_path,
            direction_model,
            n_features=direction_arrays["X_all"].shape[2],
            target_hours=direction_arrays["y_all"].shape[1],
            target_name="wind_direction",
            target_mode="residual",
            output_activation="linear",
            extra={"trained_at_utc": datetime.now(timezone.utc).isoformat()},
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
        direction_target_mode = "residual"
        model_last_trained_at_utc = datetime.now(timezone.utc).isoformat()

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

    speed_pred = predict_speed(
        model=speed_model,
        X_input=inference_input_speed["X_input"],
        forecast_speed=inference_input_speed["forecast_next24"],
        y_mean=float(speed_arrays["y_mean"][0]),
        y_std=float(speed_arrays["y_std"][0]),
        target_mode=speed_target_mode,
        constraint_eps=speed_constraint_eps,
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
        speed_target_mode=speed_target_mode,
        speed_constraint_eps=speed_constraint_eps,
        local_tz=args.local_timezone,
        test_now_local_hour=args.test_now_local_hour,
        current_day_interval_minutes=args.current_day_interval_minutes,
        device=device,
    )
    is_test_mode = args.test_now_local_hour is not None
    test_suffix = f"_test_hour_{int(args.test_now_local_hour):02d}" if is_test_mode else ""

    current_day_table_path = out_dir / f"current_day_predictions{test_suffix}.csv"
    current_day_table_csv = current_day_table.copy()
    current_day_table_csv["time_local"] = current_day_table_csv["time_local"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    current_day_table_csv.to_csv(current_day_table_path, index=False)

    current_day_plot_path = out_dir / f"current_day_predictions{test_suffix}.png"
    save_current_day_plot(
        current_day_table,
        current_day_plot_path,
        args.local_timezone,
        prediction_generated_at_utc=prediction_generated_at_utc,
        prediction_updated_at_utc=prediction_updated_at_utc,
        model_trained_at_utc=model_last_trained_at_utc,
    )
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
    git_publish = {"enabled": bool(args.git_auto_push_pages), "pushed": False, "reason": "disabled"}
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
        # Always refresh daily MAE plot from CSV when available, not only after end-of-day save.
        if daily_mae_csv_src is not None:
            daily_mae_png_refresh = out_dir / "daily_mae_history.png"
            save_daily_mae_plot(daily_mae_csv_src, daily_mae_png_refresh, local_tz=args.local_timezone)
            daily_mae_png_src = daily_mae_png_refresh
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
        "prediction_plot_png": str(plot_path),
        "current_day_table_csv": str(current_day_table_path),
        "current_day_plot_png": str(current_day_plot_path),
        "current_day_plot_archived_png": archived_current_day_plot,
        "daily_mae_history_csv": daily_mae_csv,
        "daily_mae_history_png": daily_mae_png,
        "web_dashboard_dir": None if web_publish is None else str(Path(args.web_out_dir).resolve()),
        "web_dashboard_files": web_publish,
        "web_dashboard_git_publish": git_publish,
        "speed_model_path": str(speed_model_path),
        "direction_model_path": str(direction_model_path),
        "data_refresh": refresh_info,
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
