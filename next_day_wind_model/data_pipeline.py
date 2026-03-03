from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetConfig:
    site: str = "valkenburgsemeer"
    model: str = "HARMONIE"
    window_hours: int = 72
    target_hours: int = 24


def _to_float(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_first(payload: Dict, keys: Sequence[str]) -> float | None:
    lower_map = {k.lower(): k for k in payload.keys()}
    for key in keys:
        real_key = lower_map.get(key.lower())
        if real_key is not None:
            return _to_float(payload.get(real_key))
    return None


def _load_forecasts(conn: sqlite3.Connection, site: str, model: str) -> pd.DataFrame:
    query = """
    SELECT run_ts, target_ts, payload
    FROM forecasts
    WHERE site = ?
      AND model = ?
      AND target_ts IS NOT NULL
    """
    rows = conn.execute(query, (site, model)).fetchall()
    if not rows:
        raise ValueError("No forecast rows found for selected site/model.")

    records: List[Dict] = []
    for run_ts, target_ts, payload_raw in rows:
        payload = json.loads(payload_raw) if payload_raw else {}
        records.append(
            {
                "run_ts": int(run_ts),
                "target_ts": int(target_ts),
                "forecast_avg": _extract_first(
                    payload,
                    ["WindForecastAvr", "wind_speed", "windspeed", "WS", "ff", "speed"],
                ),
                "forecast_max": _extract_first(
                    payload,
                    ["WindForecastMax", "wind_gust", "gust", "WG", "fg"],
                ),
                "forecast_dir": _extract_first(
                    payload,
                    ["WindDirection", "wind_dir", "winddirection", "WD", "DD", "dir", "direction"],
                ),
            }
        )

    forecast_df = pd.DataFrame.from_records(records)
    forecast_df["target_dt"] = pd.to_datetime(forecast_df["target_ts"], unit="ms", utc=True)

    # Keep the latest run per target timestamp.
    forecast_df = forecast_df.sort_values(["target_dt", "run_ts"], ascending=[True, False])
    forecast_df = forecast_df.drop_duplicates(subset=["target_dt"], keep="first")
    forecast_df = forecast_df.set_index("target_dt").sort_index()

    return forecast_df[["forecast_avg", "forecast_max", "forecast_dir"]]


def _load_observations(conn: sqlite3.Connection, site: str) -> pd.DataFrame:
    query = """
    SELECT ts, wind_speed, wind_gust, wind_dir, payload
    FROM observations
    WHERE site = ?
      AND ts IS NOT NULL
    """
    rows = conn.execute(query, (site,)).fetchall()
    if not rows:
        raise ValueError("No observation rows found for selected site.")

    records: List[Dict] = []
    for ts, wind_speed, wind_gust, wind_dir, payload_raw in rows:
        payload = json.loads(payload_raw) if payload_raw else {}
        actual_avg = _extract_first(
            payload,
            ["AverageWind", "wind_speed", "windspeed", "WS", "ff", "speed", "WindSpeedAvg"],
        )
        actual_max = _extract_first(
            payload,
            ["MaxWind", "wind_gust", "gust", "WG", "fg"],
        )
        actual_dir = _extract_first(
            payload,
            ["WindDirection", "wind_dir", "winddirection", "WD", "DD", "dir", "direction"],
        )
        # Fallback to DB columns when payload does not expose expected keys.
        if actual_avg is None:
            actual_avg = _to_float(wind_speed)
        if actual_max is None:
            actual_max = _to_float(wind_gust)
        if actual_dir is None:
            actual_dir = _to_float(wind_dir)
        records.append(
            {
                "obs_ts": int(ts),
                "actual_avg": actual_avg,
                "actual_max": actual_max,
                "actual_dir": actual_dir,
            }
        )

    obs_df = pd.DataFrame.from_records(records)
    obs_df["obs_dt"] = pd.to_datetime(obs_df["obs_ts"], unit="ms", utc=True)
    obs_df = obs_df.set_index("obs_dt").sort_index()

    # Observations are sub-hourly; aggregate to hourly means for stable supervision.
    hourly = obs_df.resample("1h").mean(numeric_only=True)
    return hourly[["actual_avg", "actual_max", "actual_dir"]]


def _build_aligned_hourly_frame(conn: sqlite3.Connection, cfg: DatasetConfig) -> pd.DataFrame:
    fc = _load_forecasts(conn, cfg.site, cfg.model)
    obs = _load_observations(conn, cfg.site)

    frame = fc.join(obs, how="inner")
    frame = frame.dropna(subset=["forecast_avg", "actual_avg"])
    if frame.empty:
        raise ValueError("No aligned rows after joining forecasts and observations.")
    return frame


def _add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    month = out.index.month.astype(np.float32)
    angle = (2.0 * np.pi * (month - 1.0)) / 12.0
    out["month_sin"] = np.sin(angle)
    out["month_cos"] = np.cos(angle)
    return out


def _interpolate_missing_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    feature_cols = ["forecast_avg", "forecast_max", "forecast_dir", "actual_max", "actual_dir"]
    for col in feature_cols:
        if col in out.columns:
            out[col] = out[col].interpolate(limit_direction="both")
    return out


def _fit_standardizer(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # X shape: (samples, timesteps, features)
    flat = X.reshape(-1, X.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std == 0.0] = 1.0
    return mean, std


def _apply_standardizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)


def _fit_target_scaler(y: np.ndarray) -> Tuple[float, float]:
    mean = float(y.mean())
    std = float(y.std())
    if std == 0.0:
        std = 1.0
    return mean, std


def _apply_target_scaler(y: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (y - mean) / std


def _angle_diff_deg(actual: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    return ((actual - forecast + 180.0) % 360.0) - 180.0


def _angle_add_deg(base: np.ndarray, delta: np.ndarray) -> np.ndarray:
    return (base + delta) % 360.0


def _resolve_target_mode(target_mode: str) -> str:
    mode = str(target_mode).strip().lower()
    if mode not in {"absolute", "residual"}:
        raise ValueError("target_mode must be one of: absolute, residual")
    return mode


def _build_samples(
    frame: pd.DataFrame,
    cfg: DatasetConfig,
    feature_cols: List[str],
    target_col: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_values = frame[feature_cols].to_numpy(dtype=np.float32)
    actual_values = frame[target_col].to_numpy(dtype=np.float32)
    forecast_values = frame["forecast_avg"].to_numpy(dtype=np.float32)

    X_list: List[np.ndarray] = []
    y_actual_list: List[np.ndarray] = []
    y_forecast_list: List[np.ndarray] = []
    timestamps: List[str] = []

    window = cfg.window_hours
    horizon = cfg.target_hours
    total = len(frame)

    for i in range(window - 1, total - horizon):
        x_window = feature_values[i - window + 1 : i + 1]
        y_actual_next = actual_values[i + 1 : i + 1 + horizon]
        y_forecast_next = forecast_values[i + 1 : i + 1 + horizon]

        if np.isnan(x_window).any() or np.isnan(y_actual_next).any() or np.isnan(y_forecast_next).any():
            continue

        X_list.append(x_window)
        y_actual_list.append(y_actual_next)
        y_forecast_list.append(y_forecast_next)
        timestamps.append(frame.index[i].isoformat())

    if not X_list:
        raise ValueError("No training samples could be built with the current window/horizon settings.")

    X = np.stack(X_list).astype(np.float32)
    y_actual = np.stack(y_actual_list).astype(np.float32)
    y_forecast = np.stack(y_forecast_list).astype(np.float32)
    return X, y_actual, y_forecast, np.array(timestamps)


def _build_training_frame(db_path: Path, cfg: DatasetConfig) -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    try:
        frame = _build_aligned_hourly_frame(conn, cfg)
    finally:
        conn.close()
    frame = _interpolate_missing_features(frame)
    frame = _add_calendar_features(frame)
    return frame


def _build_forecast_feature_frame(db_path: Path, cfg: DatasetConfig) -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    try:
        frame = _load_forecasts(conn, cfg.site, cfg.model)
    finally:
        conn.close()
    frame = _interpolate_missing_features(frame)
    frame = _add_calendar_features(frame)
    return frame


def _latest_observation_time(conn: sqlite3.Connection, site: str) -> pd.Timestamp:
    query = """
    SELECT MAX(ts)
    FROM observations
    WHERE site = ?
      AND ts IS NOT NULL
    """
    row = conn.execute(query, (site,)).fetchone()
    if row is None or row[0] is None:
        raise ValueError("No observations found to determine reference day for inference.")
    return pd.to_datetime(int(row[0]), unit="ms", utc=True)


def build_all_training_arrays(
    db_path: Path,
    cfg: DatasetConfig,
    target_mode: str = "absolute",
) -> Dict[str, np.ndarray | List[str]]:
    target_mode = _resolve_target_mode(target_mode)
    frame = _build_training_frame(db_path, cfg)

    feature_cols = ["forecast_avg", "forecast_max", "forecast_dir", "month_sin", "month_cos"]
    target_col = "actual_avg"
    X_raw, y_actual_raw, y_forecast_raw, timestamps = _build_samples(frame, cfg, feature_cols, target_col)
    if target_mode == "residual":
        y_target_raw = y_actual_raw - y_forecast_raw
    else:
        y_target_raw = y_actual_raw

    x_mean, x_std = _fit_standardizer(X_raw)
    y_mean, y_std = _fit_target_scaler(y_target_raw)
    X_scaled = _apply_standardizer(X_raw, x_mean, x_std)
    y_scaled = _apply_target_scaler(y_target_raw, y_mean, y_std)

    return {
        "X_all": X_scaled,
        "y_all": y_scaled,
        "y_actual_all_raw": y_actual_raw,
        "y_forecast_all_raw": y_forecast_raw,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": np.array([y_mean], dtype=np.float32),
        "y_std": np.array([y_std], dtype=np.float32),
        "timestamps": timestamps,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "target_mode": target_mode,
    }


def build_training_arrays(
    db_path: Path,
    cfg: DatasetConfig,
    target_mode: str = "absolute",
) -> Dict[str, np.ndarray | Dict | List[str]]:
    target_mode = _resolve_target_mode(target_mode)
    frame = _build_training_frame(db_path, cfg)
    feature_cols = ["forecast_avg", "forecast_max", "forecast_dir", "month_sin", "month_cos"]
    target_col = "actual_avg"
    X_raw, y_actual_raw, y_forecast_raw, timestamps = _build_samples(frame, cfg, feature_cols, target_col)
    if target_mode == "residual":
        y_target_raw = y_actual_raw - y_forecast_raw
    else:
        y_target_raw = y_actual_raw

    split_idx = int(len(X_raw) * 0.8)
    split_idx = max(split_idx, 1)
    split_idx = min(split_idx, len(X_raw) - 1)

    X_train_raw, X_val_raw = X_raw[:split_idx], X_raw[split_idx:]
    y_train_target_raw, y_val_target_raw = y_target_raw[:split_idx], y_target_raw[split_idx:]
    y_train_actual_raw, y_val_actual_raw = y_actual_raw[:split_idx], y_actual_raw[split_idx:]
    y_train_forecast_raw, y_val_forecast_raw = y_forecast_raw[:split_idx], y_forecast_raw[split_idx:]

    x_mean, x_std = _fit_standardizer(X_train_raw)
    y_mean, y_std = _fit_target_scaler(y_train_target_raw)

    X_train = _apply_standardizer(X_train_raw, x_mean, x_std)
    X_val = _apply_standardizer(X_val_raw, x_mean, x_std)
    y_train = _apply_target_scaler(y_train_target_raw, y_mean, y_std)
    y_val = _apply_target_scaler(y_val_target_raw, y_mean, y_std)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "y_actual_train_raw": y_train_actual_raw,
        "y_actual_val_raw": y_val_actual_raw,
        "y_forecast_train_raw": y_train_forecast_raw,
        "y_forecast_val_raw": y_val_forecast_raw,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": np.array([y_mean], dtype=np.float32),
        "y_std": np.array([y_std], dtype=np.float32),
        "timestamps": timestamps,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "target_mode": target_mode,
    }


def build_all_direction_training_arrays(
    db_path: Path,
    cfg: DatasetConfig,
) -> Dict[str, np.ndarray | List[str]]:
    frame = _build_training_frame(db_path, cfg)
    feature_cols = ["forecast_avg", "forecast_max", "forecast_dir", "month_sin", "month_cos"]

    feature_values = frame[feature_cols].to_numpy(dtype=np.float32)
    actual_dir_values = frame["actual_dir"].to_numpy(dtype=np.float32)
    forecast_dir_values = frame["forecast_dir"].to_numpy(dtype=np.float32)

    X_list: List[np.ndarray] = []
    y_target_list: List[np.ndarray] = []
    y_actual_list: List[np.ndarray] = []
    y_forecast_list: List[np.ndarray] = []
    timestamps: List[str] = []

    window = cfg.window_hours
    horizon = cfg.target_hours
    total = len(frame)

    for i in range(window - 1, total - horizon):
        x_window = feature_values[i - window + 1 : i + 1]
        y_actual = actual_dir_values[i + 1 : i + 1 + horizon]
        y_forecast = forecast_dir_values[i + 1 : i + 1 + horizon]
        y_target = _angle_diff_deg(y_actual, y_forecast)

        if np.isnan(x_window).any() or np.isnan(y_target).any() or np.isnan(y_actual).any() or np.isnan(y_forecast).any():
            continue

        X_list.append(x_window)
        y_target_list.append(y_target)
        y_actual_list.append(y_actual)
        y_forecast_list.append(y_forecast)
        timestamps.append(frame.index[i].isoformat())

    if not X_list:
        raise ValueError("No direction training samples could be built with current window/horizon settings.")

    X_raw = np.stack(X_list).astype(np.float32)
    y_target_raw = np.stack(y_target_list).astype(np.float32)
    y_actual_raw = np.stack(y_actual_list).astype(np.float32)
    y_forecast_raw = np.stack(y_forecast_list).astype(np.float32)

    x_mean, x_std = _fit_standardizer(X_raw)
    y_mean, y_std = _fit_target_scaler(y_target_raw)
    X_scaled = _apply_standardizer(X_raw, x_mean, x_std)
    y_scaled = _apply_target_scaler(y_target_raw, y_mean, y_std)

    return {
        "X_all": X_scaled,
        "y_all": y_scaled,
        "y_actual_all_raw": y_actual_raw,
        "y_forecast_all_raw": y_forecast_raw,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": np.array([y_mean], dtype=np.float32),
        "y_std": np.array([y_std], dtype=np.float32),
        "timestamps": np.array(timestamps),
        "feature_cols": feature_cols,
        "target_col": "actual_dir",
        "target_mode": "residual",
    }


def build_next_day_inference_input(
    db_path: Path,
    cfg: DatasetConfig,
    x_mean: np.ndarray,
    x_std: np.ndarray,
) -> Dict[str, np.ndarray | str]:
    frame = _build_forecast_feature_frame(db_path, cfg)
    feature_cols = ["forecast_avg", "forecast_max", "forecast_dir", "month_sin", "month_cos"]
    conn = sqlite3.connect(str(db_path))
    try:
        latest_obs = _latest_observation_time(conn, cfg.site)
    finally:
        conn.close()

    # Predict the next calendar day (UTC) after the most recent observation day.
    reference_day_start = latest_obs.floor("D")
    target_day_start = reference_day_start + pd.Timedelta(days=1)
    target_times = pd.date_range(
        start=target_day_start,
        periods=cfg.target_hours,
        freq="1h",
        tz="UTC",
    )
    anchor_time = target_day_start - pd.Timedelta(hours=1)
    history_times = pd.date_range(
        end=anchor_time,
        periods=cfg.window_hours,
        freq="1h",
        tz="UTC",
    )

    history_frame = frame.reindex(history_times)
    target_frame = frame.reindex(target_times)
    if history_frame[feature_cols].isna().any().any():
        raise ValueError("Missing forecast rows in history window for next-day inference.")
    if target_frame["forecast_avg"].isna().any():
        raise ValueError("Missing forecast rows in target next-day window.")

    x_window = history_frame[feature_cols].to_numpy(dtype=np.float32)
    forecast_next = target_frame["forecast_avg"].to_numpy(dtype=np.float32)
    forecast_dir_next = target_frame["forecast_dir"].to_numpy(dtype=np.float32)

    X_input = _apply_standardizer(x_window[np.newaxis, :, :], x_mean, x_std).astype(np.float32)
    return {
        "X_input": X_input,
        "forecast_next24": forecast_next,
        "forecast_dir_next24": forecast_dir_next,
        "target_times": np.array([t.isoformat() for t in target_times]),
        "anchor_time": anchor_time.isoformat(),
        "reference_observation_time": latest_obs.isoformat(),
        "prediction_day_start": target_day_start.isoformat(),
    }
