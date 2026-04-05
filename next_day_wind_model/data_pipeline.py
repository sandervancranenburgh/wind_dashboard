from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
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


def _forecast_value(payload: Dict, db_value, keys: Sequence[str]) -> float | None:
    value = _extract_first(payload, keys)
    if value is not None:
        return value
    return _to_float(db_value)


def load_forecast_vintages(conn: sqlite3.Connection, site: str, model: str) -> pd.DataFrame:
    """
    Load immutable forecast vintages from SQLite.

    run_ts is the forecast/model vintage timestamp.
    fetched_ts is when our collector actually saw that payload.
    Preserving both matters because fair historical evaluation must compare
    against the specific Harmonie vintage that was available at the time.
    """
    query = """
    SELECT run_ts, fetched_ts, target_ts, horizon_hr, wind_speed, wind_gust, wind_dir, payload
    FROM forecasts
    WHERE site = ?
      AND model = ?
      AND target_ts IS NOT NULL
    ORDER BY run_ts ASC, target_ts ASC
    """
    rows = conn.execute(query, (site, model)).fetchall()
    if not rows:
        raise ValueError("No forecast rows found for selected site/model.")

    records: List[Dict] = []
    for run_ts, fetched_ts, target_ts, horizon_hr, wind_speed, wind_gust, wind_dir, payload_raw in rows:
        payload = json.loads(payload_raw) if payload_raw else {}
        records.append(
            {
                "run_ts": int(run_ts),
                "fetched_ts": int(fetched_ts),
                "target_ts": int(target_ts),
                "horizon_hr": None if horizon_hr is None else int(horizon_hr),
                "forecast_avg": _forecast_value(
                    payload,
                    wind_speed,
                    ["WindForecastAvr", "wind_speed", "windspeed", "WS", "ff", "speed"],
                ),
                "forecast_min": _forecast_value(
                    payload,
                    None,
                    ["WindForecastMin", "wind_min", "windspeed_min", "WS_min", "ff_min", "speed_min"],
                ),
                "forecast_max": _forecast_value(
                    payload,
                    wind_gust,
                    ["WindForecastMax", "wind_gust", "gust", "WG", "fg"],
                ),
                "forecast_dir": _forecast_value(
                    payload,
                    wind_dir,
                    ["WindDirection", "wind_dir", "winddirection", "WD", "DD", "dir", "direction"],
                ),
            }
        )

    forecast_df = pd.DataFrame.from_records(records)
    forecast_df["run_dt"] = pd.to_datetime(forecast_df["run_ts"], unit="ms", utc=True)
    forecast_df["fetched_dt"] = pd.to_datetime(forecast_df["fetched_ts"], unit="ms", utc=True)
    forecast_df["target_dt"] = pd.to_datetime(forecast_df["target_ts"], unit="ms", utc=True)
    return forecast_df.sort_values(["run_dt", "target_dt"]).reset_index(drop=True)


def _collapse_latest_forecast_view(forecast_vintages: pd.DataFrame) -> pd.DataFrame:
    if forecast_vintages.empty:
        raise ValueError("No forecast vintages found for selected site/model.")

    latest_df = forecast_vintages.sort_values(
        ["target_ts", "run_ts", "fetched_ts"],
        ascending=[True, False, False],
    )
    latest_df = latest_df.drop_duplicates(subset=["target_ts"], keep="first")
    latest_df = latest_df.set_index("target_dt").sort_index()
    return latest_df[
        [
            "run_ts",
            "fetched_ts",
            "horizon_hr",
            "forecast_avg",
            "forecast_min",
            "forecast_max",
            "forecast_dir",
        ]
    ]


def load_latest_forecast_view(conn: sqlite3.Connection, site: str, model: str) -> pd.DataFrame:
    """
    Return a derived latest-per-target convenience view.

    This collapse is useful for compatibility helpers, but it is not the
    canonical forecast history. The canonical representation is the full set of
    forecast vintages returned by load_forecast_vintages().
    """
    forecast_vintages = load_forecast_vintages(conn, site, model)
    return _collapse_latest_forecast_view(forecast_vintages)


def _load_forecasts(conn: sqlite3.Connection, site: str, model: str) -> pd.DataFrame:
    return load_latest_forecast_view(conn, site, model)


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
    # Training still uses a compatibility collapse, but it now starts from the
    # full vintage-preserving forecast history rather than treating latest-per-
    # target rows as the canonical store.
    forecast_vintages = load_forecast_vintages(conn, cfg.site, cfg.model)
    fc = _collapse_latest_forecast_view(forecast_vintages)
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
    feature_cols = ["forecast_avg", "forecast_min", "forecast_max", "forecast_dir", "actual_max", "actual_dir"]
    for col in feature_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
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
        frame = load_latest_forecast_view(conn, cfg.site, cfg.model)
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


@lru_cache(maxsize=8)
def _load_vintage_lookup_bundle(db_path_str: str, site: str, model: str) -> Dict[str, object]:
    conn = sqlite3.connect(db_path_str)
    try:
        forecast_vintages = load_forecast_vintages(conn, site, model)
    finally:
        conn.close()

    forecast_vintages = forecast_vintages.sort_values(["run_ts", "fetched_ts", "target_ts"]).reset_index(drop=True)
    target_lookup: Dict[int, Dict[str, np.ndarray]] = {}
    by_target = forecast_vintages.sort_values(["target_ts", "run_ts", "fetched_ts"])
    for target_ts, group in by_target.groupby("target_ts", sort=False):
        target_lookup[int(target_ts)] = {
            "fetched_ts": group["fetched_ts"].astype(np.int64).to_numpy(),
            "run_ts": group["run_ts"].astype(np.int64).to_numpy(),
            "horizon_hr": pd.to_numeric(group["horizon_hr"], errors="coerce").to_numpy(dtype=np.float32),
            "forecast_avg": pd.to_numeric(group["forecast_avg"], errors="coerce").to_numpy(dtype=np.float32),
            "forecast_min": pd.to_numeric(group["forecast_min"], errors="coerce").to_numpy(dtype=np.float32),
            "forecast_max": pd.to_numeric(group["forecast_max"], errors="coerce").to_numpy(dtype=np.float32),
            "forecast_dir": pd.to_numeric(group["forecast_dir"], errors="coerce").to_numpy(dtype=np.float32),
        }

    run_entries: List[Dict[str, object]] = []
    by_run = forecast_vintages.sort_values(["run_ts", "target_ts"])
    for run_ts, group in by_run.groupby("run_ts", sort=False):
        target_values = group["target_ts"].astype(np.int64).to_numpy()
        run_entries.append(
            {
                "run_ts": int(run_ts),
                "available_ts": int(group["fetched_ts"].max()),
                "row_fetched_ts": group["fetched_ts"].astype(np.int64).to_numpy(),
                "target_index": {int(ts): idx for idx, ts in enumerate(target_values.tolist())},
                "forecast_avg": pd.to_numeric(group["forecast_avg"], errors="coerce").to_numpy(dtype=np.float32),
                "forecast_min": pd.to_numeric(group["forecast_min"], errors="coerce").to_numpy(dtype=np.float32),
                "forecast_max": pd.to_numeric(group["forecast_max"], errors="coerce").to_numpy(dtype=np.float32),
                "forecast_dir": pd.to_numeric(group["forecast_dir"], errors="coerce").to_numpy(dtype=np.float32),
                "horizon_hr": pd.to_numeric(group["horizon_hr"], errors="coerce").to_numpy(dtype=np.float32),
            }
        )
    run_entries.sort(key=lambda entry: (int(entry["run_ts"]), int(entry["available_ts"])))
    run_available_ts = np.asarray([int(entry["available_ts"]) for entry in run_entries], dtype=np.int64)
    return {
        "target_lookup": target_lookup,
        "run_entries": run_entries,
        "run_available_ts": run_available_ts,
    }


def _target_ms(ts: pd.Timestamp) -> int:
    return int(ts.value // 1_000_000)


def _lookup_latest_target_as_of(
    target_lookup: Dict[int, Dict[str, np.ndarray]],
    target_ts_ms: int,
    anchor_ts_ms: int,
) -> Dict[str, float | int] | None:
    """
    Return the latest forecast row for a single target that was available at anchor_ts_ms.

    anchor_ts_ms is the historical issue time we pretend to stand in.
    run_ts is the forecast vintage timestamp.
    fetched_ts is when our collector first saw that specific forecast row.

    Fairness rule: only rows with fetched_ts <= anchor_ts_ms are eligible, and
    among eligible rows we choose the latest forecast vintage (highest run_ts,
    then highest fetched_ts as a tie-break).
    """
    series = target_lookup.get(int(target_ts_ms))
    if series is None:
        return None
    eligible = series["fetched_ts"] <= int(anchor_ts_ms)
    if not np.any(eligible):
        return None
    eligible_idx = np.flatnonzero(eligible)
    order = np.lexsort(
        (
            series["fetched_ts"][eligible_idx],
            series["run_ts"][eligible_idx],
        )
    )
    pos = int(eligible_idx[order[-1]])
    return {
        "run_ts": int(series["run_ts"][pos]),
        "fetched_ts": int(series["fetched_ts"][pos]),
        "horizon_hr": float(series["horizon_hr"][pos]),
        "forecast_avg": float(series["forecast_avg"][pos]),
        "forecast_min": float(series["forecast_min"][pos]),
        "forecast_max": float(series["forecast_max"][pos]),
        "forecast_dir": float(series["forecast_dir"][pos]),
    }


def _build_history_forecast_frame(
    target_lookup: Dict[int, Dict[str, np.ndarray]],
    history_times: pd.DatetimeIndex,
    anchor_ts_ms: int,
) -> pd.DataFrame | None:
    if len(history_times) == 0:
        return pd.DataFrame(index=history_times)
    records: List[Dict[str, float | int | pd.Timestamp]] = []
    for target_time in history_times:
        row = _lookup_latest_target_as_of(target_lookup, _target_ms(target_time), anchor_ts_ms)
        if row is None:
            return None
        records.append({"target_dt": target_time, **row})

    history_frame = pd.DataFrame.from_records(records).set_index("target_dt").sort_index()
    history_frame = _interpolate_missing_features(history_frame)
    history_frame = _add_calendar_features(history_frame)
    required_cols = ["forecast_avg", "forecast_max", "forecast_dir", "month_sin", "month_cos"]
    if history_frame[required_cols].isna().any().any():
        return None
    return history_frame


def _select_latest_complete_run_frame(
    run_entries: List[Dict[str, object]],
    target_times: pd.DatetimeIndex,
    anchor_ts_ms: int,
) -> pd.DataFrame | None:
    """
    Select the latest complete Harmonie run that was actually available at anchor_ts_ms.

    A run is eligible only when every requested target timestamp exists in that
    run and each corresponding row satisfies fetched_ts <= anchor_ts_ms. This is
    the key leakage-prevention rule for fair historical training and evaluation.
    """
    target_mss = [_target_ms(ts) for ts in target_times]
    for entry in reversed(run_entries):
        if int(entry["available_ts"]) > int(anchor_ts_ms):
            continue
        index_map = entry["target_index"]
        row_fetched_ts = np.asarray(entry["row_fetched_ts"], dtype=np.int64)
        indices: List[int] = []
        complete = True
        for target_ts_ms in target_mss:
            idx = index_map.get(int(target_ts_ms))
            if idx is None:
                complete = False
                break
            if int(row_fetched_ts[int(idx)]) > int(anchor_ts_ms):
                complete = False
                break
            indices.append(int(idx))
        if complete:
            target_frame = pd.DataFrame(
                {
                    "run_ts": np.full(len(target_times), int(entry["run_ts"]), dtype=np.int64),
                    "fetched_ts": row_fetched_ts[indices],
                    "horizon_hr": np.asarray(entry["horizon_hr"], dtype=np.float32)[indices],
                    "forecast_avg": np.asarray(entry["forecast_avg"], dtype=np.float32)[indices],
                    "forecast_min": np.asarray(entry["forecast_min"], dtype=np.float32)[indices],
                    "forecast_max": np.asarray(entry["forecast_max"], dtype=np.float32)[indices],
                    "forecast_dir": np.asarray(entry["forecast_dir"], dtype=np.float32)[indices],
                },
                index=target_times,
            )
            target_frame.index.name = "target_dt"
            if target_frame[["forecast_avg", "forecast_max", "forecast_dir"]].isna().any().any():
                continue
            return target_frame
    return None


def build_anchor_forecast_context(
    db_path: Path,
    cfg: DatasetConfig,
    anchor_time: pd.Timestamp,
    history_times: pd.DatetimeIndex,
    target_times: pd.DatetimeIndex,
) -> Dict[str, object]:
    """
    Build the forecast context that would have been knowable at anchor_time.

    anchor_time:
        Historical issue time for the sample or evaluation point.
    run_ts:
        Timestamp of the forecast vintage / model run.
    fetched_ts:
        First-seen availability time recorded by our collector.

    Fairness rule:
        Only forecast rows with fetched_ts <= anchor_time are usable.
        Historical feature timestamps use the latest available row per target as
        of anchor_time, while future target timestamps use the latest complete
        run available at anchor_time.
    """
    anchor_time_utc = pd.Timestamp(anchor_time)
    if anchor_time_utc.tzinfo is None:
        anchor_time_utc = anchor_time_utc.tz_localize("UTC")
    else:
        anchor_time_utc = anchor_time_utc.tz_convert("UTC")

    history_times_utc = pd.to_datetime(history_times, utc=True)
    target_times_utc = pd.to_datetime(target_times, utc=True)
    bundle = _load_vintage_lookup_bundle(str(db_path), cfg.site, cfg.model)
    anchor_ts_ms = _target_ms(anchor_time_utc)

    history_frame = (
        pd.DataFrame(index=history_times_utc)
        if len(history_times_utc) == 0
        else _build_history_forecast_frame(bundle["target_lookup"], history_times_utc, anchor_ts_ms)
    )
    target_frame = (
        pd.DataFrame(index=target_times_utc)
        if len(target_times_utc) == 0
        else _select_latest_complete_run_frame(bundle["run_entries"], target_times_utc, anchor_ts_ms)
    )
    return {
        "anchor_time": anchor_time_utc,
        "history_frame": history_frame,
        "target_frame": target_frame,
    }


def build_anchor_forecast_timeline(
    db_path: Path,
    cfg: DatasetConfig,
    anchor_time: pd.Timestamp,
    timeline: pd.DatetimeIndex,
) -> pd.DataFrame | None:
    timeline_utc = pd.to_datetime(timeline, utc=True)
    if len(timeline_utc) == 0:
        return pd.DataFrame()

    anchor_time_utc = pd.Timestamp(anchor_time)
    if anchor_time_utc.tzinfo is None:
        anchor_time_utc = anchor_time_utc.tz_localize("UTC")
    else:
        anchor_time_utc = anchor_time_utc.tz_convert("UTC")

    history_times = timeline_utc[timeline_utc <= anchor_time_utc]
    future_times = timeline_utc[timeline_utc > anchor_time_utc]
    context = build_anchor_forecast_context(
        db_path=db_path,
        cfg=cfg,
        anchor_time=anchor_time_utc,
        history_times=history_times,
        target_times=future_times,
    )

    frames: List[pd.DataFrame] = []
    if len(history_times) > 0:
        history_frame = context["history_frame"]
        if history_frame is None:
            return None
        frames.append(history_frame.reindex(history_times))
    if len(future_times) > 0:
        target_frame = context["target_frame"]
        if target_frame is None:
            return None
        frames.append(target_frame.reindex(future_times))
    if not frames:
        return pd.DataFrame(index=timeline_utc)

    frame = pd.concat(frames).sort_index().reindex(timeline_utc)
    frame = _interpolate_missing_features(frame)
    frame = _add_calendar_features(frame)
    return frame


def _build_vintage_aware_samples(
    db_path: Path,
    cfg: DatasetConfig,
    actual_col: str,
    forecast_target_col: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bundle = _load_vintage_lookup_bundle(str(db_path), cfg.site, cfg.model)
    conn = sqlite3.connect(str(db_path))
    try:
        obs = _load_observations(conn, cfg.site)
    finally:
        conn.close()

    feature_cols = ["forecast_avg", "forecast_max", "forecast_dir", "month_sin", "month_cos"]
    X_list: List[np.ndarray] = []
    y_actual_list: List[np.ndarray] = []
    y_forecast_list: List[np.ndarray] = []
    timestamps: List[str] = []

    window = cfg.window_hours
    horizon = cfg.target_hours
    total = len(obs)
    target_lookup = bundle["target_lookup"]
    run_entries = bundle["run_entries"]
    # Forecasts are selected as they were actually available at each anchor time.
    # Historical features use the latest forecast known by that anchor, while the
    # future Harmonie baseline comes from the latest complete run available then.
    for i in range(window - 1, total - horizon):
        anchor_time = obs.index[i]
        history_times = obs.index[i - window + 1 : i + 1]
        target_times = obs.index[i + 1 : i + 1 + horizon]
        anchor_ts_ms = _target_ms(anchor_time)

        history_frame = _build_history_forecast_frame(target_lookup, history_times, anchor_ts_ms)
        if history_frame is None:
            continue
        target_frame = _select_latest_complete_run_frame(run_entries, target_times, anchor_ts_ms)
        if target_frame is None:
            continue

        actual_next = obs.iloc[i + 1 : i + 1 + horizon][actual_col].to_numpy(dtype=np.float32)
        forecast_next = target_frame[forecast_target_col].to_numpy(dtype=np.float32)
        x_window = history_frame[feature_cols].to_numpy(dtype=np.float32)

        if np.isnan(x_window).any() or np.isnan(actual_next).any() or np.isnan(forecast_next).any():
            continue

        X_list.append(x_window)
        y_actual_list.append(actual_next)
        y_forecast_list.append(forecast_next)
        timestamps.append(anchor_time.isoformat())

    if not X_list:
        raise ValueError("No vintage-aware training samples could be built with the current window/horizon settings.")

    X_raw = np.stack(X_list).astype(np.float32)
    y_actual_raw = np.stack(y_actual_list).astype(np.float32)
    y_forecast_raw = np.stack(y_forecast_list).astype(np.float32)
    return X_raw, y_actual_raw, y_forecast_raw, np.array(timestamps)


def build_all_training_arrays(
    db_path: Path,
    cfg: DatasetConfig,
    target_mode: str = "absolute",
) -> Dict[str, np.ndarray | List[str]]:
    target_mode = _resolve_target_mode(target_mode)
    feature_cols = ["forecast_avg", "forecast_max", "forecast_dir", "month_sin", "month_cos"]
    target_col = "actual_avg"
    X_raw, y_actual_raw, y_forecast_raw, timestamps = _build_vintage_aware_samples(
        db_path,
        cfg,
        actual_col=target_col,
        forecast_target_col="forecast_avg",
    )
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
    feature_cols = ["forecast_avg", "forecast_max", "forecast_dir", "month_sin", "month_cos"]
    target_col = "actual_avg"
    X_raw, y_actual_raw, y_forecast_raw, timestamps = _build_vintage_aware_samples(
        db_path,
        cfg,
        actual_col=target_col,
        forecast_target_col="forecast_avg",
    )
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
    feature_cols = ["forecast_avg", "forecast_max", "forecast_dir", "month_sin", "month_cos"]
    X_raw, y_actual_raw, y_forecast_raw, timestamps = _build_vintage_aware_samples(
        db_path,
        cfg,
        actual_col="actual_dir",
        forecast_target_col="forecast_dir",
    )
    y_target_raw = _angle_diff_deg(y_actual_raw, y_forecast_raw)

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
        "target_col": "actual_dir",
        "target_mode": "residual",
    }


def build_next_day_inference_input(
    db_path: Path,
    cfg: DatasetConfig,
    x_mean: np.ndarray,
    x_std: np.ndarray,
) -> Dict[str, np.ndarray | str]:
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
    anchor_ts_ms = _target_ms(anchor_time)

    context = build_anchor_forecast_context(
        db_path=db_path,
        cfg=cfg,
        anchor_time=anchor_time,
        history_times=history_times,
        target_times=target_times,
    )
    history_frame = context["history_frame"]
    target_frame = context["target_frame"]
    if history_frame is None or history_frame[feature_cols].isna().any().any():
        raise ValueError("Missing anchor-time forecast history for next-day inference.")
    if target_frame is None or target_frame[["forecast_avg", "forecast_max", "forecast_dir"]].isna().any().any():
        raise ValueError("Missing anchor-time forecast run for next-day inference target window.")

    x_window = history_frame[feature_cols].to_numpy(dtype=np.float32)
    forecast_next = target_frame["forecast_avg"].to_numpy(dtype=np.float32)
    forecast_min_next = target_frame["forecast_min"].to_numpy(dtype=np.float32)
    forecast_max_next = target_frame["forecast_max"].to_numpy(dtype=np.float32)
    forecast_dir_next = target_frame["forecast_dir"].to_numpy(dtype=np.float32)

    X_input = _apply_standardizer(x_window[np.newaxis, :, :], x_mean, x_std).astype(np.float32)
    return {
        "X_input": X_input,
        "forecast_next24": forecast_next,
        "forecast_min_next24": forecast_min_next,
        "forecast_max_next24": forecast_max_next,
        "forecast_dir_next24": forecast_dir_next,
        "target_run_ts": target_frame["run_ts"].to_numpy(dtype=np.int64),
        "target_fetched_ts": target_frame["fetched_ts"].to_numpy(dtype=np.int64),
        "target_horizon_hr": target_frame["horizon_hr"].to_numpy(dtype=np.float32),
        "anchor_forecast_dir": np.float32(history_frame["forecast_dir"].iloc[-1]),
        "target_times": np.array([t.isoformat() for t in target_times]),
        "anchor_time": anchor_time.isoformat(),
        "reference_observation_time": latest_obs.isoformat(),
        "prediction_day_start": target_day_start.isoformat(),
    }
