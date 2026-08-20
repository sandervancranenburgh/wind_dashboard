from __future__ import annotations

import json
import math
import re
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from next_day_wind_model.operational_update import local_day_utc_bounds, write_bytes_if_changed


FORECAST_COLUMNS = (
    "forecast_wind_speed",
    "forecast_wind_min",
    "forecast_wind_max",
    "forecast_wind_dir_deg",
    "lstm_pred_wind_speed_full",
    "lstm_pred_wind_dir_deg_full",
    "lstm_pred_wind_speed",
    "lstm_pred_wind_dir_deg",
)


def _truthy_series(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def _payload_value(payload: dict[str, Any], *keys: str) -> Any:
    lower = {str(key).lower(): key for key in payload}
    for candidate in keys:
        key = lower.get(candidate.lower())
        if key is not None and payload.get(key) is not None:
            return payload.get(key)
    return None


def _as_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def load_current_day_observations(
    db_path: Path,
    *,
    site: str,
    local_timezone: str,
    now_utc: datetime | None = None,
) -> pd.DataFrame:
    """Load only one DST-aware local day of observations.

    The normalized DB columns supply average/max/direction. Payload JSON is
    parsed only for this bounded day so the existing MinWind presentation can
    be retained until that value is promoted to a normalized column.
    """
    start_ms, end_ms = local_day_utc_bounds(local_timezone, now_utc=now_utc)
    resolved = db_path.resolve()
    conn = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            """
            SELECT ts, wind_speed, wind_gust, wind_dir, payload
            FROM observations
            WHERE site = ? AND ts >= ? AND ts < ?
            ORDER BY ts
            """,
            (site, start_ms, end_ms),
        ).fetchall()
    finally:
        conn.close()

    records: list[dict[str, Any]] = []
    for ts, wind_speed, wind_gust, wind_dir, payload_raw in rows:
        payload: dict[str, Any] = {}
        if payload_raw:
            try:
                loaded = json.loads(payload_raw)
            except json.JSONDecodeError:
                loaded = {}
            if isinstance(loaded, dict):
                payload = loaded
        average = _payload_value(payload, "AverageWind", "WindSpeedAvg")
        minimum = _payload_value(payload, "MinWind", "WindSpeedMin")
        maximum = _payload_value(payload, "MaxWind", "WindSpeedMax")
        direction = _payload_value(payload, "WindDirection")
        records.append(
            {
                "obs_ts": int(ts),
                "actual_avg": _as_float(wind_speed if average is None else average),
                "actual_min": _as_float(minimum),
                "actual_max": _as_float(wind_gust if maximum is None else maximum),
                "actual_dir": _as_float(wind_dir if direction is None else direction),
            }
        )
    if not records:
        return pd.DataFrame(
            columns=["actual_avg", "actual_min", "actual_max", "actual_dir"],
            index=pd.DatetimeIndex([], tz=ZoneInfo(local_timezone)),
        )
    frame = pd.DataFrame.from_records(records)
    frame["obs_time"] = pd.to_datetime(frame["obs_ts"], unit="ms", utc=True).dt.tz_convert(
        ZoneInfo(local_timezone)
    )
    frame = frame.set_index("obs_time").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame[["actual_avg", "actual_min", "actual_max", "actual_dir"]]


def compose_current_day_table(
    cached_table: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    local_timezone: str,
    now_utc: datetime | None,
    build_plot_frame: Callable[..., pd.DataFrame],
) -> pd.DataFrame:
    frame = cached_table.copy()
    if "time_local" not in frame.columns:
        raise ValueError("cached current-day table lacks time_local")
    frame["time_local"] = pd.to_datetime(frame["time_local"], errors="coerce", utc=True).dt.tz_convert(
        ZoneInfo(local_timezone)
    )
    frame = frame.dropna(subset=["time_local"]).copy()
    if frame.empty:
        raise ValueError("cached current-day table has no usable rows")

    if "is_forecast_grid" in frame.columns:
        grid_mask = _truthy_series(frame["is_forecast_grid"])
    else:
        grid_mask = pd.to_numeric(frame.get("forecast_wind_speed"), errors="coerce").notna()
    grid = frame[grid_mask].copy().sort_values("time_local")
    grid = grid.drop_duplicates(subset=["time_local"], keep="last")
    if grid.empty:
        raise ValueError("cached current-day table has no forecast grid")
    missing = [column for column in FORECAST_COLUMNS if column not in grid.columns]
    if missing:
        raise ValueError(f"cached current-day table lacks columns: {', '.join(missing)}")

    utc_now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    local_now = utc_now.astimezone(ZoneInfo(local_timezone))
    now_hour = pd.Timestamp(local_now).floor("h")
    dense_times = pd.DatetimeIndex(grid["time_local"])
    forecast_values: dict[str, np.ndarray] = {}
    for column in FORECAST_COLUMNS:
        values = pd.to_numeric(grid[column], errors="coerce").to_numpy(dtype=np.float32)
        if column in {"lstm_pred_wind_speed", "lstm_pred_wind_dir_deg"}:
            values = values.copy()
            values[dense_times < now_hour] = np.nan
        forecast_values[column] = values

    return build_plot_frame(
        dense_times,
        forecast_values,
        observations,
        now_local=local_now,
        future_start=now_hour + pd.Timedelta(hours=1),
    )


def _metadata_value(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    return None if value is None else str(value)


def _table_csv_bytes(table: pd.DataFrame) -> bytes:
    serializable = table.copy()
    serializable["time_local"] = pd.to_datetime(serializable["time_local"]).dt.strftime(
        "%Y-%m-%dT%H:%M:%S%z"
    )
    return serializable.to_csv(index=False).encode("utf-8")


def _copy_if_changed(source: Path, destination: Path) -> bool:
    return write_bytes_if_changed(destination, source.read_bytes())


def _measured_dashboard_index_bytes(
    existing: bytes,
    *,
    generated_at_utc: str,
    local_timezone: str,
) -> bytes:
    """Advance only current-day cache tokens in an existing dashboard page."""
    generated = datetime.fromisoformat(generated_at_utc.replace("Z", "+00:00"))
    if generated.tzinfo is None:
        generated = generated.replace(tzinfo=timezone.utc)
    generated = generated.astimezone(timezone.utc)
    version = generated.isoformat()
    cache_token = str(int(generated.timestamp() * 1_000_000))
    local_label = generated.astimezone(ZoneInfo(local_timezone)).strftime(
        "%d %B %Y %H:%M:%S %Z"
    )
    text = existing.decode("utf-8")

    meta_pattern = re.compile(
        r'(<p\b[^>]*\bdata-dashboard-version=)"[^"]*"([^>]*>)Last updated:.*?(</p>)',
        re.DOTALL,
    )
    text, meta_count = meta_pattern.subn(
        lambda match: (
            f'{match.group(1)}"{version}"{match.group(2)}'
            f"Last updated: {local_label}{match.group(3)}"
        ),
        text,
        count=1,
    )
    current_asset_pattern = re.compile(
        r'(current_day_predictions(?:_mobile)?\.png\?v=)[^"\'&<>\s]+'
    )
    text, current_asset_count = current_asset_pattern.subn(
        lambda match: match.group(1) + cache_token,
        text,
    )
    current_json_pattern = re.compile(
        r'(current_day_interactive_data\.json\?v=)[^"\'&<>\s]+'
    )
    text, current_json_count = current_json_pattern.subn(
        lambda match: match.group(1) + cache_token,
        text,
    )
    text, version_count = re.subn(
        r'(currentVersion:\s*)"[^"]*"',
        lambda match: match.group(1) + json.dumps(version),
        text,
        count=1,
    )
    if meta_count != 1 or current_asset_count < 2 or current_json_count < 1 or version_count != 1:
        raise ValueError("dashboard index lacks expected current-day version markers")
    return text.encode("utf-8")


def _dashboard_index_version(existing: bytes) -> str | None:
    match = re.search(r'data-dashboard-version="([^"]+)"', existing.decode("utf-8"))
    if match is None:
        return None
    value = match.group(1).strip()
    return value or None


def run_measured_only_stage(
    *,
    args: Any,
    db_path: Path,
    out_dir: Path,
    build_plot_frame: Callable[..., pd.DataFrame],
    save_current_day_plot: Callable[..., None],
    load_prediction_history: Callable[..., list[pd.DataFrame]],
    write_interactive_assets: Callable[..., dict[str, str]],
    load_harmonie_metadata: Callable[..., tuple[Any, str]],
    auto_push: Callable[..., dict[str, Any]],
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    current_csv = out_dir / "current_day_predictions.csv"
    metadata_path = out_dir / "metadata_update.json"
    if not current_csv.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("measured-only mode requires cached current-day CSV and metadata")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("cached metadata must be a JSON object")
    cached_table = pd.read_csv(current_csv)
    observations = load_current_day_observations(
        db_path,
        site=args.site,
        local_timezone=args.local_timezone,
        now_utc=now_utc,
    )
    if observations.empty:
        raise ValueError("measured-only mode found no observations for the current local day")

    composed = compose_current_day_table(
        cached_table,
        observations,
        local_timezone=args.local_timezone,
        now_utc=now_utc,
        build_plot_frame=build_plot_frame,
    )
    current_csv_changed = write_bytes_if_changed(current_csv, _table_csv_bytes(composed))

    prediction_generated_at_utc = (
        _metadata_value(metadata, "prediction_generated_at_utc")
        or _metadata_value(metadata, "trained_at_utc")
        or datetime.now(timezone.utc).isoformat()
    )
    prediction_updated_at_utc = _metadata_value(metadata, "prediction_updated_at_utc")
    model_trained_at_utc = (
        _metadata_value(metadata, "model_last_trained_at_utc")
        or _metadata_value(metadata, "trained_at_utc")
    )
    harmonie_time_utc = _metadata_value(metadata, "harmonie_fetched_at_utc")
    harmonie_time_kind = _metadata_value(metadata, "harmonie_time_kind") or "fetched"
    if harmonie_time_utc is None:
        harmonie_time_utc, harmonie_time_kind = load_harmonie_metadata(db_path, args.site)
    plot_update_interval_minutes = int(getattr(args, "plot_update_interval_minutes", 6))
    harmonie_update_interval_minutes = int(getattr(args, "harmonie_update_interval_minutes", 60))
    plot_updated_at_utc = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    target_day = pd.to_datetime(composed["time_local"]).dt.tz_convert(
        ZoneInfo(args.local_timezone)
    ).iloc[0].date()
    prior_tables = load_prediction_history(
        out_dir=out_dir,
        target_day_local=target_day,
        local_tz=args.local_timezone,
        max_snapshots=16,
    )
    live_metric = metadata.get("current_day_live_monitoring_metric")
    if not isinstance(live_metric, dict):
        live_metric = None

    current_png = out_dir / "current_day_predictions.png"
    current_mobile_png = out_dir / "current_day_predictions_mobile.png"
    artifact_changes: dict[str, bool] = {"current_day_predictions.csv": current_csv_changed}
    with tempfile.TemporaryDirectory(prefix="wind-measured-render-") as temporary_dir:
        temporary = Path(temporary_dir)
        temporary_png = temporary / current_png.name
        temporary_mobile_png = temporary / current_mobile_png.name
        save_current_day_plot(
            composed,
            temporary_png,
            args.local_timezone,
            prediction_generated_at_utc=prediction_generated_at_utc,
            prediction_updated_at_utc=prediction_updated_at_utc,
            model_trained_at_utc=model_trained_at_utc,
            harmonie_time_utc=harmonie_time_utc,
            harmonie_time_kind=harmonie_time_kind,
            plot_updated_at_utc=plot_updated_at_utc,
            plot_update_interval_minutes=plot_update_interval_minutes,
            harmonie_update_interval_minutes=harmonie_update_interval_minutes,
            prior_prediction_tables=prior_tables,
            live_monitoring_metric=live_metric,
        )
        save_current_day_plot(
            composed,
            temporary_mobile_png,
            args.local_timezone,
            prediction_generated_at_utc=prediction_generated_at_utc,
            prediction_updated_at_utc=prediction_updated_at_utc,
            model_trained_at_utc=model_trained_at_utc,
            harmonie_time_utc=harmonie_time_utc,
            harmonie_time_kind=harmonie_time_kind,
            plot_updated_at_utc=plot_updated_at_utc,
            plot_update_interval_minutes=plot_update_interval_minutes,
            harmonie_update_interval_minutes=harmonie_update_interval_minutes,
            prior_prediction_tables=prior_tables,
            live_monitoring_metric=live_metric,
            mobile=True,
        )
        artifact_changes[current_png.name] = _copy_if_changed(temporary_png, current_png)
        artifact_changes[current_mobile_png.name] = _copy_if_changed(temporary_mobile_png, current_mobile_png)

        interactive_dir = temporary / "interactive"
        interactive_dir.mkdir()
        missing_next_csv = temporary / "missing_next_day.csv"
        interactive = write_interactive_assets(
            web_out_dir=interactive_dir,
            local_tz=args.local_timezone,
            current_day_csv=current_csv,
            next_day_csv=missing_next_csv,
            current_day_prior_prediction_tables=prior_tables,
            prediction_generated_at_utc=prediction_generated_at_utc,
            prediction_updated_at_utc=prediction_updated_at_utc,
            plot_updated_at_utc=plot_updated_at_utc,
            model_trained_at_utc=model_trained_at_utc,
            harmonie_time_utc=harmonie_time_utc,
            harmonie_time_kind=harmonie_time_kind,
            plot_update_interval_minutes=plot_update_interval_minutes,
            harmonie_update_interval_minutes=harmonie_update_interval_minutes,
        )

        web_out_dir = Path(args.web_out_dir)
        publish_sources = {
            "current_day_predictions.csv": current_csv,
            "current_day_predictions.png": current_png,
            "current_day_predictions_mobile.png": current_mobile_png,
        }
        if "current_day_json" in interactive:
            publish_sources["current_day_interactive_data.json"] = (
                interactive_dir / interactive["current_day_json"]
            )
        web_changes = {
            name: _copy_if_changed(source, web_out_dir / name)
            for name, source in publish_sources.items()
        }

    meaningful_web_change = any(web_changes.values())
    latest_observation_utc = observations.index.max().tz_convert("UTC").isoformat()
    web_metadata_path = Path(args.web_out_dir) / "metadata_update.json"
    try:
        web_metadata = json.loads(web_metadata_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        web_metadata = {}
    if not isinstance(web_metadata, dict):
        web_metadata = {}
    web_index_path = Path(args.web_out_dir) / "index.html"
    index_bytes = web_index_path.read_bytes()
    metadata_version = web_metadata.get("generated_at_utc")
    if not isinstance(metadata_version, str) or not metadata_version.strip():
        metadata_version = None
    metadata_refresh_needed = (
        meaningful_web_change
        or web_metadata.get("latest_observation_time_utc") != latest_observation_utc
        or metadata_version is None
    )
    index_refresh_needed = _dashboard_index_version(index_bytes) != metadata_version
    metadata_changed = False
    if metadata_refresh_needed or index_refresh_needed:
        if metadata_refresh_needed:
            generated_at = datetime.now(timezone.utc).isoformat()
            web_metadata.update(
                {
                    "static_plot_generated_at_utc": generated_at,
                    "plot_updated_at_utc": plot_updated_at_utc,
                    "generated_at_utc": generated_at,
                    "latest_observation_time_utc": latest_observation_utc,
                    "prediction_generated_at_utc": prediction_generated_at_utc,
                    "prediction_updated_at_utc": prediction_updated_at_utc,
                    "harmonie_fetched_at_utc": None if harmonie_time_utc is None else str(harmonie_time_utc),
                    "harmonie_time_kind": harmonie_time_kind,
                    "plot_update_interval_minutes": plot_update_interval_minutes,
                    "harmonie_update_interval_minutes": harmonie_update_interval_minutes,
                    "model_last_trained_at_utc": model_trained_at_utc,
                    "measured_only_update": True,
                }
            )
            metadata_changed = write_bytes_if_changed(
                web_metadata_path,
                (json.dumps(web_metadata, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            )
        else:
            generated_at = metadata_version
        web_changes["metadata_update.json"] = metadata_changed
        web_changes["index.html"] = write_bytes_if_changed(
            web_index_path,
            _measured_dashboard_index_bytes(
                index_bytes,
                generated_at_utc=generated_at,
                local_timezone=args.local_timezone,
            ),
        )

    git_publish: dict[str, Any] = {
        "enabled": bool(args.git_auto_push_pages),
        "pushed": False,
        "reason": "no_meaningful_web_change" if not any(web_changes.values()) else "disabled",
    }
    if args.git_auto_push_pages and any(web_changes.values()):
        repo_root = Path(__file__).resolve().parents[1]
        git_publish = auto_push(
            repo_root=repo_root,
            web_out_dir=Path(args.web_out_dir),
            remote=args.git_remote,
            branch=args.git_branch,
        )

    result = {
        "execution_mode": "measured_only",
        "observation_rows": int(len(observations)),
        "latest_observation_time_utc": latest_observation_utc,
        "artifact_changes": artifact_changes,
        "web_changes": web_changes,
        "git_publish": git_publish,
    }
    print("Measured-only current-day update complete.")
    print(json.dumps(result, sort_keys=True))
    return result
