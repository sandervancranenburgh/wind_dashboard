#!/usr/bin/env python3
"""Render weather-icon acceptance plots from read-only cached wind tables."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

os.environ.setdefault("WIND_PLOT_RENDERER_ONLY", "1")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from next_day_wind_model.update_model_and_predict import (
    save_current_day_plot,
    save_prediction_plot,
)
from next_day_wind_model.ecmwf_dashboard import load_ecmwf_plot_data
from next_day_wind_model.weather_conditions import (
    build_weather_timeline,
    derive_windsurfice_weather,
    weather_description,
)


def _payload_temperature(payload: dict) -> float:
    lower = {str(key).lower(): value for key, value in payload.items()}
    for key in ("temperature", "temp", "air_temperature"):
        try:
            return float(lower[key])
        except (KeyError, TypeError, ValueError):
            continue
    return float("nan")


def _windsurfice_pairs(
    db_path: Path, site: str, targets: pd.DatetimeIndex, available_at: pd.Timestamp
) -> tuple[np.ndarray, np.ndarray]:
    temperatures = np.full(len(targets), np.nan)
    codes = np.full(len(targets), np.nan)
    conn = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            """SELECT run_ts, fetched_ts, target_ts, weather_code, payload
               FROM forecasts WHERE site = ? AND target_ts >= ? AND target_ts <= ?
               AND fetched_ts <= ? ORDER BY target_ts, run_ts, fetched_ts""",
            (
                site,
                int(targets.min().timestamp() * 1000),
                int(targets.max().timestamp() * 1000),
                int(available_at.timestamp() * 1000),
            ),
        ).fetchall()
    finally:
        conn.close()
    latest: dict[int, tuple] = {}
    for row in rows:
        latest[int(row[2])] = row
    for index, target in enumerate(targets):
        row = latest.get(int(target.timestamp() * 1000))
        if row is None:
            continue
        try:
            payload = json.loads(row[4]) if row[4] else {}
        except json.JSONDecodeError:
            payload = {}
        temperatures[index] = _payload_temperature(payload)
        code = row[3]
        if code is None:
            code = derive_windsurfice_weather(payload)
        if code is not None:
            codes[index] = float(code)
    return temperatures, codes


def _weather_for_targets(
    db_path: Path, site: str, targets: pd.DatetimeIndex, available_at: pd.Timestamp
) -> pd.DataFrame:
    temperatures, codes = _windsurfice_pairs(db_path, site, targets, available_at)
    return build_weather_timeline(
        db_path,
        site=site,
        target_times=targets,
        fallback_temperature_c=temperatures,
        fallback_weather_code=codes,
        available_at=available_at,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--current-csv", type=Path, required=True)
    parser.add_argument("--next-csv", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument("--ecmwf-db", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--site", default="valkenburgsemeer")
    parser.add_argument("--timezone", default="Europe/Amsterdam")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(args.metadata_json.read_text(encoding="utf-8"))
    model_trained_at_utc = metadata.get("model_last_trained_at_utc")
    if not model_trained_at_utc:
        raise ValueError("Acceptance metadata must contain model_last_trained_at_utc")
    now = pd.Timestamp.now(tz="UTC")
    zone = ZoneInfo(args.timezone)
    current = pd.read_csv(args.current_csv)
    current["time_local"] = pd.to_datetime(current["time_local"], utc=True).dt.tz_convert(zone)
    current_hours = pd.DatetimeIndex(current["time_local"])
    exact_mask = (current_hours.minute == 0) & (current_hours.second == 0)
    exact_targets = current_hours[exact_mask].tz_convert("UTC")
    current_weather = _weather_for_targets(args.db, args.site, exact_targets, now)
    for column in current_weather.columns:
        current[column] = None if column in {"weather_source", "is_daylight"} else np.nan
        current.loc[exact_mask, column] = current_weather[column].to_numpy()

    next_day = pd.read_csv(args.next_csv)
    next_day["target_time_utc"] = pd.to_datetime(next_day["target_time_utc"], utc=True)
    next_day["target_time_local"] = pd.to_datetime(
        next_day["target_time_local"], utc=True
    ).dt.tz_convert(zone)
    next_targets = pd.DatetimeIndex(next_day["target_time_utc"])
    next_weather = _weather_for_targets(args.db, args.site, next_targets, now)
    for column in next_weather.columns:
        next_day[column] = next_weather[column].to_numpy()

    current_diagnostics: dict[str, object] = {}
    next_diagnostics: dict[str, object] = {}
    fallback_issue_local = (
        current["time_local"].iloc[0].normalize()
        + pd.Timedelta(hours=20, minutes=42)
    )
    fallback_timestamp = fallback_issue_local.tz_convert("UTC").isoformat()
    prediction_generated_at_utc = metadata.get(
        "prediction_generated_at_utc", fallback_timestamp
    )
    prediction_updated_at_utc = metadata.get(
        "prediction_updated_at_utc", prediction_generated_at_utc
    )
    plot_updated_at_utc = metadata.get("plot_updated_at_utc", fallback_timestamp)
    harmonie_time_utc = metadata.get("harmonie_fetched_at_utc", fallback_timestamp)
    harmonie_expected_next_at_utc = metadata.get("harmonie_expected_next_at_utc")

    plot_times_utc = pd.DatetimeIndex(
        list(pd.DatetimeIndex(current["time_local"]).tz_convert("UTC"))
        + list(pd.DatetimeIndex(next_day["target_time_utc"]))
    )
    ecmwf_speed_series, ecmwf_metadata = load_ecmwf_plot_data(
        args.ecmwf_db,
        site=args.site,
        cutoff_utc=pd.Timestamp(plot_updated_at_utc),
        start_utc=plot_times_utc.min().to_pydatetime(),
        end_utc=(plot_times_utc.max() + pd.Timedelta(hours=6)).to_pydatetime(),
        local_timezone=args.timezone,
    )
    if ecmwf_speed_series.empty:
        raise ValueError(
            f"Acceptance ECMWF overlay unavailable: {ecmwf_metadata.get('reason')}"
        )
    ecmwf_metadata_text = ecmwf_metadata.get("metadata_line")
    save_current_day_plot(
        current,
        args.out_dir / "current_day_predictions_weather.png",
        local_tz=args.timezone,
        prediction_generated_at_utc=prediction_generated_at_utc,
        prediction_updated_at_utc=prediction_updated_at_utc,
        model_trained_at_utc=model_trained_at_utc,
        harmonie_time_utc=harmonie_time_utc,
        harmonie_expected_next_at_utc=harmonie_expected_next_at_utc,
        plot_updated_at_utc=plot_updated_at_utc,
        prior_prediction_tables=[],
        live_monitoring_metric={"available": False},
        spot_name="Valkenburgse meer",
        ecmwf_speed_series=ecmwf_speed_series,
        ecmwf_metadata_text=ecmwf_metadata_text,
        render_diagnostics=current_diagnostics,
    )
    save_prediction_plot(
        next_day,
        args.out_dir / "next_day_predictions_weather.png",
        local_tz=args.timezone,
        plot_updated_at_utc=plot_updated_at_utc,
        prediction_updated_at_utc=prediction_updated_at_utc,
        model_trained_at_utc=model_trained_at_utc,
        harmonie_time_utc=harmonie_time_utc,
        harmonie_expected_next_at_utc=harmonie_expected_next_at_utc,
        spot_name="Valkenburgse meer",
        ecmwf_speed_series=ecmwf_speed_series,
        render_diagnostics=next_diagnostics,
    )
    save_current_day_plot(
        current,
        args.out_dir / "current_day_predictions_weather_mobile.png",
        local_tz=args.timezone,
        prediction_generated_at_utc=prediction_generated_at_utc,
        prediction_updated_at_utc=prediction_updated_at_utc,
        model_trained_at_utc=model_trained_at_utc,
        harmonie_time_utc=harmonie_time_utc,
        harmonie_expected_next_at_utc=harmonie_expected_next_at_utc,
        plot_updated_at_utc=plot_updated_at_utc,
        prior_prediction_tables=[],
        live_monitoring_metric={"available": False},
        mobile=True,
        spot_name="Valkenburgse meer",
        ecmwf_speed_series=ecmwf_speed_series,
        ecmwf_metadata_text=ecmwf_metadata_text,
    )
    save_prediction_plot(
        next_day,
        args.out_dir / "next_day_predictions_weather_mobile.png",
        local_tz=args.timezone,
        plot_updated_at_utc=plot_updated_at_utc,
        prediction_updated_at_utc=prediction_updated_at_utc,
        model_trained_at_utc=model_trained_at_utc,
        harmonie_time_utc=harmonie_time_utc,
        harmonie_expected_next_at_utc=harmonie_expected_next_at_utc,
        mobile=True,
        spot_name="Valkenburgse meer",
        ecmwf_speed_series=ecmwf_speed_series,
    )

    local_next = next_weather.copy()
    local_next.insert(0, "time", next_targets.tz_convert(zone).strftime("%H:%M"))
    local_next["description"] = [weather_description(value) for value in local_next["weather_code"]]
    compact = local_next[local_next["time"].between("08:00", "21:00")][
        [
            "time", "forecast_temperature_c", "total_cloud_cover_pct",
            "total_precip_hourly_mm", "snowfall_hourly_cm", "visibility_m",
            "weather_code", "description", "weather_source", "is_daylight",
        ]
    ]
    print(compact.to_string(index=False, na_rep="—", float_format=lambda value: f"{value:.2f}"))
    print(json.dumps({"current": current_diagnostics, "next": next_diagnostics}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
