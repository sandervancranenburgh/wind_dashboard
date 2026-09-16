"""Bounded operational ECMWF selection for dashboard plotting."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


ECMWF_FORECAST_COLOR = "#007f7f"
ECMWF_FORECAST_LINEWIDTH = 1.9
ECMWF_CYCLE_HOURS = 6
ECMWF_ARRIVAL_SAMPLE_RUNS = 12
ECMWF_ARRIVAL_MIN_RUNS = 4
MPS_TO_KNOT = 1.943844492


def _as_utc(value: datetime | pd.Timestamp | str) -> datetime:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")
    return parsed.to_pydatetime()


def _utc_iso(value: datetime | pd.Timestamp | str) -> str:
    return _as_utc(value).isoformat().replace("+00:00", "Z")


def _format_compact_local_time(value: datetime, timezone_name: str) -> str:
    local = value.astimezone(ZoneInfo(timezone_name))
    now_local = datetime.now(ZoneInfo(timezone_name))
    if local.date() == now_local.date():
        return local.strftime("%H:%M")
    return f"{local.day} {local.strftime('%B %H:%M')}"


def _empty_metadata(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
        "metadata_line": None,
        "run_identity": None,
        "arrival_estimate_method": "unavailable",
        "arrival_sample_count": 0,
        "arrival_latency_hours": [],
    }


def _open_read_only(path: Path) -> sqlite3.Connection:
    resolved = path.expanduser().resolve()
    connection = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA busy_timeout=5000")
    return connection


def load_ecmwf_plot_data(
    archive_db: Path,
    *,
    site: str,
    cutoff_utc: datetime,
    start_utc: datetime,
    end_utc: datetime,
    local_timezone: str,
    fallback_availability_delay_hours: float = 8.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load only one selected run and bounded points needed by both plots.

    The selected run must have completed no later than cutoff_utc. Recent
    completion latencies are loaded as scalar timestamps only; forecast points
    are queried exclusively for the selected run and requested time interval.
    """

    empty = pd.DataFrame(columns=["time_utc", "time_local", "wind_speed_knots"])
    path = archive_db.expanduser()
    if not path.is_file():
        return empty, _empty_metadata("archive database unavailable")

    cutoff = _as_utc(cutoff_utc)
    start = _as_utc(start_utc)
    end = _as_utc(end_utc)
    if end <= start:
        raise ValueError("ECMWF plot interval must end after it starts")

    try:
        connection = _open_read_only(path)
    except sqlite3.Error as exc:
        return empty, _empty_metadata(f"archive database unavailable: {type(exc).__name__}")
    try:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if not {"forecast_collection_runs", "forecast_points"}.issubset(tables):
            return empty, _empty_metadata("archive tables unavailable")

        selected = connection.execute(
            """
            SELECT run_time, completed_time
            FROM forecast_collection_runs
            WHERE provider='ECMWF' AND model='IFS' AND site=? AND status='complete'
              AND completed_time IS NOT NULL
              AND julianday(run_time) <= julianday(?)
              AND julianday(completed_time) <= julianday(?)
            ORDER BY julianday(run_time) DESC
            LIMIT 1
            """,
            (site, _utc_iso(cutoff), _utc_iso(cutoff)),
        ).fetchone()
        if selected is None:
            return empty, _empty_metadata("no complete ECMWF run before cutoff")
        run_time = _as_utc(selected[0])
        completed_time = _as_utc(selected[1])

        points = pd.read_sql_query(
            """
            SELECT valid_time, wind_speed_mps, fetched_time, grid_latitude,
                   grid_longitude, model_version, resolution, source
            FROM forecast_points
            WHERE provider='ECMWF' AND model='IFS' AND site=? AND run_time=?
              AND julianday(valid_time) >= julianday(?)
              AND julianday(valid_time) <= julianday(?)
            ORDER BY julianday(valid_time)
            """,
            connection,
            params=(site, selected[0], _utc_iso(start), _utc_iso(end)),
        )
        arrival_rows = connection.execute(
            """
            SELECT run_time, completed_time
            FROM forecast_collection_runs
            WHERE provider='ECMWF' AND model='IFS' AND site=? AND status='complete'
              AND completed_time IS NOT NULL
              AND julianday(completed_time) <= julianday(?)
            ORDER BY julianday(completed_time) DESC
            LIMIT ?
            """,
            (site, _utc_iso(cutoff), ECMWF_ARRIVAL_SAMPLE_RUNS),
        ).fetchall()
    finally:
        connection.close()

    latencies = [
        (_as_utc(completed) - _as_utc(run)).total_seconds() / 3600.0
        for run, completed in arrival_rows
    ]
    latencies = [value for value in latencies if 0.0 <= value <= 48.0]
    if len(latencies) >= ECMWF_ARRIVAL_MIN_RUNS:
        estimated_latency = float(median(latencies))
        estimate_method = "median_recent_complete_run_latency"
    else:
        estimated_latency = float(fallback_availability_delay_hours)
        estimate_method = "configured_availability_delay_fallback"

    expected_run = run_time + timedelta(hours=ECMWF_CYCLE_HOURS)
    expected_fetch = expected_run + timedelta(hours=estimated_latency)
    while expected_fetch <= cutoff:
        expected_run += timedelta(hours=ECMWF_CYCLE_HOURS)
        expected_fetch = expected_run + timedelta(hours=estimated_latency)

    run_identity = f"{_utc_iso(run_time)}|{_utc_iso(completed_time)}"
    metadata: dict[str, Any] = {
        "available": not points.empty,
        "reason": None if not points.empty else "selected run has no points in plot interval",
        "run_identity": run_identity,
        "run_time_utc": _utc_iso(run_time),
        "completed_time_utc": _utc_iso(completed_time),
        "last_fetch_utc": _utc_iso(completed_time),
        "next_expected_run_utc": _utc_iso(expected_run),
        "next_expected_fetch_utc": _utc_iso(expected_fetch),
        "arrival_estimate_method": estimate_method,
        "arrival_sample_count": len(latencies),
        "arrival_latency_hours": [float(value) for value in latencies],
        "estimated_latency_hours": estimated_latency,
        "information_cutoff_utc": _utc_iso(cutoff),
        "database_open_mode": "read-only/query-only",
        "rows_loaded": int(len(points)),
    }
    metadata["metadata_line"] = (
        "Last ECMWF fetch: "
        f"{_format_compact_local_time(completed_time, local_timezone)} - "
        "Next expected fetch: ~"
        f"{_format_compact_local_time(expected_fetch, local_timezone)}"
    )
    if points.empty:
        return empty, metadata

    points["time_utc"] = pd.to_datetime(points["valid_time"], utc=True, errors="coerce")
    points["wind_speed_knots"] = (
        pd.to_numeric(points["wind_speed_mps"], errors="coerce") * MPS_TO_KNOT
    )
    points = points.dropna(subset=["time_utc", "wind_speed_knots"]).copy()
    points["time_local"] = points["time_utc"].dt.tz_convert(ZoneInfo(local_timezone))
    if points.empty:
        metadata["available"] = False
        metadata["reason"] = "selected run points are invalid"
        return empty, metadata

    first = points.iloc[0]
    metadata.update(
        {
            "available": True,
            "model_version": first["model_version"],
            "resolution": first["resolution"],
            "source": first["source"],
            "selected_grid": [
                float(first["grid_latitude"]),
                float(first["grid_longitude"]),
            ],
            "rows_loaded": int(len(points)),
        }
    )
    return points[["time_utc", "time_local", "wind_speed_knots"]].reset_index(drop=True), metadata
