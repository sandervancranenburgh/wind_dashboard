"""Read-only, issue-time-correct ECMWF wind-speed evaluation and coverage."""
from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from next_day_wind_model.ecmwf_dashboard import MPS_TO_KNOT

REPORT_VERSION = 1
SECTORS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def load_ecmwf_evaluation(archive_db: Path | None, *, site: str,
                          issue_times, target_times) -> pd.DataFrame:
    """Return one result per input pair, preserving order (including duplicates).

    Select one complete vintage per issue, then interpolate its scalar speeds.
    Never substitute an older vintage to fill a missing point in the chosen run.
    Run metadata is small; points are read only for selected runs and the target
    interval plus its possible three-hour interpolation brackets.
    """
    issues = pd.DatetimeIndex(pd.to_datetime(issue_times, utc=True, errors="coerce", format="mixed"))
    targets = pd.DatetimeIndex(pd.to_datetime(target_times, utc=True, errors="coerce", format="mixed"))
    if len(issues) != len(targets):
        raise ValueError("Issue and target times must have equal lengths")
    result = pd.DataFrame({
        "ecmwf_speed_knots": np.full(len(issues), np.nan),
        "ecmwf_run_time": [None] * len(issues),
        "ecmwf_completed_time": [None] * len(issues),
        "ecmwf_missing_reason": ["invalid timestamp"] * len(issues),
    })
    valid = ~issues.isna() & ~targets.isna()
    if not valid.any():
        return result
    result.loc[valid, "ecmwf_missing_reason"] = "archive unavailable"
    if archive_db is None or not Path(archive_db).is_file():
        return result
    try:
        with closing(sqlite3.connect(Path(archive_db).resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            conn.execute("PRAGMA query_only=ON")
            conn.execute("PRAGMA busy_timeout=5000")
            runs = conn.execute("""SELECT run_time, completed_time FROM forecast_collection_runs
                WHERE provider='ECMWF' AND model='IFS' AND site=? AND status='complete'
                  AND completed_time IS NOT NULL
                  AND julianday(run_time) <= julianday(?)
                  AND julianday(completed_time) <= julianday(?)
                ORDER BY julianday(run_time)""",
                (site, issues[valid].max().isoformat(), issues[valid].max().isoformat())).fetchall()
            parsed = [(run, completed, pd.Timestamp(run), pd.Timestamp(completed))
                      for run, completed in runs]
            selected = {}
            result.loc[valid, "ecmwf_missing_reason"] = "no complete run before issue"
            for issue in issues[valid].unique():
                eligible = [r for r in parsed if r[2] <= issue and r[3] <= issue]
                if eligible:
                    selected[issue] = eligible[-1][:2]
            positions_by_run = {}
            for i in np.flatnonzero(valid):
                selection = selected.get(issues[i])
                if selection is not None:
                    positions_by_run.setdefault(selection, []).append(i)
            for (run, completed), matched in positions_by_run.items():
                positions = np.asarray(matched, dtype=int)
                query_targets = targets[positions]
                points = pd.read_sql_query("""SELECT valid_time, fetched_time, wind_speed_mps
                    FROM forecast_points WHERE provider='ECMWF' AND model='IFS'
                      AND site=? AND run_time=?
                      AND julianday(valid_time) >= julianday(?)
                      AND julianday(valid_time) <= julianday(?)
                    ORDER BY julianday(valid_time)""", conn, params=(site, run,
                    (query_targets.min() - pd.Timedelta(hours=3)).isoformat(),
                    (query_targets.max() + pd.Timedelta(hours=3)).isoformat()))
                result.loc[positions, "ecmwf_run_time"] = run
                result.loc[positions, "ecmwf_completed_time"] = completed
                result.loc[positions, "ecmwf_missing_reason"] = "missing interpolation bracket"
                native = pd.DatetimeIndex(pd.to_datetime(points.valid_time, utc=True))
                fetched = pd.DatetimeIndex(pd.to_datetime(points.fetched_time, utc=True))
                speeds = pd.to_numeric(points.wind_speed_mps, errors="coerce").to_numpy(float)
                for i in positions:
                    target, issue = targets[i], issues[i]
                    right = native.searchsorted(target)
                    if right < len(native) and native[right] == target:
                        left = right
                    elif right == 0 or right == len(native):
                        continue
                    else:
                        left = right - 1
                        if native[right] - native[left] > pd.Timedelta(hours=3):
                            result.loc[i, "ecmwf_missing_reason"] = "native gap exceeds three hours"
                            continue
                    if pd.isna(fetched[left]) or pd.isna(fetched[right]) or max(fetched[left], fetched[right]) > issue:
                        result.loc[i, "ecmwf_missing_reason"] = "point fetched after issue"
                        continue
                    if not np.isfinite(speeds[[left, right]]).all():
                        result.loc[i, "ecmwf_missing_reason"] = "invalid wind speed"
                        continue
                    fraction = 0. if left == right else (target - native[left]) / (native[right] - native[left])
                    value = speeds[left] + fraction * (speeds[right] - speeds[left])
                    result.loc[i, "ecmwf_speed_knots"] = value * MPS_TO_KNOT
                    result.loc[i, "ecmwf_missing_reason"] = None
    except (sqlite3.Error, OSError, ValueError, pd.errors.DatabaseError):
        # Discard partial results too: an unreadable archive cannot leave a stale
        # or partially generated third curve in a newly written report.
        result.loc[valid, "ecmwf_speed_knots"] = np.nan
        result.loc[valid, ["ecmwf_run_time", "ecmwf_completed_time"]] = None
        result.loc[valid, "ecmwf_missing_reason"] = "archive unreadable"
    return result


def attach_ecmwf(cases: pd.DataFrame, archive_db: Path | None, *, site: str) -> pd.DataFrame:
    """Cases contain issue_time_utc and target_time_utc; replace old ECMWF data."""
    values = load_ecmwf_evaluation(archive_db, site=site,
        issue_times=cases.issue_time_utc, target_times=cases.target_time_utc)
    cases = cases.copy()
    for col in values:
        cases[col] = values[col].to_numpy()
    return cases


def add_sector_metrics(summary: pd.DataFrame, cases: pd.DataFrame,
                       target_sectors: pd.Series | None = None) -> pd.DataFrame:
    """Add ECMWF-only metrics without altering any existing model statistics.

    cases has actual, target_time_utc, ecmwf_speed_knots, and (for current-day)
    sector. Supplying target_sectors averages available overlapping issues first
    and assigns the existing full-sample HARMONIE sector to each target hour.
    """
    work = cases.copy()
    work["target_time_utc"] = pd.to_datetime(work.target_time_utc, utc=True)
    if "ecmwf_speed_knots" not in work:
        work["ecmwf_speed_knots"] = np.nan
    work = work.loc[np.isfinite(work[["actual", "ecmwf_speed_knots"]]).all(axis=1)].copy()
    if target_sectors is not None:
        sectors = target_sectors.copy()
        sectors.index = pd.to_datetime(sectors.index, utc=True)
        work = work.groupby("target_time_utc", as_index=False)[["actual", "ecmwf_speed_knots"]].mean()
        work["sector"] = work.target_time_utc.map(sectors)
    records = []
    for sector in summary.sector:
        group = work.loc[work.sector == sector]
        error = group.ecmwf_speed_knots - group.actual
        records.append({"sector": sector, "ecmwf_n_points": len(group),
            "ecmwf_n_target_hours": group.target_time_utc.nunique(),
            "ecmwf_mae": error.abs().mean(), "ecmwf_bias_pred_minus_actual": error.mean(),
            "ecmwf_start_utc": group.target_time_utc.min(),
            "ecmwf_end_utc": group.target_time_utc.max(),
            "ecmwf_report_version": REPORT_VERSION})
    return summary.drop(columns=[c for c in summary if c.startswith("ecmwf_")]).merge(
        pd.DataFrame(records), on="sector", how="left")


def report_is_current(path: Path) -> bool:
    try:
        frame = pd.read_csv(path, usecols=["ecmwf_report_version"])
        return not frame.empty and frame.ecmwf_report_version.eq(REPORT_VERSION).all()
    except (OSError, ValueError, pd.errors.ParserError):
        return False


def coverage_text(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        frame = pd.read_csv(path)
        good = frame.loc[frame.ecmwf_n_points.gt(0) & frame.ecmwf_mae.notna()]
        if good.empty:
            return "ECMWF evaluation is unavailable for these forecast issues."
        start = pd.to_datetime(good.ecmwf_start_utc, utc=True).min().tz_convert("Europe/Amsterdam")
        end = pd.to_datetime(good.ecmwf_end_utc, utc=True).max().tz_convert("Europe/Amsterdam")
        count = int(good.ecmwf_n_points.sum())
        hours = int(good.ecmwf_n_target_hours.sum())
        units = f"{count:,} forecast cases" if count != hours else f"{hours:,} target hours"
        return (f"ECMWF: {start:%d %b %Y}–{end:%d %b %Y}, {units}. "
                "The curves can cover different evaluation periods.")
    except (OSError, ValueError, AttributeError, pd.errors.ParserError):
        return ""
