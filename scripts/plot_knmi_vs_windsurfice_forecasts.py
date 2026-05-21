#!/usr/bin/env python3
"""Plot KNMI shadow forecasts against Windsurfice HARMONIE forecasts."""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


EXPECTED_KNMI_TABLE = "knmi_forecasts_shadow"
WINDSURFICE_TABLE = "forecasts"
OBSERVATIONS_TABLE = "observations"


@dataclass(frozen=True)
class TableMap:
    site: str | None
    model: str | None
    run_ts: str | None
    run_iso: str | None
    fetched_ts: str | None
    fetched_iso: str | None
    target_ts: str | None
    target_iso: str | None
    horizon_hr: str | None
    wind_speed: str | None
    wind_gust: str | None
    wind_dir: str | None


@dataclass(frozen=True)
class SelectedVintage:
    knmi_run_utc: pd.Timestamp
    knmi_fetched_ref_utc: pd.Timestamp | None
    windsurfice_run_utc: pd.Timestamp | None
    windsurfice_fetched_ref_utc: pd.Timestamp | None
    vintage_delta_minutes: float | None
    close_vintage: bool
    overlap_rows: int
    warning: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create visual verification plots comparing Windsurfice HARMONIE and KNMI HARMONIE shadow forecasts.",
    )
    parser.add_argument("--db", type=Path, default=Path("data/wind_data_all_sites.db"))
    parser.add_argument("--site", default="valkenburgsemeer")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/knmi_validation/plots"))
    parser.add_argument("--timezone", default="Europe/Amsterdam")
    parser.add_argument("--date", default=None, help="Local date YYYY-MM-DD. Default: today in --timezone.")
    parser.add_argument("--knmi-run-ts", default=None, help="Specific KNMI run timestamp, e.g. 2026-05-21T18:00:00Z.")
    parser.add_argument("--max-vintage-delta-minutes", type=float, default=120.0)
    parser.add_argument("--include-observations", action="store_true")
    parser.add_argument("--include-gust", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({quote_ident(table)})").fetchall()
    if not rows:
        raise SystemExit(f"Required table {table!r} was not found in the SQLite database.")
    return [str(row[1]) for row in rows]


def first_present(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    column_list = list(columns)
    column_set = set(column_list)
    lower_to_real = {column.lower(): column for column in column_list}
    for candidate in candidates:
        if candidate in column_set:
            return candidate
        real = lower_to_real.get(candidate.lower())
        if real:
            return real
    return None


def map_table(columns: list[str]) -> TableMap:
    return TableMap(
        site=first_present(columns, ("site",)),
        model=first_present(columns, ("model",)),
        run_ts=first_present(columns, ("run_ts", "forecast_run_ts", "issued_ts")),
        run_iso=first_present(columns, ("run_iso", "issued_iso")),
        fetched_ts=first_present(columns, ("fetched_ts", "fetched_at", "forecast_fetched_ts", "created_at")),
        fetched_iso=first_present(columns, ("fetched_iso", "fetched_at_iso", "created_iso")),
        target_ts=first_present(columns, ("target_ts", "ts", "timestamp")),
        target_iso=first_present(columns, ("target_iso", "iso_time", "target_time_utc")),
        horizon_hr=first_present(columns, ("horizon_hr", "horizon", "forecast_horizon_hr")),
        wind_speed=first_present(columns, ("wind_speed", "forecast_avg", "WindForecastAvr", "wind_speed_10m_knots")),
        wind_gust=first_present(columns, ("wind_gust", "forecast_max", "WindForecastMax", "gust")),
        wind_dir=first_present(columns, ("wind_dir", "wind_direction", "forecast_dir", "WindDirection")),
    )


def require_mapping(table: str, mapping: TableMap) -> None:
    missing = []
    if mapping.site is None:
        missing.append("site")
    if mapping.target_ts is None and mapping.target_iso is None:
        missing.append("target timestamp")
    if mapping.wind_speed is None:
        missing.append("wind speed")
    if missing:
        raise SystemExit(f"Could not map required columns for {table}: {', '.join(missing)}")


def parse_timestamp(value: Any) -> pd.Timestamp:
    if value is None or pd.isna(value):
        return pd.NaT
    if isinstance(value, pd.Timestamp):
        return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
    if isinstance(value, (int, float, np.integer, np.floating)):
        unit = "ms" if abs(float(value)) >= 10_000_000_000 else "s"
        return pd.to_datetime(value, unit=unit, utc=True, errors="coerce")
    text = str(value).strip()
    if not text:
        return pd.NaT
    try:
        numeric = float(text)
    except ValueError:
        return pd.to_datetime(text, utc=True, errors="coerce")
    unit = "ms" if abs(numeric) >= 10_000_000_000 else "s"
    return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")


def parse_timestamp_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.to_datetime(series, utc=True, errors="coerce")
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        max_abs = numeric.abs().max()
        unit = "ms" if pd.notna(max_abs) and max_abs >= 10_000_000_000 else "s"
        return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    non_null = series.notna().sum()
    numeric = pd.to_numeric(series, errors="coerce")
    if non_null and numeric.notna().sum() >= non_null * 0.95:
        max_abs = numeric.abs().max()
        unit = "ms" if pd.notna(max_abs) and max_abs >= 10_000_000_000 else "s"
        return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")


def iso_z(value: Any) -> str | None:
    ts = parse_timestamp(value)
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).isoformat().replace("+00:00", "Z")


def select_sql_columns(mapping: TableMap) -> list[tuple[str, str]]:
    pairs = [
        ("site", mapping.site),
        ("model", mapping.model),
        ("run_ts_raw", mapping.run_ts),
        ("run_iso_raw", mapping.run_iso),
        ("fetched_ts_raw", mapping.fetched_ts),
        ("fetched_iso_raw", mapping.fetched_iso),
        ("target_ts_raw", mapping.target_ts),
        ("target_iso_raw", mapping.target_iso),
        ("horizon_hr", mapping.horizon_hr),
        ("wind_speed", mapping.wind_speed),
        ("wind_gust", mapping.wind_gust),
        ("wind_dir", mapping.wind_dir),
    ]
    return [(alias, column) for alias, column in pairs if column is not None]


def load_forecast_table(conn: sqlite3.Connection, table: str, site: str, model: str | None = "HARMONIE") -> pd.DataFrame:
    columns = table_columns(conn, table)
    mapping = map_table(columns)
    require_mapping(table, mapping)
    selected = select_sql_columns(mapping)
    sql_columns = ", ".join(f"{quote_ident(column)} AS {quote_ident(alias)}" for alias, column in selected)
    where = [f"{quote_ident(mapping.site)} = ?"] if mapping.site is not None else []
    params: list[Any] = [site] if mapping.site is not None else []
    if model is not None and mapping.model is not None:
        where.append(f"{quote_ident(mapping.model)} = ?")
        params.append(model)
    where_sql = " WHERE " + " AND ".join(where) if where else ""
    frame = pd.read_sql_query(f"SELECT {sql_columns} FROM {quote_ident(table)}{where_sql}", conn, params=params)
    if frame.empty:
        return frame

    frame["target_dt"] = (
        parse_timestamp_series(frame["target_ts_raw"])
        if "target_ts_raw" in frame.columns
        else parse_timestamp_series(frame["target_iso_raw"])
    )
    frame["run_dt"] = pd.NaT
    if "run_ts_raw" in frame.columns:
        frame["run_dt"] = parse_timestamp_series(frame["run_ts_raw"])
    elif "run_iso_raw" in frame.columns:
        frame["run_dt"] = parse_timestamp_series(frame["run_iso_raw"])
    frame["fetched_dt"] = pd.NaT
    if "fetched_ts_raw" in frame.columns:
        frame["fetched_dt"] = parse_timestamp_series(frame["fetched_ts_raw"])
    elif "fetched_iso_raw" in frame.columns:
        frame["fetched_dt"] = parse_timestamp_series(frame["fetched_iso_raw"])

    for column in ("wind_speed", "wind_gust", "wind_dir", "horizon_hr"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["target_dt", "wind_speed"]).sort_values(["run_dt", "fetched_dt", "target_dt"])
    return frame


def load_observations(conn: sqlite3.Connection, site: str) -> pd.DataFrame:
    columns = table_columns(conn, OBSERVATIONS_TABLE)
    mapping = map_table(columns)
    require_mapping(OBSERVATIONS_TABLE, mapping)
    selected = select_sql_columns(mapping)
    sql_columns = ", ".join(f"{quote_ident(column)} AS {quote_ident(alias)}" for alias, column in selected)
    where = f" WHERE {quote_ident(mapping.site)} = ?" if mapping.site is not None else ""
    params = [site] if mapping.site is not None else []
    frame = pd.read_sql_query(f"SELECT {sql_columns} FROM {quote_ident(OBSERVATIONS_TABLE)}{where}", conn, params=params)
    if frame.empty:
        return frame
    frame["target_dt"] = (
        parse_timestamp_series(frame["target_ts_raw"])
        if "target_ts_raw" in frame.columns
        else parse_timestamp_series(frame["target_iso_raw"])
    )
    for column in ("wind_speed", "wind_gust", "wind_dir"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["target_dt"]).sort_values("target_dt")


def local_day_bounds(day: date, zone: ZoneInfo) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(datetime.combine(day, datetime.min.time()), tz=zone)
    end = start + pd.Timedelta(days=1)
    return start, end


def choose_knmi_run(knmi: pd.DataFrame, selected_day: date, next_day: date, zone: ZoneInfo, requested: str | None) -> pd.Timestamp:
    if requested:
        requested_ts = parse_timestamp(requested)
        if pd.isna(requested_ts):
            raise SystemExit(f"Could not parse --knmi-run-ts: {requested!r}")
        run_rows = knmi[knmi["run_dt"].eq(requested_ts)]
        if run_rows.empty:
            raise SystemExit(f"No KNMI shadow rows found for run {iso_z(requested_ts)}.")
        return pd.Timestamp(requested_ts)

    start_local, _ = local_day_bounds(selected_day, zone)
    _, next_end_local = local_day_bounds(next_day, zone)
    candidates = knmi[
        (knmi["target_dt"] >= start_local.tz_convert("UTC"))
        & (knmi["target_dt"] < next_end_local.tz_convert("UTC"))
        & knmi["run_dt"].notna()
    ]
    if candidates.empty:
        raise SystemExit(f"No KNMI shadow runs cover {selected_day} / {next_day}.")
    return pd.Timestamp(candidates["run_dt"].max())


def best_windsurfice_snapshot(
    windsurfice: pd.DataFrame,
    knmi_run: pd.DataFrame,
    max_delta_minutes: float,
) -> tuple[pd.DataFrame, SelectedVintage]:
    knmi_targets = set(knmi_run["target_dt"].dropna())
    if not knmi_targets:
        raise SystemExit("Selected KNMI run has no target timestamps.")
    knmi_fetched_ref = first_valid_timestamp(knmi_run["fetched_dt"])
    knmi_run_ref = first_valid_timestamp(knmi_run["run_dt"])
    reference = knmi_fetched_ref if knmi_fetched_ref is not None else knmi_run_ref
    if reference is None:
        raise SystemExit("Selected KNMI run has neither fetched nor run timestamp metadata.")

    candidates: list[dict[str, Any]] = []
    for (run_dt, fetched_dt), group in windsurfice.groupby(["run_dt", "fetched_dt"], dropna=False):
        overlap = group[group["target_dt"].isin(knmi_targets)]
        if overlap.empty:
            continue
        fetch_ref = first_valid_timestamp(overlap["fetched_dt"])
        run_ref = first_valid_timestamp(overlap["run_dt"])
        candidate_ref = fetch_ref if fetch_ref is not None else run_ref
        if candidate_ref is None:
            continue
        delta = abs((candidate_ref - reference) / pd.Timedelta(minutes=1))
        candidates.append(
            {
                "run_dt": run_ref,
                "fetched_dt": fetch_ref,
                "candidate_ref": candidate_ref,
                "delta": float(delta),
                "overlap": int(len(overlap)),
                "frame": overlap.copy(),
            }
        )
    if not candidates:
        raise SystemExit("No Windsurfice forecast snapshot overlaps the selected KNMI target timestamps.")
    candidates.sort(key=lambda item: (item["delta"], -item["overlap"]))
    best = candidates[0]
    close = best["delta"] <= float(max_delta_minutes)
    warning = None if close else (
        f"Closest Windsurfice snapshot is {best['delta']:.1f} minutes from KNMI reference, "
        f"outside tolerance {max_delta_minutes:.1f} minutes."
    )
    selected = SelectedVintage(
        knmi_run_utc=pd.Timestamp(knmi_run_ref),
        knmi_fetched_ref_utc=knmi_fetched_ref,
        windsurfice_run_utc=best["run_dt"],
        windsurfice_fetched_ref_utc=best["fetched_dt"],
        vintage_delta_minutes=best["delta"],
        close_vintage=close,
        overlap_rows=best["overlap"],
        warning=warning,
    )
    return best["frame"], selected


def first_valid_timestamp(series: pd.Series) -> pd.Timestamp | None:
    valid = series.dropna()
    if valid.empty:
        return None
    return pd.Timestamp(valid.iloc[0])


def circular_diff_deg(left: pd.Series, right: pd.Series) -> pd.Series:
    left_num = pd.to_numeric(left, errors="coerce")
    right_num = pd.to_numeric(right, errors="coerce")
    return ((left_num - right_num + 180.0) % 360.0) - 180.0


def merge_comparison(knmi: pd.DataFrame, windsurfice: pd.DataFrame, zone: ZoneInfo) -> pd.DataFrame:
    left = knmi[
        ["target_dt", "horizon_hr", "wind_speed", "wind_gust", "wind_dir"]
    ].rename(
        columns={
            "horizon_hr": "knmi_horizon_hr",
            "wind_speed": "knmi_wind_speed",
            "wind_gust": "knmi_wind_gust",
            "wind_dir": "knmi_wind_dir",
        }
    )
    right = windsurfice[
        ["target_dt", "horizon_hr", "wind_speed", "wind_gust", "wind_dir"]
    ].rename(
        columns={
            "horizon_hr": "windsurfice_horizon_hr",
            "wind_speed": "windsurfice_wind_speed",
            "wind_gust": "windsurfice_wind_gust",
            "wind_dir": "windsurfice_wind_dir",
        }
    )
    merged = left.merge(right, on="target_dt", how="inner").sort_values("target_dt")
    merged["target_local"] = merged["target_dt"].dt.tz_convert(zone)
    merged["speed_diff_knots"] = merged["knmi_wind_speed"] - merged["windsurfice_wind_speed"]
    merged["wind_dir_circular_diff_deg"] = circular_diff_deg(merged["knmi_wind_dir"], merged["windsurfice_wind_dir"])
    if "knmi_wind_gust" in merged and "windsurfice_wind_gust" in merged:
        merged["gust_diff_knots"] = merged["knmi_wind_gust"] - merged["windsurfice_wind_gust"]
    return merged


def day_frame(frame: pd.DataFrame, day: date, zone: ZoneInfo) -> pd.DataFrame:
    start, end = local_day_bounds(day, zone)
    return frame[(frame["target_local"] >= start) & (frame["target_local"] < end)].copy()


def observation_day_frame(obs: pd.DataFrame, day: date, zone: ZoneInfo) -> pd.DataFrame:
    if obs.empty:
        return obs
    start, end = local_day_bounds(day, zone)
    out = obs.copy()
    out["target_local"] = out["target_dt"].dt.tz_convert(zone)
    return out[(out["target_local"] >= start) & (out["target_local"] < end)].copy()


def metrics_for_day(frame: pd.DataFrame, include_gust: bool) -> dict[str, Any]:
    matched = frame.dropna(subset=["knmi_wind_speed", "windsurfice_wind_speed"])
    speed_abs = matched["speed_diff_knots"].abs()
    result: dict[str, Any] = {
        "matched_rows": int(len(matched)),
        "speed_mae": float(speed_abs.mean()) if not speed_abs.empty else None,
        "speed_median_abs_error": float(speed_abs.median()) if not speed_abs.empty else None,
        "speed_p95_abs_error": float(speed_abs.quantile(0.95)) if not speed_abs.empty else None,
        "speed_max_abs_error": float(speed_abs.max()) if not speed_abs.empty else None,
        "direction_mean_circular_difference": float(matched["wind_dir_circular_diff_deg"].mean())
        if "wind_dir_circular_diff_deg" in matched and matched["wind_dir_circular_diff_deg"].notna().any()
        else None,
    }
    if include_gust and "gust_diff_knots" in matched:
        gust_abs = matched.dropna(subset=["gust_diff_knots"])["gust_diff_knots"].abs()
        result["gust_mae"] = float(gust_abs.mean()) if not gust_abs.empty else None
    return result


def plot_day(
    frame: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    output_path: Path,
    site: str,
    day: date,
    day_label: str,
    zone_name: str,
    selected: SelectedVintage,
    include_observations: bool,
    include_gust: bool,
    show: bool,
) -> None:
    if frame.empty:
        print(f"WARNING: no overlapping rows for {day_label} plot ({day}). Skipping {output_path}.")
        return

    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.plot(
        frame["target_local"],
        frame["windsurfice_wind_speed"],
        color="#1f77b4",
        linewidth=2.0,
        label="Windsurfice HARMONIE avg",
    )
    ax.plot(
        frame["target_local"],
        frame["knmi_wind_speed"],
        color="#d62728",
        linewidth=2.0,
        linestyle="--",
        label="KNMI HARMONIE avg",
    )
    if include_gust:
        if frame["windsurfice_wind_gust"].notna().any():
            ax.plot(
                frame["target_local"],
                frame["windsurfice_wind_gust"],
                color="#6baed6",
                linewidth=1.3,
                alpha=0.85,
                label="Windsurfice gust/max",
            )
        if frame["knmi_wind_gust"].notna().any():
            ax.plot(
                frame["target_local"],
                frame["knmi_wind_gust"],
                color="#ff9896",
                linewidth=1.3,
                linestyle="--",
                alpha=0.9,
                label="KNMI gust/max",
            )
    if include_observations and not observations.empty and observations["wind_speed"].notna().any():
        ax.plot(
            observations["target_local"],
            observations["wind_speed"],
            color="#222222",
            linewidth=1.2,
            marker="o",
            markersize=2.5,
            alpha=0.75,
            label="Observed avg",
        )

    delta_text = (
        f"{selected.vintage_delta_minutes:.1f} min"
        if selected.vintage_delta_minutes is not None
        else "unknown"
    )
    windsurfice_ref = selected.windsurfice_fetched_ref_utc or selected.windsurfice_run_utc
    subtitle = (
        f"KNMI run {iso_z(selected.knmi_run_utc)} | "
        f"Windsurfice ref {iso_z(windsurfice_ref)} | vintage delta {delta_text}"
    )
    ax.set_title(f"{site} {day_label} forecast comparison ({day})\n{subtitle}")
    ax.set_xlabel(f"Local time ({zone_name})")
    ax.set_ylabel("Wind speed (knots)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="best")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=ZoneInfo(zone_name)))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2, tz=ZoneInfo(zone_name)))
    fig.autofmt_xdate()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def print_selection(selected: SelectedVintage) -> None:
    print("Selected comparison vintages")
    print("============================")
    print(f"KNMI run_ts UTC: {iso_z(selected.knmi_run_utc)}")
    print(f"KNMI fetched_ts reference UTC: {iso_z(selected.knmi_fetched_ref_utc)}")
    print(f"Windsurfice run_ts UTC: {iso_z(selected.windsurfice_run_utc)}")
    print(f"Windsurfice fetched_ts reference UTC: {iso_z(selected.windsurfice_fetched_ref_utc)}")
    print(f"Vintage delta minutes: {selected.vintage_delta_minutes}")
    print(f"Overlapping target rows: {selected.overlap_rows}")
    print(f"Comparison type: {'close-vintage' if selected.close_vintage else 'approximate'}")
    if selected.warning:
        print(f"WARNING: {selected.warning}")


def print_day_metrics(label: str, metrics: dict[str, Any], include_gust: bool) -> None:
    print(f"\n{label} diagnostics")
    print("-" * (len(label) + 12))
    print(f"matched target rows: {metrics.get('matched_rows')}")
    print(f"speed MAE: {metrics.get('speed_mae')}")
    print(f"speed median absolute error: {metrics.get('speed_median_abs_error')}")
    print(f"speed p95 absolute error: {metrics.get('speed_p95_abs_error')}")
    print(f"max speed absolute error: {metrics.get('speed_max_abs_error')}")
    print(f"direction mean circular difference: {metrics.get('direction_mean_circular_difference')}")
    if include_gust:
        print(f"gust MAE: {metrics.get('gust_mae')}")
    print("expectation: lines should overlap closely for close-vintage comparisons.")


def main() -> None:
    args = parse_args()
    zone = ZoneInfo(args.timezone)
    selected_date = (
        datetime.now(zone).date()
        if args.date is None
        else datetime.strptime(args.date, "%Y-%m-%d").date()
    )
    next_date = selected_date + pd.Timedelta(days=1)
    next_date = next_date.date() if hasattr(next_date, "date") else next_date

    conn = sqlite3.connect(args.db)
    try:
        knmi_all = load_forecast_table(conn, EXPECTED_KNMI_TABLE, args.site, model="HARMONIE")
        windsurfice_all = load_forecast_table(conn, WINDSURFICE_TABLE, args.site, model="HARMONIE")
        observations_all = load_observations(conn, args.site) if args.include_observations else pd.DataFrame()
    finally:
        conn.close()

    if knmi_all.empty:
        raise SystemExit(f"No KNMI shadow rows found for site={args.site!r}.")
    if windsurfice_all.empty:
        raise SystemExit(f"No Windsurfice forecast rows found for site={args.site!r}.")

    knmi_run_ts = choose_knmi_run(knmi_all, selected_date, next_date, zone, args.knmi_run_ts)
    knmi_run = knmi_all[knmi_all["run_dt"].eq(knmi_run_ts)].copy()
    if knmi_run.empty:
        raise SystemExit(f"No KNMI rows found for selected run {iso_z(knmi_run_ts)}.")

    windsurfice_snapshot, selected = best_windsurfice_snapshot(
        windsurfice_all,
        knmi_run,
        max_delta_minutes=args.max_vintage_delta_minutes,
    )
    comparison = merge_comparison(knmi_run, windsurfice_snapshot, zone)
    if comparison.empty:
        raise SystemExit("No overlapping target timestamps after selecting vintages.")

    current = day_frame(comparison, selected_date, zone)
    next_day = day_frame(comparison, next_date, zone)
    current_obs = observation_day_frame(observations_all, selected_date, zone)
    next_obs = observation_day_frame(observations_all, next_date, zone)

    date_label = selected_date.isoformat()
    safe_site = args.site.replace("/", "_")
    current_png = args.output_dir / f"knmi_vs_windsurfice_current_day_{safe_site}_{date_label}.png"
    next_png = args.output_dir / f"knmi_vs_windsurfice_next_day_{safe_site}_{date_label}.png"
    csv_path = args.output_dir / f"knmi_vs_windsurfice_plot_data_{safe_site}_{date_label}.csv"

    print_selection(selected)
    current_metrics = metrics_for_day(current, args.include_gust)
    next_metrics = metrics_for_day(next_day, args.include_gust)
    print_day_metrics("Current-day", current_metrics, args.include_gust)
    print_day_metrics("Next-day", next_metrics, args.include_gust)

    plot_day(
        current,
        current_obs,
        output_path=current_png,
        site=args.site,
        day=selected_date,
        day_label="current day",
        zone_name=args.timezone,
        selected=selected,
        include_observations=args.include_observations,
        include_gust=args.include_gust,
        show=args.show,
    )
    plot_day(
        next_day,
        next_obs,
        output_path=next_png,
        site=args.site,
        day=next_date,
        day_label="next day",
        zone_name=args.timezone,
        selected=selected,
        include_observations=args.include_observations,
        include_gust=args.include_gust,
        show=args.show,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_frame = comparison.copy()
    csv_frame["target_dt_utc"] = csv_frame["target_dt"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    csv_frame["target_dt_local"] = csv_frame["target_local"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    csv_frame.drop(columns=["target_dt", "target_local"]).to_csv(csv_path, index=False)

    print("\nOutputs")
    print("-------")
    print(f"Current-day plot: {current_png}")
    print(f"Next-day plot: {next_png}")
    print(f"Comparison CSV: {csv_path}")


if __name__ == "__main__":
    main()
