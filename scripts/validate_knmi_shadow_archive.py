#!/usr/bin/env python3
"""Validate the KNMI HARMONIE shadow archive against Windsurfice HARMONIE."""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from next_day_wind_model.knmi_harmonie import (
    KNOTS_PER_MPS,
    SitePoint,
    nearest_value,
    open_grib_parameter,
    parse_run_and_horizon,
)


EXPECTED_HORIZONS = set(range(61))
TABLES_TO_INSPECT = (
    "forecasts",
    "knmi_forecasts_shadow",
    "harmonie_knmi_features",
    "observations",
)
HORIZON_BANDS = [
    (-0.1, 6, "0-6"),
    (6, 12, "7-12"),
    (12, 24, "13-24"),
    (24, 48, "25-48"),
    (48, 60, "49-60"),
]
SPEED_BANDS_KT = [
    (-0.1, 5, "0-5 kt"),
    (5, 10, "5-10 kt"),
    (10, 15, "10-15 kt"),
    (15, 20, "15-20 kt"),
    (20, float("inf"), ">20 kt"),
]
DEFAULT_SITE_POINTS = {
    "valkenburgsemeer": SitePoint(site="valkenburgsemeer", lat=52.168, lon=4.437),
}
GUST_U_WIND_PARAMETER = 162
GUST_V_WIND_PARAMETER = 163
MAX_WIND_CANDIDATE_PARAMETERS = (162, 163)


@dataclass(frozen=True)
class Schema:
    table: str
    columns: tuple[str, ...]

    def has(self, column: str) -> bool:
        return column in self.columns


@dataclass(frozen=True)
class TableMap:
    site: str | None
    model: str | None
    run_ts: str | None
    fetched_ts: str | None
    target_ts: str | None
    horizon_hr: str | None
    wind_speed: str | None
    wind_gust: str | None
    wind_dir: str | None


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def table_info(conn: sqlite3.Connection, table: str) -> Schema:
    rows = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
    return Schema(table=table, columns=tuple(str(row[1]) for row in rows))


def first_present(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    column_set = set(columns)
    lower_to_real = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in column_set:
            return candidate
        real = lower_to_real.get(candidate.lower())
        if real:
            return real
    return None


def map_table(schema: Schema) -> TableMap:
    return TableMap(
        site=first_present(schema.columns, ("site",)),
        model=first_present(schema.columns, ("model",)),
        run_ts=first_present(schema.columns, ("run_ts", "forecast_run_ts", "issued_ts", "run_iso")),
        fetched_ts=first_present(schema.columns, ("fetched_ts", "forecast_fetched_ts", "fetched_iso")),
        target_ts=first_present(schema.columns, ("target_ts", "ts", "timestamp", "iso_time", "target_iso")),
        horizon_hr=first_present(schema.columns, ("horizon_hr", "horizon", "forecast_horizon_hr")),
        wind_speed=first_present(
            schema.columns,
            ("wind_speed", "WindForecastAvr", "forecast_avg", "wind_speed_10m_knots", "wind_speed_10m_mps"),
        ),
        wind_gust=first_present(schema.columns, ("wind_gust", "WindForecastMax", "forecast_max", "gust", "fg")),
        wind_dir=first_present(schema.columns, ("wind_dir", "WindDirection", "forecast_dir", "wind_dir_10m")),
    )


def parse_timestamp(value: object) -> pd.Timestamp:
    if value is None or pd.isna(value):
        return pd.NaT
    if isinstance(value, pd.Timestamp):
        return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
    if isinstance(value, (int, float)):
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


def datetime_series_to_ms(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    raw_ns = parsed.astype("int64")
    out = pd.Series(raw_ns // 1_000_000, index=series.index)
    out[parsed.isna()] = pd.NA
    return out.astype("Int64")


def timestamp_ms(ts: object) -> int | None:
    parsed = parse_timestamp(ts)
    if pd.isna(parsed):
        return None
    return int(pd.Timestamp(parsed).timestamp() * 1000)


def iso_z(value: object) -> str | None:
    ts = parse_timestamp(value)
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).isoformat().replace("+00:00", "Z")


def circular_abs_diff(a: pd.Series, b: pd.Series) -> pd.Series:
    left = pd.to_numeric(a, errors="coerce")
    right = pd.to_numeric(b, errors="coerce")
    return ((left - right + 180.0) % 360.0 - 180.0).abs()


def direction_sector(direction: object) -> str | None:
    if direction is None or pd.isna(direction):
        return None
    sectors = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
    return sectors[int(((float(direction) % 360.0) + 22.5) // 45.0) % 8]


def band_value(value: object, bands: list[tuple[float, float, str]]) -> str | None:
    if value is None or pd.isna(value):
        return None
    numeric = float(value)
    for low, high, label in bands:
        if low < numeric <= high:
            return label
    return None


def connect_read_only(path: Path) -> tuple[sqlite3.Connection, str]:
    db_path = path.expanduser().resolve()
    errors: list[str] = []
    for mode_name, uri in (
        ("mode=ro", f"file:{db_path}?mode=ro"),
        ("immutable=1", f"file:{db_path}?immutable=1"),
    ):
        try:
            conn = sqlite3.connect(uri, uri=True)
            conn.execute("PRAGMA query_only = ON")
            conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()
            conn.execute("PRAGMA table_info('forecasts')").fetchall()
        except sqlite3.Error as exc:
            errors.append(f"{mode_name}: {exc}")
            try:
                conn.close()
            except Exception:
                pass
            continue
        return conn, mode_name
    raise sqlite3.OperationalError("Could not open SQLite database read-only; " + "; ".join(errors))


def select_table_frame(
    conn: sqlite3.Connection,
    table: str,
    mapping: TableMap,
    *,
    site: str | None = None,
    model: str | None = None,
) -> pd.DataFrame:
    selected: list[tuple[str, str]] = []
    for output, column in (
        ("site", mapping.site),
        ("model", mapping.model),
        ("run_ts_raw", mapping.run_ts),
        ("fetched_ts_raw", mapping.fetched_ts),
        ("target_ts_raw", mapping.target_ts),
        ("horizon_hr", mapping.horizon_hr),
        ("wind_speed", mapping.wind_speed),
        ("wind_gust", mapping.wind_gust),
        ("wind_dir", mapping.wind_dir),
    ):
        if column is not None:
            selected.append((output, column))
    if not selected:
        return pd.DataFrame()

    sql_columns = ", ".join(f"{quote_ident(column)} AS {quote_ident(output)}" for output, column in selected)
    where: list[str] = []
    params: list[object] = []
    if site is not None and mapping.site is not None:
        where.append(f"{quote_ident(mapping.site)} = ?")
        params.append(site)
    if model is not None and mapping.model is not None:
        where.append(f"{quote_ident(mapping.model)} = ?")
        params.append(model)
    where_sql = " WHERE " + " AND ".join(where) if where else ""
    query = f"SELECT rowid AS source_rowid, {sql_columns} FROM {quote_ident(table)}{where_sql}"
    frame = pd.read_sql_query(query, conn, params=params)
    return normalize_frame(frame)


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in (
        "site",
        "model",
        "run_ts_raw",
        "fetched_ts_raw",
        "target_ts_raw",
        "horizon_hr",
        "wind_speed",
        "wind_gust",
        "wind_dir",
    ):
        if column not in out.columns:
            out[column] = pd.NA
    out["run_dt"] = parse_timestamp_series(out["run_ts_raw"])
    out["fetched_dt"] = parse_timestamp_series(out["fetched_ts_raw"])
    out["target_dt"] = parse_timestamp_series(out["target_ts_raw"])
    out["run_ms"] = datetime_series_to_ms(out["run_dt"])
    out["fetched_ms"] = datetime_series_to_ms(out["fetched_dt"])
    out["target_ms"] = datetime_series_to_ms(out["target_dt"])
    out["horizon_hr"] = pd.to_numeric(out["horizon_hr"], errors="coerce")
    out["wind_speed"] = pd.to_numeric(out["wind_speed"], errors="coerce")
    out["wind_gust"] = pd.to_numeric(out["wind_gust"], errors="coerce")
    out["wind_dir"] = pd.to_numeric(out["wind_dir"], errors="coerce")
    return out


def latest_per_target(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame.dropna(subset=["target_ms"]).copy()
    if valid.empty:
        return valid
    fallback_dt = pd.Timestamp("1900-01-01", tz="UTC")
    valid["_sort_run"] = valid["run_dt"].fillna(fallback_dt)
    valid["_sort_fetch"] = valid["fetched_dt"].fillna(fallback_dt)
    valid = valid.sort_values(["target_ms", "_sort_run", "_sort_fetch", "source_rowid"])
    return valid.drop_duplicates("target_ms", keep="last").drop(columns=["_sort_run", "_sort_fetch"])


def metric_summary(frame: pd.DataFrame, group_col: str | None = None) -> pd.DataFrame:
    matched = frame[frame["matched"]].copy()
    if matched.empty:
        columns = [
            "n",
            "matched_rows",
            "speed_abs_mean_kt",
            "speed_abs_median_kt",
            "speed_abs_p95_kt",
            "speed_abs_max_kt",
            "dir_abs_mean_deg",
            "dir_abs_median_deg",
            "dir_abs_p95_deg",
            "dir_abs_max_deg",
        ]
        return pd.DataFrame(columns=([group_col] if group_col else []) + columns)

    def summarize(part: pd.DataFrame) -> pd.Series:
        speed_abs = part["wind_speed_diff_kt"].abs()
        dir_abs = part["wind_dir_circular_diff_deg"]
        return pd.Series(
            {
                "n": int(len(part)),
                "matched_rows": int(len(part)),
                "speed_abs_mean_kt": speed_abs.mean(),
                "speed_abs_median_kt": speed_abs.median(),
                "speed_abs_p95_kt": speed_abs.quantile(0.95),
                "speed_abs_max_kt": speed_abs.max(),
                "dir_abs_mean_deg": dir_abs.mean(),
                "dir_abs_median_deg": dir_abs.median(),
                "dir_abs_p95_deg": dir_abs.quantile(0.95),
                "dir_abs_max_deg": dir_abs.max(),
            }
        )

    if group_col is None:
        return summarize(matched).to_frame().T

    rows = []
    for group_value, part in matched.groupby(group_col, dropna=False):
        row = summarize(part).to_dict()
        row[group_col] = group_value
        rows.append(row)
    value_cols = [group_col] + [column for column in rows[0] if column != group_col]
    return pd.DataFrame(rows)[value_cols]


def enrich_breakdown_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["horizon_band"] = out["knmi_horizon_hr"].map(lambda value: band_value(value, HORIZON_BANDS))
    out["wind_speed_band"] = out["knmi_wind_speed_kt"].map(lambda value: band_value(value, SPEED_BANDS_KT))
    out["direction_sector"] = out["knmi_wind_dir_deg"].map(direction_sector)
    return out


def compare_target_only(knmi: pd.DataFrame, windsurfice: pd.DataFrame) -> pd.DataFrame:
    knmi_latest = latest_per_target(knmi)
    wf_latest = latest_per_target(windsurfice)
    merged = knmi_latest.merge(
        wf_latest,
        on="target_ms",
        how="outer",
        suffixes=("_knmi", "_windsurfice"),
        indicator=True,
    )
    out = pd.DataFrame(
        {
            "mode": "target-only",
            "target_ts": merged["target_dt_knmi"].combine_first(merged["target_dt_windsurfice"]).map(iso_z),
            "knmi_run_ts": merged.get("run_dt_knmi").map(iso_z),
            "knmi_fetched_ts": merged.get("fetched_dt_knmi").map(iso_z),
            "windsurfice_run_ts": merged.get("run_dt_windsurfice").map(iso_z),
            "windsurfice_fetched_ts": merged.get("fetched_dt_windsurfice").map(iso_z),
            "knmi_horizon_hr": merged.get("horizon_hr_knmi"),
            "windsurfice_horizon_hr": merged.get("horizon_hr_windsurfice"),
            "knmi_wind_speed_kt": merged.get("wind_speed_knmi"),
            "windsurfice_wind_speed_kt": merged.get("wind_speed_windsurfice"),
            "knmi_wind_gust_kt": merged.get("wind_gust_knmi"),
            "windsurfice_wind_gust_kt": merged.get("wind_gust_windsurfice"),
            "knmi_wind_dir_deg": merged.get("wind_dir_knmi"),
            "windsurfice_wind_dir_deg": merged.get("wind_dir_windsurfice"),
            "match_kind": merged["_merge"],
        }
    )
    out["matched"] = out["match_kind"] == "both"
    out["vintage_delta_minutes"] = (
        (merged.get("run_dt_windsurfice") - merged.get("run_dt_knmi")).dt.total_seconds().abs() / 60.0
    )
    out["wind_speed_diff_kt"] = out["knmi_wind_speed_kt"] - out["windsurfice_wind_speed_kt"]
    out["wind_dir_circular_diff_deg"] = circular_abs_diff(out["knmi_wind_dir_deg"], out["windsurfice_wind_dir_deg"])
    out.loc[~out["matched"], ["wind_speed_diff_kt", "wind_dir_circular_diff_deg", "vintage_delta_minutes"]] = pd.NA
    return enrich_breakdown_columns(out)


def windsurfice_vintage_column(frame: pd.DataFrame) -> tuple[str | None, str]:
    run_present = frame["run_dt"].notna().any()
    fetched_present = frame["fetched_dt"].notna().any()
    if run_present and fetched_present:
        paired = frame.dropna(subset=["run_dt", "fetched_dt"])
        identical_share = ((paired["run_ms"] - paired["fetched_ms"]).abs() <= 60_000).mean() if not paired.empty else 0.0
        if identical_share < 0.8:
            return "run_dt", "authoritative run_ts"
        return "fetched_dt", "fallback/fetched timestamp; Windsurfice run_ts is usually fetch-time metadata here"
    if run_present:
        return "run_dt", "run_ts; no separate fetched timestamp was available"
    if fetched_present:
        return "fetched_dt", "fallback/fetched timestamp"
    return None, "no usable Windsurfice run/fetch timestamp"


def compare_closest_vintage(
    knmi: pd.DataFrame,
    windsurfice: pd.DataFrame,
    max_delta_minutes: float,
) -> tuple[pd.DataFrame, str]:
    wf_vintage_col, vintage_note = windsurfice_vintage_column(windsurfice)
    if wf_vintage_col is None:
        return pd.DataFrame(), vintage_note

    knmi_ref_col = "run_dt" if wf_vintage_col == "run_dt" and "authoritative" in vintage_note else "fetched_dt"
    max_delta = float(max_delta_minutes)
    wf = windsurfice.dropna(subset=["target_ms", wf_vintage_col]).copy()
    groups = {target: group for target, group in wf.groupby("target_ms", dropna=False)}
    rows: list[dict[str, object]] = []
    used_wf_rowids: set[int] = set()

    for knmi_row in knmi.dropna(subset=["target_ms"]).itertuples(index=False):
        target = getattr(knmi_row, "target_ms")
        ref_dt = getattr(knmi_row, knmi_ref_col)
        candidates = groups.get(target)
        best = None
        delta_minutes = None
        closest_available_delta_minutes = None
        unmatched_reason = None
        if candidates is not None and not pd.isna(ref_dt):
            candidate = candidates.copy()
            candidate["_delta_min"] = (candidate[wf_vintage_col] - ref_dt).abs().dt.total_seconds() / 60.0
            closest_available_delta_minutes = float(candidate["_delta_min"].min()) if not candidate.empty else None
            candidate = candidate[candidate["_delta_min"] <= max_delta]
            if not candidate.empty:
                candidate = candidate.sort_values(["_delta_min", wf_vintage_col, "source_rowid"])
                best = candidate.iloc[0]
                delta_minutes = float(best["_delta_min"])
                used_wf_rowids.add(int(best["source_rowid"]))
        elif candidates is None:
            unmatched_reason = "no Windsurfice row with same target_ts"
        else:
            unmatched_reason = f"missing KNMI reference timestamp ({knmi_ref_col})"

        matched = best is not None
        if not matched and unmatched_reason is None:
            unmatched_reason = f"no Windsurfice row within {max_delta:.1f} min vintage tolerance"
        rows.append(
            {
                "mode": "closest-vintage",
                "target_ts": iso_z(getattr(knmi_row, "target_dt")),
                "knmi_run_ts": iso_z(getattr(knmi_row, "run_dt")),
                "knmi_fetched_ts": iso_z(getattr(knmi_row, "fetched_dt")),
                "windsurfice_run_ts": iso_z(best["run_dt"]) if matched else None,
                "windsurfice_fetched_ts": iso_z(best["fetched_dt"]) if matched else None,
                "knmi_horizon_hr": getattr(knmi_row, "horizon_hr"),
                "windsurfice_horizon_hr": best["horizon_hr"] if matched else pd.NA,
                "knmi_wind_speed_kt": getattr(knmi_row, "wind_speed"),
                "windsurfice_wind_speed_kt": best["wind_speed"] if matched else pd.NA,
                "knmi_wind_gust_kt": getattr(knmi_row, "wind_gust"),
                "windsurfice_wind_gust_kt": best["wind_gust"] if matched else pd.NA,
                "knmi_wind_dir_deg": getattr(knmi_row, "wind_dir"),
                "windsurfice_wind_dir_deg": best["wind_dir"] if matched else pd.NA,
                "vintage_delta_minutes": delta_minutes,
                "closest_available_vintage_delta_minutes": closest_available_delta_minutes,
                "unmatched_reason": None if matched else unmatched_reason,
                "matched": matched,
                "match_kind": "both" if matched else "left_only",
                "windsurfice_source_rowid": int(best["source_rowid"]) if matched else pd.NA,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out, vintage_note
    out["wind_speed_diff_kt"] = out["knmi_wind_speed_kt"] - out["windsurfice_wind_speed_kt"]
    out["wind_dir_circular_diff_deg"] = circular_abs_diff(out["knmi_wind_dir_deg"], out["windsurfice_wind_dir_deg"])
    out.loc[~out["matched"], ["wind_speed_diff_kt", "wind_dir_circular_diff_deg"]] = pd.NA
    out.attrs["used_windsurfice_rowids"] = used_wf_rowids
    out.attrs["windsurfice_vintage_note"] = vintage_note
    out.attrs["knmi_reference_column"] = knmi_ref_col
    return enrich_breakdown_columns(out), vintage_note


def overlap_range(left: pd.DataFrame, right: pd.DataFrame) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    left_targets = left["target_dt"].dropna()
    right_targets = right["target_dt"].dropna()
    if left_targets.empty or right_targets.empty:
        return None, None
    start = max(left_targets.min(), right_targets.min())
    end = min(left_targets.max(), right_targets.max())
    if start > end:
        return None, None
    return pd.Timestamp(start), pd.Timestamp(end)


def load_observations(conn: sqlite3.Connection, schema: Schema, site: str) -> pd.DataFrame:
    mapping = map_table(schema)
    if mapping.target_ts is None:
        return pd.DataFrame()
    return select_table_frame(conn, schema.table, mapping, site=site, model=None)


def observation_join_counts(knmi: pd.DataFrame, observations: pd.DataFrame) -> tuple[int, int]:
    if knmi.empty or observations.empty:
        return 0, 0
    knmi_targets = knmi[["target_dt"]].dropna().drop_duplicates().sort_values("target_dt")
    obs_targets = observations[["target_dt"]].dropna().drop_duplicates().sort_values("target_dt")
    exact = int(knmi_targets["target_dt"].isin(set(obs_targets["target_dt"])).sum())
    nearest = pd.merge_asof(
        knmi_targets,
        obs_targets.rename(columns={"target_dt": "obs_dt"}),
        left_on="target_dt",
        right_on="obs_dt",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=30),
    )
    within_30 = int(nearest["obs_dt"].notna().sum())
    return exact, within_30


def archive_coverage(knmi: pd.DataFrame, windsurfice: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    overlap_start, overlap_end = overlap_range(knmi, windsurfice)
    exact_obs, within_30_obs = observation_join_counts(knmi, observations)
    run_count = int(knmi["run_ms"].nunique(dropna=True))
    return pd.DataFrame(
        [
            {"metric": "knmi_shadow_rows", "value": len(knmi)},
            {"metric": "knmi_distinct_run_ts", "value": run_count},
            {"metric": "knmi_min_run_ts", "value": iso_z(knmi["run_dt"].min())},
            {"metric": "knmi_max_run_ts", "value": iso_z(knmi["run_dt"].max())},
            {"metric": "knmi_distinct_target_ts", "value": int(knmi["target_ms"].nunique(dropna=True))},
            {"metric": "knmi_min_target_ts", "value": iso_z(knmi["target_dt"].min())},
            {"metric": "knmi_max_target_ts", "value": iso_z(knmi["target_dt"].max())},
            {"metric": "windsurfice_rows", "value": len(windsurfice)},
            {"metric": "windsurfice_distinct_run_ts", "value": int(windsurfice["run_ms"].nunique(dropna=True))},
            {"metric": "windsurfice_distinct_fetched_ts", "value": int(windsurfice["fetched_ms"].nunique(dropna=True))},
            {"metric": "windsurfice_distinct_target_ts", "value": int(windsurfice["target_ms"].nunique(dropna=True))},
            {"metric": "windsurfice_min_target_ts", "value": iso_z(windsurfice["target_dt"].min())},
            {"metric": "windsurfice_max_target_ts", "value": iso_z(windsurfice["target_dt"].max())},
            {"metric": "overlap_min_target_ts", "value": iso_z(overlap_start)},
            {"metric": "overlap_max_target_ts", "value": iso_z(overlap_end)},
            {"metric": "expected_knmi_rows_distinct_runs_x_61", "value": run_count * 61},
            {
                "metric": "knmi_rows_target_already_past_at_fetch",
                "value": int((knmi["target_dt"] < knmi["fetched_dt"]).fillna(False).sum()),
            },
            {"metric": "knmi_distinct_targets_exact_observation", "value": exact_obs},
            {"metric": "knmi_distinct_targets_observation_within_30_min", "value": within_30_obs},
        ]
    )


def missing_horizons_by_run(knmi: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run_dt, group in knmi.dropna(subset=["run_dt"]).groupby("run_dt"):
        horizons = set(int(value) for value in group["horizon_hr"].dropna().astype(int))
        missing = sorted(EXPECTED_HORIZONS - horizons)
        extra = sorted(horizon for horizon in horizons if horizon not in EXPECTED_HORIZONS)
        if missing or extra:
            rows.append(
                {
                    "run_ts": iso_z(run_dt),
                    "present_horizons": len(horizons & EXPECTED_HORIZONS),
                    "missing_count": len(missing),
                    "missing_horizons": ",".join(str(value) for value in missing),
                    "extra_horizons": ",".join(str(value) for value in extra),
                }
            )
    return pd.DataFrame(rows)


def duplicate_knmi_rows(knmi: pd.DataFrame) -> pd.DataFrame:
    if knmi.empty:
        return pd.DataFrame()
    dupes = (
        knmi.groupby(["run_ms", "target_ms"], dropna=False)
        .size()
        .reset_index(name="row_count")
        .query("row_count > 1")
        .copy()
    )
    if dupes.empty:
        return dupes
    dupes["run_ts"] = dupes["run_ms"].map(iso_z)
    dupes["target_ts"] = dupes["target_ms"].map(iso_z)
    return dupes[["run_ts", "target_ts", "row_count"]]


def print_schema(schema: Schema, mapping: TableMap) -> None:
    print(f"\nSchema: {schema.table}")
    if not schema.columns:
        print("  table not found or has no columns")
        return
    print("  columns: " + ", ".join(schema.columns))
    print(
        "  mapped: "
        f"site={mapping.site}, model={mapping.model}, run_ts={mapping.run_ts}, "
        f"fetched_ts={mapping.fetched_ts}, target_ts={mapping.target_ts}, "
        f"horizon_hr={mapping.horizon_hr}, wind_speed={mapping.wind_speed}, "
        f"wind_gust={mapping.wind_gust}, wind_dir={mapping.wind_dir}"
    )


def print_coverage(
    coverage: pd.DataFrame,
    missing: pd.DataFrame,
    duplicates: pd.DataFrame,
    knmi_run_count: int,
    min_run_count: int,
) -> None:
    print("\nArchive coverage")
    print("================")
    for row in coverage.itertuples(index=False):
        print(f"{row.metric}: {row.value}")
    if missing.empty:
        print("missing KNMI horizons by run_ts: none; every run has horizons 0-60")
    else:
        print("missing KNMI horizons by run_ts:")
        print(missing.to_string(index=False))
    if duplicates.empty:
        print("duplicate KNMI rows by run_ts,target_ts: none")
    else:
        print("duplicate KNMI rows by run_ts,target_ts:")
        print(duplicates.to_string(index=False))
    if knmi_run_count < min_run_count:
        print(
            f"\nWARNING: Only {knmi_run_count} KNMI runs are archived. "
            "This is useful for plumbing verification but may still be limited for replacement validation."
        )
    else:
        print(f"KNMI run count meets replacement-validation minimum: {knmi_run_count} >= {min_run_count}")


def print_vintage_note() -> None:
    print("\nVintage semantics")
    print("=================")
    print("KNMI has authoritative run_ts from the HARMONIE filename.")
    print("Windsurfice may only have fallback fetch-time run_ts metadata, not true HARMONIE run_ts.")
    print("Therefore, exact run_ts matching may not be possible.")
    print(
        "Differences in latest-vs-latest or target-only comparisons may reflect different forecast "
        "vintages rather than extraction errors."
    )


def print_metrics(name: str, comparison: pd.DataFrame, windsurfice_rows_considered: int) -> None:
    if comparison.empty:
        print(f"\n{name}: no comparison rows")
        return
    matched = int(comparison["matched"].sum())
    unmatched_knmi = int((~comparison["matched"]).sum())
    if name == "Target-only":
        unmatched_wf = int((comparison["match_kind"] == "right_only").sum())
        unmatched_knmi = int((comparison["match_kind"] == "left_only").sum())
    else:
        used = comparison.attrs.get("used_windsurfice_rowids", set())
        unmatched_wf = max(0, windsurfice_rows_considered - len(used))
    print(f"\n{name}")
    print("=" * len(name))
    print(f"matched rows: {matched}")
    print(f"unmatched KNMI rows: {unmatched_knmi}")
    print(f"unmatched Windsurfice rows: {unmatched_wf}")
    if not matched:
        print("metrics unavailable because no rows matched")
        return
    metrics = metric_summary(comparison).iloc[0]
    print(f"mean absolute speed difference: {metrics.speed_abs_mean_kt:.3f} kt")
    print(f"median absolute speed difference: {metrics.speed_abs_median_kt:.3f} kt")
    print(f"95th percentile absolute speed difference: {metrics.speed_abs_p95_kt:.3f} kt")
    print(f"max absolute speed difference: {metrics.speed_abs_max_kt:.3f} kt")
    print(f"mean circular direction difference: {metrics.dir_abs_mean_deg:.3f} deg")
    print(f"median circular direction difference: {metrics.dir_abs_median_deg:.3f} deg")
    print(f"95th percentile circular direction difference: {metrics.dir_abs_p95_deg:.3f} deg")
    print(f"max circular direction difference: {metrics.dir_abs_max_deg:.3f} deg")


def print_breakdowns(name: str, comparison: pd.DataFrame) -> None:
    if comparison.empty or not comparison["matched"].any():
        return
    print(f"\n{name} breakdowns")
    breakdowns = (
        ("By horizon band", "horizon_band"),
        ("By horizon_hr", "knmi_horizon_hr"),
        ("By KNMI run_ts", "knmi_run_ts"),
        ("By KNMI wind-speed band", "wind_speed_band"),
        ("By KNMI direction sector", "direction_sector"),
    )
    if name != "Closest-vintage":
        breakdowns = (
            ("By KNMI run_ts", "knmi_run_ts"),
            ("By horizon_hr", "knmi_horizon_hr"),
            ("By horizon band", "horizon_band"),
            ("By KNMI wind-speed band", "wind_speed_band"),
            ("By KNMI direction sector", "direction_sector"),
        )
    for label, column in breakdowns:
        summary = metric_summary(comparison, column)
        if summary.empty:
            continue
        display_cols = [
            column,
            "n",
            "speed_abs_mean_kt",
            "speed_abs_median_kt",
            "speed_abs_p95_kt",
            "speed_abs_max_kt",
            "dir_abs_mean_deg",
            "dir_abs_median_deg",
            "dir_abs_p95_deg",
            "dir_abs_max_deg",
        ]
        print(f"\n{label}:")
        print(summary[display_cols].round(3).to_string(index=False))


def closest_unmatched_rows(closest_vintage: pd.DataFrame) -> pd.DataFrame:
    if closest_vintage.empty:
        return pd.DataFrame()
    unmatched = closest_vintage[~closest_vintage["matched"]].copy()
    if unmatched.empty:
        return unmatched
    columns = [
        "target_ts",
        "knmi_run_ts",
        "knmi_fetched_ts",
        "knmi_horizon_hr",
        "horizon_band",
        "knmi_wind_speed_kt",
        "knmi_wind_dir_deg",
        "closest_available_vintage_delta_minutes",
        "unmatched_reason",
    ]
    return unmatched[[column for column in columns if column in unmatched.columns]].sort_values(
        ["knmi_run_ts", "knmi_horizon_hr", "target_ts"]
    )


def print_unmatched_diagnostics(closest_vintage: pd.DataFrame) -> pd.DataFrame:
    unmatched = closest_unmatched_rows(closest_vintage)
    print("\nClosest-vintage unmatched KNMI diagnostics")
    print("==========================================")
    print(f"unmatched KNMI rows: {len(unmatched)}")
    if unmatched.empty:
        print("No unmatched closest-vintage KNMI rows.")
        return unmatched

    print(f"min unmatched target_ts: {unmatched['target_ts'].min()}")
    print(f"max unmatched target_ts: {unmatched['target_ts'].max()}")
    if "unmatched_reason" in unmatched.columns:
        print("\nUnmatched reasons:")
        print(unmatched["unmatched_reason"].fillna("unknown").value_counts().rename_axis("reason").reset_index(name="n").to_string(index=False))

    for label, column in (
        ("By KNMI run_ts", "knmi_run_ts"),
        ("By horizon_hr", "knmi_horizon_hr"),
        ("By horizon band", "horizon_band"),
    ):
        if column not in unmatched.columns:
            continue
        summary = unmatched[column].fillna("missing").value_counts().sort_index().rename_axis(column).reset_index(name="n")
        print(f"\n{label}:")
        print(summary.to_string(index=False))

    run_counts = unmatched["knmi_run_ts"].value_counts() if "knmi_run_ts" in unmatched.columns else pd.Series(dtype=int)
    horizon_counts = unmatched["knmi_horizon_hr"].value_counts() if "knmi_horizon_hr" in unmatched.columns else pd.Series(dtype=int)
    if not run_counts.empty:
        top_run = run_counts.index[0]
        top_run_n = int(run_counts.iloc[0])
        print(
            f"\nConcentration: top unmatched run {top_run} has {top_run_n}/{len(unmatched)} "
            f"rows ({top_run_n / len(unmatched) * 100.0:.1f}%)."
        )
    if not horizon_counts.empty:
        top_horizon = horizon_counts.index[0]
        top_horizon_n = int(horizon_counts.iloc[0])
        print(
            f"Concentration: top unmatched horizon {top_horizon} has {top_horizon_n}/{len(unmatched)} "
            f"rows ({top_horizon_n / len(unmatched) * 100.0:.1f}%)."
        )
    print(
        "Likely explanation: unmatched rows do not have a Windsurfice row with the same target_ts "
        "inside the configured max-vintage-delta tolerance."
    )
    return unmatched


def unit_assumption_note(conn: sqlite3.Connection, schemas: dict[str, Schema], site: str) -> None:
    print("\nSpeed units")
    print("===========")
    print(
        "Comparison unit: knots. KNMI shadow wind_speed is produced from canonical "
        "wind_speed_10m_mps converted with KNOTS_PER_MPS."
    )
    schema = schemas.get("harmonie_knmi_features")
    if not schema or not {"wind_speed_10m_mps", "wind_speed_10m_knots", "site"}.issubset(schema.columns):
        print("Canonical KNMI unit sanity check unavailable; required columns are missing.")
        return
    query = """
        SELECT wind_speed_10m_mps, wind_speed_10m_knots
        FROM harmonie_knmi_features
        WHERE site = ?
          AND wind_speed_10m_mps IS NOT NULL
          AND wind_speed_10m_knots IS NOT NULL
        LIMIT 500
    """
    sample = pd.read_sql_query(query, conn, params=(site,))
    if sample.empty:
        print("Canonical KNMI unit sanity check unavailable; no non-null sample rows.")
        return
    expected = sample["wind_speed_10m_mps"].astype(float) * KNOTS_PER_MPS
    mae = (expected - sample["wind_speed_10m_knots"].astype(float)).abs().mean()
    print(f"Canonical KNMI m/s-to-knots sanity MAE: {mae:.6f} kt over {len(sample)} sampled rows.")


def wind_field_storage_diagnostics(
    conn: sqlite3.Connection,
    schemas: dict[str, Schema],
    site: str,
    model: str,
) -> dict[str, object]:
    print("\nWindsurfice max-wind field storage")
    print("==================================")
    result: dict[str, object] = {"knmi_persisted_max_available": False}
    forecasts_schema = schemas["forecasts"]
    print(f"forecasts wind-speed columns: wind_speed={forecasts_schema.has('wind_speed')}, wind_gust={forecasts_schema.has('wind_gust')}, wind_dir={forecasts_schema.has('wind_dir')}")
    if forecasts_schema.has("wind_speed"):
        print("Average forecast wind speed: forecasts.wind_speed, populated from payload WindForecastAvr.")
    if forecasts_schema.has("wind_gust"):
        stats = conn.execute(
            """
            SELECT COUNT(*), COUNT(wind_gust), MIN(wind_gust), MAX(wind_gust), AVG(wind_gust)
            FROM forecasts
            WHERE site = ? AND model = ?
            """,
            (site, model),
        ).fetchone()
        print(
            "Maximum/gust forecast wind: forecasts.wind_gust, populated from payload WindForecastMax "
            f"({stats[1]}/{stats[0]} non-null, range {stats[2]}..{stats[3]}, mean {stats[4]:.3f} kt)."
        )
    print("Minimum forecast wind speed: no dedicated forecasts table column is present.")
    print("Direction: forecasts.wind_dir, populated from payload WindDirection.")

    payload_keys: set[str] = set()
    for (payload_raw,) in conn.execute(
        """
        SELECT payload
        FROM forecasts
        WHERE site = ? AND model = ? AND payload IS NOT NULL
        ORDER BY run_ts DESC, target_ts ASC
        LIMIT 20
        """,
        (site, model),
    ).fetchall():
        try:
            payload = json.loads(payload_raw)
        except Exception:
            continue
        payload_keys.update(str(key) for key in payload)
    wind_payload_keys = sorted(
        key for key in payload_keys if any(token in key.lower() for token in ("wind", "gust", "forecast"))
    )
    print("Sample Windsurfice wind-related payload keys: " + (", ".join(wind_payload_keys) if wind_payload_keys else "none"))

    shadow_schema = schemas["knmi_forecasts_shadow"]
    if shadow_schema.has("wind_gust"):
        shadow_gust_count = conn.execute(
            """
            SELECT COUNT(*), COUNT(wind_gust)
            FROM knmi_forecasts_shadow
            WHERE site = ? AND model = ?
            """,
            (site, model),
        ).fetchone()
        status = "present" if shadow_gust_count[1] else "unpopulated"
        print(
            f"KNMI shadow max/gust equivalent: {status}; knmi_forecasts_shadow.wind_gust has "
            f"{shadow_gust_count[1]}/{shadow_gust_count[0]} non-null rows."
        )
    canonical_gust_like = [
        column
        for column in schemas["harmonie_knmi_features"].columns
        if any(token in column.lower() for token in ("gust", "max_wind", "wind_gust", "wind_max"))
    ]
    print(
        "Canonical KNMI gust/max columns: "
        + (", ".join(canonical_gust_like) if canonical_gust_like else "none currently persisted")
    )
    return result


def diagnose_no_overlap(knmi: pd.DataFrame, windsurfice: pd.DataFrame, required_missing: list[str]) -> None:
    start, end = overlap_range(knmi, windsurfice)
    if start is not None and end is not None and not required_missing:
        return
    print("\nNo-overlap / limited-overlap diagnostics")
    print("========================================")
    if knmi.empty:
        print("- no KNMI rows for the selected site/model")
    if windsurfice.empty:
        print("- no Windsurfice rows for the selected site/model")
    if not knmi.empty and not windsurfice.empty and (start is None or end is None):
        print("- no overlapping target_ts range")
    if knmi["target_dt"].isna().all() or windsurfice["target_dt"].isna().all():
        print("- target timestamps were not parsed")
    for item in required_missing:
        print(f"- missing required column mapping: {item}")


def site_point_from_db(conn: sqlite3.Connection, site: str) -> SitePoint | None:
    row = conn.execute(
        """
        SELECT site_lat, site_lon
        FROM harmonie_knmi_features
        WHERE site = ? AND site_lat IS NOT NULL AND site_lon IS NOT NULL
        ORDER BY run_ts DESC
        LIMIT 1
        """,
        (site,),
    ).fetchone()
    if row is None:
        return DEFAULT_SITE_POINTS.get(site)
    return SitePoint(site=site, lat=float(row[0]), lon=float(row[1]))


def discover_raw_candidates(raw_grib: Path | None, raw_tar: Path | None) -> tuple[Path | None, Path | None, str | None]:
    if raw_grib is not None:
        return raw_grib, None, None if raw_grib.exists() else f"Raw GRIB file not found: {raw_grib}"
    if raw_tar is not None:
        return None, raw_tar, None if raw_tar.exists() else f"Raw tar file not found: {raw_tar}"

    raw_root = Path("data/raw/knmi")
    gribs = sorted(
        [path for pattern in ("*.grib", "*.grb", "*.grib2") for path in raw_root.rglob(pattern)],
        key=lambda path: path.stat().st_mtime,
    )
    if gribs:
        return gribs[-1], None, None
    tars = sorted(raw_root.rglob("*.tar"), key=lambda path: path.stat().st_mtime)
    if tars:
        return None, tars[-1], None
    return None, None, None


def grib_ls_metadata(grib_path: Path) -> str:
    try:
        completed = subprocess.run(
            [
                "grib_ls",
                "-p",
                "shortName,indicatorOfParameter,level,typeOfLevel,units,name",
                str(grib_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        return "grib_ls is not installed."
    except Exception as exc:
        return f"grib_ls failed: {exc}"
    if completed.returncode != 0:
        return f"grib_ls failed with exit {completed.returncode}: {completed.stderr.strip()}"
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    interesting = [
        line
        for line in lines
        if " 162 " in f" {line} " or " 163 " in f" {line} " or " 33 " in f" {line} " or " 34 " in f" {line} "
    ]
    return "\n".join(interesting[:30] or lines[:30])


def grib_paths_from_tar(raw_tar: Path, temp_dir: Path) -> list[Path]:
    paths: list[Path] = []
    with tarfile.open(raw_tar, "r") as tar:
        members = [member for member in tar.getmembers() if member.isfile()]
        for member in members:
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            out_path = temp_dir / Path(member.name).name
            with out_path.open("wb") as handle:
                handle.write(extracted.read())
            paths.append(out_path)
    return sorted(paths)


def closest_windsurfice_max(
    windsurfice: pd.DataFrame,
    target_dt: pd.Timestamp,
    reference_dt: pd.Timestamp,
    max_delta_minutes: float,
) -> tuple[float | None, float | None]:
    target_ms_value = timestamp_ms(target_dt)
    if target_ms_value is None:
        return None, None
    candidates = windsurfice[
        (windsurfice["target_ms"].astype("Int64") == target_ms_value) & windsurfice["wind_gust"].notna()
    ].copy()
    if candidates.empty:
        return None, None
    vintage_col, _ = windsurfice_vintage_column(candidates)
    if vintage_col is None:
        return None, None
    candidates = candidates.dropna(subset=[vintage_col])
    if candidates.empty:
        return None, None
    candidates["_delta_min"] = (candidates[vintage_col] - reference_dt).abs().dt.total_seconds() / 60.0
    candidates = candidates[candidates["_delta_min"] <= float(max_delta_minutes)]
    if candidates.empty:
        return None, None
    best = candidates.sort_values(["_delta_min", vintage_col, "source_rowid"]).iloc[0]
    return float(best["wind_gust"]), float(best["_delta_min"])


def candidate_parameter_value(grib_path: Path, parameter: int, site_point: SitePoint) -> tuple[float | None, str]:
    try:
        ds = open_grib_parameter(grib_path, parameter=parameter, level=10)
    except Exception as exc:
        return None, str(exc)
    try:
        value, _, _ = nearest_value(ds, site_point)
        units = str(next(iter(ds.data_vars.values())).attrs.get("units", "unknown")) if ds.data_vars else "unknown"
        short_name = str(next(iter(ds.data_vars.keys()))) if ds.data_vars else f"param_{parameter}"
        return float(value), f"{short_name}; units={units}"
    except Exception as exc:
        return None, str(exc)
    finally:
        ds.close()


def max_wind_candidate_summary(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame()
    rows = []
    for (parameter, unit_assumption, metadata), group in records.groupby(
        ["candidate_parameter", "unit_assumption", "metadata"], dropna=False
    ):
        diffs = group["candidate_value_kt"] - group["windsurfice_wind_gust_kt"]
        abs_diff = diffs.abs()
        rows.append(
            {
                "candidate_field": f"indicatorOfParameter={parameter}",
                "level": "10 m",
                "metadata": metadata,
                "unit_assumption": unit_assumption,
                "matched_rows": int(len(group)),
                "mae_kt": abs_diff.mean(),
                "median_abs_error_kt": abs_diff.median(),
                "p95_abs_error_kt": abs_diff.quantile(0.95),
                "max_abs_error_kt": abs_diff.max(),
                "correlation": group["candidate_value_kt"].corr(group["windsurfice_wind_gust_kt"])
                if len(group) > 1
                else pd.NA,
            }
        )
    return pd.DataFrame(rows).sort_values(["mae_kt", "matched_rows"], ascending=[True, False])


def validate_max_wind_from_raw(
    conn: sqlite3.Connection,
    windsurfice: pd.DataFrame,
    site: str,
    max_delta_minutes: float,
    raw_grib: Path | None,
    raw_tar: Path | None,
) -> tuple[pd.DataFrame, str]:
    selected_grib, selected_tar, raw_error = discover_raw_candidates(raw_grib, raw_tar)
    if raw_error:
        return pd.DataFrame(), raw_error
    if selected_grib is None and selected_tar is None:
        return (
            pd.DataFrame(),
            "Cannot validate gust/max field because no raw KNMI GRIB file is available. "
            "Re-run the KNMI extractor with --keep-raw for one cycle, then run this diagnostic again.\n"
            "Suggested command: python3 scripts/knmi_extract_latest_to_db.py --keep-raw",
        )

    site_point = site_point_from_db(conn, site)
    if site_point is None:
        return pd.DataFrame(), f"No site coordinates found for site={site!r}."

    records: list[dict[str, object]] = []

    def inspect_paths(paths: list[Path], source_label: str) -> str:
        metadata_text = grib_ls_metadata(paths[0]) if paths else "no GRIB paths"
        for grib_path in paths:
            try:
                run_dt, horizon_hr, target_dt = parse_run_and_horizon(grib_path.name)
            except Exception:
                continue
            windsurfice_max, vintage_delta = closest_windsurfice_max(
                windsurfice,
                pd.Timestamp(target_dt),
                pd.Timestamp(run_dt),
                max_delta_minutes=max_delta_minutes,
            )
            if windsurfice_max is None:
                continue
            raw_values: dict[int, tuple[float, str]] = {}
            for parameter in MAX_WIND_CANDIDATE_PARAMETERS:
                raw_value, metadata = candidate_parameter_value(grib_path, parameter, site_point)
                if raw_value is None:
                    continue
                raw_values[parameter] = (float(raw_value), metadata)
                for unit_assumption, candidate_value in (
                    ("raw already knots", raw_value),
                    ("m/s converted to knots", raw_value * KNOTS_PER_MPS),
                ):
                    records.append(
                        {
                            "source": source_label,
                            "grib_path": str(grib_path),
                            "candidate_parameter": parameter,
                            "metadata": metadata,
                            "unit_assumption": unit_assumption,
                            "run_ts": iso_z(run_dt),
                            "target_ts": iso_z(target_dt),
                            "horizon_hr": int(horizon_hr),
                            "candidate_value_kt": float(candidate_value),
                            "windsurfice_wind_gust_kt": float(windsurfice_max),
                            "vintage_delta_minutes": vintage_delta,
                        }
                    )
            if all(parameter in raw_values for parameter in MAX_WIND_CANDIDATE_PARAMETERS):
                u_value, u_metadata = raw_values[GUST_U_WIND_PARAMETER]
                v_value, v_metadata = raw_values[GUST_V_WIND_PARAMETER]
                magnitude = (u_value**2 + v_value**2) ** 0.5
                metadata = f"vector magnitude of 162/163; {u_metadata}; {v_metadata}"
                for unit_assumption, candidate_value in (
                    ("raw already knots", magnitude),
                    ("m/s converted to knots", magnitude * KNOTS_PER_MPS),
                    ("knots converted to m/s", magnitude / KNOTS_PER_MPS),
                ):
                    records.append(
                        {
                            "source": source_label,
                            "grib_path": str(grib_path),
                            "candidate_parameter": "162+163",
                            "metadata": metadata,
                            "unit_assumption": unit_assumption,
                            "run_ts": iso_z(run_dt),
                            "target_ts": iso_z(target_dt),
                            "horizon_hr": int(horizon_hr),
                            "candidate_value_kt": float(candidate_value),
                            "windsurfice_wind_gust_kt": float(windsurfice_max),
                            "vintage_delta_minutes": vintage_delta,
                        }
                    )
        return metadata_text

    if selected_grib is not None:
        metadata_text = inspect_paths([selected_grib], str(selected_grib))
    else:
        with TemporaryDirectory() as tmp:
            paths = grib_paths_from_tar(selected_tar, Path(tmp))
            metadata_text = inspect_paths(paths, str(selected_tar))

    print("\nRaw GRIB metadata sample")
    print("========================")
    print(metadata_text)
    summary = max_wind_candidate_summary(pd.DataFrame(records))
    if summary.empty:
        return summary, "No raw candidate 162/163 rows could be matched to Windsurfice WindForecastMax."
    return summary, "Raw candidate comparison completed."


def print_max_wind_validation(summary: pd.DataFrame, message: str) -> dict[str, object]:
    print("\nKNMI max/gust validation")
    print("========================")
    print(message)
    result = {"status": "unresolved", "message": message}
    if summary.empty:
        print("Max/gust field status: unresolved.")
        return result
    print(summary.round(3).to_string(index=False))
    best = summary.iloc[0]
    if int(best["matched_rows"]) >= 10 and float(best["mae_kt"]) <= 2.0:
        result["status"] = "candidate"
        print(
            f"Best candidate is {best['candidate_field']} ({best['unit_assumption']}) with "
            f"MAE {best['mae_kt']:.3f} kt. Treat as a candidate until confirmed over more runs."
        )
    else:
        print("No candidate clearly matches Windsurfice WindForecastMax yet.")
    return result


def persisted_max_wind_summary(closest_vintage: pd.DataFrame) -> pd.DataFrame:
    if closest_vintage.empty:
        return pd.DataFrame()
    required = {"knmi_wind_gust_kt", "windsurfice_wind_gust_kt", "matched"}
    if not required.issubset(closest_vintage.columns):
        return pd.DataFrame()
    matched = closest_vintage[
        closest_vintage["matched"]
        & closest_vintage["knmi_wind_gust_kt"].notna()
        & closest_vintage["windsurfice_wind_gust_kt"].notna()
    ].copy()
    if matched.empty:
        return pd.DataFrame()
    diff = matched["knmi_wind_gust_kt"].astype(float) - matched["windsurfice_wind_gust_kt"].astype(float)
    abs_diff = diff.abs()
    return pd.DataFrame(
        [
            {
                "candidate_field": "knmi_forecasts_shadow.wind_gust",
                "matched_rows": int(len(matched)),
                "mae_kt": abs_diff.mean(),
                "median_abs_error_kt": abs_diff.median(),
                "p95_abs_error_kt": abs_diff.quantile(0.95),
                "max_abs_error_kt": abs_diff.max(),
                "correlation": matched["knmi_wind_gust_kt"].astype(float).corr(
                    matched["windsurfice_wind_gust_kt"].astype(float)
                )
                if len(matched) > 1
                else pd.NA,
            }
        ]
    )


def print_persisted_max_wind_validation(summary: pd.DataFrame) -> dict[str, object]:
    print("\nPersisted KNMI max/gust validation")
    print("==================================")
    if summary.empty:
        print("No persisted KNMI wind_gust rows are available in closest-vintage matches.")
        return {"status": "unresolved", "message": "No persisted KNMI wind_gust rows available."}
    print(summary.round(3).to_string(index=False))
    best = summary.iloc[0]
    if int(best["matched_rows"]) >= 20 and float(best["mae_kt"]) <= 2.0:
        print("Persisted KNMI wind_gust validates against Windsurfice WindForecastMax for the available matched rows.")
        return {"status": "validated", "message": "Persisted KNMI wind_gust validates against Windsurfice."}
    print("Persisted KNMI wind_gust is present, but the available evidence is not strong enough yet.")
    return {"status": "partially validated", "message": "Persisted KNMI wind_gust needs more evidence."}


def print_final_interpretation(
    coverage: pd.DataFrame,
    closest_vintage: pd.DataFrame,
    max_wind_result: dict[str, object],
) -> None:
    coverage_map = dict(zip(coverage["metric"], coverage["value"]))
    missing_expected = coverage_map.get("knmi_shadow_rows") != coverage_map.get("expected_knmi_rows_distinct_runs_x_61")
    closest_metrics = metric_summary(closest_vintage).iloc[0] if not closest_vintage.empty and closest_vintage["matched"].any() else None

    print("\nValidation interpretation")
    print("=========================")
    if closest_metrics is not None and float(closest_metrics["speed_abs_mean_kt"]) < 0.75:
        print("10 m average wind speed/direction: validated for plumbing and extraction against closest Windsurfice vintages.")
    else:
        print("10 m average wind speed/direction: partially validated; inspect closest-vintage errors before relying on it.")
    print("Forecast vintage handling: closest-vintage comparison is strongly preferred over target-only comparison.")
    print(f"Max/gust field: {max_wind_result.get('status', 'unresolved')}.")
    print("Archive completeness: " + ("incomplete" if missing_expected else "complete for archived KNMI runs."))
    print(
        "Observation joins: exact joins unavailable, 30-minute joins available "
        f"({coverage_map.get('knmi_distinct_targets_observation_within_30_min')} distinct KNMI targets)."
    )
    if max_wind_result.get("status") == "validated":
        print(
            "Recommendation: continue the KNMI shadow fetch; do not replace Windsurfice until this gust mapping "
            "has been confirmed over more archived days."
        )
    else:
        print(
            "Recommendation: continue the KNMI shadow fetch; do not replace Windsurfice until max/gust is resolved "
            "and more days are accumulated."
        )


def write_outputs(
    output_dir: Path,
    coverage: pd.DataFrame,
    target_only: pd.DataFrame,
    closest_vintage: pd.DataFrame,
    unmatched_closest: pd.DataFrame,
    max_wind_summary: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(output_dir / "knmi_validation_archive_coverage.csv", index=False)
    if not target_only.empty:
        target_only.to_csv(output_dir / "knmi_validation_matches_target_only.csv", index=False)
    if not closest_vintage.empty:
        closest_vintage.to_csv(output_dir / "knmi_validation_matches_closest_vintage.csv", index=False)
    if not unmatched_closest.empty:
        unmatched_closest.to_csv(output_dir / "knmi_validation_unmatched_closest_vintage.csv", index=False)
    if not max_wind_summary.empty:
        max_wind_summary.to_csv(output_dir / "knmi_validation_max_wind_candidates.csv", index=False)

    summaries = []
    for frame in (target_only, closest_vintage):
        if frame.empty:
            continue
        by_run = metric_summary(frame, "knmi_run_ts")
        if not by_run.empty:
            by_run.insert(0, "mode", frame["mode"].iloc[0])
            summaries.append(by_run)
    if summaries:
        pd.concat(summaries, ignore_index=True).to_csv(output_dir / "knmi_validation_summary_by_run.csv", index=False)

    for filename, column in (
        ("knmi_validation_summary_by_horizon.csv", "knmi_horizon_hr"),
        ("knmi_validation_summary_by_horizon_band.csv", "horizon_band"),
        ("knmi_validation_summary_by_speed_band.csv", "wind_speed_band"),
        ("knmi_validation_summary_by_direction_sector.csv", "direction_sector"),
    ):
        parts = []
        for frame in (target_only, closest_vintage):
            if frame.empty:
                continue
            summary = metric_summary(frame, column)
            if not summary.empty:
                summary.insert(0, "mode", frame["mode"].iloc[0])
                parts.append(summary)
        if parts:
            pd.concat(parts, ignore_index=True).to_csv(output_dir / filename, index=False)

    closest_outputs = (
        ("knmi_validation_closest_by_run.csv", "knmi_run_ts"),
        ("knmi_validation_closest_by_horizon.csv", "knmi_horizon_hr"),
        ("knmi_validation_closest_by_horizon_band.csv", "horizon_band"),
        ("knmi_validation_closest_by_speed_band.csv", "wind_speed_band"),
        ("knmi_validation_closest_by_direction_sector.csv", "direction_sector"),
    )
    if not closest_vintage.empty:
        for filename, column in closest_outputs:
            summary = metric_summary(closest_vintage, column)
            if not summary.empty:
                summary.to_csv(output_dir / filename, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate direct KNMI HARMONIE shadow forecasts against Windsurfice HARMONIE."
    )
    parser.add_argument("--db", type=Path, default=Path("data/wind_data_all_sites.db"))
    parser.add_argument("--site", default="valkenburgsemeer")
    parser.add_argument("--model", default="HARMONIE")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/knmi_validation"))
    parser.add_argument("--max-vintage-delta-minutes", type=float, default=90.0)
    parser.add_argument("--min-run-count", type=int, default=12)
    parser.add_argument("--write-csv", action="store_true")
    parser.add_argument("--mode", choices=("target-only", "closest-vintage", "all"), default="all")
    parser.add_argument("--validate-max-wind", action="store_true")
    parser.add_argument("--raw-grib", type=Path)
    parser.add_argument("--raw-tar", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conn, sqlite_open_mode = connect_read_only(args.db)
    try:
        schemas = {table: table_info(conn, table) for table in TABLES_TO_INSPECT}
        mappings = {table: map_table(schema) for table, schema in schemas.items()}

        print("KNMI shadow archive validation")
        print("==============================")
        print(f"Database: {args.db}")
        print(f"SQLite open mode: read-only {sqlite_open_mode}")
        if sqlite_open_mode == "immutable=1":
            print("Note: immutable read-only mode was used because normal mode=ro failed during schema inspection.")
        print(f"Site: {args.site}")
        print(f"Model: {args.model}")
        for table in TABLES_TO_INSPECT:
            print_schema(schemas[table], mappings[table])

        required_missing = []
        for table, required in (
            ("knmi_forecasts_shadow", ("site", "target_ts", "run_ts", "fetched_ts", "wind_speed", "wind_dir")),
            ("forecasts", ("site", "target_ts", "wind_speed", "wind_dir")),
        ):
            mapping = mappings[table]
            for attr in required:
                if getattr(mapping, attr) is None:
                    required_missing.append(f"{table}.{attr}")

        knmi = select_table_frame(
            conn,
            "knmi_forecasts_shadow",
            mappings["knmi_forecasts_shadow"],
            site=args.site,
            model=args.model,
        )
        windsurfice = select_table_frame(
            conn,
            "forecasts",
            mappings["forecasts"],
            site=args.site,
            model=args.model,
        )
        observations = load_observations(conn, schemas["observations"], args.site)
        coverage = archive_coverage(knmi, windsurfice, observations)
        missing = missing_horizons_by_run(knmi)
        duplicates = duplicate_knmi_rows(knmi)
        knmi_run_count = int(knmi["run_ms"].nunique(dropna=True))

        print_coverage(coverage, missing, duplicates, knmi_run_count, args.min_run_count)
        print_vintage_note()
        unit_assumption_note(conn, schemas, args.site)
        wind_field_storage_diagnostics(conn, schemas, args.site, args.model)
        diagnose_no_overlap(knmi, windsurfice, required_missing)

        target_only = pd.DataFrame()
        closest_vintage = pd.DataFrame()
        unmatched_closest = pd.DataFrame()
        max_wind_summary = pd.DataFrame()
        persisted_max_wind = pd.DataFrame()
        max_wind_result: dict[str, object] = {
            "status": "unresolved",
            "message": "No KNMI max/gust equivalent is currently persisted in the shadow archive.",
        }

        if args.mode in ("target-only", "all"):
            print("\nTarget-only selection: latest KNMI row per target_ts versus latest Windsurfice row per target_ts.")
            print("This mode is useful but can mix forecast vintages.")
            target_only = compare_target_only(knmi, windsurfice)
            print_metrics("Target-only", target_only, len(latest_per_target(windsurfice)))
            print_breakdowns("Target-only", target_only)

        if args.mode in ("closest-vintage", "all"):
            closest_vintage, vintage_note = compare_closest_vintage(
                knmi,
                windsurfice,
                max_delta_minutes=args.max_vintage_delta_minutes,
            )
            print("\nClosest-vintage selection:")
            print(f"Windsurfice timestamp used: {vintage_note}")
            if not closest_vintage.empty:
                print(f"KNMI reference timestamp used: {closest_vintage.attrs.get('knmi_reference_column')}")
            else:
                print("No suitable Windsurfice vintage timestamp exists; closest-vintage comparison skipped.")
            print_metrics("Closest-vintage", closest_vintage, len(windsurfice))
            print_breakdowns("Closest-vintage", closest_vintage)
            unmatched_closest = print_unmatched_diagnostics(closest_vintage)
            persisted_max_wind = persisted_max_wind_summary(closest_vintage)

        if args.validate_max_wind:
            persisted_result = print_persisted_max_wind_validation(persisted_max_wind)
            max_wind_summary, max_wind_message = validate_max_wind_from_raw(
                conn,
                windsurfice,
                args.site,
                max_delta_minutes=args.max_vintage_delta_minutes,
                raw_grib=args.raw_grib,
                raw_tar=args.raw_tar,
            )
            raw_result = print_max_wind_validation(max_wind_summary, max_wind_message)
            max_wind_result = persisted_result if persisted_result.get("status") == "validated" else raw_result
        else:
            max_wind_result = print_max_wind_validation(
                pd.DataFrame(),
                "Not attempted in this run. Pass --validate-max-wind with --raw-grib or --raw-tar, "
                "or keep one raw KNMI cycle with: python3 scripts/knmi_extract_latest_to_db.py --keep-raw",
            )

        print_final_interpretation(coverage, closest_vintage, max_wind_result)

        if args.write_csv:
            write_outputs(
                args.output_dir,
                coverage,
                target_only,
                closest_vintage,
                unmatched_closest,
                max_wind_summary,
            )
            print(f"\nCSV outputs written under: {args.output_dir}")
        else:
            print("\nCSV output disabled; pass --write-csv to save match and summary files.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
