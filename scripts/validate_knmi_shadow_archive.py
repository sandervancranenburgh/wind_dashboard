#!/usr/bin/env python3
"""Validate the KNMI HARMONIE shadow archive against Windsurfice HARMONIE."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from next_day_wind_model.knmi_harmonie import KNOTS_PER_MPS


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
    for column in ("site", "model", "run_ts_raw", "fetched_ts_raw", "target_ts_raw", "horizon_hr", "wind_speed", "wind_dir"):
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
        if candidates is not None and not pd.isna(ref_dt):
            candidate = candidates.copy()
            candidate["_delta_min"] = (candidate[wf_vintage_col] - ref_dt).abs().dt.total_seconds() / 60.0
            candidate = candidate[candidate["_delta_min"] <= max_delta]
            if not candidate.empty:
                candidate = candidate.sort_values(["_delta_min", wf_vintage_col, "source_rowid"])
                best = candidate.iloc[0]
                delta_minutes = float(best["_delta_min"])
                used_wf_rowids.add(int(best["source_rowid"]))

        matched = best is not None
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
                "knmi_wind_dir_deg": getattr(knmi_row, "wind_dir"),
                "windsurfice_wind_dir_deg": best["wind_dir"] if matched else pd.NA,
                "vintage_delta_minutes": delta_minutes,
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
        f"horizon_hr={mapping.horizon_hr}, wind_speed={mapping.wind_speed}, wind_dir={mapping.wind_dir}"
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
    for label, column in (
        ("By KNMI run_ts", "knmi_run_ts"),
        ("By horizon_hr", "knmi_horizon_hr"),
        ("By horizon band", "horizon_band"),
        ("By KNMI wind-speed band", "wind_speed_band"),
        ("By KNMI direction sector", "direction_sector"),
    ):
        summary = metric_summary(comparison, column)
        if summary.empty:
            continue
        display_cols = [
            column,
            "matched_rows",
            "speed_abs_mean_kt",
            "speed_abs_p95_kt",
            "dir_abs_mean_deg",
            "dir_abs_p95_deg",
        ]
        print(f"\n{label}:")
        print(summary[display_cols].round(3).to_string(index=False))


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


def write_outputs(
    output_dir: Path,
    coverage: pd.DataFrame,
    target_only: pd.DataFrame,
    closest_vintage: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(output_dir / "knmi_validation_archive_coverage.csv", index=False)
    if not target_only.empty:
        target_only.to_csv(output_dir / "knmi_validation_matches_target_only.csv", index=False)
    if not closest_vintage.empty:
        closest_vintage.to_csv(output_dir / "knmi_validation_matches_closest_vintage.csv", index=False)

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
        diagnose_no_overlap(knmi, windsurfice, required_missing)

        target_only = pd.DataFrame()
        closest_vintage = pd.DataFrame()

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

        if args.write_csv:
            write_outputs(args.output_dir, coverage, target_only, closest_vintage)
            print(f"\nCSV outputs written under: {args.output_dir}")
        else:
            print("\nCSV output disabled; pass --write-csv to save match and summary files.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
