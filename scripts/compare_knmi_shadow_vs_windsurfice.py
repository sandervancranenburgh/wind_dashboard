#!/usr/bin/env python3
"""Compare latest Windsurfice forecast snapshot with latest KNMI shadow snapshot."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from db_store import DB_FILENAME
from next_day_wind_model.knmi_harmonie import SHADOW_TABLE_NAME, create_knmi_forecasts_shadow_table


def iso_from_ms(value: int | float | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.to_datetime(int(value), unit="ms", utc=True).isoformat().replace("+00:00", "Z")


def circular_abs_diff(a: pd.Series, b: pd.Series) -> pd.Series:
    return ((a.astype(float) - b.astype(float) + 180.0) % 360.0 - 180.0).abs()


def latest_run_ts(conn: sqlite3.Connection, table: str, site: str, model: str) -> int | None:
    if table == "forecasts":
        row = conn.execute(
            "SELECT MAX(run_ts) FROM forecasts WHERE site = ? AND model = ?",
            (site, model),
        ).fetchone()
    else:
        row = conn.execute(
            f"SELECT MAX(run_ts) FROM {SHADOW_TABLE_NAME} WHERE site = ? AND model = ?",
            (site, model),
        ).fetchone()
    return None if row is None or row[0] is None else int(row[0])


def load_snapshot(conn: sqlite3.Connection, table: str, site: str, model: str, run_ts: int) -> pd.DataFrame:
    query = f"""
        SELECT
            site,
            model,
            run_ts,
            run_iso,
            fetched_ts,
            fetched_iso,
            target_ts,
            target_iso,
            horizon_hr,
            wind_speed,
            wind_dir
        FROM {table}
        WHERE site = ?
          AND model = ?
          AND run_ts = ?
        ORDER BY target_ts
    """
    return pd.read_sql_query(query, conn, params=(site, model, int(run_ts)))


def snapshot_metadata(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {
            "rows": 0,
            "run_ts": None,
            "fetched_min": None,
            "fetched_max": None,
            "target_min": None,
            "target_max": None,
        }
    return {
        "rows": len(frame),
        "run_ts": iso_from_ms(frame["run_ts"].iloc[0]),
        "fetched_min": iso_from_ms(frame["fetched_ts"].min()),
        "fetched_max": iso_from_ms(frame["fetched_ts"].max()),
        "target_min": iso_from_ms(frame["target_ts"].min()),
        "target_max": iso_from_ms(frame["target_ts"].max()),
    }


def compare_snapshots(windsurfice: pd.DataFrame, knmi: pd.DataFrame) -> pd.DataFrame:
    merged = windsurfice.merge(
        knmi,
        on="target_ts",
        how="outer",
        suffixes=("_windsurfice", "_knmi"),
        indicator=True,
    ).sort_values("target_ts")
    matched = merged["_merge"] == "both"
    merged["wind_speed_diff_knots"] = merged["wind_speed_knmi"] - merged["wind_speed_windsurfice"]
    merged["wind_dir_circular_diff"] = circular_abs_diff(merged["wind_dir_knmi"], merged["wind_dir_windsurfice"])
    merged.loc[~matched, ["wind_speed_diff_knots", "wind_dir_circular_diff"]] = pd.NA
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare latest Windsurfice forecast snapshot against latest KNMI shadow forecast snapshot.",
    )
    parser.add_argument("--db", type=Path, default=Path("data") / DB_FILENAME)
    parser.add_argument("--site", default="valkenburgsemeer")
    parser.add_argument("--model", default="HARMONIE")
    parser.add_argument("--sample-rows", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conn = sqlite3.connect(args.db)
    try:
        create_knmi_forecasts_shadow_table(conn)
        windsurfice_run_ts = latest_run_ts(conn, "forecasts", args.site, args.model)
        knmi_run_ts = latest_run_ts(conn, SHADOW_TABLE_NAME, args.site, args.model)
        if windsurfice_run_ts is None:
            raise SystemExit(f"No Windsurfice forecast rows found in forecasts for site={args.site!r}.")
        if knmi_run_ts is None:
            raise SystemExit(f"No KNMI shadow rows found in {SHADOW_TABLE_NAME} for site={args.site!r}.")

        windsurfice = load_snapshot(conn, "forecasts", args.site, args.model, windsurfice_run_ts)
        knmi = load_snapshot(conn, SHADOW_TABLE_NAME, args.site, args.model, knmi_run_ts)
    finally:
        conn.close()

    comparison = compare_snapshots(windsurfice, knmi)
    matched = comparison["_merge"] == "both"
    unmatched_windsurfice = comparison["_merge"] == "left_only"
    unmatched_knmi = comparison["_merge"] == "right_only"

    speed_abs = comparison.loc[matched, "wind_speed_diff_knots"].abs()
    dir_abs = comparison.loc[matched, "wind_dir_circular_diff"]

    print("\nKNMI shadow versus Windsurfice latest snapshot")
    print("==============================================")
    print(f"Database path: {args.db}")
    print(f"Site: {args.site}")
    print(f"Model: {args.model}")
    print("Unit conversion used: none in comparison; KNMI shadow wind_speed is already 10 m m/s converted to knots.")
    print("\nWindsurfice snapshot metadata:")
    for key, value in snapshot_metadata(windsurfice).items():
        print(f"  {key}: {value}")
    print("\nKNMI shadow snapshot metadata:")
    for key, value in snapshot_metadata(knmi).items():
        print(f"  {key}: {value}")
    print("\nMatch summary:")
    print(f"  Windsurfice rows: {len(windsurfice)}")
    print(f"  KNMI shadow rows: {len(knmi)}")
    print(f"  Matched rows by exact UTC target_ts: {int(matched.sum())}")
    print(f"  Unmatched Windsurfice rows: {int(unmatched_windsurfice.sum())}")
    print(f"  Unmatched KNMI rows: {int(unmatched_knmi.sum())}")
    if matched.any():
        print(f"  Mean absolute forecast_avg speed difference: {speed_abs.mean():.3f} knots")
        print(f"  Max absolute forecast_avg speed difference: {speed_abs.max():.3f} knots")
        print(f"  Mean circular direction difference: {dir_abs.mean():.3f} deg")
        print(f"  Max circular direction difference: {dir_abs.max():.3f} deg")
    else:
        print("  Difference metrics unavailable because no target timestamps matched.")

    if args.sample_rows > 0:
        sample = comparison.loc[
            matched,
            [
                "target_iso_windsurfice",
                "horizon_hr_windsurfice",
                "horizon_hr_knmi",
                "wind_speed_windsurfice",
                "wind_speed_knmi",
                "wind_speed_diff_knots",
                "wind_dir_windsurfice",
                "wind_dir_knmi",
                "wind_dir_circular_diff",
            ],
        ].head(args.sample_rows)
        print("\nMatched row sample:")
        print(sample.to_string(index=False) if not sample.empty else "No matched rows.")


if __name__ == "__main__":
    main()
