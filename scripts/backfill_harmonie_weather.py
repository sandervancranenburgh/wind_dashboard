#!/usr/bin/env python3
"""Development-only weather-code backfill for an isolated SQLite database."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from db_store import init_db
from next_day_wind_model.knmi_harmonie import (
    SitePoint,
    extract_tar_features,
    upsert_harmonie_knmi_features,
)
from next_day_wind_model.weather_conditions import derive_windsurfice_weather


PRODUCTION_NAMES = {"wind_data.db", "wind_data_all_sites.db"}


def guarded_development_db(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path.name in PRODUCTION_NAMES or not path.is_relative_to(Path("/tmp")):
        raise ValueError(
            "Backfill refuses production database names and paths; use an explicit database under /tmp."
        )
    return path


def backfill_forecasts(conn: sqlite3.Connection) -> int:
    rows = conn.execute("SELECT rowid, payload FROM forecasts").fetchall()
    updates: list[tuple[int | None, int]] = []
    for rowid, payload_raw in rows:
        try:
            payload = json.loads(payload_raw) if payload_raw else {}
        except (TypeError, json.JSONDecodeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        updates.append((derive_windsurfice_weather(payload), int(rowid)))
    conn.executemany("UPDATE forecasts SET weather_code = ? WHERE rowid = ?", updates)
    conn.commit()
    return len(updates)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="Development SQLite path under /tmp.")
    parser.add_argument("--p1-tar", action="append", default=[], help="Downloaded P1 tar to reprocess; repeatable.")
    parser.add_argument("--site", default="valkenburgsemeer")
    parser.add_argument("--lat", type=float, default=52.168)
    parser.add_argument("--lon", type=float, default=4.437)
    args = parser.parse_args()

    db_path = guarded_development_db(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        init_db(conn)
        forecasts = backfill_forecasts(conn)
        p1_rows = 0
        for tar_value in args.p1_tar:
            result = extract_tar_features(
                Path(tar_value), SitePoint(args.site, args.lat, args.lon)
            )
            p1_rows += upsert_harmonie_knmi_features(conn, result.frame)
    finally:
        conn.close()
    print(f"backfilled_forecasts={forecasts} reprocessed_p1_rows={p1_rows} db={db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
