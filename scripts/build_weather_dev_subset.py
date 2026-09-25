#!/usr/bin/env python3
"""Build a small, isolated weather-development DB from read-only source rows."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from db_store import init_db
from scripts.backfill_harmonie_weather import guarded_development_db


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--site", default="valkenburgsemeer")
    parser.add_argument("--model", default="HARMONIE")
    args = parser.parse_args()

    source_path = args.source.expanduser().resolve()
    destination = guarded_development_db(args.db)
    if source_path == destination:
        raise ValueError("Source and destination must differ.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(f"file:{source_path}?mode=ro&immutable=1", uri=True)
    target = sqlite3.connect(destination)
    try:
        init_db(target)
        latest_observation = source.execute(
            "SELECT MAX(ts) FROM observations WHERE site = ?", (args.site,)
        ).fetchone()[0]
        if latest_observation is None:
            raise ValueError(f"No observations found for {args.site!r}.")
        day_ms = 86_400_000
        obs_start = int(latest_observation) - 2 * day_ms
        target_start = int(latest_observation) - day_ms
        target_end = int(latest_observation) + 3 * day_ms
        observations = source.execute(
            """SELECT site, ts, iso_time, wind_speed, wind_gust, wind_dir, payload
               FROM observations WHERE site = ? AND ts >= ? AND ts <= ? ORDER BY ts""",
            (args.site, obs_start, int(latest_observation)),
        ).fetchall()
        target.executemany(
            "INSERT OR REPLACE INTO observations VALUES(?,?,?,?,?,?,?)", observations
        )
        forecasts = source.execute(
            """
            WITH ranked AS (
                SELECT site, model, run_ts, run_iso, fetched_ts, fetched_iso,
                       target_ts, target_iso, horizon_hr, wind_speed, wind_gust,
                       wind_dir, payload,
                       ROW_NUMBER() OVER (
                           PARTITION BY target_ts ORDER BY run_ts DESC, fetched_ts DESC
                       ) AS row_number
                FROM forecasts
                WHERE site = ? AND model = ? AND target_ts >= ? AND target_ts <= ?
            )
            SELECT site, model, run_ts, run_iso, fetched_ts, fetched_iso,
                   target_ts, target_iso, horizon_hr, wind_speed, wind_gust,
                   wind_dir, payload
            FROM ranked WHERE row_number = 1 ORDER BY target_ts
            """,
            (args.site, args.model, target_start, target_end),
        ).fetchall()
        target.executemany(
            """INSERT OR REPLACE INTO forecasts(
               site, model, run_ts, run_iso, fetched_ts, fetched_iso, target_ts,
               target_iso, horizon_hr, wind_speed, wind_gust, wind_dir, payload)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            forecasts,
        )
        target.commit()
    finally:
        source.close()
        target.close()
    print(
        f"observations={len(observations)} forecasts={len(forecasts)} "
        f"latest_observation_ms={latest_observation} db={destination}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
