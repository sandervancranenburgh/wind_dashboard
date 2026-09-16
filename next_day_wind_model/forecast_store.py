"""Development store for provider-neutral proof-of-concept forecasts.

This schema is intentionally separate from ``db_store.py`` and the production
``forecasts`` table.  It can be created only in an explicitly development-safe
SQLite path.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from next_day_wind_model.forecast_provider import ForecastPoint, ForecastValue


TABLE_NAME = "forecast_points"
RUNS_TABLE = "forecast_collection_runs"
DOWNLOADS_TABLE = "forecast_downloads"
PRODUCTION_DATABASE_NAMES = {"wind_data.db", "wind_data_all_sites.db"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class ArchiveCoverage:
    expected_wind_leads: frozenset[int]
    stored_wind_leads: frozenset[int]
    expected_gust_leads: frozenset[int]
    stored_gust_leads: frozenset[int]

    @property
    def missing_wind_leads(self) -> frozenset[int]:
        return self.expected_wind_leads - self.stored_wind_leads

    @property
    def missing_gust_leads(self) -> frozenset[int]:
        return self.expected_gust_leads - self.stored_gust_leads

    @property
    def complete(self) -> bool:
        return not self.missing_wind_leads and not self.missing_gust_leads


def validate_development_db_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.name.lower() in PRODUCTION_DATABASE_NAMES:
        raise ValueError(
            f"Refusing production database name {resolved.name!r}; "
            "use a dedicated development .sqlite file"
        )
    return resolved


def connect_development_db(path: Path) -> sqlite3.Connection:
    safe_path = validate_development_db_path(path)
    safe_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(safe_path)
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def create_forecast_points_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            model_version TEXT,
            resolution TEXT,
            source TEXT NOT NULL,
            run_time TEXT NOT NULL,
            valid_time TEXT NOT NULL,
            lead_time_hours INTEGER NOT NULL,
            fetched_time TEXT NOT NULL,
            site TEXT NOT NULL,
            site_latitude REAL NOT NULL,
            site_longitude REAL NOT NULL,
            grid_latitude REAL NOT NULL,
            grid_longitude REAL NOT NULL,
            u10_mps REAL NOT NULL,
            v10_mps REAL NOT NULL,
            wind_speed_mps REAL NOT NULL,
            wind_direction_deg REAL NOT NULL,
            wind_gust_mps REAL,
            metadata_json TEXT NOT NULL,
            PRIMARY KEY (provider, model, run_time, valid_time, site)
        )
        """
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_site_valid "
        f"ON {TABLE_NAME}(site, valid_time)"
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {RUNS_TABLE} (
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            run_time TEXT NOT NULL,
            site TEXT NOT NULL,
            expected_wind_leads_json TEXT NOT NULL,
            expected_gust_leads_json TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('pending', 'partial', 'complete', 'failed')),
            attempts INTEGER NOT NULL DEFAULT 0,
            first_seen_time TEXT NOT NULL,
            last_attempt_time TEXT NOT NULL,
            updated_time TEXT NOT NULL,
            completed_time TEXT,
            last_error TEXT,
            PRIMARY KEY (provider, model, run_time, site)
        )
        """
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {DOWNLOADS_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            run_time TEXT NOT NULL,
            site TEXT NOT NULL,
            sites_json TEXT NOT NULL DEFAULT '[]',
            field_group TEXT NOT NULL,
            lead_hours_json TEXT NOT NULL,
            bytes_downloaded INTEGER NOT NULL,
            retrieve_calls INTEGER NOT NULL,
            retrieved_time TEXT NOT NULL,
            UNIQUE(provider, model, run_time, site, field_group, lead_hours_json)
        )
        """
    )
    download_columns = {
        str(row[1]) for row in conn.execute(f"PRAGMA table_info({DOWNLOADS_TABLE})").fetchall()
    }
    if "sites_json" not in download_columns:
        conn.execute(
            f"ALTER TABLE {DOWNLOADS_TABLE} ADD COLUMN sites_json TEXT NOT NULL DEFAULT '[]'"
        )
    conn.commit()


def archive_coverage(
    conn: sqlite3.Connection,
    *,
    provider: str,
    model: str,
    run_time: str,
    site: str,
    expected_wind_leads: Sequence[int],
    expected_gust_leads: Sequence[int] = (),
) -> ArchiveCoverage:
    expected_wind = frozenset(int(step) for step in expected_wind_leads)
    expected_gust = frozenset(int(step) for step in expected_gust_leads)
    stored_wind = stored_lead_hours(
        conn,
        provider=provider,
        model=model,
        run_time=run_time,
        site=site,
    )
    stored_gust = stored_lead_hours(
        conn,
        provider=provider,
        model=model,
        run_time=run_time,
        site=site,
        require_gust=True,
    )
    return ArchiveCoverage(expected_wind, frozenset(stored_wind), expected_gust, frozenset(stored_gust))


def begin_collection_attempt(
    conn: sqlite3.Connection,
    *,
    provider: str,
    model: str,
    run_time: str,
    site: str,
    expected_wind_leads: Sequence[int],
    expected_gust_leads: Sequence[int],
) -> None:
    create_forecast_points_table(conn)
    now = _utc_now_iso()
    wind_json = json.dumps(sorted({int(step) for step in expected_wind_leads}))
    gust_json = json.dumps(sorted({int(step) for step in expected_gust_leads}))
    conn.execute(
        f"""
        INSERT INTO {RUNS_TABLE} (
            provider, model, run_time, site,
            expected_wind_leads_json, expected_gust_leads_json,
            status, attempts, first_seen_time, last_attempt_time, updated_time
        ) VALUES (?, ?, ?, ?, ?, ?, 'pending', 1, ?, ?, ?)
        ON CONFLICT(provider, model, run_time, site) DO UPDATE SET
            expected_wind_leads_json=excluded.expected_wind_leads_json,
            expected_gust_leads_json=excluded.expected_gust_leads_json,
            status='pending',
            attempts={RUNS_TABLE}.attempts + 1,
            last_attempt_time=excluded.last_attempt_time,
            updated_time=excluded.updated_time,
            completed_time=NULL,
            last_error=NULL
        """,
        (provider, model, run_time, site, wind_json, gust_json, now, now, now),
    )
    conn.commit()


def finish_collection_attempt(
    conn: sqlite3.Connection,
    *,
    provider: str,
    model: str,
    run_time: str,
    site: str,
    status: str,
    error: str | None = None,
) -> None:
    if status not in {"partial", "complete", "failed"}:
        raise ValueError(f"invalid terminal collection status: {status}")
    now = _utc_now_iso()
    cursor = conn.execute(
        f"""
        UPDATE {RUNS_TABLE}
        SET status = ?, updated_time = ?,
            completed_time = CASE WHEN ? = 'complete' THEN ? ELSE NULL END,
            last_error = ?
        WHERE provider = ? AND model = ? AND run_time = ? AND site = ?
        """,
        (status, now, status, now, error, provider, model, run_time, site),
    )
    if cursor.rowcount != 1:
        raise ValueError("collection attempt was not started")
    conn.commit()


def collection_run_row(
    conn: sqlite3.Connection,
    *,
    provider: str,
    model: str,
    run_time: str,
    site: str,
) -> sqlite3.Row | tuple | None:
    create_forecast_points_table(conn)
    return conn.execute(
        f"SELECT * FROM {RUNS_TABLE} WHERE provider = ? AND model = ? AND run_time = ? AND site = ?",
        (provider, model, run_time, site),
    ).fetchone()


def stored_lead_hours(
    conn: sqlite3.Connection,
    *,
    provider: str,
    model: str,
    run_time: str,
    site: str,
    require_gust: bool = False,
) -> set[int]:
    create_forecast_points_table(conn)
    rows = conn.execute(
        f"""
        SELECT lead_time_hours
        FROM {TABLE_NAME}
        WHERE provider = ? AND model = ? AND run_time = ? AND site = ?
          AND (? = 0 OR wind_gust_mps IS NOT NULL)
        """,
        (provider, model, run_time, site, int(require_gust)),
    ).fetchall()
    return {int(row[0]) for row in rows}


def upsert_forecast_points(conn: sqlite3.Connection, points: Iterable[ForecastPoint]) -> int:
    create_forecast_points_table(conn)
    rows = []
    for point in points:
        rows.append(
            (
                point.provider,
                point.model,
                point.model_version,
                point.resolution,
                point.source,
                point.run_time.isoformat().replace("+00:00", "Z"),
                point.valid_time.isoformat().replace("+00:00", "Z"),
                point.lead_time_hours,
                point.fetched_time.isoformat().replace("+00:00", "Z"),
                point.site.name,
                point.site.latitude,
                point.site.longitude,
                point.grid_latitude,
                point.grid_longitude,
                point.u10_mps,
                point.v10_mps,
                point.wind_speed_mps,
                point.wind_direction_deg,
                point.wind_gust_mps,
                json.dumps(dict(point.metadata), ensure_ascii=False, sort_keys=True),
            )
        )
    if not rows:
        return 0

    conn.executemany(
        f"""
        INSERT INTO {TABLE_NAME} (
            provider, model, model_version, resolution, source,
            run_time, valid_time, lead_time_hours, fetched_time,
            site, site_latitude, site_longitude, grid_latitude, grid_longitude,
            u10_mps, v10_mps, wind_speed_mps, wind_direction_deg,
            wind_gust_mps, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(provider, model, run_time, valid_time, site) DO UPDATE SET
            model_version=excluded.model_version,
            resolution=excluded.resolution,
            source=excluded.source,
            lead_time_hours=excluded.lead_time_hours,
            fetched_time=MIN(forecast_points.fetched_time, excluded.fetched_time),
            site_latitude=excluded.site_latitude,
            site_longitude=excluded.site_longitude,
            grid_latitude=excluded.grid_latitude,
            grid_longitude=excluded.grid_longitude,
            u10_mps=excluded.u10_mps,
            v10_mps=excluded.v10_mps,
            wind_speed_mps=excluded.wind_speed_mps,
            wind_direction_deg=excluded.wind_direction_deg,
            wind_gust_mps=excluded.wind_gust_mps,
            metadata_json=excluded.metadata_json
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def upsert_forecast_values(conn: sqlite3.Connection, values: Iterable[ForecastValue]) -> int:
    """Merge independently retrieved optional scalar values into point rows."""
    create_forecast_points_table(conn)
    prepared = []
    for value in values:
        if value.variable != "wind_gust_10m":
            raise ValueError(f"unsupported point scalar variable: {value.variable}")
        run_iso = value.run_time.isoformat().replace("+00:00", "Z")
        valid_iso = value.valid_time.isoformat().replace("+00:00", "Z")
        existing = conn.execute(
            f"""
            SELECT grid_latitude, grid_longitude, metadata_json
            FROM {TABLE_NAME}
            WHERE provider = ? AND model = ? AND run_time = ? AND valid_time = ? AND site = ?
            """,
            (value.provider, value.model, run_iso, valid_iso, value.site.name),
        ).fetchone()
        if existing is None:
            raise ValueError(
                "cannot store optional forecast value before its U/V point: "
                f"{value.provider}/{value.model} {run_iso} lead {value.lead_time_hours}"
            )
        if abs(float(existing[0]) - value.grid_latitude) > 1e-6 or abs(float(existing[1]) - value.grid_longitude) > 1e-6:
            raise ValueError("optional forecast value uses a different grid point than U/V")
        metadata = json.loads(existing[2])
        metadata.update(
            {
                "gust_unit": value.unit,
                "gust_metadata": dict(value.metadata),
            }
        )
        prepared.append(
            (
                value.value,
                value.model_version,
                value.resolution,
                json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                value.provider,
                value.model,
                run_iso,
                valid_iso,
                value.site.name,
            )
        )

    if not prepared:
        return 0
    conn.executemany(
        f"""
        UPDATE {TABLE_NAME}
        SET wind_gust_mps = ?,
            model_version = COALESCE(?, model_version),
            resolution = COALESCE(?, resolution),
            metadata_json = ?
        WHERE provider = ? AND model = ? AND run_time = ? AND valid_time = ? AND site = ?
        """,
        prepared,
    )
    conn.commit()
    return len(prepared)


def record_download(
    conn: sqlite3.Connection,
    *,
    provider: str,
    model: str,
    run_time: str,
    site: str,
    sites: Sequence[str] | None = None,
    field_group: str,
    lead_hours: Sequence[int],
    bytes_downloaded: int,
    retrieve_calls: int = 1,
) -> None:
    create_forecast_points_table(conn)
    leads_json = json.dumps(sorted({int(step) for step in lead_hours}))
    sites_json = json.dumps(sorted({str(item) for item in (sites or (site,))}))
    conn.execute(
        f"""
        INSERT INTO {DOWNLOADS_TABLE} (
            provider, model, run_time, site, sites_json, field_group, lead_hours_json,
            bytes_downloaded, retrieve_calls, retrieved_time
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(provider, model, run_time, site, field_group, lead_hours_json) DO UPDATE SET
            sites_json=excluded.sites_json,
            bytes_downloaded=excluded.bytes_downloaded,
            retrieve_calls=excluded.retrieve_calls,
            retrieved_time=excluded.retrieved_time
        """,
        (
            provider,
            model,
            run_time,
            site,
            sites_json,
            field_group,
            leads_json,
            int(bytes_downloaded),
            int(retrieve_calls),
            _utc_now_iso(),
        ),
    )
    conn.commit()
