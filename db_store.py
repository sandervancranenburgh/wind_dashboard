import os
import json
import sqlite3
from datetime import datetime, timezone
from typing import Iterable, Dict, Any, Optional, Tuple


DB_FILENAME = "wind_data.db"
FORECASTS_TABLE = "forecasts"


def db_path(out_dir: str) -> str:
    return os.path.join(out_dir, DB_FILENAME)


def connect_db(out_dir: str) -> sqlite3.Connection:
    path = db_path(out_dir)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _create_forecasts_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {FORECASTS_TABLE} (
            site TEXT NOT NULL,
            model TEXT NOT NULL,
            run_ts INTEGER NOT NULL,
            run_iso TEXT,
            fetched_ts INTEGER NOT NULL,
            fetched_iso TEXT,
            target_ts INTEGER NOT NULL,
            target_iso TEXT,
            horizon_hr INTEGER,
            wind_speed REAL,
            wind_gust REAL,
            wind_dir REAL,
            payload TEXT,
            PRIMARY KEY (site, model, run_ts, target_ts)
        )
        """
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_fc_site_model_run ON {FORECASTS_TABLE}(site, model, run_ts)"
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_fc_site_model_target ON {FORECASTS_TABLE}(site, model, target_ts)"
    )


def _forecast_schema_matches(conn: sqlite3.Connection) -> bool:
    if not _table_exists(conn, FORECASTS_TABLE):
        return False

    expected = {
        "site": ("TEXT", 1, 1),
        "model": ("TEXT", 1, 2),
        "run_ts": ("INTEGER", 1, 3),
        "run_iso": ("TEXT", 0, 0),
        "fetched_ts": ("INTEGER", 1, 0),
        "fetched_iso": ("TEXT", 0, 0),
        "target_ts": ("INTEGER", 1, 4),
        "target_iso": ("TEXT", 0, 0),
        "horizon_hr": ("INTEGER", 0, 0),
        "wind_speed": ("REAL", 0, 0),
        "wind_gust": ("REAL", 0, 0),
        "wind_dir": ("REAL", 0, 0),
        "payload": ("TEXT", 0, 0),
    }

    rows = conn.execute(f"PRAGMA table_info({FORECASTS_TABLE})").fetchall()
    actual = {
        row[1]: (
            str(row[2]).upper(),
            int(row[3]),
            int(row[5]),
        )
        for row in rows
    }
    return actual == expected


def _next_backup_table_name(conn: sqlite3.Connection, base_name: str) -> str:
    suffix = 0
    while True:
        candidate = base_name if suffix == 0 else f"{base_name}_{suffix}"
        if not _table_exists(conn, candidate):
            return candidate
        suffix += 1


def _migrate_forecasts_table(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, FORECASTS_TABLE):
        _create_forecasts_table(conn)
        return
    if _forecast_schema_matches(conn):
        _create_forecasts_table(conn)
        return

    legacy_table = _next_backup_table_name(conn, 'forecasts_legacy')
    conn.execute(f"ALTER TABLE {FORECASTS_TABLE} RENAME TO {legacy_table}")
    _create_forecasts_table(conn)

    legacy_columns = {row[1] for row in conn.execute(f"PRAGMA table_info({legacy_table})").fetchall()}
    if not {"site", "run_ts", "target_ts"}.issubset(legacy_columns):
        return

    model_expr = "COALESCE(model, 'unknown')" if 'model' in legacy_columns else "'unknown'"
    run_iso_expr = 'run_iso' if 'run_iso' in legacy_columns else 'NULL'
    target_iso_expr = 'target_iso' if 'target_iso' in legacy_columns else 'NULL'
    horizon_expr = (
        'horizon_hr'
        if 'horizon_hr' in legacy_columns
        else 'CAST(ROUND((target_ts - run_ts) / 3600000.0) AS INTEGER)'
    )
    wind_speed_expr = 'wind_speed' if 'wind_speed' in legacy_columns else 'NULL'
    wind_gust_expr = 'wind_gust' if 'wind_gust' in legacy_columns else 'NULL'
    wind_dir_expr = 'wind_dir' if 'wind_dir' in legacy_columns else 'NULL'
    payload_expr = 'payload' if 'payload' in legacy_columns else 'NULL'

    # Legacy DBs did not persist fetch time separately, so migration reuses the
    # stored run time. New writes keep run_ts (forecast vintage) separate from
    # fetched_ts (when our collector actually saw that vintage).
    conn.execute(
        f"""
        INSERT OR REPLACE INTO {FORECASTS_TABLE} (
            site, model, run_ts, run_iso, fetched_ts, fetched_iso,
            target_ts, target_iso, horizon_hr, wind_speed, wind_gust, wind_dir, payload
        )
        SELECT
            site,
            {model_expr},
            run_ts,
            {run_iso_expr},
            run_ts,
            {run_iso_expr},
            target_ts,
            {target_iso_expr},
            {horizon_expr},
            {wind_speed_expr},
            {wind_gust_expr},
            {wind_dir_expr},
            {payload_expr}
        FROM {legacy_table}
        WHERE run_ts IS NOT NULL
          AND target_ts IS NOT NULL
        """
    )


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS observations (
            site TEXT NOT NULL,
            ts INTEGER NOT NULL,
            iso_time TEXT,
            wind_speed REAL,
            wind_gust REAL,
            wind_dir REAL,
            payload TEXT,
            PRIMARY KEY (site, ts)
        )
        """
    )
    _migrate_forecasts_table(conn)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_obs_ts ON observations(ts)")
    # Predictions table (per horizon)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            site TEXT NOT NULL,
            model_name TEXT,
            generated_ts INTEGER NOT NULL,
            generated_iso TEXT,
            horizon_hr INTEGER NOT NULL,
            target_ts INTEGER NOT NULL,
            target_iso TEXT,
            pred_wind_speed REAL,
            payload TEXT,
            PRIMARY KEY (site, model_name, generated_ts, horizon_hr)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_target ON predictions(target_ts)")
    conn.commit()


def _iso_utc_from_ms(ms: Optional[int]) -> Optional[str]:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _extract_first(d: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    # Case-insensitive key search
    lower = {k.lower(): k for k in d.keys()}
    for k in keys:
        real = lower.get(k.lower())
        if real is not None:
            return d.get(real)
    return None


def _as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def guess_timestamp_ms(d: Dict[str, Any]) -> Optional[int]:
    t = _extract_first(d, ["timestamp", "time", "UnixTime", "ts", "dt"])  # seconds or ms
    if t is None:
        return None
    s = str(t)
    try:
        return int(t) if len(s) > 10 else int(t) * 1000
    except Exception:
        return None


def extract_wind_triplet(d: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    # Heuristics for common key names (case-insensitive)
    spd = _extract_first(
        d,
        [
            "wind_speed",
            "windspeed",
            "WS",
            "WS10m",
            "ff",
            "speed",
            "WindSpeedAvg",
            "wind",
        ],
    )
    gst = _extract_first(d, ["wind_gust", "gust", "WG", "GUST", "fg"])
    dire = _extract_first(d, ["wind_dir", "winddirection", "WD", "DD", "dir", "direction"])
    return _as_float(spd), _as_float(gst), _as_float(dire)


def upsert_observations(conn: sqlite3.Connection, site: str, rows: Iterable[Dict[str, Any]]) -> int:
    cur = conn.cursor()
    inserted = 0
    for r in rows:
        ts = guess_timestamp_ms(r)
        if ts is None:
            continue
        spd, gst, dire = extract_wind_triplet(r)
        iso = _iso_utc_from_ms(ts)
        cur.execute(
            """
            INSERT INTO observations(site, ts, iso_time, wind_speed, wind_gust, wind_dir, payload)
            VALUES(?,?,?,?,?,?,json(?))
            ON CONFLICT(site, ts) DO UPDATE SET
                iso_time=excluded.iso_time,
                wind_speed=COALESCE(excluded.wind_speed, observations.wind_speed),
                wind_gust=COALESCE(excluded.wind_gust, observations.wind_gust),
                wind_dir=COALESCE(excluded.wind_dir, observations.wind_dir),
                payload=excluded.payload
            """,
            (site, ts, iso, spd, gst, dire, json.dumps(r, ensure_ascii=False)),
        )
        inserted += 1
    conn.commit()
    return inserted


def upsert_forecasts(
    conn: sqlite3.Connection,
    site: str,
    model: str,
    rows: Iterable[Dict[str, Any]],
    run_ts_ms: int,
    fetched_ts_ms: int,
) -> int:
    """
    Persist immutable forecast vintages.

    run_ts_ms is the forecast/model vintage timestamp.
    fetched_ts_ms is when this system actually observed that vintage.
    Keeping both matters because fair backtests must compare predictions against
    the exact Harmonie vintage that was available at the time.
    """
    if run_ts_ms is None:
        raise ValueError('run_ts_ms must be provided explicitly for forecast upserts.')
    if fetched_ts_ms is None:
        raise ValueError('fetched_ts_ms must be provided explicitly for forecast upserts.')

    run_iso = _iso_utc_from_ms(run_ts_ms)
    fetched_iso = _iso_utc_from_ms(fetched_ts_ms)

    cur = conn.cursor()
    inserted = 0
    for r in rows:
        target_ts = guess_timestamp_ms(r)
        if target_ts is None:
            continue
        horizon_ms = target_ts - run_ts_ms
        horizon_hr = int(round(horizon_ms / 1000 / 3600))
        spd, gst, dire = extract_wind_triplet(r)
        cur.execute(
            f"""
            INSERT INTO {FORECASTS_TABLE}(
                site, model, run_ts, run_iso, fetched_ts, fetched_iso, target_ts, target_iso, horizon_hr,
                wind_speed, wind_gust, wind_dir, payload
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,json(?))
            ON CONFLICT(site, model, run_ts, target_ts) DO UPDATE SET
                run_iso=excluded.run_iso,
                fetched_ts=excluded.fetched_ts,
                fetched_iso=excluded.fetched_iso,
                target_iso=excluded.target_iso,
                horizon_hr=excluded.horizon_hr,
                wind_speed=COALESCE(excluded.wind_speed, {FORECASTS_TABLE}.wind_speed),
                wind_gust=COALESCE(excluded.wind_gust, {FORECASTS_TABLE}.wind_gust),
                wind_dir=COALESCE(excluded.wind_dir, {FORECASTS_TABLE}.wind_dir),
                payload=excluded.payload
            """,
            (
                site,
                model,
                run_ts_ms,
                run_iso,
                fetched_ts_ms,
                fetched_iso,
                target_ts,
                _iso_utc_from_ms(target_ts),
                horizon_hr,
                spd,
                gst,
                dire,
                json.dumps(r, ensure_ascii=False),
            ),
        )
        inserted += 1
    conn.commit()
    return inserted


def upsert_predictions(
    conn: sqlite3.Connection,
    site: str,
    model_name: str,
    generated_ts_ms: int,
    preds: Dict[int, float],  # {horizon_hr: pred_wind_speed}
) -> int:
    gen_iso = _iso_utc_from_ms(generated_ts_ms)
    cur = conn.cursor()
    n = 0
    for horizon_hr, value in preds.items():
        target_ts = generated_ts_ms + int(horizon_hr) * 3600 * 1000
        cur.execute(
            """
            INSERT INTO predictions(
                site, model_name, generated_ts, generated_iso,
                horizon_hr, target_ts, target_iso, pred_wind_speed, payload
            ) VALUES (?,?,?,?,?,?,?,?,json(?))
            ON CONFLICT(site, model_name, generated_ts, horizon_hr) DO UPDATE SET
                target_ts=excluded.target_ts,
                target_iso=excluded.target_iso,
                pred_wind_speed=excluded.pred_wind_speed,
                payload=excluded.payload
            """,
            (
                site,
                model_name,
                generated_ts_ms,
                gen_iso,
                int(horizon_hr),
                target_ts,
                _iso_utc_from_ms(target_ts),
                float(value) if value is not None else None,
                json.dumps({"horizon_hr": int(horizon_hr)}),
            ),
        )
        n += 1
    conn.commit()
    return n
