import os
import json
import math
import shutil
import sqlite3
from datetime import datetime, timezone
from typing import Iterable, Dict, Any, Optional, Tuple, List


DB_FILENAME = "wind_data_all_sites.db"
LEGACY_DB_FILENAME = "wind_data.db"
FORECASTS_TABLE = "forecasts"
PREDICTION_LOG_TABLE = "prediction_log"
HOUR_MS = 3_600_000
CURRENT_DAY_PLOT_INTERVAL_MS = 6 * 60 * 1000
PREDICTION_LOG_EVAL_COLUMNS = {
    "model_error": "REAL",
    "harmonie_error": "REAL",
    "model_abs_error": "REAL",
    "harmonie_abs_error": "REAL",
    "model_sq_error": "REAL",
    "harmonie_sq_error": "REAL",
}
SPOT_TO_SITE = {
    "Valkenburgse meer": "valkenburgsemeer",
    "Oostvoornse meer": "oostvoorne",
}
SURF_EXPERIENCE_OPTIONAL_COLUMNS = {
    "visibility": "TEXT NOT NULL DEFAULT 'private'",
    "min_measured_wind_speed": "REAL",
    "mean_measured_direction": "REAL",
    "mean_measured_direction_label": "TEXT",
    "avg_forecast_temperature": "REAL",
    "min_forecast_temperature": "REAL",
    "max_forecast_temperature": "REAL",
}
USER_PROFILE_OPTIONAL_COLUMNS = {
    "public_username": "TEXT",
}


def db_path(out_dir: str) -> str:
    return os.path.join(out_dir, DB_FILENAME)


def _legacy_db_path(out_dir: str) -> str:
    return os.path.join(out_dir, LEGACY_DB_FILENAME)


def _ensure_named_shared_db(out_dir: str) -> str:
    path = db_path(out_dir)
    legacy_path = _legacy_db_path(out_dir)
    if not os.path.exists(path) and os.path.exists(legacy_path):
        shutil.copy2(legacy_path, path)
    return path


def connect_db(out_dir: str) -> sqlite3.Connection:
    path = _ensure_named_shared_db(out_dir)
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


def _create_prediction_log_table(conn: sqlite3.Connection) -> None:
    # Canonical issued-prediction log.
    #
    # One row = one target timestamp for one issued forecast. We persist the
    # selected Harmonie run/fetched metadata alongside the model prediction so
    # later evaluation can reconstruct the exact baseline and availability
    # context that was used operationally at issuance time. Realised actuals and
    # per-row errors live on the same row because they are attributes of that
    # issued prediction record, not a separate forecasting event.
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {PREDICTION_LOG_TABLE} (
            site TEXT NOT NULL,
            model_type TEXT NOT NULL,
            prediction_kind TEXT NOT NULL,
            model_name TEXT,
            model_version TEXT,
            model_artifact TEXT,
            issued_ts INTEGER NOT NULL,
            issued_iso TEXT,
            anchor_ts INTEGER NOT NULL,
            anchor_iso TEXT,
            target_ts INTEGER NOT NULL,
            target_iso TEXT,
            horizon_hr REAL,
            prediction_value REAL,
            harmonie_value REAL,
            harmonie_run_ts INTEGER,
            harmonie_run_iso TEXT,
            harmonie_fetched_ts INTEGER,
            harmonie_fetched_iso TEXT,
            actual_value REAL,
            model_error REAL,
            harmonie_error REAL,
            model_abs_error REAL,
            harmonie_abs_error REAL,
            model_sq_error REAL,
            harmonie_sq_error REAL,
            run_context TEXT,
            metadata_json TEXT,
            PRIMARY KEY (site, model_type, prediction_kind, issued_ts, target_ts)
        )
        """
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_predlog_site_type_issued ON {PREDICTION_LOG_TABLE}(site, model_type, issued_ts)"
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_predlog_site_type_target ON {PREDICTION_LOG_TABLE}(site, model_type, target_ts)"
    )


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    if not _table_exists(conn, table_name):
        return set()
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row[1]) for row in rows}


def _migrate_prediction_log_table(conn: sqlite3.Connection) -> None:
    _create_prediction_log_table(conn)
    existing_columns = _table_columns(conn, PREDICTION_LOG_TABLE)
    for column_name, column_type in PREDICTION_LOG_EVAL_COLUMNS.items():
        if column_name in existing_columns:
            continue
        conn.execute(
            f"ALTER TABLE {PREDICTION_LOG_TABLE} ADD COLUMN {column_name} {column_type}"
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
    # Legacy predictions table retained for backward compatibility with older
    # per-horizon exports. The canonical durable store for issued forecasts is
    # prediction_log, created below.
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
    _migrate_prediction_log_table(conn)
    init_account_db(conn)
    conn.commit()


def init_account_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            username_norm TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_ts INTEGER NOT NULL,
            created_iso TEXT NOT NULL,
            last_login_ts INTEGER,
            last_login_iso TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id INTEGER PRIMARY KEY,
            public_username TEXT,
            rider_name TEXT,
            rider_weight INTEGER,
            default_spot TEXT,
            updated_ts INTEGER NOT NULL,
            updated_iso TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS surf_experiences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            submitted_ts INTEGER NOT NULL,
            submitted_iso TEXT NOT NULL,
            rider TEXT NOT NULL,
            spot TEXT NOT NULL,
            date TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL,
            start_ts INTEGER NOT NULL,
            end_ts INTEGER NOT NULL,
            session_rating INTEGER NOT NULL,
            rider_review TEXT,
            rider_weight INTEGER,
            wing_size INTEGER NOT NULL,
            foil_size INTEGER NOT NULL,
            rider_notes TEXT,
            measured_wind_data_json TEXT NOT NULL,
            measured_wind_status TEXT NOT NULL,
            measured_wind_point_count INTEGER NOT NULL DEFAULT 0,
            avg_measured_wind_speed REAL,
            max_measured_wind_speed REAL,
            min_measured_wind_speed REAL,
            avg_measured_wind_dir REAL,
            mean_measured_direction REAL,
            mean_measured_direction_label TEXT,
            avg_forecast_temperature REAL,
            min_forecast_temperature REAL,
            max_forecast_temperature REAL,
            visibility TEXT NOT NULL DEFAULT 'private',
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """
    )
    existing_columns = _table_columns(conn, "surf_experiences")
    for column_name, column_type in SURF_EXPERIENCE_OPTIONAL_COLUMNS.items():
        if column_name not in existing_columns:
            conn.execute(f"ALTER TABLE surf_experiences ADD COLUMN {column_name} {column_type}")
    existing_profile_columns = _table_columns(conn, "user_profiles")
    for column_name, column_type in USER_PROFILE_OPTIONAL_COLUMNS.items():
        if column_name not in existing_profile_columns:
            conn.execute(f"ALTER TABLE user_profiles ADD COLUMN {column_name} {column_type}")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_profiles_user ON user_profiles(user_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_experiences_user_date ON surf_experiences(user_id, date, start_time)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_experiences_visibility_date ON surf_experiences(visibility, date, start_time)")
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


def _as_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _json_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _abs_or_none(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return abs(float(value))


def _square_or_none(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    value_f = float(value)
    return value_f * value_f


def _submission_visibility(value: Any) -> str:
    return "public" if str(value or "").strip().lower() == "public" else "private"


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
            "WindForecastAvr",
            "wind",
        ],
    )
    gst = _extract_first(d, ["wind_gust", "gust", "WG", "GUST", "fg", "WindForecastMax"])
    dire = _extract_first(d, ["wind_dir", "winddirection", "WD", "DD", "dir", "direction", "WindDirection"])
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
                fetched_ts=MIN({FORECASTS_TABLE}.fetched_ts, excluded.fetched_ts),
                fetched_iso=CASE
                    WHEN excluded.fetched_ts < {FORECASTS_TABLE}.fetched_ts THEN excluded.fetched_iso
                    ELSE {FORECASTS_TABLE}.fetched_iso
                END,
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


def log_prediction_batch(
    conn: sqlite3.Connection,
    rows: Iterable[Dict[str, Any]],
) -> int:
    """
    Append immutable issued-prediction rows to the canonical prediction log.

    Each row represents one target timestamp for one issued forecast and stores
    the exact Harmonie value/run/fetched metadata used at issuance time. The log
    is append-only by issued_ts; duplicate inserts for the same issued row are
    ignored rather than mutating prior operational records.
    """
    cur = conn.cursor()
    inserted = 0
    for row in rows:
        site = row.get("site")
        model_type = row.get("model_type")
        prediction_kind = row.get("prediction_kind")
        issued_ts = _as_int(row.get("issued_ts"))
        anchor_ts = _as_int(row.get("anchor_ts"))
        target_ts = _as_int(row.get("target_ts"))
        if not site or not model_type or not prediction_kind:
            raise ValueError("Prediction log rows require site, model_type, and prediction_kind.")
        if issued_ts is None or anchor_ts is None or target_ts is None:
            raise ValueError("Prediction log rows require issued_ts, anchor_ts, and target_ts.")

        cur.execute(
            f"""
            INSERT INTO {PREDICTION_LOG_TABLE}(
                site,
                model_type,
                prediction_kind,
                model_name,
                model_version,
                model_artifact,
                issued_ts,
                issued_iso,
                anchor_ts,
                anchor_iso,
                target_ts,
                target_iso,
                horizon_hr,
                prediction_value,
                harmonie_value,
                harmonie_run_ts,
                harmonie_run_iso,
                harmonie_fetched_ts,
                harmonie_fetched_iso,
                actual_value,
                run_context,
                metadata_json
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(site, model_type, prediction_kind, issued_ts, target_ts) DO NOTHING
            """,
            (
                site,
                model_type,
                prediction_kind,
                row.get("model_name"),
                row.get("model_version"),
                row.get("model_artifact"),
                issued_ts,
                row.get("issued_iso") or _iso_utc_from_ms(issued_ts),
                anchor_ts,
                row.get("anchor_iso") or _iso_utc_from_ms(anchor_ts),
                target_ts,
                row.get("target_iso") or _iso_utc_from_ms(target_ts),
                _as_float(row.get("horizon_hr")),
                _as_float(row.get("prediction_value")),
                _as_float(row.get("harmonie_value")),
                _as_int(row.get("harmonie_run_ts")),
                row.get("harmonie_run_iso") or _iso_utc_from_ms(_as_int(row.get("harmonie_run_ts"))),
                _as_int(row.get("harmonie_fetched_ts")),
                row.get("harmonie_fetched_iso") or _iso_utc_from_ms(_as_int(row.get("harmonie_fetched_ts"))),
                _as_float(row.get("actual_value")),
                row.get("run_context"),
                _json_text(row.get("metadata_json")),
            ),
        )
        inserted += max(int(cur.rowcount), 0)
    conn.commit()
    return inserted


def _extract_actual_observation_value(payload_raw: Optional[str], wind_speed: Any) -> Optional[float]:
    payload: Dict[str, Any] = {}
    if payload_raw:
        try:
            payload = json.loads(payload_raw)
        except json.JSONDecodeError:
            payload = {}
    actual = _extract_first(
        payload,
        ["AverageWind", "wind_speed", "windspeed", "WS", "ff", "speed", "WindSpeedAvg"],
    )
    if actual is None:
        actual = wind_speed
    return _as_float(actual)


def _load_hourly_observation_lookup(
    conn: sqlite3.Connection,
    site: str,
    include_partial_hour: bool = False,
) -> tuple[Dict[int, float], Optional[int]]:
    rows = conn.execute(
        """
        SELECT ts, wind_speed, payload
        FROM observations
        WHERE site = ?
          AND ts IS NOT NULL
        ORDER BY ts
        """,
        (site,),
    ).fetchall()
    if not rows:
        return {}, None

    bucket_sums: Dict[int, float] = {}
    bucket_counts: Dict[int, int] = {}
    max_obs_ts: Optional[int] = None
    for ts, wind_speed, payload_raw in rows:
        ts_ms = _as_int(ts)
        if ts_ms is None:
            continue
        actual_value = _extract_actual_observation_value(payload_raw, wind_speed)
        if actual_value is None:
            continue
        bucket_ts = (ts_ms // HOUR_MS) * HOUR_MS
        bucket_sums[bucket_ts] = bucket_sums.get(bucket_ts, 0.0) + float(actual_value)
        bucket_counts[bucket_ts] = bucket_counts.get(bucket_ts, 0) + 1
        if max_obs_ts is None or ts_ms > max_obs_ts:
            max_obs_ts = ts_ms

    if not bucket_sums or max_obs_ts is None:
        return {}, None

    if include_partial_hour:
        latest_resolvable_target_ts = (max_obs_ts // HOUR_MS) * HOUR_MS
    else:
        latest_resolvable_target_ts = ((max_obs_ts - 1) // HOUR_MS) * HOUR_MS if max_obs_ts > 0 else None

    actual_lookup = {
        bucket_ts: float(bucket_sums[bucket_ts] / max(bucket_counts[bucket_ts], 1))
        for bucket_ts in sorted(bucket_sums)
        if latest_resolvable_target_ts is not None and bucket_ts <= latest_resolvable_target_ts
    }
    return actual_lookup, latest_resolvable_target_ts


def materialize_prediction_log_evaluation(
    conn: sqlite3.Connection,
    site: Optional[str] = None,
    model_type: Optional[str] = None,
    prediction_kind: Optional[str] = None,
    include_partial_hour: bool = False,
) -> int:
    """
    Attach realised hourly actuals and fair model-vs-Harmonie errors to prediction_log.

    The join key is the exact hourly target_ts. Actual observations are
    aggregated to hourly means using the same convention as the rest of the
    refactor, and rows are updated only when that hourly actual can be resolved
    from the observation store.
    """
    init_db(conn)
    filters = ["target_ts IS NOT NULL"]
    params: list[Any] = []
    if site is not None:
        filters.append("site = ?")
        params.append(site)
    if model_type is not None:
        filters.append("model_type = ?")
        params.append(model_type)
    if prediction_kind is not None:
        filters.append("prediction_kind = ?")
        params.append(prediction_kind)
    filters.append(
        "(actual_value IS NULL OR model_error IS NULL OR harmonie_error IS NULL OR model_abs_error IS NULL OR harmonie_abs_error IS NULL OR model_sq_error IS NULL OR harmonie_sq_error IS NULL)"
    )

    rows = conn.execute(
        f"""
        SELECT rowid, site, target_ts, prediction_value, harmonie_value
        FROM {PREDICTION_LOG_TABLE}
        WHERE {" AND ".join(filters)}
        ORDER BY site ASC, target_ts ASC
        """,
        params,
    ).fetchall()
    if not rows:
        return 0

    site_actuals: Dict[str, Dict[int, float]] = {}
    updates: list[tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], int]] = []
    for rowid, row_site, target_ts, prediction_value, harmonie_value in rows:
        if row_site not in site_actuals:
            actual_lookup, _ = _load_hourly_observation_lookup(
                conn,
                row_site,
                include_partial_hour=include_partial_hour,
            )
            site_actuals[row_site] = actual_lookup
        actual_value = site_actuals[row_site].get(int(target_ts))
        if actual_value is None:
            continue
        pred_value = _as_float(prediction_value)
        harmonie_val = _as_float(harmonie_value)
        model_error = None if pred_value is None else float(pred_value - actual_value)
        harmonie_error = None if harmonie_val is None else float(harmonie_val - actual_value)
        updates.append(
            (
                float(actual_value),
                model_error,
                harmonie_error,
                _abs_or_none(model_error),
                _abs_or_none(harmonie_error),
                _square_or_none(model_error),
                _square_or_none(harmonie_error),
                int(rowid),
            )
        )

    if not updates:
        return 0

    conn.executemany(
        f"""
        UPDATE {PREDICTION_LOG_TABLE}
        SET
            actual_value = ?,
            model_error = ?,
            harmonie_error = ?,
            model_abs_error = ?,
            harmonie_abs_error = ?,
            model_sq_error = ?,
            harmonie_sq_error = ?
        WHERE rowid = ?
        """,
        updates,
    )
    conn.commit()
    return int(len(updates))


def load_prediction_evaluation_summary(
    conn: sqlite3.Connection,
    site: Optional[str] = None,
    model_type: Optional[str] = None,
    prediction_kind: Optional[str] = None,
    issued_ts_from_ms: Optional[int] = None,
    issued_ts_to_ms: Optional[int] = None,
    target_ts_from_ms: Optional[int] = None,
    target_ts_to_ms: Optional[int] = None,
    min_horizon_hr: Optional[float] = None,
    max_horizon_hr: Optional[float] = None,
) -> list[Dict[str, Any]]:
    """
    Return lightweight realised-evaluation summaries from the canonical log.

    Summaries are grouped by site/model_type/prediction_kind and compare the
    logged model predictions against the logged Harmonie baselines on the same
    realised rows.
    """
    filters = [
        "actual_value IS NOT NULL",
        "model_abs_error IS NOT NULL",
        "harmonie_abs_error IS NOT NULL",
    ]
    params: list[Any] = []
    if site is not None:
        filters.append("site = ?")
        params.append(site)
    if model_type is not None:
        filters.append("model_type = ?")
        params.append(model_type)
    if prediction_kind is not None:
        filters.append("prediction_kind = ?")
        params.append(prediction_kind)
    if issued_ts_from_ms is not None:
        filters.append("issued_ts >= ?")
        params.append(int(issued_ts_from_ms))
    if issued_ts_to_ms is not None:
        filters.append("issued_ts <= ?")
        params.append(int(issued_ts_to_ms))
    if target_ts_from_ms is not None:
        filters.append("target_ts >= ?")
        params.append(int(target_ts_from_ms))
    if target_ts_to_ms is not None:
        filters.append("target_ts <= ?")
        params.append(int(target_ts_to_ms))
    if min_horizon_hr is not None:
        filters.append("horizon_hr >= ?")
        params.append(float(min_horizon_hr))
    if max_horizon_hr is not None:
        filters.append("horizon_hr <= ?")
        params.append(float(max_horizon_hr))

    rows = conn.execute(
        f"""
        SELECT
            site,
            model_type,
            prediction_kind,
            COUNT(*) AS n_rows,
            AVG(model_abs_error) AS mae_model,
            AVG(harmonie_abs_error) AS mae_harmonie,
            AVG(model_sq_error) AS mse_model,
            AVG(harmonie_sq_error) AS mse_harmonie,
            AVG(harmonie_abs_error - model_abs_error) AS mae_improvement_vs_harmonie
        FROM {PREDICTION_LOG_TABLE}
        WHERE {" AND ".join(filters)}
        GROUP BY site, model_type, prediction_kind
        ORDER BY site ASC, model_type ASC, prediction_kind ASC
        """,
        params,
    ).fetchall()
    return [
        {
            "site": row[0],
            "model_type": row[1],
            "prediction_kind": row[2],
            "n_rows": int(row[3]),
            "mae_model": None if row[4] is None else float(row[4]),
            "mae_harmonie": None if row[5] is None else float(row[5]),
            "mse_model": None if row[6] is None else float(row[6]),
            "mse_harmonie": None if row[7] is None else float(row[7]),
            "mae_improvement_vs_harmonie": None if row[8] is None else float(row[8]),
        }
        for row in rows
    ]


def _load_prediction_log_realized_rows(
    conn: sqlite3.Connection,
    *,
    site: Optional[str] = None,
    model_type: Optional[str] = None,
    prediction_kind: Optional[str] = None,
    model_artifact: Optional[str] = None,
    model_version: Optional[str] = None,
    issued_ts_from_ms: Optional[int] = None,
    issued_ts_to_ms: Optional[int] = None,
    target_ts_from_ms: Optional[int] = None,
    target_ts_to_ms: Optional[int] = None,
    min_horizon_hr: Optional[float] = None,
    max_horizon_hr: Optional[float] = None,
    frozen_next_day: bool = False,
) -> list[tuple[int, int, Optional[float], float, float, float, float, float, float]]:
    filters = [
        "pl.actual_value IS NOT NULL",
        "pl.model_error IS NOT NULL",
        "pl.harmonie_error IS NOT NULL",
        "pl.model_abs_error IS NOT NULL",
        "pl.harmonie_abs_error IS NOT NULL",
        "pl.model_sq_error IS NOT NULL",
        "pl.harmonie_sq_error IS NOT NULL",
    ]
    params: list[Any] = []
    if site is not None:
        filters.append("pl.site = ?")
        params.append(site)
    if model_type is not None:
        filters.append("pl.model_type = ?")
        params.append(model_type)
    if prediction_kind is not None:
        filters.append("pl.prediction_kind = ?")
        params.append(prediction_kind)
    if model_artifact is not None:
        filters.append("pl.model_artifact = ?")
        params.append(model_artifact)
    if model_version is not None:
        filters.append("pl.model_version = ?")
        params.append(model_version)
    if issued_ts_from_ms is not None:
        filters.append("pl.issued_ts >= ?")
        params.append(int(issued_ts_from_ms))
    if issued_ts_to_ms is not None:
        filters.append("pl.issued_ts <= ?")
        params.append(int(issued_ts_to_ms))
    if target_ts_from_ms is not None:
        filters.append("pl.target_ts >= ?")
        params.append(int(target_ts_from_ms))
    if target_ts_to_ms is not None:
        filters.append("pl.target_ts <= ?")
        params.append(int(target_ts_to_ms))
    if min_horizon_hr is not None:
        filters.append("pl.horizon_hr >= ?")
        params.append(float(min_horizon_hr))
    if max_horizon_hr is not None:
        filters.append("pl.horizon_hr <= ?")
        params.append(float(max_horizon_hr))

    if frozen_next_day:
        first_issue_filters = ["1 = 1"]
        first_issue_params: list[Any] = []
        if site is not None:
            first_issue_filters.append("site = ?")
            first_issue_params.append(site)
        if model_type is not None:
            first_issue_filters.append("model_type = ?")
            first_issue_params.append(model_type)
        if prediction_kind is not None:
            first_issue_filters.append("prediction_kind = ?")
            first_issue_params.append(prediction_kind)
        if model_artifact is not None:
            first_issue_filters.append("model_artifact = ?")
            first_issue_params.append(model_artifact)
        if model_version is not None:
            first_issue_filters.append("model_version = ?")
            first_issue_params.append(model_version)

        rows = conn.execute(
            f"""
            WITH first_issue AS (
                SELECT
                    site,
                    model_type,
                    prediction_kind,
                    anchor_ts,
                    target_ts,
                    MIN(issued_ts) AS issued_ts
                FROM {PREDICTION_LOG_TABLE}
                WHERE {" AND ".join(first_issue_filters)}
                GROUP BY site, model_type, prediction_kind, anchor_ts, target_ts
            )
            SELECT
                pl.issued_ts,
                pl.target_ts,
                pl.horizon_hr,
                pl.model_error,
                pl.harmonie_error,
                pl.model_abs_error,
                pl.harmonie_abs_error,
                pl.model_sq_error,
                pl.harmonie_sq_error
            FROM {PREDICTION_LOG_TABLE} AS pl
            INNER JOIN first_issue AS fi
                ON pl.site = fi.site
               AND pl.model_type = fi.model_type
               AND pl.prediction_kind = fi.prediction_kind
               AND pl.anchor_ts = fi.anchor_ts
               AND pl.target_ts = fi.target_ts
               AND pl.issued_ts = fi.issued_ts
            WHERE {" AND ".join(filters)}
            ORDER BY pl.issued_ts ASC, pl.target_ts ASC
            """,
            [*first_issue_params, *params],
        ).fetchall()
    else:
        rows = conn.execute(
            f"""
            SELECT
                pl.issued_ts,
                pl.target_ts,
                pl.horizon_hr,
                pl.model_error,
                pl.harmonie_error,
                pl.model_abs_error,
                pl.harmonie_abs_error,
                pl.model_sq_error,
                pl.harmonie_sq_error
            FROM {PREDICTION_LOG_TABLE} AS pl
            WHERE {" AND ".join(filters)}
            ORDER BY pl.issued_ts ASC, pl.target_ts ASC
            """,
            params,
        ).fetchall()

    return [
        (
            int(row[0]),
            int(row[1]),
            None if row[2] is None else float(row[2]),
            float(row[3]),
            float(row[4]),
            float(row[5]),
            float(row[6]),
            float(row[7]),
            float(row[8]),
        )
        for row in rows
    ]


def _utc_day_string(ms: int) -> str:
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def _safe_pct_improvement(baseline: Optional[float], candidate: Optional[float]) -> Optional[float]:
    if baseline is None or candidate is None:
        return None
    baseline_f = float(baseline)
    if abs(baseline_f) < 1e-9:
        return None
    return float((baseline_f - float(candidate)) / baseline_f * 100.0)


def _finalize_error_summary(
    *,
    count: int,
    model_abs_sum: float,
    harmonie_abs_sum: float,
    model_sq_sum: float,
    harmonie_sq_sum: float,
    model_error_sum: float = 0.0,
    harmonie_error_sum: float = 0.0,
    model_win_count: int = 0,
    harmonie_win_count: int = 0,
    tie_count: int = 0,
) -> Dict[str, Any]:
    if count <= 0:
        return {
            "count": 0,
            "model_mae": None,
            "harmonie_mae": None,
            "model_rmse": None,
            "harmonie_rmse": None,
            "model_bias": None,
            "harmonie_bias": None,
            "mae_improvement": None,
            "rmse_improvement": None,
            "mae_improvement_pct": None,
            "rmse_improvement_pct": None,
            "model_win_count_vs_harmonie": 0,
            "harmonie_win_count_vs_model": 0,
            "tie_count": 0,
            "model_win_rate_vs_harmonie": None,
            "harmonie_win_rate_vs_model": None,
            "tie_rate": None,
        }

    model_mae = float(model_abs_sum / count)
    harmonie_mae = float(harmonie_abs_sum / count)
    model_rmse = float(math.sqrt(model_sq_sum / count))
    harmonie_rmse = float(math.sqrt(harmonie_sq_sum / count))
    return {
        "count": int(count),
        "model_mae": model_mae,
        "harmonie_mae": harmonie_mae,
        "model_rmse": model_rmse,
        "harmonie_rmse": harmonie_rmse,
        "model_bias": float(model_error_sum / count),
        "harmonie_bias": float(harmonie_error_sum / count),
        "mae_improvement": float(harmonie_mae - model_mae),
        "rmse_improvement": float(harmonie_rmse - model_rmse),
        "mae_improvement_pct": _safe_pct_improvement(harmonie_mae, model_mae),
        "rmse_improvement_pct": _safe_pct_improvement(harmonie_rmse, model_rmse),
        "model_win_count_vs_harmonie": int(model_win_count),
        "harmonie_win_count_vs_model": int(harmonie_win_count),
        "tie_count": int(tie_count),
        "model_win_rate_vs_harmonie": float(model_win_count / count),
        "harmonie_win_rate_vs_model": float(harmonie_win_count / count),
        "tie_rate": float(tie_count / count),
    }


def _load_next_day_realized_rows(
    conn: sqlite3.Connection,
    site: Optional[str] = None,
    prediction_kind: str = "wind_speed",
    model_artifact: Optional[str] = None,
    model_version: Optional[str] = None,
    issued_ts_from_ms: Optional[int] = None,
    issued_ts_to_ms: Optional[int] = None,
    target_ts_from_ms: Optional[int] = None,
    target_ts_to_ms: Optional[int] = None,
) -> list[tuple[int, int, Optional[float], float, float, float, float, float, float]]:
    """
    Load realised frozen next-day evaluation rows from prediction_log.

    A next-day row is the canonical daily issuance row written with
    model_type='next_day'. These rows come from the vintage-aware next-day
    prediction path and therefore exclude rolling intraday/current-day contexts.
    When the same anchor_ts/target_ts was rerun later, we keep the earliest
    issued_ts branch so frozen next-day reporting does not drift with reruns.
    """
    return _load_prediction_log_realized_rows(
        conn,
        site=site,
        model_type="next_day",
        prediction_kind=prediction_kind,
        model_artifact=model_artifact,
        model_version=model_version,
        issued_ts_from_ms=issued_ts_from_ms,
        issued_ts_to_ms=issued_ts_to_ms,
        target_ts_from_ms=target_ts_from_ms,
        target_ts_to_ms=target_ts_to_ms,
        frozen_next_day=True,
    )


def load_next_day_realized_detail_rows(
    conn: sqlite3.Connection,
    site: Optional[str] = None,
    prediction_kind: str = "wind_speed",
    issued_day_utc: Optional[str] = None,
) -> list[Dict[str, Any]]:
    """
    Load canonical frozen next-day realised detail rows for reporting/export.

    These rows are the earliest issued branch per (anchor_ts, target_ts), which
    keeps daily next-day reporting tied to the frozen operational issuance
    rather than later reruns or dashboard snapshots.
    """
    first_issue_filters = [
        "model_type = 'next_day'",
        "prediction_kind = ?",
    ]
    first_issue_params: list[Any] = [prediction_kind]
    if site is not None:
        first_issue_filters.append("site = ?")
        first_issue_params.append(site)

    filters = [
        "pl.actual_value IS NOT NULL",
        "pl.model_abs_error IS NOT NULL",
        "pl.harmonie_abs_error IS NOT NULL",
    ]
    params: list[Any] = []
    if issued_day_utc is not None:
        filters.append("strftime('%Y-%m-%d', pl.issued_ts / 1000, 'unixepoch') = ?")
        params.append(str(issued_day_utc))

    rows = conn.execute(
        f"""
        WITH first_issue AS (
            SELECT
                site,
                prediction_kind,
                anchor_ts,
                target_ts,
                MIN(issued_ts) AS issued_ts
            FROM {PREDICTION_LOG_TABLE}
            WHERE {" AND ".join(first_issue_filters)}
            GROUP BY site, prediction_kind, anchor_ts, target_ts
        )
        SELECT
            pl.issued_ts,
            pl.issued_iso,
            pl.anchor_ts,
            pl.anchor_iso,
            pl.target_ts,
            pl.target_iso,
            pl.horizon_hr,
            pl.actual_value,
            pl.prediction_value,
            pl.harmonie_value,
            pl.model_abs_error,
            pl.harmonie_abs_error,
            pl.model_sq_error,
            pl.harmonie_sq_error
        FROM {PREDICTION_LOG_TABLE} AS pl
        INNER JOIN first_issue AS fi
            ON pl.site = fi.site
           AND pl.prediction_kind = fi.prediction_kind
           AND pl.anchor_ts = fi.anchor_ts
           AND pl.target_ts = fi.target_ts
           AND pl.issued_ts = fi.issued_ts
        WHERE {" AND ".join(filters)}
        ORDER BY pl.issued_ts ASC, pl.target_ts ASC
        """,
        [*first_issue_params, *params],
    ).fetchall()
    return [
        {
            "issued_ts": int(row[0]),
            "issued_iso": row[1],
            "anchor_ts": int(row[2]),
            "anchor_iso": row[3],
            "target_ts": int(row[4]),
            "target_iso": row[5],
            "horizon_hr": None if row[6] is None else float(row[6]),
            "actual_value": None if row[7] is None else float(row[7]),
            "prediction_value": None if row[8] is None else float(row[8]),
            "harmonie_value": None if row[9] is None else float(row[9]),
            "model_abs_error": None if row[10] is None else float(row[10]),
            "harmonie_abs_error": None if row[11] is None else float(row[11]),
            "model_sq_error": None if row[12] is None else float(row[12]),
            "harmonie_sq_error": None if row[13] is None else float(row[13]),
        }
        for row in rows
    ]


def summarize_next_day_vs_harmonie(
    conn: sqlite3.Connection,
    site: Optional[str] = None,
    prediction_kind: str = "wind_speed",
    model_artifact: Optional[str] = None,
    model_version: Optional[str] = None,
    issued_ts_from_ms: Optional[int] = None,
    issued_ts_to_ms: Optional[int] = None,
    target_ts_from_ms: Optional[int] = None,
    target_ts_to_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Summarize realised next-day model-vs-Harmonie performance from prediction_log.

    The summary uses only canonical realised next-day rows, i.e. frozen daily
    next-day issuances logged with model_type='next_day'. It does not mix in
    intraday rows, website snapshots, or ex-post reconstructed latest views.
    """
    rows = _load_next_day_realized_rows(
        conn,
        site=site,
        prediction_kind=prediction_kind,
        model_artifact=model_artifact,
        model_version=model_version,
        issued_ts_from_ms=issued_ts_from_ms,
        issued_ts_to_ms=issued_ts_to_ms,
        target_ts_from_ms=target_ts_from_ms,
        target_ts_to_ms=target_ts_to_ms,
    )
    summary = _finalize_error_summary(
        count=len(rows),
        model_error_sum=sum(row[3] for row in rows),
        harmonie_error_sum=sum(row[4] for row in rows),
        model_abs_sum=sum(row[5] for row in rows),
        harmonie_abs_sum=sum(row[6] for row in rows),
        model_sq_sum=sum(row[7] for row in rows),
        harmonie_sq_sum=sum(row[8] for row in rows),
        model_win_count=sum(1 for row in rows if row[5] < row[6]),
        harmonie_win_count=sum(1 for row in rows if row[6] < row[5]),
        tie_count=sum(1 for row in rows if row[5] == row[6]),
    )
    summary.update(
        {
            "site": site,
            "model_type": "next_day",
            "prediction_kind": prediction_kind,
            "model_artifact": model_artifact,
            "model_version": model_version,
            "issued_ts_from_ms": issued_ts_from_ms,
            "issued_ts_to_ms": issued_ts_to_ms,
            "target_ts_from_ms": target_ts_from_ms,
            "target_ts_to_ms": target_ts_to_ms,
        }
    )
    return summary


def summarize_next_day_vs_harmonie_by_issued_day(
    conn: sqlite3.Connection,
    site: Optional[str] = None,
    prediction_kind: str = "wind_speed",
    model_artifact: Optional[str] = None,
    model_version: Optional[str] = None,
    issued_ts_from_ms: Optional[int] = None,
    issued_ts_to_ms: Optional[int] = None,
    target_ts_from_ms: Optional[int] = None,
    target_ts_to_ms: Optional[int] = None,
) -> list[Dict[str, Any]]:
    """
    Group realised next-day model-vs-Harmonie performance by issued UTC day.

    This is the canonical frozen next-day issuance view: each group contains
    only model_type='next_day' rows whose realised actuals have been
    materialized into prediction_log.
    """
    rows = _load_next_day_realized_rows(
        conn,
        site=site,
        prediction_kind=prediction_kind,
        model_artifact=model_artifact,
        model_version=model_version,
        issued_ts_from_ms=issued_ts_from_ms,
        issued_ts_to_ms=issued_ts_to_ms,
        target_ts_from_ms=target_ts_from_ms,
        target_ts_to_ms=target_ts_to_ms,
    )
    grouped: Dict[str, Dict[str, float | int]] = {}
    for issued_ts, _target_ts, _horizon_hr, model_error, harmonie_error, model_abs, harmonie_abs, model_sq, harmonie_sq in rows:
        issued_day = _utc_day_string(issued_ts)
        bucket = grouped.setdefault(
            issued_day,
            {
                "count": 0,
                "model_error_sum": 0.0,
                "harmonie_error_sum": 0.0,
                "model_abs_sum": 0.0,
                "harmonie_abs_sum": 0.0,
                "model_sq_sum": 0.0,
                "harmonie_sq_sum": 0.0,
                "model_win_count": 0,
                "harmonie_win_count": 0,
                "tie_count": 0,
            },
        )
        bucket["count"] += 1
        bucket["model_error_sum"] += model_error
        bucket["harmonie_error_sum"] += harmonie_error
        bucket["model_abs_sum"] += model_abs
        bucket["harmonie_abs_sum"] += harmonie_abs
        bucket["model_sq_sum"] += model_sq
        bucket["harmonie_sq_sum"] += harmonie_sq
        if model_abs < harmonie_abs:
            bucket["model_win_count"] += 1
        elif harmonie_abs < model_abs:
            bucket["harmonie_win_count"] += 1
        else:
            bucket["tie_count"] += 1

    out: list[Dict[str, Any]] = []
    for issued_day in sorted(grouped):
        bucket = grouped[issued_day]
        summary = _finalize_error_summary(
            count=int(bucket["count"]),
            model_error_sum=float(bucket["model_error_sum"]),
            harmonie_error_sum=float(bucket["harmonie_error_sum"]),
            model_abs_sum=float(bucket["model_abs_sum"]),
            harmonie_abs_sum=float(bucket["harmonie_abs_sum"]),
            model_sq_sum=float(bucket["model_sq_sum"]),
            harmonie_sq_sum=float(bucket["harmonie_sq_sum"]),
            model_win_count=int(bucket["model_win_count"]),
            harmonie_win_count=int(bucket["harmonie_win_count"]),
            tie_count=int(bucket["tie_count"]),
        )
        summary.update(
            {
                "issued_day_utc": issued_day,
                "site": site,
                "model_type": "next_day",
                "prediction_kind": prediction_kind,
                "model_artifact": model_artifact,
                "model_version": model_version,
            }
        )
        out.append(summary)
    return out


def summarize_next_day_vs_harmonie_by_horizon(
    conn: sqlite3.Connection,
    site: Optional[str] = None,
    prediction_kind: str = "wind_speed",
    model_artifact: Optional[str] = None,
    model_version: Optional[str] = None,
    issued_ts_from_ms: Optional[int] = None,
    issued_ts_to_ms: Optional[int] = None,
    target_ts_from_ms: Optional[int] = None,
    target_ts_to_ms: Optional[int] = None,
) -> list[Dict[str, Any]]:
    """
    Group realised next-day model-vs-Harmonie performance by forecast lead time.

    Lead time comes from the logged next-day horizon_hr on canonical
    model_type='next_day' rows, so the summary preserves the same frozen daily
    issuance semantics as the underlying prediction_log.
    """
    rows = _load_next_day_realized_rows(
        conn,
        site=site,
        prediction_kind=prediction_kind,
        model_artifact=model_artifact,
        model_version=model_version,
        issued_ts_from_ms=issued_ts_from_ms,
        issued_ts_to_ms=issued_ts_to_ms,
        target_ts_from_ms=target_ts_from_ms,
        target_ts_to_ms=target_ts_to_ms,
    )
    grouped: Dict[int, Dict[str, float | int]] = {}
    for _issued_ts, _target_ts, horizon_hr, model_error, harmonie_error, model_abs, harmonie_abs, model_sq, harmonie_sq in rows:
        if horizon_hr is None:
            continue
        horizon_key = int(round(horizon_hr))
        bucket = grouped.setdefault(
            horizon_key,
            {
                "count": 0,
                "model_error_sum": 0.0,
                "harmonie_error_sum": 0.0,
                "model_abs_sum": 0.0,
                "harmonie_abs_sum": 0.0,
                "model_sq_sum": 0.0,
                "harmonie_sq_sum": 0.0,
                "model_win_count": 0,
                "harmonie_win_count": 0,
                "tie_count": 0,
            },
        )
        bucket["count"] += 1
        bucket["model_error_sum"] += model_error
        bucket["harmonie_error_sum"] += harmonie_error
        bucket["model_abs_sum"] += model_abs
        bucket["harmonie_abs_sum"] += harmonie_abs
        bucket["model_sq_sum"] += model_sq
        bucket["harmonie_sq_sum"] += harmonie_sq
        if model_abs < harmonie_abs:
            bucket["model_win_count"] += 1
        elif harmonie_abs < model_abs:
            bucket["harmonie_win_count"] += 1
        else:
            bucket["tie_count"] += 1

    out: list[Dict[str, Any]] = []
    for horizon_hr in sorted(grouped):
        bucket = grouped[horizon_hr]
        summary = _finalize_error_summary(
            count=int(bucket["count"]),
            model_error_sum=float(bucket["model_error_sum"]),
            harmonie_error_sum=float(bucket["harmonie_error_sum"]),
            model_abs_sum=float(bucket["model_abs_sum"]),
            harmonie_abs_sum=float(bucket["harmonie_abs_sum"]),
            model_sq_sum=float(bucket["model_sq_sum"]),
            harmonie_sq_sum=float(bucket["harmonie_sq_sum"]),
            model_win_count=int(bucket["model_win_count"]),
            harmonie_win_count=int(bucket["harmonie_win_count"]),
            tie_count=int(bucket["tie_count"]),
        )
        summary.update(
            {
                "horizon_hr": int(horizon_hr),
                "site": site,
                "model_type": "next_day",
                "prediction_kind": prediction_kind,
                "model_artifact": model_artifact,
                "model_version": model_version,
            }
        )
        out.append(summary)
    return out


def summarize_prediction_log_vs_harmonie(
    conn: sqlite3.Connection,
    *,
    site: Optional[str] = None,
    model_type: Optional[str] = None,
    prediction_kind: Optional[str] = "wind_speed",
    model_artifact: Optional[str] = None,
    model_version: Optional[str] = None,
    issued_ts_from_ms: Optional[int] = None,
    issued_ts_to_ms: Optional[int] = None,
    target_ts_from_ms: Optional[int] = None,
    target_ts_to_ms: Optional[int] = None,
    min_horizon_hr: Optional[float] = None,
    max_horizon_hr: Optional[float] = None,
    frozen_next_day: bool = False,
) -> Dict[str, Any]:
    rows = _load_prediction_log_realized_rows(
        conn,
        site=site,
        model_type=model_type,
        prediction_kind=prediction_kind,
        model_artifact=model_artifact,
        model_version=model_version,
        issued_ts_from_ms=issued_ts_from_ms,
        issued_ts_to_ms=issued_ts_to_ms,
        target_ts_from_ms=target_ts_from_ms,
        target_ts_to_ms=target_ts_to_ms,
        min_horizon_hr=min_horizon_hr,
        max_horizon_hr=max_horizon_hr,
        frozen_next_day=frozen_next_day,
    )
    summary = _finalize_error_summary(
        count=len(rows),
        model_error_sum=sum(row[3] for row in rows),
        harmonie_error_sum=sum(row[4] for row in rows),
        model_abs_sum=sum(row[5] for row in rows),
        harmonie_abs_sum=sum(row[6] for row in rows),
        model_sq_sum=sum(row[7] for row in rows),
        harmonie_sq_sum=sum(row[8] for row in rows),
        model_win_count=sum(1 for row in rows if row[5] < row[6]),
        harmonie_win_count=sum(1 for row in rows if row[6] < row[5]),
        tie_count=sum(1 for row in rows if row[5] == row[6]),
    )
    summary.update(
        {
            "site": site,
            "model_type": model_type,
            "prediction_kind": prediction_kind,
            "model_artifact": model_artifact,
            "model_version": model_version,
            "issued_ts_from_ms": issued_ts_from_ms,
            "issued_ts_to_ms": issued_ts_to_ms,
            "target_ts_from_ms": target_ts_from_ms,
            "target_ts_to_ms": target_ts_to_ms,
            "min_horizon_hr": min_horizon_hr,
            "max_horizon_hr": max_horizon_hr,
            "frozen_next_day": bool(frozen_next_day),
        }
    )
    return summary


def summarize_prediction_log_vs_harmonie_by_issued_day(
    conn: sqlite3.Connection,
    *,
    site: Optional[str] = None,
    model_type: Optional[str] = None,
    prediction_kind: Optional[str] = "wind_speed",
    model_artifact: Optional[str] = None,
    model_version: Optional[str] = None,
    issued_ts_from_ms: Optional[int] = None,
    issued_ts_to_ms: Optional[int] = None,
    target_ts_from_ms: Optional[int] = None,
    target_ts_to_ms: Optional[int] = None,
    min_horizon_hr: Optional[float] = None,
    max_horizon_hr: Optional[float] = None,
    frozen_next_day: bool = False,
) -> list[Dict[str, Any]]:
    rows = _load_prediction_log_realized_rows(
        conn,
        site=site,
        model_type=model_type,
        prediction_kind=prediction_kind,
        model_artifact=model_artifact,
        model_version=model_version,
        issued_ts_from_ms=issued_ts_from_ms,
        issued_ts_to_ms=issued_ts_to_ms,
        target_ts_from_ms=target_ts_from_ms,
        target_ts_to_ms=target_ts_to_ms,
        min_horizon_hr=min_horizon_hr,
        max_horizon_hr=max_horizon_hr,
        frozen_next_day=frozen_next_day,
    )
    grouped: Dict[str, Dict[str, float | int]] = {}
    for issued_ts, _target_ts, _horizon_hr, model_error, harmonie_error, model_abs, harmonie_abs, model_sq, harmonie_sq in rows:
        issued_day = _utc_day_string(issued_ts)
        bucket = grouped.setdefault(
            issued_day,
            {
                "count": 0,
                "model_error_sum": 0.0,
                "harmonie_error_sum": 0.0,
                "model_abs_sum": 0.0,
                "harmonie_abs_sum": 0.0,
                "model_sq_sum": 0.0,
                "harmonie_sq_sum": 0.0,
                "model_win_count": 0,
                "harmonie_win_count": 0,
                "tie_count": 0,
            },
        )
        bucket["count"] += 1
        bucket["model_error_sum"] += model_error
        bucket["harmonie_error_sum"] += harmonie_error
        bucket["model_abs_sum"] += model_abs
        bucket["harmonie_abs_sum"] += harmonie_abs
        bucket["model_sq_sum"] += model_sq
        bucket["harmonie_sq_sum"] += harmonie_sq
        if model_abs < harmonie_abs:
            bucket["model_win_count"] += 1
        elif harmonie_abs < model_abs:
            bucket["harmonie_win_count"] += 1
        else:
            bucket["tie_count"] += 1

    out: list[Dict[str, Any]] = []
    for issued_day in sorted(grouped):
        bucket = grouped[issued_day]
        summary = _finalize_error_summary(
            count=int(bucket["count"]),
            model_error_sum=float(bucket["model_error_sum"]),
            harmonie_error_sum=float(bucket["harmonie_error_sum"]),
            model_abs_sum=float(bucket["model_abs_sum"]),
            harmonie_abs_sum=float(bucket["harmonie_abs_sum"]),
            model_sq_sum=float(bucket["model_sq_sum"]),
            harmonie_sq_sum=float(bucket["harmonie_sq_sum"]),
            model_win_count=int(bucket["model_win_count"]),
            harmonie_win_count=int(bucket["harmonie_win_count"]),
            tie_count=int(bucket["tie_count"]),
        )
        summary.update(
            {
                "issued_day_utc": issued_day,
                "site": site,
                "model_type": model_type,
                "prediction_kind": prediction_kind,
                "model_artifact": model_artifact,
                "model_version": model_version,
                "frozen_next_day": bool(frozen_next_day),
            }
        )
        out.append(summary)
    return out


def summarize_prediction_log_vs_harmonie_by_horizon(
    conn: sqlite3.Connection,
    *,
    site: Optional[str] = None,
    model_type: Optional[str] = None,
    prediction_kind: Optional[str] = "wind_speed",
    model_artifact: Optional[str] = None,
    model_version: Optional[str] = None,
    issued_ts_from_ms: Optional[int] = None,
    issued_ts_to_ms: Optional[int] = None,
    target_ts_from_ms: Optional[int] = None,
    target_ts_to_ms: Optional[int] = None,
    min_horizon_hr: Optional[float] = None,
    max_horizon_hr: Optional[float] = None,
    frozen_next_day: bool = False,
) -> list[Dict[str, Any]]:
    rows = _load_prediction_log_realized_rows(
        conn,
        site=site,
        model_type=model_type,
        prediction_kind=prediction_kind,
        model_artifact=model_artifact,
        model_version=model_version,
        issued_ts_from_ms=issued_ts_from_ms,
        issued_ts_to_ms=issued_ts_to_ms,
        target_ts_from_ms=target_ts_from_ms,
        target_ts_to_ms=target_ts_to_ms,
        min_horizon_hr=min_horizon_hr,
        max_horizon_hr=max_horizon_hr,
        frozen_next_day=frozen_next_day,
    )
    grouped: Dict[int, Dict[str, float | int]] = {}
    for _issued_ts, _target_ts, horizon_hr, model_error, harmonie_error, model_abs, harmonie_abs, model_sq, harmonie_sq in rows:
        if horizon_hr is None:
            continue
        horizon_key = int(round(horizon_hr))
        bucket = grouped.setdefault(
            horizon_key,
            {
                "count": 0,
                "model_error_sum": 0.0,
                "harmonie_error_sum": 0.0,
                "model_abs_sum": 0.0,
                "harmonie_abs_sum": 0.0,
                "model_sq_sum": 0.0,
                "harmonie_sq_sum": 0.0,
                "model_win_count": 0,
                "harmonie_win_count": 0,
                "tie_count": 0,
            },
        )
        bucket["count"] += 1
        bucket["model_error_sum"] += model_error
        bucket["harmonie_error_sum"] += harmonie_error
        bucket["model_abs_sum"] += model_abs
        bucket["harmonie_abs_sum"] += harmonie_abs
        bucket["model_sq_sum"] += model_sq
        bucket["harmonie_sq_sum"] += harmonie_sq
        if model_abs < harmonie_abs:
            bucket["model_win_count"] += 1
        elif harmonie_abs < model_abs:
            bucket["harmonie_win_count"] += 1
        else:
            bucket["tie_count"] += 1

    out: list[Dict[str, Any]] = []
    for horizon_hr in sorted(grouped):
        bucket = grouped[horizon_hr]
        summary = _finalize_error_summary(
            count=int(bucket["count"]),
            model_error_sum=float(bucket["model_error_sum"]),
            harmonie_error_sum=float(bucket["harmonie_error_sum"]),
            model_abs_sum=float(bucket["model_abs_sum"]),
            harmonie_abs_sum=float(bucket["harmonie_abs_sum"]),
            model_sq_sum=float(bucket["model_sq_sum"]),
            harmonie_sq_sum=float(bucket["harmonie_sq_sum"]),
            model_win_count=int(bucket["model_win_count"]),
            harmonie_win_count=int(bucket["harmonie_win_count"]),
            tie_count=int(bucket["tie_count"]),
        )
        summary.update(
            {
                "horizon_hr": int(horizon_hr),
                "site": site,
                "model_type": model_type,
                "prediction_kind": prediction_kind,
                "model_artifact": model_artifact,
                "model_version": model_version,
                "frozen_next_day": bool(frozen_next_day),
            }
        )
        out.append(summary)
    return out


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


def _now_ms() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


def _username_norm(username: str) -> str:
    return username.strip().casefold()


def create_user(conn: sqlite3.Connection, username: str, password_hash: str) -> int:
    now = _now_ms()
    cur = conn.execute(
        """
        INSERT INTO users(username, username_norm, password_hash, created_ts, created_iso)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            username.strip(),
            _username_norm(username),
            password_hash,
            now,
            _iso_utc_from_ms(now),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def get_user_by_username(conn: sqlite3.Connection, username: str) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        """
        SELECT id, username, username_norm, password_hash, created_ts, created_iso, last_login_ts, last_login_iso
        FROM users
        WHERE username_norm = ?
        """,
        (_username_norm(username),),
    ).fetchone()
    if row is None:
        return None
    return {
        "id": int(row[0]),
        "username": row[1],
        "username_norm": row[2],
        "password_hash": row[3],
        "created_ts": row[4],
        "created_iso": row[5],
        "last_login_ts": row[6],
        "last_login_iso": row[7],
    }


def get_user_by_id(conn: sqlite3.Connection, user_id: int) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        """
        SELECT id, username, username_norm, password_hash, created_ts, created_iso, last_login_ts, last_login_iso
        FROM users
        WHERE id = ?
        """,
        (int(user_id),),
    ).fetchone()
    if row is None:
        return None
    return {
        "id": int(row[0]),
        "username": row[1],
        "username_norm": row[2],
        "password_hash": row[3],
        "created_ts": row[4],
        "created_iso": row[5],
        "last_login_ts": row[6],
        "last_login_iso": row[7],
    }


def mark_user_login(conn: sqlite3.Connection, user_id: int) -> None:
    now = _now_ms()
    conn.execute(
        """
        UPDATE users
        SET last_login_ts = ?, last_login_iso = ?
        WHERE id = ?
        """,
        (now, _iso_utc_from_ms(now), int(user_id)),
    )
    conn.commit()


def get_user_profile(conn: sqlite3.Connection, user_id: int) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        """
        SELECT user_id, public_username, rider_name, rider_weight, default_spot, updated_ts, updated_iso
        FROM user_profiles
        WHERE user_id = ?
        """,
        (int(user_id),),
    ).fetchone()
    if row is None:
        return None
    return {
        "user_id": int(row[0]),
        "public_username": row[1] or "",
        "rider_name": row[2] or "",
        "rider_weight": row[3],
        "default_spot": row[4] or "",
        "updated_ts": row[5],
        "updated_iso": row[6],
    }


def _normalize_profile_identity(value: str | None) -> str:
    return (value or "").strip().casefold()


def find_user_profile_identity_conflicts(
    conn: sqlite3.Connection,
    user_id: int,
    *,
    public_username: str | None = None,
    rider_name: str | None = None,
) -> Dict[str, int]:
    requested_values = {
        "public_username": _normalize_profile_identity(public_username),
        "rider_name": _normalize_profile_identity(rider_name),
    }
    requested_values = {key: value for key, value in requested_values.items() if value}
    if not requested_values:
        return {}

    conflicts: Dict[str, int] = {}
    rows = conn.execute(
        """
        SELECT user_id, public_username, rider_name
        FROM user_profiles
        WHERE user_id != ?
        """,
        (int(user_id),),
    ).fetchall()
    for row in rows:
        other_user_id = int(row[0])
        if (
            "public_username" in requested_values
            and _normalize_profile_identity(row[1]) == requested_values["public_username"]
        ):
            conflicts["public_username"] = other_user_id
        if "rider_name" in requested_values and _normalize_profile_identity(row[2]) == requested_values["rider_name"]:
            conflicts["rider_name"] = other_user_id
    return conflicts


def upsert_user_profile(
    conn: sqlite3.Connection,
    user_id: int,
    public_username: str,
    rider_name: str,
    rider_weight: Optional[int],
    default_spot: str,
) -> None:
    now = _now_ms()
    conn.execute(
        """
        INSERT INTO user_profiles(
            user_id, public_username, rider_name, rider_weight, default_spot, updated_ts, updated_iso
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            public_username = excluded.public_username,
            rider_name = excluded.rider_name,
            rider_weight = excluded.rider_weight,
            default_spot = excluded.default_spot,
            updated_ts = excluded.updated_ts,
            updated_iso = excluded.updated_iso
        """,
        (
            int(user_id),
            public_username.strip() or None,
            rider_name.strip() or None,
            rider_weight,
            default_spot or None,
            now,
            _iso_utc_from_ms(now),
        ),
    )
    conn.commit()


def _extract_observation_measurement(
    payload_raw: Optional[str],
    fallback_value: Any,
    payload_keys: Iterable[str],
) -> Optional[float]:
    payload: Dict[str, Any] = {}
    if payload_raw:
        try:
            payload = json.loads(payload_raw)
        except json.JSONDecodeError:
            payload = {}
    value = _extract_first(payload, payload_keys)
    if value is None:
        value = fallback_value
    return _as_float(value)


def _wind_direction_label(direction_deg: Optional[float]) -> Optional[str]:
    if direction_deg is None:
        return None
    labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    index = int(((float(direction_deg) % 360.0) + 22.5) // 45.0) % len(labels)
    return labels[index]


def _wind_direction_display(direction_deg: Optional[float], label: Optional[str] = None) -> Optional[str]:
    if direction_deg is None:
        return None
    direction_label = label or _wind_direction_label(direction_deg)
    if not direction_label:
        return f"{float(direction_deg):.0f} deg"
    return f"{direction_label} ({float(direction_deg):.0f} deg)"


def get_forecast_temperature_for_session(
    conn: sqlite3.Connection,
    spot: str,
    start_ts_ms: int,
    end_ts_ms: int,
) -> Dict[str, Any]:
    site = SPOT_TO_SITE.get(spot)
    if site is None:
        return {
            "status": "unavailable",
            "reason": f"No forecast site is configured for {spot}.",
            "site": None,
            "summary": {
                "avg_temperature": None,
                "min_temperature": None,
                "max_temperature": None,
            },
        }

    rows = conn.execute(
        f"""
        SELECT target_ts, payload
        FROM {FORECASTS_TABLE}
        WHERE site = ?
          AND target_ts >= ?
          AND target_ts <= ?
        ORDER BY target_ts, fetched_ts DESC
        """,
        (site, int(start_ts_ms) - HOUR_MS // 2, int(end_ts_ms) + HOUR_MS // 2),
    ).fetchall()

    temperatures_by_target: Dict[int, float] = {}
    for target_ts, payload_raw in rows:
        if int(target_ts) in temperatures_by_target:
            continue
        temperature = _extract_observation_measurement(
            payload_raw,
            None,
            ["Temperature", "temperature", "temp", "air_temperature"],
        )
        if temperature is not None:
            temperatures_by_target[int(target_ts)] = float(temperature)

    temperatures = list(temperatures_by_target.values())
    summary = {
        "avg_temperature": None if not temperatures else sum(temperatures) / len(temperatures),
        "min_temperature": None if not temperatures else min(temperatures),
        "max_temperature": None if not temperatures else max(temperatures),
    }
    return {
        "status": "ok" if temperatures else "unavailable",
        "reason": None if temperatures else "No forecast temperature found for the selected spot and time window.",
        "site": site,
        "summary": summary,
    }


def get_prediction_lines_for_session(
    conn: sqlite3.Connection,
    spot: str,
    start_ts_ms: int,
    end_ts_ms: int,
) -> Dict[str, Any]:
    site = SPOT_TO_SITE.get(spot)
    if site is None:
        return {
            "status": "unavailable",
            "reason": f"No prediction site is configured for {spot}.",
            "site": None,
            "issued_ts": None,
            "issued_iso": None,
            "records": [],
        }

    target_from = int(start_ts_ms) - HOUR_MS
    target_to = int(end_ts_ms) + HOUR_MS
    issue_row = conn.execute(
        f"""
        SELECT issued_ts, issued_iso
        FROM {PREDICTION_LOG_TABLE}
        WHERE site = ?
          AND prediction_kind = 'wind_speed'
          AND issued_ts <= ?
          AND target_ts >= ?
          AND target_ts <= ?
          AND prediction_value IS NOT NULL
          AND harmonie_value IS NOT NULL
        ORDER BY issued_ts DESC
        LIMIT 1
        """,
        (site, int(start_ts_ms), target_from, target_to),
    ).fetchone()
    if issue_row is None:
        return {
            "status": "unavailable",
            "reason": "No issued wind-speed prediction found before the session start.",
            "site": site,
            "issued_ts": None,
            "issued_iso": None,
            "records": [],
        }

    issued_ts, issued_iso = issue_row
    rows = conn.execute(
        f"""
        SELECT target_ts, target_iso, prediction_value, harmonie_value
        FROM {PREDICTION_LOG_TABLE}
        WHERE site = ?
          AND prediction_kind = 'wind_speed'
          AND issued_ts = ?
          AND target_ts >= ?
          AND target_ts <= ?
          AND prediction_value IS NOT NULL
          AND harmonie_value IS NOT NULL
        ORDER BY target_ts
        """,
        (site, int(issued_ts), target_from, target_to),
    ).fetchall()
    hourly_records = [
        {
            "timestamp": int(target_ts),
            "iso_time": target_iso,
            "superlocal_wind_speed": float(prediction_value),
            "harmonie_wind_speed": float(harmonie_value),
        }
        for target_ts, target_iso, prediction_value, harmonie_value in rows
    ]
    records = _interpolate_prediction_records_to_plot_grid(hourly_records, int(start_ts_ms), int(end_ts_ms))
    return {
        "status": "ok" if records else "unavailable",
        "reason": None if records else "No prediction targets found for the selected session window.",
        "site": site,
        "issued_ts": int(issued_ts),
        "issued_iso": issued_iso,
        "records": records,
    }


def _interpolate_prediction_records_to_plot_grid(
    hourly_records: List[Dict[str, Any]],
    start_ts_ms: int,
    end_ts_ms: int,
) -> List[Dict[str, Any]]:
    if not hourly_records:
        return []
    ordered = sorted(hourly_records, key=lambda record: int(record["timestamp"]))
    dense_records: List[Dict[str, Any]] = []
    grid_ts = (int(start_ts_ms) // CURRENT_DAY_PLOT_INTERVAL_MS) * CURRENT_DAY_PLOT_INTERVAL_MS
    if grid_ts < int(start_ts_ms):
        grid_ts += CURRENT_DAY_PLOT_INTERVAL_MS
    idx = 0
    while grid_ts <= int(end_ts_ms):
        while idx + 1 < len(ordered) and int(ordered[idx + 1]["timestamp"]) <= grid_ts:
            idx += 1
        prev_record = ordered[idx]
        next_record = ordered[idx + 1] if idx + 1 < len(ordered) else None
        if int(prev_record["timestamp"]) > grid_ts:
            grid_ts += CURRENT_DAY_PLOT_INTERVAL_MS
            continue

        def interp_value(key: str) -> float:
            prev_value = float(prev_record[key])
            if next_record is None or int(next_record["timestamp"]) == int(prev_record["timestamp"]):
                return prev_value
            next_value = float(next_record[key])
            fraction = (grid_ts - int(prev_record["timestamp"])) / (
                int(next_record["timestamp"]) - int(prev_record["timestamp"])
            )
            return prev_value + (next_value - prev_value) * fraction

        dense_records.append(
            {
                "timestamp": int(grid_ts),
                "iso_time": _iso_utc_from_ms(int(grid_ts)),
                "superlocal_wind_speed": interp_value("superlocal_wind_speed"),
                "harmonie_wind_speed": interp_value("harmonie_wind_speed"),
            }
        )
        grid_ts += CURRENT_DAY_PLOT_INTERVAL_MS
    return dense_records


def get_measured_wind_for_session(
    conn: sqlite3.Connection,
    spot: str,
    start_ts_ms: int,
    end_ts_ms: int,
) -> Dict[str, Any]:
    site = SPOT_TO_SITE.get(spot)
    if site is None:
        return {
            "status": "unavailable",
            "reason": f"No measured wind site is configured for {spot}.",
            "site": None,
            "records": [],
            "summary": {
                "point_count": 0,
                "avg_wind_speed": None,
                "max_wind_speed": None,
                "min_wind_speed": None,
                "mean_wind_dir": None,
                "mean_wind_dir_label": None,
            },
        }

    rows = conn.execute(
        """
        SELECT ts, iso_time, wind_speed, wind_gust, wind_dir, payload
        FROM observations
        WHERE site = ?
          AND ts >= ?
          AND ts <= ?
        ORDER BY ts
        """,
        (site, int(start_ts_ms), int(end_ts_ms)),
    ).fetchall()
    prior_row = conn.execute(
        """
        SELECT ts, iso_time, wind_speed, wind_gust, wind_dir, payload
        FROM observations
        WHERE site = ?
          AND ts < ?
        ORDER BY ts DESC
        LIMIT 1
        """,
        (site, int(start_ts_ms)),
    ).fetchone()

    def observation_record(row: Any) -> Dict[str, Any]:
        ts, iso_time, wind_speed, wind_gust, wind_dir, payload_raw = row
        speed = _extract_observation_measurement(
            payload_raw,
            wind_speed,
            ["AverageWind", "wind_speed", "windspeed", "WS", "ff", "speed", "WindSpeedAvg"],
        )
        gust = _extract_observation_measurement(
            payload_raw,
            wind_gust,
            ["MaxWind", "WindSpeedMax", "wind_gust", "gust", "WG", "GUST", "fg"],
        )
        minimum = _extract_observation_measurement(
            payload_raw,
            None,
            ["MinWind", "WindSpeedMin"],
        )
        direction = _extract_observation_measurement(
            payload_raw,
            wind_dir,
            ["WindDirection", "wind_dir", "winddirection", "WD", "DD", "dir", "direction"],
        )
        return {
            "timestamp": int(ts),
            "iso_time": iso_time,
            "measured_wind_speed": speed,
            "measured_wind_gust": gust,
            "measured_wind_min": minimum,
            "measured_wind_max": gust,
            "measured_wind_direction": direction,
        }

    records = [observation_record(row) for row in rows]
    plot_source_records = ([observation_record(prior_row)] if prior_row is not None else []) + records
    plot_records = _forward_fill_observation_records_to_plot_grid(plot_source_records, int(start_ts_ms), int(end_ts_ms))
    speeds = [float(record["measured_wind_speed"]) for record in records if record["measured_wind_speed"] is not None]
    gusts = [float(record["measured_wind_gust"]) for record in records if record["measured_wind_gust"] is not None]
    directions = [
        float(record["measured_wind_direction"]) % 360.0
        for record in records
        if record["measured_wind_direction"] is not None
    ]
    avg_dir = None
    if directions:
        sin_sum = sum(math.sin(math.radians(value)) for value in directions)
        cos_sum = sum(math.cos(math.radians(value)) for value in directions)
        avg_dir = (math.degrees(math.atan2(sin_sum, cos_sum)) + 360.0) % 360.0

    rolling_standard_deviations = []
    for index, record in enumerate(records):
        window_start = int(record["timestamp"]) - 15 * 60 * 1000
        window_speeds = [
            float(candidate["measured_wind_speed"])
            for candidate in records[: index + 1]
            if int(candidate["timestamp"]) >= window_start and candidate["measured_wind_speed"] is not None
        ]
        if len(window_speeds) < 3:
            continue
        window_mean = sum(window_speeds) / len(window_speeds)
        variance = sum((value - window_mean) ** 2 for value in window_speeds) / (len(window_speeds) - 1)
        rolling_standard_deviations.append(math.sqrt(variance))

    summary = {
        "point_count": len(records),
        "avg_wind_speed": None if not speeds else sum(speeds) / len(speeds),
        "max_wind_speed": None if not speeds else max(speeds),
        "max_wind_speed_kind": "average_wind",
        "max_wind_gust": None if not gusts else max(gusts),
        "min_wind_speed": None if not speeds else min(speeds),
        "wind_variability": (
            None
            if not rolling_standard_deviations
            else sum(rolling_standard_deviations) / len(rolling_standard_deviations)
        ),
        "wind_variability_kind": "mean_15min_rolling_sample_standard_deviation",
        "mean_wind_dir": avg_dir,
        "mean_wind_dir_label": _wind_direction_label(avg_dir),
    }
    status = "ok" if records else "unavailable"
    result = {
        "status": status,
        "reason": None if records else "No measured wind observations found for the selected spot and time window.",
        "site": site,
        "records": records,
        "plot_records": plot_records,
        "summary": summary,
    }
    return result


def _forward_fill_observation_records_to_plot_grid(
    records: List[Dict[str, Any]],
    start_ts_ms: int,
    end_ts_ms: int,
) -> List[Dict[str, Any]]:
    if not records:
        return []
    ordered = sorted(records, key=lambda record: int(record["timestamp"]))
    dense_records: List[Dict[str, Any]] = []
    grid_ts = (int(start_ts_ms) // CURRENT_DAY_PLOT_INTERVAL_MS) * CURRENT_DAY_PLOT_INTERVAL_MS
    if grid_ts < int(start_ts_ms):
        grid_ts += CURRENT_DAY_PLOT_INTERVAL_MS
    idx = 0
    latest: Dict[str, Any] | None = None
    while grid_ts <= int(end_ts_ms):
        while idx < len(ordered) and int(ordered[idx]["timestamp"]) <= grid_ts:
            latest = ordered[idx]
            idx += 1
        if latest is not None:
            dense_records.append(
                {
                    "timestamp": int(grid_ts),
                    "iso_time": _iso_utc_from_ms(int(grid_ts)),
                    "measured_wind_speed": latest.get("measured_wind_speed"),
                    "measured_wind_gust": latest.get("measured_wind_gust"),
                    "measured_wind_min": latest.get("measured_wind_min"),
                    "measured_wind_max": latest.get("measured_wind_max"),
                    "measured_wind_direction": latest.get("measured_wind_direction"),
                }
            )
        grid_ts += CURRENT_DAY_PLOT_INTERVAL_MS
    return dense_records


def create_surf_experience(conn: sqlite3.Connection, experience: Dict[str, Any]) -> int:
    now = _now_ms()
    measured = experience["measured_wind"]
    summary = measured.get("summary") or {}
    forecast_temperature = get_forecast_temperature_for_session(
        conn,
        experience["spot"],
        int(experience["start_ts"]),
        int(experience["end_ts"]),
    )
    temperature_summary = forecast_temperature.get("summary") or {}
    cur = conn.execute(
        """
        INSERT INTO surf_experiences(
            user_id, submitted_ts, submitted_iso, rider, spot, date, start_time, end_time,
            start_ts, end_ts, session_rating, rider_review, rider_weight, wing_size,
            foil_size, rider_notes, measured_wind_data_json, measured_wind_status,
            measured_wind_point_count, avg_measured_wind_speed, max_measured_wind_speed,
            min_measured_wind_speed, avg_measured_wind_dir, mean_measured_direction,
            mean_measured_direction_label, avg_forecast_temperature,
            min_forecast_temperature, max_forecast_temperature, visibility
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, json(?), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(experience["user_id"]),
            now,
            _iso_utc_from_ms(now),
            experience["rider"],
            experience["spot"],
            experience["date"],
            experience["start_time"],
            experience["end_time"],
            int(experience["start_ts"]),
            int(experience["end_ts"]),
            int(experience["session_rating"]),
            experience.get("rider_review") or None,
            experience.get("rider_weight"),
            int(experience["wing_size"]),
            int(experience["foil_size"]),
            experience.get("rider_notes") or None,
            json.dumps(measured, ensure_ascii=False, sort_keys=True),
            measured.get("status") or "unavailable",
            int(summary.get("point_count") or 0),
            summary.get("avg_wind_speed"),
            summary.get("max_wind_speed"),
            summary.get("min_wind_speed"),
            summary.get("mean_wind_dir"),
            summary.get("mean_wind_dir"),
            summary.get("mean_wind_dir_label"),
            temperature_summary.get("avg_temperature"),
            temperature_summary.get("min_temperature"),
            temperature_summary.get("max_temperature"),
            _submission_visibility(experience.get("visibility")),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def update_surf_experience(
    conn: sqlite3.Connection,
    experience_id: int,
    user_id: int,
    experience: Dict[str, Any],
) -> bool:
    cur = conn.execute(
        """
        UPDATE surf_experiences
        SET rider = ?,
            spot = ?,
            date = ?,
            start_time = ?,
            end_time = ?,
            start_ts = ?,
            end_ts = ?,
            session_rating = ?,
            rider_review = ?,
            rider_weight = ?,
            wing_size = ?,
            foil_size = ?,
            rider_notes = ?,
            visibility = ?
        WHERE id = ?
          AND user_id = ?
        """,
        (
            experience["rider"],
            experience["spot"],
            experience["date"],
            experience["start_time"],
            experience["end_time"],
            int(experience["start_ts"]),
            int(experience["end_ts"]),
            int(experience["session_rating"]),
            experience.get("rider_review") or None,
            experience.get("rider_weight"),
            int(experience["wing_size"]),
            int(experience["foil_size"]),
            experience.get("rider_notes") or None,
            _submission_visibility(experience.get("visibility")),
            int(experience_id),
            int(user_id),
        ),
    )
    if int(cur.rowcount) <= 0:
        conn.commit()
        return False
    refreshed = refresh_surf_experience_measured_wind(conn, int(experience_id), user_id=int(user_id))
    if not refreshed:
        conn.commit()
    return True


def refresh_surf_experience_measured_wind(
    conn: sqlite3.Connection,
    experience_id: int,
    user_id: Optional[int] = None,
) -> bool:
    if user_id is None:
        row = conn.execute(
            """
            SELECT id, spot, start_ts, end_ts
            FROM surf_experiences
            WHERE id = ?
            """,
            (int(experience_id),),
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT id, spot, start_ts, end_ts
            FROM surf_experiences
            WHERE id = ?
              AND user_id = ?
            """,
            (int(experience_id), int(user_id)),
        ).fetchone()
    if row is None:
        return False

    _, spot, start_ts, end_ts = row
    measured = get_measured_wind_for_session(conn, spot, int(start_ts), int(end_ts))
    summary = measured.get("summary") or {}
    forecast_temperature = get_forecast_temperature_for_session(conn, spot, int(start_ts), int(end_ts))
    temperature_summary = forecast_temperature.get("summary") or {}
    conn.execute(
        """
        UPDATE surf_experiences
        SET measured_wind_data_json = json(?),
            measured_wind_status = ?,
            measured_wind_point_count = ?,
            avg_measured_wind_speed = ?,
            max_measured_wind_speed = ?,
            min_measured_wind_speed = ?,
            avg_measured_wind_dir = ?,
            mean_measured_direction = ?,
            mean_measured_direction_label = ?,
            avg_forecast_temperature = ?,
            min_forecast_temperature = ?,
            max_forecast_temperature = ?
        WHERE id = ?
        """,
        (
            json.dumps(measured, ensure_ascii=False, sort_keys=True),
            measured.get("status") or "unavailable",
            int(summary.get("point_count") or 0),
            summary.get("avg_wind_speed"),
            summary.get("max_wind_speed"),
            summary.get("min_wind_speed"),
            summary.get("mean_wind_dir"),
            summary.get("mean_wind_dir"),
            summary.get("mean_wind_dir_label"),
            temperature_summary.get("avg_temperature"),
            temperature_summary.get("min_temperature"),
            temperature_summary.get("max_temperature"),
            int(experience_id),
        ),
    )
    conn.commit()
    return True


def backfill_surf_experience_measured_summaries(
    conn: sqlite3.Connection,
    user_id: Optional[int] = None,
) -> int:
    if user_id is None:
        rows = conn.execute(
            """
            SELECT id, spot
            FROM surf_experiences
            WHERE avg_measured_wind_speed IS NULL
               OR max_measured_wind_speed IS NULL
               OR min_measured_wind_speed IS NULL
               OR mean_measured_direction IS NULL
               OR mean_measured_direction_label IS NULL
               OR avg_forecast_temperature IS NULL
               OR COALESCE(json_extract(measured_wind_data_json, '$.summary.max_wind_speed_kind'), '') != 'average_wind'
               OR json_extract(measured_wind_data_json, '$.summary.max_wind_gust') IS NULL
            """
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT id, spot
            FROM surf_experiences
            WHERE user_id = ?
              AND (
                avg_measured_wind_speed IS NULL
                OR max_measured_wind_speed IS NULL
                OR min_measured_wind_speed IS NULL
                OR mean_measured_direction IS NULL
                OR mean_measured_direction_label IS NULL
                OR avg_forecast_temperature IS NULL
                OR COALESCE(json_extract(measured_wind_data_json, '$.summary.max_wind_speed_kind'), '') != 'average_wind'
                OR json_extract(measured_wind_data_json, '$.summary.max_wind_gust') IS NULL
              )
            """,
            (int(user_id),),
        ).fetchall()

    refreshed = 0
    for experience_id, spot in rows:
        if spot not in SPOT_TO_SITE:
            continue
        if refresh_surf_experience_measured_wind(conn, int(experience_id), user_id=user_id):
            refreshed += 1
    return refreshed


def delete_surf_experience(conn: sqlite3.Connection, user_id: int, experience_id: int) -> bool:
    cur = conn.execute(
        """
        DELETE FROM surf_experiences
        WHERE id = ?
          AND user_id = ?
        """,
        (int(experience_id), int(user_id)),
    )
    conn.commit()
    return int(cur.rowcount) > 0


def list_surf_experiences(
    conn: sqlite3.Connection,
    user_id: int,
    sort_key: str = "date",
    sort_dir: str = "desc",
    scope: str = "mine",
) -> List[Dict[str, Any]]:
    allowed_sort = {
        "date": "e.date",
        "visibility": "e.visibility COLLATE NOCASE",
        "rider": "COALESCE(NULLIF(TRIM(p.public_username), ''), 'Unknown rider') COLLATE NOCASE",
        "spot": "e.spot COLLATE NOCASE",
        "start_time": "e.start_time",
        "end_time": "e.end_time",
        "session_rating": "e.session_rating",
        "wing_size": "e.wing_size",
        "foil_size": "e.foil_size",
        "avg_measured_wind_speed": "e.avg_measured_wind_speed",
        "max_measured_wind_speed": "e.max_measured_wind_speed",
        "min_measured_wind_speed": "e.min_measured_wind_speed",
        "max_measured_wind_gust": "CAST(json_extract(e.measured_wind_data_json, '$.summary.max_wind_gust') AS REAL)",
        "wind_variability": "CAST(json_extract(e.measured_wind_data_json, '$.summary.wind_variability') AS REAL)",
        "mean_measured_direction": "e.mean_measured_direction",
        "avg_forecast_temperature": "e.avg_forecast_temperature",
    }
    order_expr = allowed_sort.get(sort_key, allowed_sort["date"])
    direction = "ASC" if sort_dir.lower() == "asc" else "DESC"
    where_clause = "e.user_id = ?" if scope != "all" else "(e.visibility = 'public' OR e.user_id = ?)"
    rows = conn.execute(
        f"""
        SELECT e.id, e.user_id, e.visibility,
               COALESCE(NULLIF(TRIM(p.public_username), ''), 'Unknown rider'),
               e.submitted_iso, e.rider, e.spot, e.date, e.start_time, e.end_time,
               e.session_rating, e.rider_review, e.wing_size, e.foil_size, e.rider_notes,
               e.measured_wind_status, e.measured_wind_point_count,
               e.avg_measured_wind_speed, e.max_measured_wind_speed, e.min_measured_wind_speed,
               CAST(json_extract(e.measured_wind_data_json, '$.summary.max_wind_gust') AS REAL),
               CAST(json_extract(e.measured_wind_data_json, '$.summary.wind_variability') AS REAL),
               COALESCE(e.mean_measured_direction, e.avg_measured_wind_dir),
               e.mean_measured_direction_label, e.avg_forecast_temperature
        FROM surf_experiences AS e
        LEFT JOIN user_profiles AS p ON p.user_id = e.user_id
        WHERE {where_clause}
        ORDER BY {order_expr} {direction}, e.start_time {direction}, e.id {direction}
        """,
        (int(user_id),),
    ).fetchall()
    experiences = []
    for row in rows:
        mean_direction = row[22]
        mean_direction_label = row[23] or _wind_direction_label(mean_direction)
        is_owner = int(row[1]) == int(user_id)
        experiences.append(
            {
                "id": int(row[0]),
                "user_id": int(row[1]),
                "visibility": _submission_visibility(row[2]),
                "is_owner": is_owner,
                "submitted_by": row[3],
                "rider_display": row[3],
                "submitted_iso": row[4],
                "rider": row[5] if is_owner else None,
                "spot": row[6],
                "date": row[7],
                "start_time": row[8],
                "end_time": row[9],
                "session_rating": int(row[10]),
                "rider_review": row[11] or "",
                "wing_size": int(row[12]),
                "foil_size": int(row[13]),
                "rider_notes": (row[14] or "") if is_owner else "",
                "measured_wind_status": row[15],
                "measured_wind_point_count": int(row[16]),
                "avg_measured_wind_speed": row[17],
                "max_measured_wind_speed": row[18],
                "min_measured_wind_speed": row[19],
                "max_measured_wind_gust": row[20],
                "wind_variability": row[21],
                "mean_measured_direction": mean_direction,
                "mean_measured_direction_label": mean_direction_label,
                "mean_measured_direction_display": _wind_direction_display(mean_direction, mean_direction_label),
                "avg_forecast_temperature": row[24],
            }
        )
    return experiences


def _get_surf_experience(
    conn: sqlite3.Connection,
    user_id: int,
    experience_id: int,
    allow_public: bool,
) -> Optional[Dict[str, Any]]:
    access_clause = "(e.user_id = ? OR e.visibility = 'public')" if allow_public else "e.user_id = ?"
    row = conn.execute(
        f"""
        SELECT e.id, e.user_id, e.visibility,
               COALESCE(NULLIF(TRIM(p.public_username), ''), 'Unknown rider'),
               e.submitted_iso, e.rider, e.spot, e.date, e.start_time, e.end_time,
               e.session_rating, e.rider_review, e.rider_weight, e.wing_size, e.foil_size,
               e.rider_notes, e.measured_wind_data_json, e.measured_wind_status,
               e.measured_wind_point_count, e.avg_measured_wind_speed,
               e.max_measured_wind_speed, e.min_measured_wind_speed,
               COALESCE(e.mean_measured_direction, e.avg_measured_wind_dir),
               e.mean_measured_direction_label, e.avg_forecast_temperature,
               e.min_forecast_temperature, e.max_forecast_temperature
        FROM surf_experiences AS e
        LEFT JOIN user_profiles AS p ON p.user_id = e.user_id
        WHERE {access_clause}
          AND e.id = ?
        """,
        (int(user_id), int(experience_id)),
    ).fetchone()
    if row is None:
        return None
    measured_raw = row[16] or "{}"
    try:
        measured = json.loads(measured_raw)
    except json.JSONDecodeError:
        measured = {"status": "unavailable", "records": [], "summary": {}}
    mean_direction = row[22]
    mean_direction_label = row[23] or _wind_direction_label(mean_direction)
    is_owner = int(row[1]) == int(user_id)
    return {
        "id": int(row[0]),
        "user_id": int(row[1]),
        "visibility": _submission_visibility(row[2]),
        "is_owner": is_owner,
        "submitted_by": row[3],
        "rider_display": row[3],
        "submitted_iso": row[4],
        "rider": row[5] if is_owner else None,
        "spot": row[6],
        "date": row[7],
        "start_time": row[8],
        "end_time": row[9],
        "session_rating": int(row[10]),
        "rider_review": row[11] or "",
        "rider_weight": row[12] if is_owner else None,
        "wing_size": int(row[13]),
        "foil_size": int(row[14]),
        "rider_notes": (row[15] or "") if is_owner else "",
        "measured_wind": measured,
        "measured_wind_status": row[17],
        "measured_wind_point_count": int(row[18]),
        "avg_measured_wind_speed": row[19],
        "max_measured_wind_speed": row[20],
        "min_measured_wind_speed": row[21],
        "mean_measured_direction": mean_direction,
        "mean_measured_direction_label": mean_direction_label,
        "mean_measured_direction_display": _wind_direction_display(mean_direction, mean_direction_label),
        "avg_forecast_temperature": row[24],
        "min_forecast_temperature": row[25],
        "max_forecast_temperature": row[26],
    }


def get_surf_experience(conn: sqlite3.Connection, user_id: int, experience_id: int) -> Optional[Dict[str, Any]]:
    return _get_surf_experience(conn, user_id, experience_id, allow_public=False)


def get_visible_surf_experience(conn: sqlite3.Connection, user_id: int, experience_id: int) -> Optional[Dict[str, Any]]:
    return _get_surf_experience(conn, user_id, experience_id, allow_public=True)
