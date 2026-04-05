import os
import json
import math
import sqlite3
from datetime import datetime, timezone
from typing import Iterable, Dict, Any, Optional, Tuple


DB_FILENAME = "wind_data.db"
FORECASTS_TABLE = "forecasts"
PREDICTION_LOG_TABLE = "prediction_log"
HOUR_MS = 3_600_000
PREDICTION_LOG_EVAL_COLUMNS = {
    "model_error": "REAL",
    "harmonie_error": "REAL",
    "model_abs_error": "REAL",
    "harmonie_abs_error": "REAL",
    "model_sq_error": "REAL",
    "harmonie_sq_error": "REAL",
}


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
