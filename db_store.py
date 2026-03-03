import os
import json
import sqlite3
from datetime import datetime, timezone
from typing import Iterable, Dict, Any, Optional, Tuple


DB_FILENAME = "windsurfice.db"


def db_path(out_dir: str) -> str:
    return os.path.join(out_dir, DB_FILENAME)


def connect_db(out_dir: str) -> sqlite3.Connection:
    path = db_path(out_dir)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


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
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS forecasts (
            site TEXT NOT NULL,
            model TEXT,
            run_ts INTEGER NOT NULL,
            run_iso TEXT,
            target_ts INTEGER NOT NULL,
            target_iso TEXT,
            horizon_hr INTEGER,
            wind_speed REAL,
            wind_gust REAL,
            wind_dir REAL,
            payload TEXT,
            PRIMARY KEY (site, run_ts, target_ts)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_obs_ts ON observations(ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fc_target ON forecasts(target_ts)")
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


def _to_epoch_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


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
    run_ts_ms: Optional[int] = None,
) -> int:
    if run_ts_ms is None:
        run_ts_ms = _to_epoch_ms(datetime.now(timezone.utc))
    run_iso = _iso_utc_from_ms(run_ts_ms)

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
            """
            INSERT INTO forecasts(
                site, model, run_ts, run_iso, target_ts, target_iso, horizon_hr,
                wind_speed, wind_gust, wind_dir, payload
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,json(?))
            ON CONFLICT(site, run_ts, target_ts) DO UPDATE SET
                model=excluded.model,
                target_iso=excluded.target_iso,
                horizon_hr=excluded.horizon_hr,
                wind_speed=COALESCE(excluded.wind_speed, forecasts.wind_speed),
                wind_gust=COALESCE(excluded.wind_gust, forecasts.wind_gust),
                wind_dir=COALESCE(excluded.wind_dir, forecasts.wind_dir),
                payload=excluded.payload
            """,
            (
                site,
                model,
                run_ts_ms,
                run_iso,
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
