#!/usr/bin/env python3
import os
import sys
from datetime import datetime, timezone
from typing import Any, Iterable

import pandas as pd
import requests

from db_store import connect_db, init_db, upsert_observations, upsert_forecasts

# --- Endpoints (Valkenburgse Meer / node7) ---
OBS_URL = "https://1.windsurfice.com/PHP_scripts/Wind_laatste_dag.php?Site=windsurfice-v25-node7&WaterTemp=0&Direction=1"
FC_URL = "https://1.windsurfice.com/PHP_scripts/GetForecastDB.php"
SITE = "valkenburgsemeer"
MODEL = "HARMONIE"
LAT = "52.1603"
LON = "4.44197"

RUN_TS_CANDIDATE_KEYS = (
    "run_ts",
    "runTs",
    "run_timestamp",
    "runTimestamp",
    "forecast_run_ts",
    "forecastRunTs",
    "model_run_ts",
    "modelRunTs",
    "run_time",
    "runTime",
    "runtime",
    "forecast_run_time",
    "forecastRunTime",
    "model_run_time",
    "modelRunTime",
    "issue_time",
    "issueTime",
    "issued_at",
    "issuedAt",
    "reference_time",
    "referenceTime",
    "base_time",
    "baseTime",
    "generation_time",
    "generationTime",
    "generated_at",
    "generatedAt",
    "created_at",
    "createdAt",
)

# --- Headers ---
# Optional cookie can be passed via LOCAL_WIND_COOKIE env var.
HEADERS = {
    "User-Agent": os.getenv(
        "LOCAL_WIND_UA",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124 Safari/537.36",
    ),
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://windsurfice.com/en/locations/valkenburgsemeer",
    "Origin": "https://windsurfice.com",
}
_cookie = os.getenv("LOCAL_WIND_COOKIE")
if _cookie:
    HEADERS["Cookie"] = _cookie


def ts_to_iso_any(t):
    """Convert seconds or milliseconds epoch to ISO-8601 UTC (with trailing 'Z')."""
    if t is None:
        return None
    s = str(t)
    try:
        ms = int(t) if len(s) > 10 else int(t) * 1000
        dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    except Exception:
        return None


def normalize_timestamp_ms(value: Any) -> int | None:
    """Parse epoch seconds/millis or ISO-8601-like strings into UTC epoch ms."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        i = int(value)
        return i if len(str(abs(i))) > 10 else i * 1000

    text = str(value).strip()
    if not text:
        return None
    try:
        i = int(text)
        return i if len(str(abs(i))) > 10 else i * 1000
    except ValueError:
        pass
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.astimezone(timezone.utc).timestamp() * 1000)


def _extract_first_case_insensitive(d: dict[str, Any], keys: Iterable[str]) -> Any:
    lower_map = {k.lower(): k for k in d.keys()}
    for key in keys:
        real_key = lower_map.get(key.lower())
        if real_key is not None:
            return d.get(real_key)
    return None


def _extract_forecast_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return [row for row in payload["data"] if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def _iter_run_metadata_dicts(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, dict):
        yield payload
        for key in ("meta", "metadata", "header", "headers", "info", "forecast", "model"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                yield nested
    rows = _extract_forecast_rows(payload)
    for row in rows[:3]:
        yield row


def resolve_forecast_run_ts(payload: Any, fetched_ts_ms: int) -> tuple[int, str]:
    """
    Resolve the forecast vintage timestamp.

    run_ts is the model/run vintage, while fetched_ts is when our collector saw
    that payload. We store both because forecast vintages must remain immutable
    for fair backtesting and later model-vs-Harmonie comparisons.

    Windsurfice payloads do not always expose explicit run metadata. When they do
    not, we fall back to fetched_ts for now; that keeps the fallback explicit and
    can be replaced later if we switch to a source with authoritative run times.
    """
    for candidate_dict in _iter_run_metadata_dicts(payload):
        candidate_value = _extract_first_case_insensitive(candidate_dict, RUN_TS_CANDIDATE_KEYS)
        candidate_ts_ms = normalize_timestamp_ms(candidate_value)
        if candidate_ts_ms is not None:
            return candidate_ts_ms, "explicit"
    return fetched_ts_ms, "fallback_fetch_time"


def add_iso_time(records):
    """Add an iso_time column if a timestamp-like field exists."""
    if records is None:
        return []
    if isinstance(records, dict):
        records = [records]
    for d in records:
        if not isinstance(d, dict):
            continue
        t = d.get("timestamp") or d.get("time") or d.get("UnixTime") or d.get("ts") or d.get("dt")
        d["iso_time"] = ts_to_iso_any(t)
    return records


def fetch_json(url, name, method="GET", params=None, data=None):
    """Fetch JSON with browser-like headers; support GET/POST; show helpful preview."""
    print(f"Fetching {name} -> {url}")
    try:
        if method.upper() == "POST":
            r = requests.post(url, headers=HEADERS, params=params, data=data, timeout=30)
        else:
            r = requests.get(url, headers=HEADERS, params=params, timeout=30)
    except requests.RequestException as e:
        raise RuntimeError(f"{name}: request failed: {e}") from e

    ct = r.headers.get("content-type", "")
    print(f"Status: {r.status_code}  Content-Type: {ct}")
    if r.status_code != 200:
        preview = (r.text or "")[:400].replace("\n", " ")
        raise RuntimeError(f"{name}: HTTP {r.status_code}. Body starts with: {preview}")

    try:
        return r.json()
    except Exception:
        preview = (r.text or "")[:400].replace("\n", " ")
        raise RuntimeError(f"{name}: response was not JSON. Starts with: {preview}")


def save_csv(data, prefix, out_dir="."):
    df = pd.DataFrame(data)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    filename = os.path.join(out_dir, f"{stamp}_{prefix}.csv")
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} rows -> {filename}")


def main(out_dir="."):
    # 1) Observations (last 24 h)
    obs = fetch_json(OBS_URL, "observations (last day)")
    obs = add_iso_time(obs)
    save_csv(obs, "valkenburgsemeer_last_day", out_dir)
    try:
        conn = connect_db(out_dir)
        init_db(conn)
        n_obs = upsert_observations(conn, SITE, obs)
        print(f"Upserted {n_obs} observation rows into SQLite")
    except Exception as e:
        print(f"Failed to upsert observations into DB: {e}")

    # 2) Forecast (HARMONIE)
    fc_payload = fetch_json(
        FC_URL,
        "forecast (HARMONIE)",
        method="POST",
        data={"lat": LAT, "lon": LON},
    )
    fetched_ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    run_ts_ms, run_ts_quality = resolve_forecast_run_ts(fc_payload, fetched_ts_ms)
    print(
        "Resolved forecast run timestamp "
        f"{ts_to_iso_any(run_ts_ms)} (quality={run_ts_quality}, fetched_at={ts_to_iso_any(fetched_ts_ms)})"
    )

    fc_rows = add_iso_time(_extract_forecast_rows(fc_payload))
    fc_rows_csv = []
    for row in fc_rows:
        row_csv = dict(row)
        row_csv["forecast_run_ts"] = run_ts_ms
        row_csv["forecast_run_iso"] = ts_to_iso_any(run_ts_ms)
        row_csv["forecast_fetched_ts"] = fetched_ts_ms
        row_csv["forecast_fetched_iso"] = ts_to_iso_any(fetched_ts_ms)
        row_csv["forecast_run_ts_quality"] = run_ts_quality
        fc_rows_csv.append(row_csv)
    save_csv(fc_rows_csv, "valkenburgsemeer_forecast", out_dir)
    try:
        conn = connect_db(out_dir)
        init_db(conn)
        n_fc = upsert_forecasts(
            conn,
            SITE,
            MODEL,
            fc_rows,
            run_ts_ms=run_ts_ms,
            fetched_ts_ms=fetched_ts_ms,
        )
        print(f"Upserted {n_fc} forecast rows into SQLite")
    except Exception as e:
        print(f"Failed to upsert forecasts into DB: {e}")


if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else "./data"
    os.makedirs(outdir, exist_ok=True)
    main(outdir)
