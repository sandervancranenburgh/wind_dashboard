#!/usr/bin/env python3
import os
import sys
import json
from datetime import datetime, timezone
import requests
import pandas as pd
from db_store import connect_db, init_db, upsert_observations, upsert_forecasts

# --- Endpoints (Valkenburgse Meer / node7) ---
OBS_URL = "https://1.windsurfice.com/PHP_scripts/Wind_laatste_dag.php?Site=windsurfice-v25-node7&WaterTemp=0&Direction=1"
FC_URL  = "https://1.windsurfice.com/PHP_scripts/GetForecastDB.php"
SITE = "valkenburgsemeer"
MODEL = "HARMONIE"
LAT = "52.1603"
LON = "4.44197"

# --- Headers: copy the same trick that worked for you in the browser ---
# If needed, put your cookie in the WIND_SURFICE_COOKIE env var to avoid hardcoding it.
HEADERS = {
    "User-Agent": os.getenv("WINDSURFICE_UA", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                                             "AppleWebKit/537.36 (KHTML, like Gecko) "
                                             "Chrome/124 Safari/537.36"),
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://windsurfice.com/en/locations/valkenburgsemeer",
    "Origin": "https://windsurfice.com",
}
_cookie = os.getenv("WINDSURFICE_COOKIE")
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

def add_iso_time(records):
    """Add an iso_time column if a timestamp-like field exists."""
    if records is None:
        return []
    if isinstance(records, dict):
        records = [records]
    for d in records:
        if not isinstance(d, dict):
            continue
        t = (d.get("timestamp") or d.get("time") or d.get("UnixTime") or
             d.get("ts") or d.get("dt"))
        d["iso_time"] = ts_to_iso_any(t)
    return records

def fetch_json(url, name, method="GET", params=None, data=None):
    """Fetch JSON with browser-like headers; support GET/POST; show helpful preview."""
    print(f"Fetching {name} → {url}")
    try:
        if method.upper() == "POST":
            # print("DEBUG sending POST data:", data)
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
    # Data is expected to already include iso_time if desired
    df = pd.DataFrame(data)

    # filename starts with sortable UTC timestamp
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M")
    filename = os.path.join(out_dir, f"{stamp}_{prefix}.csv")

    df.to_csv(filename, index=False)
    print(f"💾 Saved {len(df)} rows → {filename}")

def main(out_dir="."):
    # 1) Observations (last 24 h)
    obs = fetch_json(OBS_URL, "observations (last day)")
    obs = add_iso_time(obs)
    save_csv(obs, "valkenburgsemeer_last_day", out_dir)
    try:
        conn = connect_db(out_dir)
        init_db(conn)
        n_obs = upsert_observations(conn, SITE, obs)
        print(f"📥 Upserted {n_obs} observation rows into SQLite")
    except Exception as e:
        print(f"⚠️ Failed to upsert observations into DB: {e}")

    # 2) Forecast (HARMONIE)
    fc = fetch_json(
        FC_URL,
        "forecast (HARMONIE)",
        method="POST",
        data={"lat": LAT, "lon": LON},
    )
    # Forecast JSON shape may differ; still try to add iso_time if a timestamp-ish field exists
    if isinstance(fc, dict) and "data" in fc and isinstance(fc["data"], list):
        fc_rows = fc["data"]
    elif isinstance(fc, list):
        fc_rows = fc
    else:
        try:
            fc_rows = pd.DataFrame(fc).to_dict(orient="records")
        except Exception:
            fc_rows = [fc]

    fc_rows = add_iso_time(fc_rows)
    save_csv(fc_rows, "valkenburgsemeer_forecast", out_dir)
    try:
        conn = connect_db(out_dir)
        init_db(conn)
        n_fc = upsert_forecasts(conn, SITE, MODEL, fc_rows)
        print(f"📥 Upserted {n_fc} forecast rows into SQLite")
    except Exception as e:
        print(f"⚠️ Failed to upsert forecasts into DB: {e}")


if __name__ == "__main__":

    outdir = sys.argv[1] if len(sys.argv) > 1 else "./data"
    os.makedirs(outdir, exist_ok=True)
    main(outdir)
