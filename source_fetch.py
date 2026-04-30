#!/usr/bin/env python3
import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import pandas as pd
import requests

from db_store import connect_db, init_db, upsert_observations, upsert_forecasts

# --- Windsurfice endpoints ---
OBS_URL = "https://1.windsurfice.com/PHP_scripts/Wind_laatste_dag.php"
FC_URL = "https://1.windsurfice.com/PHP_scripts/GetForecastDB.php"


@dataclass(frozen=True)
class SiteConfig:
    site: str
    display_name: str
    windsurfice_obs_site: str
    forecast_lat: str
    forecast_lon: str
    referer: str

    @property
    def observation_params(self) -> dict[str, str | int]:
        return {
            "Site": self.windsurfice_obs_site,
            "WaterTemp": 0,
            "Direction": 1,
        }

    @property
    def forecast_payload(self) -> dict[str, str]:
        return {
            "lat": self.forecast_lat,
            "lon": self.forecast_lon,
        }


SITES = {
    "valkenburgsemeer": SiteConfig(
        site="valkenburgsemeer",
        display_name="Valkenburgse Meer",
        windsurfice_obs_site="windsurfice-v25-node7",
        forecast_lat="52.1603",
        forecast_lon="4.44197",
        referer="https://windsurfice.com/en/locations/valkenburgsemeer",
    ),
    "oostvoorne": SiteConfig(
        site="oostvoorne",
        display_name="Oostvoornse Meer",
        windsurfice_obs_site="windsurfice-v25-node6",
        forecast_lat="51.9278",
        forecast_lon="4.05502",
        referer="https://windsurfice.com/en/locations/oostvoorne",
    ),
}
MODEL = "HARMONIE"

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

BASE_HEADERS = {
    "User-Agent": os.getenv(
        "LOCAL_WIND_UA",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124 Safari/537.36",
    ),
    "Accept": "application/json,text/plain,*/*",
    "Origin": "https://windsurfice.com",
}


def build_headers(site_config: SiteConfig) -> dict[str, str]:
    """Build browser-like headers for the selected Windsurfice location."""
    headers = dict(BASE_HEADERS)
    headers["Referer"] = site_config.referer
    cookie = os.getenv("LOCAL_WIND_COOKIE")
    if cookie:
        headers["Cookie"] = cookie
    return headers


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


def fetch_json(url, name, method="GET", params=None, data=None, headers=None):
    """Fetch JSON with browser-like headers; support GET/POST; show helpful preview."""
    print(f"Fetching {name} -> {url}")
    request_headers = headers or BASE_HEADERS
    try:
        if method.upper() == "POST":
            r = requests.post(url, headers=request_headers, params=params, data=data, timeout=30)
        else:
            r = requests.get(url, headers=request_headers, params=params, timeout=30)
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
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(data)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    filename = os.path.join(out_dir, f"{stamp}_{prefix}.csv")
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} rows -> {filename}")


def fetch_site(site_config: SiteConfig, out_dir="."):
    headers = build_headers(site_config)
    site = site_config.site
    site_out_dir = os.path.join(out_dir, site)
    print(f"Fetching Windsurfice data for {site_config.display_name} ({site})")

    # 1) Observations (last 24 h)
    obs = fetch_json(
        OBS_URL,
        "observations (last day)",
        params=site_config.observation_params,
        headers=headers,
    )
    obs = add_iso_time(obs)
    save_csv(obs, "last_day", site_out_dir)
    try:
        conn = connect_db(out_dir)
        init_db(conn)
        n_obs = upsert_observations(conn, site, obs)
        print(f"Upserted {n_obs} observation rows into SQLite")
    except Exception as e:
        print(f"Failed to upsert observations into DB: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass

    # 2) Forecast (HARMONIE)
    fc_payload = fetch_json(
        FC_URL,
        "forecast (HARMONIE)",
        method="POST",
        data=site_config.forecast_payload,
        headers=headers,
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
    save_csv(fc_rows_csv, "forecast", site_out_dir)
    try:
        conn = connect_db(out_dir)
        init_db(conn)
        n_fc = upsert_forecasts(
            conn,
            site,
            MODEL,
            fc_rows,
            run_ts_ms=run_ts_ms,
            fetched_ts_ms=fetched_ts_ms,
        )
        print(f"Upserted {n_fc} forecast rows into SQLite")
    except Exception as e:
        print(f"Failed to upsert forecasts into DB: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Windsurfice observations and forecasts.")
    parser.add_argument(
        "out_dir",
        nargs="?",
        default="./data",
        help="Root directory for per-site CSV snapshot folders and wind_data_all_sites.db.",
    )
    parser.add_argument(
        "--site",
        default=None,
        choices=sorted(SITES),
        help="Configured site to fetch. If omitted, all configured sites are fetched.",
    )
    parser.add_argument(
        "--all-sites",
        action="store_true",
        help="Fetch every configured site. This is the default when --site is omitted.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    os.makedirs(args.out_dir, exist_ok=True)

    site_configs = list(SITES.values()) if args.all_sites or args.site is None else [SITES[args.site]]
    for site_config in site_configs:
        fetch_site(site_config, args.out_dir)


if __name__ == "__main__":
    main()
