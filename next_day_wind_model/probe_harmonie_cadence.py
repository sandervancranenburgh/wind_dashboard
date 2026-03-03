#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

FC_URL = "https://1.windsurfice.com/PHP_scripts/GetForecastDB.php"
DEFAULT_LAT = "52.1603"
DEFAULT_LON = "4.44197"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe HARMONIE forecast cadence and log when source data changes.",
    )
    parser.add_argument("--lat", default=DEFAULT_LAT, help="Latitude passed to GetForecastDB.php")
    parser.add_argument("--lon", default=DEFAULT_LON, help="Longitude passed to GetForecastDB.php")
    parser.add_argument(
        "--interval-minutes",
        type=float,
        default=60.0,
        help="Sleep interval between probes when iterations > 1.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of probe iterations (1 = run once).",
    )
    parser.add_argument(
        "--log-csv",
        default="next_day_wind_model/artifacts/harmonie_cadence_probe.csv",
        help="CSV log path for probe samples.",
    )
    parser.add_argument(
        "--state-json",
        default="next_day_wind_model/artifacts/harmonie_cadence_state.json",
        help="State file used to detect changes between runs.",
    )
    return parser.parse_args()


def _to_ms(value: Any) -> int | None:
    if value is None:
        return None
    try:
        i = int(value)
        if len(str(abs(i))) <= 10:
            return i * 1000
        return i
    except Exception:
        return None


def _extract_rows(payload: Any) -> list[dict]:
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return [r for r in payload["data"] if isinstance(r, dict)]
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def _rows_signature(rows: list[dict]) -> tuple[str, int | None, int | None]:
    compact = []
    target_mss: list[int] = []
    for r in rows:
        ts_ms = _to_ms(r.get("timestamp") or r.get("time") or r.get("UnixTime") or r.get("ts") or r.get("dt"))
        if ts_ms is not None:
            target_mss.append(ts_ms)
        compact.append(
            {
                "ts": ts_ms,
                "avr": r.get("WindForecastAvr"),
                "max": r.get("WindForecastMax"),
                "dir": r.get("WindDirection"),
                "tmp": r.get("Temperature"),
            }
        )
    compact.sort(key=lambda x: (x["ts"] is None, x["ts"]))
    digest = hashlib.sha1(json.dumps(compact, separators=(",", ":"), sort_keys=True).encode("utf-8")).hexdigest()
    if not target_mss:
        return digest, None, None
    return digest, min(target_mss), max(target_mss)


def _iso(ms: int | None) -> str | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    header = not path.exists()
    df.to_csv(path, mode="a", index=False, header=header)


def probe_once(lat: str, lon: str, prev_sig: str | None) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    try:
        r = requests.post(FC_URL, headers=HEADERS, data={"lat": lat, "lon": lon}, timeout=30)
        r.raise_for_status()
        payload = r.json()
        rows = _extract_rows(payload)
        sig, min_ms, max_ms = _rows_signature(rows)
        changed = prev_sig is not None and sig != prev_sig
        return {
            "fetched_at_utc": now,
            "ok": True,
            "http_status": r.status_code,
            "row_count": len(rows),
            "signature_sha1": sig,
            "changed_vs_prev": changed,
            "min_target_utc": _iso(min_ms),
            "max_target_utc": _iso(max_ms),
            "error": "",
        }
    except Exception as e:
        return {
            "fetched_at_utc": now,
            "ok": False,
            "http_status": None,
            "row_count": 0,
            "signature_sha1": "",
            "changed_vs_prev": False,
            "min_target_utc": None,
            "max_target_utc": None,
            "error": str(e),
        }


def main() -> None:
    args = parse_args()
    log_csv = Path(args.log_csv)
    state_json = Path(args.state_json)

    state = _load_state(state_json)
    prev_sig = state.get("last_signature_sha1")
    interval_s = max(args.interval_minutes * 60.0, 1.0)
    iterations = max(args.iterations, 1)

    for i in range(iterations):
        row = probe_once(args.lat, args.lon, prev_sig)
        _append_csv(log_csv, row)
        print(json.dumps(row))

        if row.get("ok") and row.get("signature_sha1"):
            prev_sig = row["signature_sha1"]
            state = {
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "last_signature_sha1": row["signature_sha1"],
                "last_max_target_utc": row.get("max_target_utc"),
                "last_row_count": row.get("row_count"),
            }
            _save_state(state_json, state)

        if i < iterations - 1:
            time.sleep(interval_s)


if __name__ == "__main__":
    main()
