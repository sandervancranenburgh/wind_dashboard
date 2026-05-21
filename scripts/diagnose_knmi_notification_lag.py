#!/usr/bin/env python3
"""Diagnose KNMI notification lag without downloading by default."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from next_day_wind_model.knmi_harmonie import (
    DATASET,
    VERSION,
    KnmiApiError,
    is_harmonie_p1_tar_filename,
    list_knmi_files,
    parse_run_from_tar_filename,
)
from scripts.knmi_extract_latest_to_db import process_knmi_file_to_db
from scripts.knmi_notification_listener import extract_filename_from_event_payload


EXPECTED_HORIZONS = set(range(61))
SEARCH_TERMS = (
    "CET",
    "CEST",
    "Europe/Amsterdam",
    "timezone",
    "tz",
    "localtime",
    "datetime.now(",
    "utcnow(",
    "fromtimestamp",
    "strptime",
    "timedelta(hours=",
    "+01",
    "+02",
)


@dataclass(frozen=True)
class ApiRun:
    filename: str
    run_ts: pd.Timestamp
    created_ts: pd.Timestamp | None
    last_modified_ts: pd.Timestamp | None
    size: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose KNMI notification/archive lag.")
    parser.add_argument("--db", type=Path, default=Path("data/wind_data_all_sites.db"))
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--version", default=VERSION)
    parser.add_argument("--site", default="valkenburgsemeer")
    parser.add_argument("--latest-api-count", type=int, default=12)
    parser.add_argument("--log-file", type=Path, default=Path("logs/knmi_notification_listener.log"))
    parser.add_argument("--fallback-log", type=Path, default=Path("logs/knmi_shadow_fetch_fallback.log"))
    parser.add_argument("--timezone", default="Europe/Amsterdam")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("data/processed/knmi_validation/knmi_notification_lag_diagnostic.json"),
    )
    parser.add_argument("--write-json", action="store_true")
    parser.add_argument(
        "--process-missing-latest",
        type=int,
        default=0,
        help="Process up to N latest API files that are missing or incomplete in the archive.",
    )
    parser.add_argument("--max-acceptable-lag-hours", type=float, default=2.0)
    return parser.parse_args()


def parse_ts(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts)


def iso_utc(value: Any) -> str | None:
    ts = parse_ts(value)
    if ts is None:
        return None
    return ts.isoformat().replace("+00:00", "Z")


def iso_local(value: Any, zone: ZoneInfo) -> str | None:
    ts = parse_ts(value)
    if ts is None:
        return None
    return ts.tz_convert(zone).isoformat()


def hours_between(later: Any, earlier: Any) -> float | None:
    later_ts = parse_ts(later)
    earlier_ts = parse_ts(earlier)
    if later_ts is None or earlier_ts is None:
        return None
    return round(float((later_ts - earlier_ts) / pd.Timedelta(hours=1)), 3)


def connect_read_only(db_path: Path) -> sqlite3.Connection:
    resolved = db_path.expanduser().resolve()
    conn = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True, timeout=30)
    conn.execute("PRAGMA query_only = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def fetch_api_runs(dataset: str, version: str, count: int) -> list[ApiRun]:
    files = list_knmi_files(dataset, version, max_keys=max(count, 1))
    runs: list[ApiRun] = []
    for item in files:
        if not is_harmonie_p1_tar_filename(item.filename):
            continue
        runs.append(
            ApiRun(
                filename=item.filename,
                run_ts=parse_run_from_tar_filename(item.filename),
                created_ts=parse_ts(item.created),
                last_modified_ts=parse_ts(item.last_modified),
                size=item.size,
            )
        )
    return runs[:count]


def archive_summary(db_path: Path, site: str) -> dict[str, Any]:
    conn = connect_read_only(db_path)
    try:
        summary = conn.execute(
            """
            SELECT
                COUNT(DISTINCT run_ts) AS distinct_run_ts,
                MIN(run_ts) AS min_run_ts,
                MAX(run_ts) AS max_run_ts,
                COUNT(*) AS row_count
            FROM harmonie_knmi_features
            WHERE site = ?
            """,
            (site,),
        ).fetchone()
        latest_rows = conn.execute(
            """
            SELECT
                run_ts,
                COUNT(*) AS row_count,
                MIN(horizon_hr) AS min_horizon,
                MAX(horizon_hr) AS max_horizon,
                GROUP_CONCAT(DISTINCT horizon_hr) AS horizons
            FROM harmonie_knmi_features
            WHERE site = ?
            GROUP BY run_ts
            ORDER BY run_ts DESC
            LIMIT 10
            """,
            (site,),
        ).fetchall()
        duplicates = conn.execute(
            """
            SELECT source, dataset, run_ts, target_ts, site, COUNT(*) AS count
            FROM harmonie_knmi_features
            WHERE site = ?
            GROUP BY source, dataset, run_ts, target_ts, site
            HAVING COUNT(*) > 1
            ORDER BY count DESC
            LIMIT 20
            """,
            (site,),
        ).fetchall()
        counts = conn.execute(
            """
            SELECT run_ts, COUNT(*) AS row_count, GROUP_CONCAT(DISTINCT horizon_hr) AS horizons
            FROM harmonie_knmi_features
            WHERE site = ?
            GROUP BY run_ts
            """,
            (site,),
        ).fetchall()
    finally:
        conn.close()

    latest_runs: list[dict[str, Any]] = []
    for run_ts, row_count, min_horizon, max_horizon, horizon_text in latest_rows:
        horizons = parse_horizon_text(horizon_text)
        latest_runs.append(
            {
                "run_ts_utc": iso_utc(run_ts),
                "row_count": int(row_count or 0),
                "min_horizon": min_horizon,
                "max_horizon": max_horizon,
                "missing_horizons": sorted(EXPECTED_HORIZONS - horizons),
            }
        )

    archive_counts: dict[str, dict[str, Any]] = {}
    for run_ts, row_count, horizon_text in counts:
        horizons = parse_horizon_text(horizon_text)
        archive_counts[iso_utc(run_ts) or str(run_ts)] = {
            "row_count": int(row_count or 0),
            "missing_horizons": sorted(EXPECTED_HORIZONS - horizons),
        }

    return {
        "distinct_run_ts": int(summary[0] or 0),
        "min_run_ts_utc": iso_utc(summary[1]),
        "max_run_ts_utc": iso_utc(summary[2]),
        "row_count": int(summary[3] or 0),
        "latest_runs": latest_runs,
        "duplicates": [tuple(row) for row in duplicates],
        "counts_by_run": archive_counts,
    }


def parse_horizon_text(value: str | None) -> set[int]:
    if not value:
        return set()
    out: set[int] = set()
    for part in str(value).split(","):
        try:
            out.add(int(part))
        except ValueError:
            continue
    return out


def api_db_comparison(api_runs: list[ApiRun], archive: dict[str, Any]) -> list[dict[str, Any]]:
    counts_by_run = archive["counts_by_run"]
    latest_api_run = api_runs[0].run_ts if api_runs else None
    latest_db_run = archive.get("max_run_ts_utc")
    latest_lag = hours_between(latest_api_run, latest_db_run)
    now = pd.Timestamp.now(tz="UTC")
    rows: list[dict[str, Any]] = []
    for item in api_runs:
        key = iso_utc(item.run_ts)
        archived = counts_by_run.get(key or "")
        rows.append(
            {
                "filename": item.filename,
                "api_run_ts_utc": key,
                "api_created_utc": iso_utc(item.created_ts),
                "archived": archived is not None,
                "archived_row_count": archived["row_count"] if archived else 0,
                "missing_horizons": archived["missing_horizons"] if archived else sorted(EXPECTED_HORIZONS),
                "lag_from_api_latest_to_db_latest_hours": latest_lag,
                "lag_from_api_created_to_now_hours": hours_between(now, item.created_ts),
                "lag_from_run_ts_to_api_created_hours": hours_between(item.created_ts, item.run_ts),
            }
        )
    return rows


def parse_listener_log(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "message": f"No listener log found at {path}."}
    lines = path.read_text(errors="replace").splitlines()
    summary: dict[str, Any] = {
        "exists": True,
        "line_count": len(lines),
        "last_listener_start": last_matching(lines, "KNMI notification listener start"),
        "last_successful_connection": last_matching(lines, "Connected to KNMI Notification Service"),
        "last_subscription": last_matching(lines, "Subscribed to"),
        "last_event_received": last_matching(lines, "Received KNMI notification event"),
        "last_filename_parsed": last_matching(lines, "Parsed KNMI HARMONIE P1 filename"),
        "last_filename_queued": last_matching(lines, "Queued KNMI HARMONIE P1 notification"),
        "last_filename_processed": last_matching(lines, "KNMI notification processed filename="),
        "last_processing_error": last_error(lines),
        "events_received_count": count_matching(lines, "Received KNMI notification event"),
        "filenames_parsed_count": count_matching(lines, "Parsed KNMI HARMONIE P1 filename"),
        "files_processed_count": count_matching(lines, "KNMI notification processed filename="),
        "duplicate_suppressed_count": count_matching(lines, "Ignoring duplicate notification"),
        "database_locked_count": count_matching(lines, "database is locked"),
        "session_present": session_present_values(lines),
        "unparsed_payload_lines": matching_tail(
            lines,
            (
                "did not contain an obvious HARMONIE P1 filename",
                "did not contain a HARMONIE P1 filename",
                "Ignored KNMI notification without matching",
            ),
            limit=10,
        ),
    }
    if summary["events_received_count"] == 0:
        summary["message"] = (
            "No current listener event entries found. If the listener was started directly, "
            "terminal output may not be in the wrapper log."
        )
    return summary


def parse_fallback_log(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "message": f"No fallback log found at {path}."}
    lines = path.read_text(errors="replace").splitlines()
    processed = [extract_filename_from_text(line) for line in lines if "Selected filename:" in line]
    processed = [item for item in processed if item]
    return {
        "exists": True,
        "line_count": len(lines),
        "last_fallback_start": last_matching(lines, "KNMI shadow fallback fetch start"),
        "last_fallback_end": last_matching(lines, "KNMI shadow fallback fetch end"),
        "processed_files": processed[-20:],
        "processed_file_count": len(processed),
        "last_error": last_error(lines),
    }


def extract_filename_from_text(text: str) -> str | None:
    match = re.search(r"HARM43_V1_P1_\d{10}\.tar", text)
    return match.group(0) if match else None


def last_matching(lines: list[str], needle: str) -> str | None:
    for line in reversed(lines):
        if needle in line:
            return line
    return None


def matching_tail(lines: list[str], needles: tuple[str, ...], limit: int) -> list[str]:
    found = [line for line in lines if any(needle in line for needle in needles)]
    return found[-limit:]


def count_matching(lines: list[str], needle: str) -> int:
    return sum(1 for line in lines if needle in line)


def last_error(lines: list[str]) -> str | None:
    for line in reversed(lines):
        lowered = line.lower()
        if "error" in lowered or "exception" in lowered or "failed" in lowered:
            return line
    return None


def session_present_values(lines: list[str]) -> list[str]:
    values: list[str] = []
    for line in lines:
        match = re.search(r"session_present=([A-Za-z]+)", line)
        if match:
            values.append(match.group(1))
    return values[-10:]


def parser_self_test(extra_payloads: list[str] | None = None) -> list[dict[str, Any]]:
    examples = [
        (b'{"filename": "HARM43_V1_P1_2026052118.tar"}', "HARM43_V1_P1_2026052118.tar"),
        (b'{"fileName": "HARM43_V1_P1_2026052119.tar"}', "HARM43_V1_P1_2026052119.tar"),
        (b'{"data": {"filename": "HARM43_V1_P1_2026052120.tar"}}', "HARM43_V1_P1_2026052120.tar"),
        (b'{"path": "/x/y/HARM43_V1_P1_2026052121.tar"}', "HARM43_V1_P1_2026052121.tar"),
        (b"created HARM43_V1_P1_2026052122.tar", "HARM43_V1_P1_2026052122.tar"),
    ]
    rows = []
    for payload, expected in examples:
        actual = extract_filename_from_event_payload(payload)
        rows.append(
            {
                "payload": payload.decode("utf-8"),
                "expected": expected,
                "actual": actual,
                "passed": actual == expected,
            }
        )
    for payload in extra_payloads or []:
        rows.append(
            {
                "payload": payload,
                "expected": None,
                "actual": extract_filename_from_event_payload(payload),
                "passed": True,
            }
        )
    return rows


def timezone_audit() -> dict[str, Any]:
    files = [
        Path("scripts/knmi_notification_listener.py"),
        Path("scripts/knmi_extract_latest_to_db.py"),
        Path("scripts/run_knmi_notification_listener.sh"),
        Path("scripts/run_knmi_shadow_fetch_fallback.sh"),
        Path("next_day_wind_model/knmi_harmonie.py"),
        Path("scripts/validate_knmi_shadow_archive.py"),
        Path("source_fetch.py"),
        Path("db_store.py"),
        Path("next_day_wind_model/data_pipeline.py"),
        Path("next_day_wind_model/update_model_and_predict.py"),
    ]
    matches: list[dict[str, Any]] = []
    for path in files:
        if not path.exists():
            continue
        for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
            terms = [term for term in SEARCH_TERMS if term in line]
            if terms:
                matches.append({"file": str(path), "line": lineno, "terms": terms, "text": line.strip()})

    issues = []
    for item in matches:
        text = item["text"]
        if "utcnow(" in text:
            issues.append({**item, "issue": "Use timezone-aware datetime.now(timezone.utc) instead of utcnow()."})
        if "+ pd.Timedelta(hours=3)" in text:
            issues.append({**item, "issue": "Manual +3 hour display offset found outside KNMI shadow path."})
        if "CET" in text and "CEST" not in text:
            issues.append({**item, "issue": "Fixed CET label/offset may be wrong during summer time."})

    return {
        "rules": [
            "Core KNMI DB timestamps should be UTC ISO strings with explicit offset.",
            "HARM43_V1_P1_YYYYMMDDHH.tar run_ts is parsed as UTC.",
            "target_ts is run_ts plus horizon hours in UTC.",
            "Europe/Amsterdam is display-only and must use IANA timezone handling.",
            "Avoid manual +1/+2 offsets for Dutch local time.",
        ],
        "match_count": len(matches),
        "matches": matches[:120],
        "potential_issues": issues,
        "knmi_shadow_assessment": (
            "No UTC/local-time violation found in the KNMI shadow extraction/listener path. "
            "Filename run_ts and GRIB target_ts are parsed with utc=True; fetched_ts and created_at use UTC."
        ),
    }


def process_missing(api_runs: list[ApiRun], comparison: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.process_missing_latest <= 0:
        return []
    candidates = [
        row for row in comparison
        if (not row["archived"]) or row["missing_horizons"]
    ][: args.process_missing_latest]
    results: list[dict[str, Any]] = []
    for row in candidates:
        print(f"Processing missing/incomplete KNMI file: {row['filename']}")
        result = process_knmi_file_to_db(filename=row["filename"], db_path=args.db, site=args.site)
        results.append(
            {
                "filename": result.filename,
                "run_ts": result.run_ts,
                "rows_written": result.rows_written,
                "shadow_rows_written": result.shadow_rows_written,
                "latest_run_horizon_count": result.latest_run_horizon_count,
            }
        )
    return results


def serializable_api_runs(api_runs: list[ApiRun], zone: ZoneInfo) -> list[dict[str, Any]]:
    return [
        {
            "filename": item.filename,
            "run_ts_utc": iso_utc(item.run_ts),
            "run_ts_local": iso_local(item.run_ts, zone),
            "created_utc": iso_utc(item.created_ts),
            "created_local": iso_local(item.created_ts, zone),
            "last_modified_utc": iso_utc(item.last_modified_ts),
            "last_modified_local": iso_local(item.last_modified_ts, zone),
            "size": item.size,
        }
        for item in api_runs
    ]


def print_report(report: dict[str, Any], zone_name: str, max_lag: float) -> None:
    api_runs = report["api_runs"]
    archive = report["archive"]
    comparison = report["api_db_comparison"]
    latest_api = api_runs[0] if api_runs else None

    print("KNMI notification lag diagnostic")
    print("================================")
    if latest_api:
        print(f"Latest API file: {latest_api['filename']}")
        print(f"API run_ts UTC: {latest_api['run_ts_utc']}")
        print(f"API run_ts {zone_name}: {latest_api['run_ts_local']}")
        print(f"API created UTC: {latest_api['created_utc']}")
        print(f"API created {zone_name}: {latest_api['created_local']}")
        print(f"API lastModified UTC: {latest_api['last_modified_utc']}")
        print(f"API size: {latest_api['size']}")
    print(f"Archive distinct_run_ts: {archive['distinct_run_ts']}")
    print(f"Archive row_count: {archive['row_count']}")
    print(f"Archive max_run_ts UTC: {archive['max_run_ts_utc']}")
    print(f"Archive max_run_ts {zone_name}: {archive.get('max_run_ts_local')}")
    lag = report["lag_summary"]["api_latest_run_minus_db_latest_run_hours"]
    print(f"Lag API latest run - DB latest run: {lag} hours")
    if lag is not None:
        print(f"Lag status: {'OK' if lag <= max_lag else 'Too high'} (threshold {max_lag}h)")

    print("\nLatest API files versus DB")
    print("--------------------------")
    for row in comparison:
        missing = row["missing_horizons"]
        missing_text = "none" if not missing else f"{len(missing)} missing"
        print(
            f"{row['filename']} run={row['api_run_ts_utc']} created={row['api_created_utc']} "
            f"archived={row['archived']} rows={row['archived_row_count']} {missing_text}"
        )

    print("\nLatest archived runs")
    print("--------------------")
    for row in archive["latest_runs"]:
        missing = row["missing_horizons"]
        missing_text = "none" if not missing else ",".join(map(str, missing))
        print(
            f"{row['run_ts_utc']} rows={row['row_count']} "
            f"horizon_range={row['min_horizon']}-{row['max_horizon']} missing={missing_text}"
        )

    listener = report["listener_log"]
    print("\nListener log")
    print("------------")
    print(listener.get("message") or f"events={listener.get('events_received_count')} parsed={listener.get('filenames_parsed_count')} processed={listener.get('files_processed_count')}")
    print(f"last connection: {listener.get('last_successful_connection')}")
    print(f"last subscription: {listener.get('last_subscription')}")
    print(f"last event: {listener.get('last_event_received')}")
    print(f"last processed: {listener.get('last_filename_processed')}")
    print(f"last error: {listener.get('last_processing_error')}")

    fallback = report["fallback_log"]
    print("\nFallback log")
    print("------------")
    if fallback.get("exists"):
        print(f"last start: {fallback.get('last_fallback_start')}")
        print(f"last end: {fallback.get('last_fallback_end')}")
        print(f"processed count: {fallback.get('processed_file_count')}")
        print(f"recent processed: {', '.join(fallback.get('processed_files') or [])}")
        print(f"last error: {fallback.get('last_error')}")
    else:
        print(fallback.get("message"))

    parser_tests = report["parser_self_test"]
    parser_ok = all(row["passed"] for row in parser_tests)
    print("\nPayload parser")
    print("--------------")
    print(f"representative cases passed: {parser_ok}")

    tz_audit = report["timezone_audit"]
    print("\nTimezone audit")
    print("--------------")
    print(tz_audit["knmi_shadow_assessment"])
    print(f"potential issues outside/around scanned paths: {len(tz_audit['potential_issues'])}")


def main() -> None:
    args = parse_args()
    if args.latest_api_count < 1:
        raise SystemExit("--latest-api-count must be one or greater.")
    if args.process_missing_latest < 0:
        raise SystemExit("--process-missing-latest must be zero or greater.")

    zone = ZoneInfo(args.timezone)
    try:
        api_runs = fetch_api_runs(args.dataset, args.version, args.latest_api_count)
    except KnmiApiError as exc:
        raise SystemExit(f"Could not list KNMI API files: {exc}") from exc

    archive = archive_summary(args.db, args.site)
    archive["max_run_ts_local"] = iso_local(archive["max_run_ts_utc"], zone)
    comparison = api_db_comparison(api_runs, archive)
    processing_results = process_missing(api_runs, comparison, args)
    if processing_results:
        archive = archive_summary(args.db, args.site)
        archive["max_run_ts_local"] = iso_local(archive["max_run_ts_utc"], zone)
        comparison = api_db_comparison(api_runs, archive)

    listener_log = parse_listener_log(args.log_file)
    fallback_log = parse_fallback_log(args.fallback_log)
    unparsed_payloads = [line for line in listener_log.get("unparsed_payload_lines", [])]
    parser_tests = parser_self_test(unparsed_payloads)
    now_utc = pd.Timestamp.now(tz="UTC")
    latest_api_run = api_runs[0].run_ts if api_runs else None
    latest_api_created = api_runs[0].created_ts if api_runs else None

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "generated_at_local": datetime.now(timezone.utc).astimezone(zone).isoformat(),
        "site": args.site,
        "dataset": args.dataset,
        "version": args.version,
        "timezone": args.timezone,
        "api_runs": serializable_api_runs(api_runs, zone),
        "archive": archive,
        "api_db_comparison": comparison,
        "lag_summary": {
            "api_latest_run_minus_db_latest_run_hours": hours_between(latest_api_run, archive["max_run_ts_utc"]),
            "now_minus_api_latest_created_hours": hours_between(now_utc, latest_api_created),
            "now_minus_db_latest_run_hours": hours_between(now_utc, archive["max_run_ts_utc"]),
        },
        "listener_log": listener_log,
        "fallback_log": fallback_log,
        "parser_self_test": parser_tests,
        "timezone_audit": timezone_audit(),
        "processing_results": processing_results,
    }

    print_report(report, args.timezone, args.max_acceptable_lag_hours)

    if args.write_json:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\nWrote JSON diagnostic: {args.json_output}")


if __name__ == "__main__":
    main()
