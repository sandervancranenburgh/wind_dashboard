#!/usr/bin/env python3
"""Extract latest KNMI HARMONIE P1 wind features into a separate SQLite table."""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from db_store import DB_FILENAME
from next_day_wind_model.knmi_harmonie import (
    DATASET,
    SOURCE,
    VERSION,
    KnmiApiError,
    KnmiExtractionError,
    SitePoint,
    cleanup_raw_harmonie_tars,
    create_harmonie_knmi_features_table,
    create_knmi_forecasts_shadow_table,
    ensure_downloaded_tar,
    extract_tar_features,
    knmi_archive_diagnostic,
    latest_harmonie_knmi_rows,
    latest_shadow_rows,
    list_knmi_files,
    parse_run_from_tar_filename,
    parse_utc_timestamp,
    recent_knmi_runs,
    select_latest_tar_file,
    upsert_knmi_forecasts_shadow,
    upsert_harmonie_knmi_features,
    utc_now_iso,
    write_knmi_rows_to_production_forecasts,
)


# TODO: Move KNMI extraction coordinates into shared site configuration once
# the project has source-specific site metadata. These are the verified point
# coordinates used for the KNMI/Windsurfice comparison.
DEFAULT_SITE_POINTS = {
    "valkenburgsemeer": SitePoint(site="valkenburgsemeer", lat=52.168, lon=4.437),
}


@dataclass(frozen=True)
class KnmiProcessResult:
    filename: str
    run_ts: str | None
    tar_path: Path
    selected_size: int | None
    horizons_extracted: int
    rows_written: int
    shadow_rows_written: int
    production_rows_written: int
    db_path: Path
    site: SitePoint
    errors: tuple[str, ...]
    latest_rows: Any
    latest_shadow_rows: Any
    cleanup_result: Any
    archive_diagnostic: dict[str, Any] | None
    latest_run_horizon_count: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download/extract the latest KNMI HARMONIE P1 tar and write feature rows to SQLite.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--db", type=Path, default=None, help=f"SQLite DB path. Default: data/{DB_FILENAME}")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/knmi/harmonie_arome_cy43_p1"))
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--version", default=VERSION)
    parser.add_argument("--site", default="valkenburgsemeer", choices=sorted(DEFAULT_SITE_POINTS))
    parser.add_argument("--site-lat", type=float, default=None)
    parser.add_argument("--site-lon", type=float, default=None)
    parser.add_argument("--max-files", type=int, default=10)
    parser.add_argument(
        "--filename",
        default=None,
        help="Process a specific KNMI tar filename, downloading it first if needed.",
    )
    parser.add_argument(
        "--latest-count",
        type=int,
        default=None,
        help="Process the latest N HARMONIE P1 tar files, oldest first. Useful for idempotent fallback catch-up.",
    )
    parser.add_argument(
        "--tar-path",
        type=Path,
        default=None,
        help="Use an existing local tar file instead of listing/downloading from KNMI.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep extracting other horizons if one GRIB member fails.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Only print latest canonical rows from harmonie_knmi_features.",
    )
    parser.add_argument("--inspect-shadow", action="store_true", help="Only print latest rows from knmi_forecasts_shadow.")
    parser.add_argument("--inspect-runs", action="store_true", help="Only print recent KNMI runs stored in the DB.")
    parser.add_argument("--diagnose-archive", action="store_true", help="Only print archive depth/training-readiness diagnostics.")
    parser.add_argument("--archive-diagnostic", action="store_true", help="Alias for --diagnose-archive.")
    parser.add_argument("--inspect-limit", type=int, default=5)
    parser.add_argument("--timezone", default="Europe/Amsterdam", help="Timezone for local diagnostic display.")
    parser.add_argument(
        "--include-observation-joinability",
        action="store_true",
        help="Include slower observation joinability counts in archive diagnostics.",
    )
    parser.add_argument("--model", default="HARMONIE", help="Forecast model label for shadow rows.")
    parser.add_argument(
        "--skip-shadow",
        action="store_true",
        help="Write only harmonie_knmi_features and skip the compatibility shadow table.",
    )
    parser.add_argument(
        "--write-production",
        action="store_true",
        help="DANGEROUS: also write KNMI-derived rows into the production forecasts table.",
    )
    parser.add_argument(
        "--skip-archive-diagnostic",
        action="store_true",
        help="Do not print the concise archive diagnostic after a successful write.",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep the processed raw tar after a successful DB write. By default it is deleted.",
    )
    parser.add_argument(
        "--raw-retention-runs",
        type=int,
        default=None,
        help=(
            "After a successful DB write, keep only the latest N HARM43_V1_P1_*.tar files "
            "in --raw-dir and delete older matching tar files. With --keep-raw, the currently "
            "processed tar is also retained even if it is older than the latest N."
        ),
    )
    parser.add_argument(
        "--cleanup-dry-run",
        action="store_true",
        help="Print raw tar cleanup actions after successful extraction without deleting files.",
    )
    return parser.parse_args()


def site_point_from_args(args: argparse.Namespace) -> SitePoint:
    return site_point_from_name(args.site, args.site_lat, args.site_lon)


def db_path_from_args(args: argparse.Namespace) -> Path:
    return args.db if args.db is not None else args.data_dir / DB_FILENAME


def site_point_from_name(site: str, site_lat: float | None = None, site_lon: float | None = None) -> SitePoint:
    if site not in DEFAULT_SITE_POINTS:
        choices = ", ".join(sorted(DEFAULT_SITE_POINTS))
        raise ValueError(f"Unknown site {site!r}. Known sites: {choices}")
    default = DEFAULT_SITE_POINTS[site]
    if site_lat is None and site_lon is None:
        return default
    if site_lat is None or site_lon is None:
        raise ValueError("site_lat and site_lon must be provided together.")
    return SitePoint(site=site, lat=float(site_lat), lon=float(site_lon))


def connect_sqlite_with_timeout(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=60)
    conn.execute("PRAGMA busy_timeout = 60000")
    return conn


def sqlite_retry_delay(attempt: int) -> float:
    return min(2.0 * attempt, 10.0)


def print_latest_rows(db_path: Path, site: str, limit: int) -> None:
    conn = connect_sqlite_with_timeout(db_path)
    try:
        rows = latest_harmonie_knmi_rows(conn, site=site, limit=limit)
    finally:
        conn.close()
    if rows.empty:
        print(f"No harmonie_knmi_features rows found for site={site!r} in {db_path}.")
        return
    print(rows.to_string(index=False))


def print_latest_shadow_rows(db_path: Path, site: str, limit: int) -> None:
    conn = connect_sqlite_with_timeout(db_path)
    try:
        rows = latest_shadow_rows(conn, site=site, limit=limit)
    finally:
        conn.close()
    if rows.empty:
        print(f"No knmi_forecasts_shadow rows found for site={site!r} in {db_path}.")
        return
    print(rows.to_string(index=False))


def print_recent_runs(db_path: Path, site: str, limit: int) -> None:
    conn = connect_sqlite_with_timeout(db_path)
    try:
        rows = recent_knmi_runs(conn, site=site, limit=limit)
    finally:
        conn.close()
    if rows.empty:
        print(f"No KNMI runs found for site={site!r} in {db_path}.")
        return
    print(rows.to_string(index=False))


def local_iso(value: Any, timezone_name: str) -> str | None:
    if value is None:
        return None
    return parse_utc_timestamp(value).tz_convert(ZoneInfo(timezone_name)).isoformat()


def lag_hours(later: Any, earlier: Any) -> float | None:
    if later is None or earlier is None:
        return None
    delta = parse_utc_timestamp(later) - parse_utc_timestamp(earlier)
    return round(delta.total_seconds() / 3600, 3)


def print_archive_diagnostic(
    db_path: Path,
    site: str,
    timezone_name: str = "Europe/Amsterdam",
    *,
    include_observation_joinability: bool = False,
) -> None:
    conn = connect_sqlite_with_timeout(db_path)
    try:
        diagnostic = knmi_archive_diagnostic(
            conn,
            site=site,
            ensure_table=False,
            include_observation_joinability=include_observation_joinability,
        )
    finally:
        conn.close()
    print("\nKNMI archive diagnostic")
    print("=======================")
    for key in (
        "distinct_run_ts",
        "distinct_target_dates",
        "min_run_ts",
        "max_run_ts",
        "min_target_ts",
        "max_target_ts",
        "row_count",
        "past_target_rows",
        "rows_joinable_to_observations_exact",
        "rows_joinable_to_observations_30min",
        "diagnostic_now_ts",
    ):
        value = diagnostic.get(key)
        if key.startswith("rows_joinable") and value is None:
            value = "not calculated (use --include-observation-joinability)"
        print(f"{key}: {value}")
    print(f"max_run_ts_local_{timezone_name}: {local_iso(diagnostic.get('max_run_ts'), timezone_name)}")
    print(
        f"diagnostic_now_ts_local_{timezone_name}: "
        f"{local_iso(diagnostic.get('diagnostic_now_ts'), timezone_name)}"
    )
    print(
        "latest_archived_run_age_hours: "
        f"{lag_hours(diagnostic.get('diagnostic_now_ts'), diagnostic.get('max_run_ts'))}"
    )
    if int(diagnostic.get("distinct_run_ts") or 0) <= 1:
        print(
            "Only one KNMI run is archived. This is sufficient for extraction verification, "
            "but not for model training/evaluation."
        )


def select_tar(args: argparse.Namespace) -> tuple[Path, str, str | None, int | None]:
    if args.tar_path is not None:
        tar_path = args.tar_path
        run_ts = parse_run_from_tar_filename(tar_path.name).isoformat()
        return tar_path, tar_path.name, run_ts, None

    if args.filename is not None:
        tar_path = ensure_downloaded_tar(args.filename, args.raw_dir, args.dataset, args.version)
        run_ts = parse_run_from_tar_filename(args.filename).isoformat()
        return tar_path, args.filename, run_ts, None

    files = list_knmi_files(args.dataset, args.version, max_keys=args.max_files)
    latest = select_latest_tar_file(files)
    tar_path = ensure_downloaded_tar(latest.filename, args.raw_dir, args.dataset, args.version)
    run_ts = parse_run_from_tar_filename(latest.filename).isoformat()
    return tar_path, latest.filename, run_ts, latest.size


def select_latest_tar_filenames(
    *,
    dataset: str = DATASET,
    version: str = VERSION,
    max_files: int = 10,
    latest_count: int = 1,
) -> list[str]:
    if latest_count < 1:
        raise ValueError("--latest-count must be one or greater.")
    files = list_knmi_files(dataset, version, max_keys=max(max_files, latest_count))
    tar_files = [item for item in files if parseable_harmonie_filename(item.filename)]
    if not tar_files:
        raise KnmiApiError("No KNMI HARMONIE tar files returned by the API.")
    return [item.filename for item in reversed(tar_files[:latest_count])]


def parseable_harmonie_filename(filename: str) -> bool:
    try:
        parse_run_from_tar_filename(filename)
    except ValueError:
        return False
    return True


def print_cleanup_summary(args: argparse.Namespace, tar_path: Path, result: Any) -> None:

    print("\nRaw tar cleanup")
    print("===============")
    processed_deleted = any(path.resolve() == tar_path.resolve() for path in result.deleted)
    processed_would_delete = any(path.resolve() == tar_path.resolve() for path in result.would_delete)
    if args.keep_raw:
        print(f"Keeping raw tar because --keep-raw was specified: {tar_path}")
    elif args.cleanup_dry_run and processed_would_delete:
        print(f"Would delete raw tar after successful extraction: {tar_path}")
    elif processed_deleted:
        print(f"Deleted raw tar after successful extraction: {tar_path}")
    elif args.raw_retention_runs is not None:
        print(f"Retained processed raw tar under --raw-retention-runs policy: {tar_path}")
    else:
        print(f"No processed raw tar deleted. It was not found as a matching HARMONIE P1 tar in {args.raw_dir}.")

    if args.raw_retention_runs is not None:
        print(f"Raw retention runs: {args.raw_retention_runs}")
        retained = [path for path in result.retained if path.exists() or args.cleanup_dry_run]
        if retained:
            print("Retained matching tar files:")
            for path in retained:
                print(f"- {path}")
        else:
            print("Retained matching tar files: none")

    retention_deleted = [
        path for path in result.deleted if not path.resolve() == tar_path.resolve()
    ]
    if retention_deleted:
        print("Deleted older matching tar files:")
        for path in retention_deleted:
            print(f"- {path}")
    if args.cleanup_dry_run and result.would_delete:
        print("Dry-run deletion candidates:")
        for path in result.would_delete:
            print(f"- {path}")
    if result.warnings:
        print("Cleanup warnings:")
        for warning in result.warnings:
            print(f"- {warning}")


def process_knmi_file_to_db(
    filename: str | None,
    db_path: Path,
    site: str,
    keep_raw: bool = False,
    raw_retention_runs: int | None = None,
    cleanup_dry_run: bool = False,
    *,
    raw_dir: Path = Path("data/raw/knmi/harmonie_arome_cy43_p1"),
    dataset: str = DATASET,
    version: str = VERSION,
    site_lat: float | None = None,
    site_lon: float | None = None,
    max_files: int = 10,
    tar_path: Path | None = None,
    continue_on_error: bool = False,
    model: str = "HARMONIE",
    skip_shadow: bool = False,
    write_production: bool = False,
    skip_archive_diagnostic: bool = False,
    include_observation_joinability: bool = False,
    db_retries: int = 5,
) -> KnmiProcessResult:
    """Download/extract one KNMI HARMONIE P1 tar and write shadow rows to SQLite."""
    if raw_retention_runs is not None and raw_retention_runs < 0:
        raise ValueError("raw_retention_runs must be zero or greater.")

    site_point = site_point_from_name(site, site_lat, site_lon)
    args = argparse.Namespace(
        tar_path=tar_path,
        filename=filename,
        raw_dir=raw_dir,
        dataset=dataset,
        version=version,
        max_files=max_files,
    )
    fetched_ts = utc_now_iso()
    tar_path, selected_filename, run_ts, selected_size = select_tar(args)
    result = extract_tar_features(
        tar_path,
        site_point,
        source=SOURCE,
        dataset=dataset,
        fetched_ts=fetched_ts,
        continue_on_error=continue_on_error,
    )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    last_locked_error: sqlite3.OperationalError | None = None
    for attempt in range(1, db_retries + 1):
        conn = connect_sqlite_with_timeout(db_path)
        try:
            create_harmonie_knmi_features_table(conn)
            create_knmi_forecasts_shadow_table(conn)
            rows_written = upsert_harmonie_knmi_features(conn, result.frame)
            shadow_rows_written = 0 if skip_shadow else upsert_knmi_forecasts_shadow(conn, result.frame, model=model)
            production_rows_written = (
                write_knmi_rows_to_production_forecasts(conn, result.frame, model=model)
                if write_production
                else 0
            )
            sample = latest_harmonie_knmi_rows(conn, site=site_point.site, limit=5)
            shadow_sample = latest_shadow_rows(conn, site=site_point.site, limit=5)
            diagnostic = (
                None
                if skip_archive_diagnostic
                else knmi_archive_diagnostic(
                    conn,
                    site=site_point.site,
                    include_observation_joinability=include_observation_joinability,
                )
            )
            recent_runs = recent_knmi_runs(conn, site=site_point.site, limit=1)
            latest_run_horizon_count = None
            if not recent_runs.empty:
                latest_run_horizon_count = int(recent_runs.iloc[0]["horizon_count"])
            break
        except sqlite3.OperationalError as exc:
            conn.close()
            if "database is locked" not in str(exc).lower() or attempt >= db_retries:
                raise
            last_locked_error = exc
            time.sleep(sqlite_retry_delay(attempt))
            continue
        finally:
            try:
                conn.close()
            except UnboundLocalError:
                pass
    else:
        raise last_locked_error or sqlite3.OperationalError("SQLite write failed.")

    cleanup_result = cleanup_raw_harmonie_tars(
        raw_dir,
        processed_tar=tar_path,
        keep_processed=keep_raw,
        retention_runs=raw_retention_runs,
        dry_run=cleanup_dry_run,
    )

    return KnmiProcessResult(
        filename=selected_filename,
        run_ts=run_ts,
        tar_path=tar_path,
        selected_size=selected_size,
        horizons_extracted=len(result.frame),
        rows_written=rows_written,
        shadow_rows_written=shadow_rows_written,
        production_rows_written=production_rows_written,
        db_path=db_path,
        site=site_point,
        errors=tuple(result.errors),
        latest_rows=sample,
        latest_shadow_rows=shadow_sample,
        cleanup_result=cleanup_result,
        archive_diagnostic=diagnostic,
        latest_run_horizon_count=latest_run_horizon_count,
    )


def print_process_result(args: argparse.Namespace, result: KnmiProcessResult) -> None:
    print_cleanup_summary(args, result.tar_path, result.cleanup_result)

    print("\nKNMI HARMONIE feature extraction")
    print("================================")
    print(f"Selected filename: {result.filename}")
    print(f"Run timestamp: {result.run_ts}")
    if result.selected_size is not None:
        print(f"Selected file size: {result.selected_size}")
    print(f"Tar path: {result.tar_path}")
    print(f"Horizons extracted: {result.horizons_extracted}")
    print(f"Canonical feature rows written: {result.rows_written}")
    print(f"Shadow forecast rows written: {result.shadow_rows_written}")
    if args.write_production:
        print(f"Production forecast rows written: {result.production_rows_written}")
    print(f"Database path: {result.db_path}")
    print(f"Site: {result.site.site} ({result.site.lat}, {result.site.lon})")
    if result.errors:
        print(f"Partial extraction errors: {len(result.errors)}")
        for error in result.errors[:5]:
            print(f"- {error}")
    print("\nLatest harmonie_knmi_features rows:")
    if result.latest_rows.empty:
        print("No rows found after write.")
    else:
        print(result.latest_rows.to_string(index=False))
    print("\nLatest knmi_forecasts_shadow rows:")
    if result.latest_shadow_rows.empty:
        print("No shadow rows found after write.")
    else:
        print(result.latest_shadow_rows.to_string(index=False))
    if not args.skip_archive_diagnostic and result.archive_diagnostic is not None:
        print("\nKNMI archive diagnostic")
        print("=======================")
        for key in (
            "distinct_run_ts",
            "distinct_target_dates",
            "min_run_ts",
            "max_run_ts",
            "min_target_ts",
            "max_target_ts",
            "row_count",
            "past_target_rows",
            "rows_joinable_to_observations_exact",
            "rows_joinable_to_observations_30min",
            "diagnostic_now_ts",
        ):
            value = result.archive_diagnostic.get(key)
            if key.startswith("rows_joinable") and value is None:
                value = "not calculated (use --include-observation-joinability)"
            print(f"{key}: {value}")
        print(
            f"max_run_ts_local_{args.timezone}: "
            f"{local_iso(result.archive_diagnostic.get('max_run_ts'), args.timezone)}"
        )
        print(
            f"diagnostic_now_ts_local_{args.timezone}: "
            f"{local_iso(result.archive_diagnostic.get('diagnostic_now_ts'), args.timezone)}"
        )
        print(
            "latest_archived_run_age_hours: "
            f"{lag_hours(result.archive_diagnostic.get('diagnostic_now_ts'), result.archive_diagnostic.get('max_run_ts'))}"
        )
        if int(result.archive_diagnostic.get("distinct_run_ts") or 0) <= 1:
            print(
                "Only one KNMI run is archived. This is sufficient for extraction verification, "
                "but not for model training/evaluation."
            )


def main() -> None:
    args = parse_args()
    if args.raw_retention_runs is not None and args.raw_retention_runs < 0:
        raise SystemExit("--raw-retention-runs must be zero or greater.")
    site = site_point_from_args(args)
    db_path = db_path_from_args(args)

    if args.inspect:
        print_latest_rows(db_path, site=site.site, limit=args.inspect_limit)
        return
    if args.inspect_shadow:
        print_latest_shadow_rows(db_path, site=site.site, limit=args.inspect_limit)
        return
    if args.inspect_runs:
        print_recent_runs(db_path, site=site.site, limit=args.inspect_limit)
        return
    if args.diagnose_archive or args.archive_diagnostic:
        print_archive_diagnostic(
            db_path,
            site=site.site,
            timezone_name=args.timezone,
            include_observation_joinability=args.include_observation_joinability,
        )
        return

    if args.latest_count is not None and (args.filename is not None or args.tar_path is not None):
        raise SystemExit("--latest-count cannot be combined with --filename or --tar-path.")
    if args.latest_count is not None and args.latest_count < 1:
        raise SystemExit("--latest-count must be one or greater.")

    try:
        filenames = (
            select_latest_tar_filenames(
                dataset=args.dataset,
                version=args.version,
                max_files=args.max_files,
                latest_count=args.latest_count,
            )
            if args.latest_count is not None
            else [args.filename]
        )
        for index, filename in enumerate(filenames, start=1):
            if len(filenames) > 1:
                print(f"\nProcessing KNMI file {index}/{len(filenames)}: {filename}")
            result = process_knmi_file_to_db(
                filename=filename,
                db_path=db_path,
                site=site.site,
                keep_raw=args.keep_raw,
                raw_retention_runs=args.raw_retention_runs,
                cleanup_dry_run=args.cleanup_dry_run,
                raw_dir=args.raw_dir,
                dataset=args.dataset,
                version=args.version,
                site_lat=args.site_lat,
                site_lon=args.site_lon,
                max_files=args.max_files,
                tar_path=args.tar_path,
                continue_on_error=args.continue_on_error,
                model=args.model,
                skip_shadow=args.skip_shadow,
                write_production=args.write_production,
                skip_archive_diagnostic=args.skip_archive_diagnostic,
                include_observation_joinability=args.include_observation_joinability,
            )
            print_process_result(args, result)
    except (KnmiApiError, KnmiExtractionError, FileNotFoundError, ValueError, sqlite3.OperationalError) as exc:
        raise SystemExit(f"KNMI extraction failed: {exc}") from exc


if __name__ == "__main__":
    main()
