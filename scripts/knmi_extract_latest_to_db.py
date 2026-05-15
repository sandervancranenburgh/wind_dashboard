#!/usr/bin/env python3
"""Extract latest KNMI HARMONIE P1 wind features into a separate SQLite table."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

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
    create_harmonie_knmi_features_table,
    ensure_downloaded_tar,
    extract_tar_features,
    latest_harmonie_knmi_rows,
    list_knmi_files,
    parse_run_from_tar_filename,
    select_latest_tar_file,
    upsert_harmonie_knmi_features,
    utc_now_iso,
)


# TODO: Move KNMI extraction coordinates into shared site configuration once
# the project has source-specific site metadata. These are the verified point
# coordinates used for the KNMI/Windsurfice comparison.
DEFAULT_SITE_POINTS = {
    "valkenburgsemeer": SitePoint(site="valkenburgsemeer", lat=52.168, lon=4.437),
}


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
        "--tar-path",
        type=Path,
        default=None,
        help="Use an existing tar file instead of listing/downloading from KNMI.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep extracting other horizons if one GRIB member fails.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Only print latest rows from harmonie_knmi_features.",
    )
    parser.add_argument("--inspect-limit", type=int, default=5)
    return parser.parse_args()


def site_point_from_args(args: argparse.Namespace) -> SitePoint:
    default = DEFAULT_SITE_POINTS[args.site]
    if args.site_lat is None and args.site_lon is None:
        return default
    if args.site_lat is None or args.site_lon is None:
        raise ValueError("--site-lat and --site-lon must be provided together.")
    return SitePoint(site=args.site, lat=float(args.site_lat), lon=float(args.site_lon))


def db_path_from_args(args: argparse.Namespace) -> Path:
    return args.db if args.db is not None else args.data_dir / DB_FILENAME


def print_latest_rows(db_path: Path, site: str, limit: int) -> None:
    conn = sqlite3.connect(db_path)
    try:
        rows = latest_harmonie_knmi_rows(conn, site=site, limit=limit)
    finally:
        conn.close()
    if rows.empty:
        print(f"No harmonie_knmi_features rows found for site={site!r} in {db_path}.")
        return
    print(rows.to_string(index=False))


def select_tar(args: argparse.Namespace) -> tuple[Path, str, str | None, int | None]:
    if args.tar_path is not None:
        tar_path = args.tar_path
        run_ts = parse_run_from_tar_filename(tar_path.name).isoformat()
        return tar_path, tar_path.name, run_ts, None

    files = list_knmi_files(args.dataset, args.version, max_keys=args.max_files)
    latest = select_latest_tar_file(files)
    tar_path = ensure_downloaded_tar(latest.filename, args.raw_dir, args.dataset, args.version)
    run_ts = parse_run_from_tar_filename(latest.filename).isoformat()
    return tar_path, latest.filename, run_ts, latest.size


def main() -> None:
    args = parse_args()
    site = site_point_from_args(args)
    db_path = db_path_from_args(args)

    if args.inspect:
        print_latest_rows(db_path, site=site.site, limit=args.inspect_limit)
        return

    fetched_ts = utc_now_iso()
    try:
        tar_path, selected_filename, run_ts, selected_size = select_tar(args)
        result = extract_tar_features(
            tar_path,
            site,
            source=SOURCE,
            dataset=args.dataset,
            fetched_ts=fetched_ts,
            continue_on_error=args.continue_on_error,
        )
    except (KnmiApiError, KnmiExtractionError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"KNMI extraction failed: {exc}") from exc

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        create_harmonie_knmi_features_table(conn)
        rows_written = upsert_harmonie_knmi_features(conn, result.frame)
        sample = latest_harmonie_knmi_rows(conn, site=site.site, limit=5)
    finally:
        conn.close()

    print("\nKNMI HARMONIE feature extraction")
    print("================================")
    print(f"Selected filename: {selected_filename}")
    print(f"Run timestamp: {run_ts}")
    if selected_size is not None:
        print(f"Selected file size: {selected_size}")
    print(f"Tar path: {tar_path}")
    print(f"Horizons extracted: {len(result.frame)}")
    print(f"Rows written: {rows_written}")
    print(f"Database path: {db_path}")
    print(f"Site: {site.site} ({site.lat}, {site.lon})")
    if result.errors:
        print(f"Partial extraction errors: {len(result.errors)}")
        for error in result.errors[:5]:
            print(f"- {error}")
    print("\nLatest harmonie_knmi_features rows:")
    if sample.empty:
        print("No rows found after write.")
    else:
        print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
