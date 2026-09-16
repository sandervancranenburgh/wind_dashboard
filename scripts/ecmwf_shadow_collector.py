#!/usr/bin/env python3
"""Collect deterministic ECMWF wind into an isolated shadow archive."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from next_day_wind_model.ecmwf_open_data import (
    MODEL_VERSION,
    RESOLUTION,
    SOURCE,
    EcmwfOpenDataProvider,
)
from next_day_wind_model.ecmwf_shadow import (
    CollectorAlreadyRunning,
    EcmwfShadowCollector,
    MultiSiteCollectionResult,
    cleanup_staging,
    exclusive_collector_lock,
    ifs_oper_lead_hours,
)
from next_day_wind_model.forecast_provider import ForecastRun, ForecastSite
from next_day_wind_model.forecast_store import (
    connect_development_db,
    create_forecast_points_table,
    validate_development_db_path,
)
from next_day_wind_model.shadow_config import (
    DEFAULT_CONFIG_PATH,
    ShadowExperimentConfig,
    load_shadow_config,
    select_sites,
)


DEFAULT_DATA_DIR = Path("data/dev/ecmwf_shadow_archive")
DEFAULT_CONFIG = REPO_ROOT / DEFAULT_CONFIG_PATH
STATE_FILENAME = "collector_state.json"


@dataclass(frozen=True)
class CycleOutcome:
    result: MultiSiteCollectionResult | None
    reason: str
    sites: tuple[ForecastSite, ...]
    steps: tuple[int, ...]
    db_path: Path

    @property
    def exit_code(self) -> int:
        if self.result is None:
            return 0
        return 0 if self.result.status == "complete" else 2


def parse_run(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y%m%d%H").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("run must use UTC format YYYYMMDDHH") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeatedly archive missing ECMWF IFS 10 m wind and gust in an isolated SQLite database."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--db", type=Path, default=None, help="Must be located inside --data-dir.")
    parser.add_argument("--horizon-hours", type=int, default=None, help="Override the configured horizon.")
    parser.add_argument("--site", action="append", dest="sites", help="Configured site ID; repeat as needed.")
    parser.add_argument("--latitude", type=float, default=None, help="Custom coordinate for exactly one --site.")
    parser.add_argument("--longitude", type=float, default=None, help="Custom coordinate for exactly one --site.")
    parser.add_argument("--source", choices=("ecmwf", "aws", "google", "azure"), default=None)
    parser.add_argument("--run", type=parse_run, help="Collect an explicit UTC run (YYYYMMDDHH).")
    parser.add_argument("--without-gust", action="store_true", help="Archive U/V only.")
    parser.add_argument("--dry-run", action="store_true", help="Use HEAD and a memory snapshot; write no files or rows.")
    parser.add_argument("--keep-grib", action="store_true", help="Retain selected global GRIB files after point archival.")
    parser.add_argument("--watch", action="store_true", help="Continue polling; default is one check.")
    parser.add_argument("--poll-interval-seconds", type=int, default=900)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.watch and args.run is not None:
        parser.error("--watch and --run cannot be combined")
    if args.poll_interval_seconds < 60:
        parser.error("--poll-interval-seconds must be at least 60")
    if args.horizon_hours is not None:
        try:
            ifs_oper_lead_hours(args.horizon_hours)
        except ValueError as exc:
            parser.error(str(exc))
    if (args.latitude is None) != (args.longitude is None):
        parser.error("--latitude and --longitude must be supplied together")
    if args.latitude is not None and (not args.sites or len(args.sites) != 1):
        parser.error("custom coordinates require exactly one --site")
    return args


def _safe_paths(data_dir: Path, db_arg: Path | None) -> tuple[Path, Path]:
    resolved_data = data_dir.expanduser().resolve()
    resolved_db = validate_development_db_path(
        db_arg if db_arg is not None else resolved_data / "ecmwf_shadow.sqlite"
    )
    if not resolved_db.is_relative_to(resolved_data):
        raise ValueError("shadow database must be located inside --data-dir")
    return resolved_data, resolved_db


def _dry_run_connection(db_path: Path) -> sqlite3.Connection:
    memory = sqlite3.connect(":memory:")
    if db_path.is_file():
        source = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            source.backup(memory)
        finally:
            source.close()
    create_forecast_points_table(memory)
    return memory


def _explicit_run(provider: EcmwfOpenDataProvider, run_time: datetime) -> ForecastRun:
    return ForecastRun(
        provider=provider.provider_name,
        model=provider.model_name,
        run_time=run_time,
        source=SOURCE,
        model_version=MODEL_VERSION,
        resolution=RESOLUTION,
        provider_reference=f"{run_time:%Y%m%d%H}",
    )


def _print_result(result: MultiSiteCollectionResult, db_path: Path, steps: tuple[int, ...], dry_run: bool) -> None:
    run = result.run.run_time.isoformat().replace("+00:00", "Z")
    print(f"ECMWF IFS run: {run}")
    print(f"Requested wind leads ({len(steps)}): {list(steps)}")
    for site_result in result.site_results:
        print(f"Site: {site_result.site.name}")
        print(f"  Missing U/V before: {sorted(site_result.before.missing_wind_leads)}")
        print(f"  Missing gust before: {sorted(site_result.before.missing_gust_leads)}")
        if not dry_run:
            print(
                f"  Rows written: U/V={site_result.wind_rows_written}, "
                f"gust={site_result.gust_rows_written}"
            )
            print(f"  Status: {site_result.status}")
            print(f"  Remaining U/V: {sorted(site_result.after.missing_wind_leads)}")
            print(f"  Remaining gust: {sorted(site_result.after.missing_gust_leads)}")
    if dry_run:
        print("Dry run: no database, lock, staging, or download files were written.")
        return
    print(f"Shared official-client retrieve calls: {result.retrieve_calls}")
    print(f"Shared download total: {result.bytes_downloaded / 1_000_000:.3f} MB")
    print(f"Overall status: {result.status}")
    print(f"Shadow database: {db_path}")


def _configured_sites(args: argparse.Namespace, config: ShadowExperimentConfig) -> tuple[ForecastSite, ...]:
    if args.latitude is not None:
        return (ForecastSite(args.sites[0], args.latitude, args.longitude),)
    return select_sites(config, args.sites)


def _eligible_sites(
    conn: sqlite3.Connection,
    sites: tuple[ForecastSite, ...],
    maximum: int | None,
) -> tuple[ForecastSite, ...]:
    if maximum is None:
        return sites
    eligible: list[ForecastSite] = []
    for site in sites:
        count = int(
            conn.execute(
                """
                SELECT COUNT(DISTINCT run_time)
                FROM forecast_collection_runs
                WHERE provider = 'ECMWF' AND model = 'IFS' AND site = ? AND status = 'complete'
                """,
                (site.name,),
            ).fetchone()[0]
        )
        if count >= maximum:
            logging.warning(
                "Skipping %s: completed-run safeguard reached (%d/%d)",
                site.name,
                count,
                maximum,
            )
        else:
            eligible.append(site)
    return tuple(eligible)


def _write_state(data_dir: Path, outcome: CycleOutcome) -> None:
    result = outcome.result
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    payload = {
        "last_successful_execution_utc": now,
        "reason": outcome.reason,
        "database": str(outcome.db_path),
        "sites": [site.name for site in outcome.sites],
        "horizon_hours": outcome.steps[-1],
        "run_time_utc": (
            result.run.run_time.isoformat().replace("+00:00", "Z") if result is not None else None
        ),
        "status": result.status if result is not None else outcome.reason,
        "retrieve_calls": result.retrieve_calls if result is not None else 0,
        "bytes_downloaded": result.bytes_downloaded if result is not None else 0,
    }
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / STATE_FILENAME
    partial = data_dir / f".{STATE_FILENAME}.{os.getpid()}.part"
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(target)


def run_cycle(
    args: argparse.Namespace,
    provider: EcmwfOpenDataProvider,
    config: ShadowExperimentConfig,
) -> CycleOutcome:
    data_dir, db_path = _safe_paths(args.data_dir, args.db)
    steps = ifs_oper_lead_hours(
        args.horizon_hours if args.horizon_hours is not None else config.horizon_hours
    )
    sites = _configured_sites(args, config)
    if args.dry_run:
        conn = _dry_run_connection(db_path)
    else:
        removed = cleanup_staging(data_dir / "staging", keep_grib=args.keep_grib)
        if removed:
            logging.info("Removed %d orphaned ECMWF staging file(s)", removed)
        conn = connect_development_db(db_path)

    try:
        create_forecast_points_table(conn)
        sites = _eligible_sites(conn, sites, config.max_completed_runs_per_site)
        if not sites:
            return CycleOutcome(None, "run_cap_reached", sites, steps, db_path)
        collector = EcmwfShadowCollector(
            provider=provider,
            conn=conn,
            work_dir=data_dir / "staging",
            keep_grib=args.keep_grib,
        )
        run = _explicit_run(provider, args.run) if args.run is not None else provider.latest_run(steps)
        result = collector.collect_run_sites(
            run,
            sites,
            steps,
            include_gust=not args.without_gust,
            dry_run=args.dry_run,
        )
    finally:
        conn.close()
    _print_result(result, db_path, steps, args.dry_run)
    return CycleOutcome(result, "dry_run" if args.dry_run else "collected", sites, steps, db_path)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    data_dir, _ = _safe_paths(args.data_dir, args.db)
    provider: EcmwfOpenDataProvider | None = None
    provider_source: str | None = None

    while True:
        try:
            config = load_shadow_config(args.config)
            now = datetime.now(timezone.utc)
            if config.expired(now):
                logging.warning(
                    "ECMWF shadow experiment expired at %s; no download or database write was attempted",
                    config.experiment_end_utc.isoformat().replace("+00:00", "Z"),
                )
                return 0
            selected_source = args.source or config.open_data_source
            if provider is None or provider_source != selected_source:
                provider = EcmwfOpenDataProvider(source=selected_source)
                provider_source = selected_source
            if args.dry_run:
                outcome = run_cycle(args, provider, config)
            else:
                with exclusive_collector_lock(data_dir / "ecmwf_shadow.lock"):
                    outcome = run_cycle(args, provider, config)
                    _write_state(data_dir, outcome)
            exit_code = outcome.exit_code
        except CollectorAlreadyRunning as exc:
            logging.info("%s", exc)
            exit_code = 0
        except Exception:
            logging.exception("ECMWF shadow collection failed")
            exit_code = 1

        if not args.watch:
            return exit_code
        try:
            time.sleep(args.poll_interval_seconds)
        except KeyboardInterrupt:
            return 130


if __name__ == "__main__":
    raise SystemExit(main())
