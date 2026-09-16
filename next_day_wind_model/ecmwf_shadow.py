"""Restart-safe, development-only ECMWF shadow collection orchestration."""

from __future__ import annotations

import fcntl
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from next_day_wind_model.forecast_provider import ForecastRun, ForecastSite
from next_day_wind_model.forecast_store import (
    ArchiveCoverage,
    archive_coverage,
    begin_collection_attempt,
    finish_collection_attempt,
    record_download,
    upsert_forecast_points,
    upsert_forecast_values,
)


class CollectorAlreadyRunning(RuntimeError):
    pass


def ifs_oper_lead_hours(horizon_hours: int) -> tuple[int, ...]:
    """Return current public deterministic IFS output steps through a horizon."""
    horizon = int(horizon_hours)
    if horizon < 0 or horizon > 360:
        raise ValueError("horizon_hours must be between 0 and 360")
    early = tuple(range(0, min(horizon, 144) + 1, 3))
    late = tuple(range(150, horizon + 1, 6)) if horizon >= 150 else ()
    return early + late


def ifs_gust_lead_hours(wind_leads: Sequence[int]) -> tuple[int, ...]:
    """Gust is an interval maximum and has no meaningful analysis at step zero."""
    return tuple(sorted({int(step) for step in wind_leads if int(step) > 0}))


@contextmanager
def exclusive_collector_lock(path: Path) -> Iterator[None]:
    """Prevent a scheduler and an event trigger from downloading the same run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CollectorAlreadyRunning(f"collector lock is already held: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True)
class CollectionResult:
    run: ForecastRun
    site: ForecastSite
    before: ArchiveCoverage
    after: ArchiveCoverage
    wind_rows_written: int = 0
    gust_rows_written: int = 0
    bytes_downloaded: int = 0
    retrieve_calls: int = 0
    attempted: bool = False

    @property
    def status(self) -> str:
        return "complete" if self.after.complete else "partial"


@dataclass(frozen=True)
class MultiSiteCollectionResult:
    run: ForecastRun
    site_results: tuple[CollectionResult, ...]
    bytes_downloaded: int = 0
    retrieve_calls: int = 0

    @property
    def status(self) -> str:
        return "complete" if all(result.after.complete for result in self.site_results) else "partial"

    def for_site(self, site_name: str) -> CollectionResult:
        for result in self.site_results:
            if result.site.name == site_name:
                return result
        raise KeyError(site_name)


def _run_iso(run: ForecastRun) -> str:
    return run.run_time.isoformat().replace("+00:00", "Z")


def _artifact_bytes(artifacts: Sequence[Path]) -> int:
    return sum(path.stat().st_size for path in artifacts if path.is_file())


def _remove_artifacts(artifacts: Sequence[Path], work_dir: Path) -> None:
    root = work_dir.resolve()
    for artifact in artifacts:
        resolved = artifact.resolve()
        if resolved.parent != root:
            raise ValueError(f"refusing to remove artifact outside collector work directory: {resolved}")
        resolved.unlink(missing_ok=True)


def cleanup_staging(work_dir: Path, *, keep_grib: bool = False) -> int:
    """Remove disposable leftovers from a killed attempt within staging only."""
    if not work_dir.exists():
        return 0
    patterns = ("ecmwf_ifs_*.grib2.part",) if keep_grib else (
        "ecmwf_ifs_*.grib2.part",
        "ecmwf_ifs_*.grib2",
    )
    removed = 0
    for pattern in patterns:
        for path in work_dir.glob(pattern):
            if path.is_file():
                _remove_artifacts((path,), work_dir)
                removed += 1
    return removed


class EcmwfShadowCollector:
    """Collect only archive gaps, committing U/V before optional gust."""

    def __init__(
        self,
        *,
        provider: Any,
        conn: sqlite3.Connection,
        work_dir: Path,
        keep_grib: bool = False,
    ) -> None:
        self.provider = provider
        self.conn = conn
        self.work_dir = work_dir
        self.keep_grib = keep_grib

    def coverage(
        self,
        run: ForecastRun,
        site: ForecastSite,
        wind_leads: Sequence[int],
        gust_leads: Sequence[int],
    ) -> ArchiveCoverage:
        return archive_coverage(
            self.conn,
            provider=run.provider,
            model=run.model,
            run_time=_run_iso(run),
            site=site.name,
            expected_wind_leads=wind_leads,
            expected_gust_leads=gust_leads,
        )

    def collect_latest(
        self,
        site: ForecastSite,
        wind_leads: Sequence[int],
        *,
        include_gust: bool = True,
        dry_run: bool = False,
    ) -> CollectionResult:
        run = self.provider.latest_run(wind_leads)
        return self.collect_run(run, site, wind_leads, include_gust=include_gust, dry_run=dry_run)

    def collect_latest_sites(
        self,
        sites: Sequence[ForecastSite],
        wind_leads: Sequence[int],
        *,
        include_gust: bool = True,
        dry_run: bool = False,
    ) -> MultiSiteCollectionResult:
        run = self.provider.latest_run(wind_leads)
        return self.collect_run_sites(
            run,
            sites,
            wind_leads,
            include_gust=include_gust,
            dry_run=dry_run,
        )

    def collect_run(
        self,
        run: ForecastRun,
        site: ForecastSite,
        wind_leads: Sequence[int],
        *,
        include_gust: bool = True,
        dry_run: bool = False,
    ) -> CollectionResult:
        multi = self.collect_run_sites(
            run,
            (site,),
            wind_leads,
            include_gust=include_gust,
            dry_run=dry_run,
        )
        result = multi.site_results[0]
        return CollectionResult(
            run=result.run,
            site=result.site,
            before=result.before,
            after=result.after,
            wind_rows_written=result.wind_rows_written,
            gust_rows_written=result.gust_rows_written,
            bytes_downloaded=multi.bytes_downloaded,
            retrieve_calls=multi.retrieve_calls,
            attempted=result.attempted,
        )

    def collect_run_sites(
        self,
        run: ForecastRun,
        sites: Sequence[ForecastSite],
        wind_leads: Sequence[int],
        *,
        include_gust: bool = True,
        dry_run: bool = False,
    ) -> MultiSiteCollectionResult:
        wind_steps = tuple(sorted({int(step) for step in wind_leads}))
        if not wind_steps or wind_steps[0] < 0:
            raise ValueError("wind_leads must contain non-negative integer hours")
        unique_sites = tuple(sites)
        if not unique_sites:
            raise ValueError("sites cannot be empty")
        if len({site.name for site in unique_sites}) != len(unique_sites):
            raise ValueError("site names must be unique")
        gust_steps = ifs_gust_lead_hours(wind_steps) if include_gust else ()
        before = {
            site.name: self.coverage(run, site, wind_steps, gust_steps)
            for site in unique_sites
        }
        active_sites = tuple(site for site in unique_sites if not before[site.name].complete)
        if dry_run or not active_sites:
            return MultiSiteCollectionResult(
                run=run,
                site_results=tuple(
                    CollectionResult(
                        run=run,
                        site=site,
                        before=before[site.name],
                        after=before[site.name],
                    )
                    for site in unique_sites
                ),
            )

        run_iso = _run_iso(run)
        for site in active_sites:
            begin_collection_attempt(
                self.conn,
                provider=run.provider,
                model=run.model,
                run_time=run_iso,
                site=site.name,
                expected_wind_leads=wind_steps,
                expected_gust_leads=gust_steps,
            )

        wind_rows = {site.name: 0 for site in unique_sites}
        gust_rows = {site.name: 0 for site in unique_sites}
        bytes_downloaded = 0
        retrieve_calls = 0
        try:
            wind_sites = tuple(site for site in active_sites if before[site.name].missing_wind_leads)
            missing_wind = tuple(
                sorted(
                    set().union(*(before[site.name].missing_wind_leads for site in wind_sites))
                    if wind_sites
                    else set()
                )
            )
            if missing_wind:
                batch = self.provider.fetch_wind_sites(run, wind_sites, missing_wind, self.work_dir)
                wind_site_by_name = {site.name: site for site in wind_sites}
                selected = tuple(
                    point
                    for point in batch.points
                    if point.site.name in wind_site_by_name
                    and point.lead_time_hours in before[point.site.name].missing_wind_leads
                )
                if any(
                    point.provider != run.provider
                    or point.model != run.model
                    or point.run_time != run.run_time
                    or point.site != wind_site_by_name[point.site.name]
                    for point in selected
                ):
                    raise ValueError("wind batch contains inconsistent forecast identity")
                artifact_size = _artifact_bytes(batch.artifacts)
                upsert_forecast_points(self.conn, selected)
                for point in selected:
                    wind_rows[point.site.name] += 1
                record_download(
                    self.conn,
                    provider=run.provider,
                    model=run.model,
                    run_time=run_iso,
                    site="__shared__",
                    sites=tuple(site.name for site in wind_sites),
                    field_group="10u_10v",
                    lead_hours=missing_wind,
                    bytes_downloaded=artifact_size,
                )
                bytes_downloaded += artifact_size
                retrieve_calls += 1
                if not self.keep_grib:
                    _remove_artifacts(batch.artifacts, self.work_dir)

            current = {
                site.name: self.coverage(run, site, wind_steps, gust_steps)
                for site in unique_sites
            }
            gust_sites = tuple(
                site
                for site in active_sites
                if current[site.name].missing_gust_leads & current[site.name].stored_wind_leads
            )
            missing_gust = tuple(
                sorted(
                    set().union(
                        *(
                            current[site.name].missing_gust_leads & current[site.name].stored_wind_leads
                            for site in gust_sites
                        )
                    )
                    if gust_sites
                    else set()
                )
            )
            if missing_gust:
                batch = self.provider.fetch_gust_values_sites(run, gust_sites, missing_gust, self.work_dir)
                gust_site_by_name = {site.name: site for site in gust_sites}
                selected = tuple(
                    value
                    for value in batch.values
                    if value.site.name in gust_site_by_name
                    and value.lead_time_hours in current[value.site.name].missing_gust_leads
                )
                if any(
                    value.provider != run.provider
                    or value.model != run.model
                    or value.run_time != run.run_time
                    or value.site != gust_site_by_name[value.site.name]
                    or value.variable != "wind_gust_10m"
                    for value in selected
                ):
                    raise ValueError("gust batch contains inconsistent forecast identity")
                artifact_size = _artifact_bytes(batch.artifacts)
                upsert_forecast_values(self.conn, selected)
                for value in selected:
                    gust_rows[value.site.name] += 1
                record_download(
                    self.conn,
                    provider=run.provider,
                    model=run.model,
                    run_time=run_iso,
                    site="__shared__",
                    sites=tuple(site.name for site in gust_sites),
                    field_group="10fg",
                    lead_hours=missing_gust,
                    bytes_downloaded=artifact_size,
                )
                bytes_downloaded += artifact_size
                retrieve_calls += 1
                if not self.keep_grib:
                    _remove_artifacts(batch.artifacts, self.work_dir)

            after = {
                site.name: self.coverage(run, site, wind_steps, gust_steps)
                for site in unique_sites
            }
            for site in active_sites:
                site_after = after[site.name]
                finish_collection_attempt(
                    self.conn,
                    provider=run.provider,
                    model=run.model,
                    run_time=run_iso,
                    site=site.name,
                    status="complete" if site_after.complete else "partial",
                    error=None if site_after.complete else (
                        f"missing U/V leads {sorted(site_after.missing_wind_leads)}; "
                        f"missing gust leads {sorted(site_after.missing_gust_leads)}"
                    ),
                )
            return MultiSiteCollectionResult(
                run=run,
                site_results=tuple(
                    CollectionResult(
                        run=run,
                        site=site,
                        before=before[site.name],
                        after=after[site.name],
                        wind_rows_written=wind_rows[site.name],
                        gust_rows_written=gust_rows[site.name],
                        attempted=site in active_sites,
                    )
                    for site in unique_sites
                ),
                bytes_downloaded=bytes_downloaded,
                retrieve_calls=retrieve_calls,
            )
        except Exception as exc:
            for site in active_sites:
                site_after = self.coverage(run, site, wind_steps, gust_steps)
                has_any_data = bool(site_after.stored_wind_leads or site_after.stored_gust_leads)
                finish_collection_attempt(
                    self.conn,
                    provider=run.provider,
                    model=run.model,
                    run_time=run_iso,
                    site=site.name,
                    status="partial" if has_any_data else "failed",
                    error=str(exc)[:2000],
                )
            raise
