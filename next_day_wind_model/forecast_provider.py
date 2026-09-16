"""Small provider-neutral boundary for numerical weather forecasts.

The operational KNMI pipeline predates this module and deliberately does not
depend on it yet.  New providers can return :class:`ForecastPoint` instances,
while adapters let the existing HARMONIE output participate without changing
the production ingestion path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


def utc_datetime(value: datetime) -> datetime:
    """Return an aware UTC datetime; provider libraries often return naive UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def wind_speed_direction(u_mps: float, v_mps: float) -> tuple[float, float]:
    """Return vector speed (m/s) and meteorological direction-from (degrees)."""
    speed = math.hypot(float(u_mps), float(v_mps))
    direction = (270.0 - math.degrees(math.atan2(float(v_mps), float(u_mps)))) % 360.0
    return speed, direction


@dataclass(frozen=True)
class ForecastSite:
    name: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class ForecastRun:
    provider: str
    model: str
    run_time: datetime
    source: str
    model_version: str | None = None
    resolution: str | None = None
    provider_reference: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_time", utc_datetime(self.run_time))


@dataclass(frozen=True)
class ForecastPoint:
    """A normalized wind forecast at one site, run, and valid time."""

    provider: str
    model: str
    run_time: datetime
    valid_time: datetime
    lead_time_hours: int
    site: ForecastSite
    grid_latitude: float
    grid_longitude: float
    u10_mps: float
    v10_mps: float
    wind_speed_mps: float
    wind_direction_deg: float
    source: str
    fetched_time: datetime
    wind_gust_mps: float | None = None
    model_version: str | None = None
    resolution: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        run_time = utc_datetime(self.run_time)
        valid_time = utc_datetime(self.valid_time)
        fetched_time = utc_datetime(self.fetched_time)
        object.__setattr__(self, "run_time", run_time)
        object.__setattr__(self, "valid_time", valid_time)
        object.__setattr__(self, "fetched_time", fetched_time)
        expected = run_time + timedelta(hours=int(self.lead_time_hours))
        if abs((valid_time - expected).total_seconds()) > 1:
            raise ValueError(
                "valid_time must equal run_time + lead_time_hours: "
                f"{valid_time.isoformat()} != {expected.isoformat()}"
            )

    @classmethod
    def from_uv(
        cls,
        *,
        provider: str,
        model: str,
        run_time: datetime,
        valid_time: datetime,
        lead_time_hours: int,
        site: ForecastSite,
        grid_latitude: float,
        grid_longitude: float,
        u10_mps: float,
        v10_mps: float,
        source: str,
        fetched_time: datetime,
        wind_gust_mps: float | None = None,
        model_version: str | None = None,
        resolution: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ForecastPoint":
        speed, direction = wind_speed_direction(u10_mps, v10_mps)
        return cls(
            provider=provider,
            model=model,
            run_time=run_time,
            valid_time=valid_time,
            lead_time_hours=int(lead_time_hours),
            site=site,
            grid_latitude=float(grid_latitude),
            grid_longitude=float(grid_longitude),
            u10_mps=float(u10_mps),
            v10_mps=float(v10_mps),
            wind_speed_mps=speed,
            wind_direction_deg=direction,
            wind_gust_mps=None if wind_gust_mps is None else float(wind_gust_mps),
            source=source,
            fetched_time=fetched_time,
            model_version=model_version,
            resolution=resolution,
            metadata=metadata or {},
        )


@dataclass(frozen=True)
class ForecastBatch:
    run: ForecastRun
    points: tuple[ForecastPoint, ...]
    artifacts: tuple[Path, ...] = ()


@dataclass(frozen=True)
class ForecastValue:
    """One provider-neutral scalar value at a forecast point.

    This complements the compact wind-oriented ``ForecastPoint``.  It lets a
    collector retrieve an optional field, such as gust, independently from
    the mandatory U/V pair without downloading U/V again after a partial run.
    """

    provider: str
    model: str
    run_time: datetime
    valid_time: datetime
    lead_time_hours: int
    site: ForecastSite
    grid_latitude: float
    grid_longitude: float
    variable: str
    value: float
    unit: str
    source: str
    fetched_time: datetime
    model_version: str | None = None
    resolution: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        run_time = utc_datetime(self.run_time)
        valid_time = utc_datetime(self.valid_time)
        fetched_time = utc_datetime(self.fetched_time)
        object.__setattr__(self, "run_time", run_time)
        object.__setattr__(self, "valid_time", valid_time)
        object.__setattr__(self, "fetched_time", fetched_time)
        object.__setattr__(self, "value", float(self.value))
        expected = run_time + timedelta(hours=int(self.lead_time_hours))
        if abs((valid_time - expected).total_seconds()) > 1:
            raise ValueError(
                "valid_time must equal run_time + lead_time_hours: "
                f"{valid_time.isoformat()} != {expected.isoformat()}"
            )
        if not self.variable or not self.unit:
            raise ValueError("variable and unit must be non-empty")


@dataclass(frozen=True)
class ForecastValueBatch:
    run: ForecastRun
    values: tuple[ForecastValue, ...]
    artifacts: tuple[Path, ...] = ()


class ForecastProvider(Protocol):
    """The intentionally small contract needed by generic orchestration."""

    provider_name: str
    model_name: str

    def latest_run(self, required_lead_hours: Sequence[int]) -> ForecastRun:
        ...

    def fetch(
        self,
        run: ForecastRun,
        site: ForecastSite,
        lead_hours: Sequence[int],
        work_dir: Path,
    ) -> ForecastBatch:
        ...


def fetch_latest_forecasts(
    provider: ForecastProvider,
    site: ForecastSite,
    lead_hours: Sequence[int],
    work_dir: Path,
) -> ForecastBatch:
    """Detect the newest complete run, fetch it, and validate the batch boundary."""
    requested = tuple(sorted({int(step) for step in lead_hours}))
    if not requested or requested[0] < 0:
        raise ValueError("lead_hours must contain one or more non-negative integer hours")

    work_dir.mkdir(parents=True, exist_ok=True)
    run = provider.latest_run(requested)
    batch = provider.fetch(run, site, requested, work_dir)
    if batch.run != run:
        raise ValueError("provider returned a batch for a different run")

    actual = {point.lead_time_hours for point in batch.points}
    missing = set(requested) - actual
    if missing:
        raise ValueError(f"provider omitted requested lead hours: {sorted(missing)}")
    for point in batch.points:
        if point.provider != run.provider or point.model != run.model or point.run_time != run.run_time:
            raise ValueError("provider returned a point with inconsistent run identity")
    return batch
