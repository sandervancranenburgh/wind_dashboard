"""Configuration boundary for the isolated ECMWF shadow experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from next_day_wind_model.forecast_provider import ForecastSite


DEFAULT_CONFIG_PATH = Path("config/ecmwf_shadow_experiment.json")


def _utc_datetime(value: str) -> datetime:
    text = str(value).strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError("experiment_end_utc must include a timezone")
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class ShadowSiteConfig:
    site_id: str
    display_name: str
    latitude: float
    longitude: float
    observation_site: str | None = None
    timezone_name: str = "Europe/Amsterdam"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ShadowSiteConfig":
        site = cls(
            site_id=str(value["site_id"]).strip(),
            display_name=str(value["display_name"]).strip(),
            latitude=float(value["latitude"]),
            longitude=float(value["longitude"]),
            observation_site=(
                str(value["observation_site"]).strip()
                if value.get("observation_site") is not None
                else None
            ),
            timezone_name=str(value.get("timezone", "Europe/Amsterdam")).strip(),
        )
        if not site.site_id or not site.display_name:
            raise ValueError("site_id and display_name must not be empty")
        if not -90.0 <= site.latitude <= 90.0:
            raise ValueError(f"invalid latitude for {site.site_id}: {site.latitude}")
        if not -180.0 <= site.longitude <= 180.0:
            raise ValueError(f"invalid longitude for {site.site_id}: {site.longitude}")
        return site

    def forecast_site(self) -> ForecastSite:
        return ForecastSite(self.site_id, self.latitude, self.longitude)


@dataclass(frozen=True)
class ShadowExperimentConfig:
    experiment_end_utc: datetime | None
    horizon_hours: int
    max_completed_runs_per_site: int | None
    sites: tuple[ShadowSiteConfig, ...]
    comparison_site_id: str
    comparison_timezone: str
    comparison_hour_local: int
    availability_delay_hours: float
    open_data_source: str

    @property
    def forecast_sites(self) -> tuple[ForecastSite, ...]:
        return tuple(site.forecast_site() for site in self.sites)

    def site(self, site_id: str) -> ShadowSiteConfig:
        for site in self.sites:
            if site.site_id == site_id:
                return site
        raise KeyError(f"site is not configured: {site_id}")

    def expired(self, now: datetime) -> bool:
        if now.tzinfo is None:
            raise ValueError("now must include a timezone")
        if self.experiment_end_utc is None:
            return False
        return now.astimezone(timezone.utc) >= self.experiment_end_utc


def _positive_optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive or null")
    return parsed


def config_from_mapping(value: Mapping[str, Any]) -> ShadowExperimentConfig:
    sites = tuple(ShadowSiteConfig.from_mapping(item) for item in value.get("sites", ()))
    if not sites:
        raise ValueError("at least one shadow site must be configured")
    site_ids = [site.site_id for site in sites]
    if len(site_ids) != len(set(site_ids)):
        raise ValueError("shadow site identifiers must be unique")
    horizon = int(value.get("horizon_hours", 120))
    if horizon < 0 or horizon > 360:
        raise ValueError("horizon_hours must be between 0 and 360")
    comparison = value.get("comparison", {})
    comparison_site_id = str(comparison.get("site_id", site_ids[0]))
    if comparison_site_id not in set(site_ids):
        raise ValueError("comparison.site_id must refer to a configured site")
    hour = int(comparison.get("snapshot_hour_local", 9))
    if not 0 <= hour <= 23:
        raise ValueError("comparison.snapshot_hour_local must be between 0 and 23")
    delay = float(value.get("availability_delay_hours", 8.0))
    if delay < 0:
        raise ValueError("availability_delay_hours must not be negative")
    source = str(value.get("open_data_source", "ecmwf"))
    if source not in {"ecmwf", "aws", "google", "azure"}:
        raise ValueError("open_data_source must be ecmwf, aws, google, or azure")
    end_value = value.get("experiment_end_utc")
    return ShadowExperimentConfig(
        experiment_end_utc=(
            None if end_value is None else _utc_datetime(str(end_value))
        ),
        horizon_hours=horizon,
        max_completed_runs_per_site=_positive_optional_int(
            value.get("max_completed_runs_per_site"), "max_completed_runs_per_site"
        ),
        sites=sites,
        comparison_site_id=comparison_site_id,
        comparison_timezone=str(comparison.get("timezone", "Europe/Amsterdam")),
        comparison_hour_local=hour,
        availability_delay_hours=delay,
        open_data_source=source,
    )


def load_shadow_config(path: Path | str = DEFAULT_CONFIG_PATH) -> ShadowExperimentConfig:
    config_path = Path(path).expanduser()
    with config_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("shadow configuration root must be an object")
    return config_from_mapping(value)


def select_sites(
    config: ShadowExperimentConfig,
    requested_site_ids: Sequence[str] | None,
) -> tuple[ForecastSite, ...]:
    if not requested_site_ids:
        return config.forecast_sites
    requested = tuple(dict.fromkeys(str(site_id) for site_id in requested_site_ids))
    unknown = sorted(set(requested) - {site.site_id for site in config.sites})
    if unknown:
        raise ValueError(f"unconfigured site(s): {', '.join(unknown)}")
    return tuple(config.site(site_id).forecast_site() for site_id in requested)
