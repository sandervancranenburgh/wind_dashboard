"""Canonical site configuration shared by collectors, models, and web views."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SITE_REGISTRY_PATH = REPO_ROOT / "config" / "sites.json"
SOURCE_NAMES = ("windsurfice", "knmi_p1", "ecmwf", "superlocal_model")


def _required_text(value: Mapping[str, Any], key: str, context: str) -> str:
    text = str(value.get(key, "")).strip()
    if not text:
        raise ValueError(f"{context}.{key} must not be empty")
    return text


def _coordinate(value: Mapping[str, Any], key: str, low: float, high: float, context: str) -> float:
    try:
        parsed = float(value[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{context}.{key} must be numeric") from exc
    if not low <= parsed <= high:
        raise ValueError(f"{context}.{key} must be between {low:g} and {high:g}")
    return parsed


@dataclass(frozen=True)
class WindsurficeSite:
    enabled: bool
    observation_station: str
    forecast_latitude: float
    forecast_longitude: float
    referer_slug: str


@dataclass(frozen=True)
class PointSource:
    enabled: bool
    latitude: float
    longitude: float


@dataclass(frozen=True)
class EcmwfSite(PointSource):
    observation_site: str


@dataclass(frozen=True)
class SuperlocalModelSite:
    enabled: bool
    publish: bool


@dataclass(frozen=True)
class SiteDefinition:
    site_id: str
    display_name: str
    rider_spot_value: str
    aliases: tuple[str, ...]
    timezone: str
    windsurfice: WindsurficeSite
    knmi_p1: PointSource
    ecmwf: EcmwfSite
    superlocal_model: SuperlocalModelSite

    def source_enabled(self, source: str) -> bool:
        if source not in SOURCE_NAMES:
            raise ValueError(f"unknown site source: {source}")
        return bool(getattr(self, source).enabled)


@dataclass(frozen=True)
class SiteRegistry:
    schema_version: int
    sites: tuple[SiteDefinition, ...]

    @property
    def site_ids(self) -> tuple[str, ...]:
        return tuple(site.site_id for site in self.sites)

    def site(self, site_id: str) -> SiteDefinition:
        canonical = self.canonical_site_id(site_id)
        for site in self.sites:
            if site.site_id == canonical:
                return site
        raise KeyError(f"unknown site: {site_id}")

    def canonical_site_id(self, value: str) -> str:
        candidate = str(value or "").strip()
        if candidate in self.site_ids:
            return candidate
        folded = candidate.casefold()
        for site in self.sites:
            names = (site.display_name, site.rider_spot_value, *site.aliases)
            if folded in {name.casefold() for name in names}:
                return site.site_id
        raise KeyError(f"unknown site or alias: {value}")

    def enabled_sites(self, source: str) -> tuple[SiteDefinition, ...]:
        return tuple(site for site in self.sites if site.source_enabled(source))

    def display_name(self, site_id: str) -> str:
        return self.site(site_id).display_name

    def spot_to_site(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for site in self.sites:
            for name in (site.rider_spot_value, site.display_name, *site.aliases):
                result[name] = site.site_id
        return result

    def rider_spot_values(self) -> tuple[str, ...]:
        return tuple(site.rider_spot_value for site in self.sites)


def _point_source(value: Mapping[str, Any], context: str) -> PointSource:
    return PointSource(
        enabled=bool(value.get("enabled", False)),
        latitude=_coordinate(value, "latitude", -90.0, 90.0, context),
        longitude=_coordinate(value, "longitude", -180.0, 180.0, context),
    )


def _site_from_mapping(value: Mapping[str, Any], index: int) -> SiteDefinition:
    context = f"sites[{index}]"
    site_id = _required_text(value, "site_id", context)
    if site_id != site_id.lower() or not site_id.replace("_", "").isalnum():
        raise ValueError(f"{context}.site_id must be lowercase letters, digits, or underscores")
    aliases_raw = value.get("aliases", ())
    if not isinstance(aliases_raw, list):
        raise ValueError(f"{context}.aliases must be an array")
    aliases = tuple(str(alias).strip() for alias in aliases_raw if str(alias).strip())

    wind_raw = value.get("windsurfice")
    knmi_raw = value.get("knmi_p1")
    ecmwf_raw = value.get("ecmwf")
    model_raw = value.get("superlocal_model")
    for name, source in (
        ("windsurfice", wind_raw),
        ("knmi_p1", knmi_raw),
        ("ecmwf", ecmwf_raw),
        ("superlocal_model", model_raw),
    ):
        if not isinstance(source, Mapping):
            raise ValueError(f"{context}.{name} must be an object")

    wind_context = f"{context}.windsurfice"
    windsurfice = WindsurficeSite(
        enabled=bool(wind_raw.get("enabled", False)),
        observation_station=_required_text(wind_raw, "observation_station", wind_context),
        forecast_latitude=_coordinate(wind_raw, "forecast_latitude", -90.0, 90.0, wind_context),
        forecast_longitude=_coordinate(wind_raw, "forecast_longitude", -180.0, 180.0, wind_context),
        referer_slug=_required_text(wind_raw, "referer_slug", wind_context),
    )
    knmi = _point_source(knmi_raw, f"{context}.knmi_p1")
    ecmwf_point = _point_source(ecmwf_raw, f"{context}.ecmwf")
    ecmwf = EcmwfSite(
        enabled=ecmwf_point.enabled,
        latitude=ecmwf_point.latitude,
        longitude=ecmwf_point.longitude,
        observation_site=_required_text(ecmwf_raw, "observation_site", f"{context}.ecmwf"),
    )
    model = SuperlocalModelSite(
        enabled=bool(model_raw.get("enabled", False)),
        publish=bool(model_raw.get("publish", False)),
    )
    if model.publish and not model.enabled:
        raise ValueError(f"{context}.superlocal_model.publish requires enabled=true")
    if model.enabled and not windsurfice.enabled:
        raise ValueError(f"{context}.superlocal_model requires windsurfice.enabled=true")

    return SiteDefinition(
        site_id=site_id,
        display_name=_required_text(value, "display_name", context),
        rider_spot_value=_required_text(value, "rider_spot_value", context),
        aliases=aliases,
        timezone=_required_text(value, "timezone", context),
        windsurfice=windsurfice,
        knmi_p1=knmi,
        ecmwf=ecmwf,
        superlocal_model=model,
    )


def registry_from_mapping(value: Mapping[str, Any]) -> SiteRegistry:
    try:
        schema_version = int(value["schema_version"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("schema_version must be an integer") from exc
    if schema_version != 1:
        raise ValueError(f"unsupported site registry schema_version: {schema_version}")
    sites_raw = value.get("sites")
    if not isinstance(sites_raw, list) or not sites_raw:
        raise ValueError("sites must be a non-empty array")
    sites = tuple(_site_from_mapping(item, index) for index, item in enumerate(sites_raw))
    site_ids = [site.site_id for site in sites]
    if len(site_ids) != len(set(site_ids)):
        raise ValueError("site_id values must be unique")

    aliases: dict[str, str] = {}
    for site in sites:
        for name in (site.display_name, site.rider_spot_value, *site.aliases):
            folded = name.casefold()
            owner = aliases.setdefault(folded, site.site_id)
            if owner != site.site_id:
                raise ValueError(f"site alias {name!r} is shared by {owner!r} and {site.site_id!r}")
    return SiteRegistry(schema_version=schema_version, sites=sites)


@lru_cache(maxsize=None)
def load_site_registry(path: Path | str = DEFAULT_SITE_REGISTRY_PATH) -> SiteRegistry:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError("site registry root must be an object")
    return registry_from_mapping(value)


def all_site_ids() -> tuple[str, ...]:
    return load_site_registry().site_ids


def display_name(site_id: str) -> str:
    return load_site_registry().display_name(site_id)


def spot_to_site_map() -> dict[str, str]:
    return load_site_registry().spot_to_site()


def rider_spot_values() -> tuple[str, ...]:
    return load_site_registry().rider_spot_values()


def enabled_sites(source: str) -> tuple[SiteDefinition, ...]:
    return load_site_registry().enabled_sites(source)


def validate_site_ids(values: Iterable[str]) -> tuple[str, ...]:
    registry = load_site_registry()
    return tuple(registry.canonical_site_id(value) for value in values)
