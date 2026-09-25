"""Weather classification, source selection, solar state, and icon metadata.

The classifier follows Open-Meteo's published WMO weather-code thresholds.  It
is intentionally pure: callers decide which meteorological source/vintage is
valid and pass one hour's values here.
"""

from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ICON_DIR = Path(__file__).with_name("weather_icons")
SITE_COORDINATES = {
    "valkenburgsemeer": (52.168, 4.437),
    "oostvoorne": (51.930, 4.050),
}

WEATHER_DESCRIPTIONS = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

_DAY_NIGHT_ASSETS = {
    "clear": ("clear-day.png", "clear-night.png"),
    "mostly-clear": ("mostly-clear-day.png", "mostly-clear-night.png"),
    "partly-cloudy": ("partly-cloudy-day.png", "partly-cloudy-night.png"),
}
_STATIC_ASSETS = {
    "overcast": "overcast.png",
    "fog": "fog.png",
    "drizzle": "drizzle.png",
    "rain": "rain.png",
    "sleet": "sleet.png",
    "snow": "snow.png",
    "thunderstorms": "thunderstorms.png",
}

WMO_CATEGORY = {
    0: "clear", 1: "mostly-clear", 2: "partly-cloudy", 3: "overcast",
    45: "fog", 48: "fog",
    51: "drizzle", 53: "drizzle", 55: "drizzle",
    56: "sleet", 57: "sleet", 66: "sleet", 67: "sleet",
    61: "rain", 63: "rain", 65: "rain",
    80: "rain", 81: "rain", 82: "rain",
    71: "snow", 73: "snow", 75: "snow", 77: "snow",
    85: "snow", 86: "snow",
    95: "thunderstorms", 96: "thunderstorms", 99: "thunderstorms",
}


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _severity(value: float, thresholds: Iterable[float], codes: Iterable[int]) -> int:
    threshold_values = tuple(thresholds)
    code_values = tuple(codes)
    for threshold, code in zip(threshold_values, code_values):
        if value < threshold:
            return code
    return code_values[-1]


def derive_weather_code(
    *,
    cloud_cover_pct: Any = None,
    precipitation_mm: Any = None,
    rain_mm: Any = None,
    snow_water_equivalent_mm: Any = None,
    snowfall_cm: Any = None,
    visibility_m: Any = None,
    thunderstorm: bool | None = None,
    hail: bool | None = None,
    heavy_hail: bool | None = None,
    freezing_precipitation: bool | None = None,
    convective_precipitation: bool | None = None,
    snow_showers: bool | None = None,
) -> int | None:
    """Return one WMO code using Open-Meteo thresholds and precedence.

    The optional thunder/freezing/convective switches are explicit inputs.  In
    particular, freezing precipitation and thunderstorms are never inferred
    from temperature, humidity, wind, or gusts.
    """
    precip = _finite(precipitation_mm)
    if precip is None:
        precip = _finite(rain_mm)
    precip = None if precip is None else max(0.0, precip)

    snow_cm = _finite(snowfall_cm)
    if snow_cm is None:
        snow_we = _finite(snow_water_equivalent_mm)
        snow_cm = None if snow_we is None else max(0.0, snow_we) * 0.7
    else:
        snow_cm = max(0.0, snow_cm)

    if thunderstorm:
        return 99 if heavy_hail else (96 if hail else 95)
    if freezing_precipitation and precip is not None and precip >= 0.01:
        return 56 if precip < 1.3 else (66 if precip < 7.6 else 67)
    if convective_precipitation:
        if snow_showers or (snow_cm is not None and snow_cm >= 0.01):
            return 85 if (snow_cm or 0.0) < 0.8 else 86
        if precip is not None and precip >= 0.01:
            return 80 if precip < 1.3 else (81 if precip < 7.6 else 82)
    if snow_cm is not None and snow_cm >= 0.01:
        return 71 if snow_cm < 0.2 else (73 if snow_cm < 0.8 else 75)
    if precip is not None and precip >= 0.01:
        if precip < 0.5:
            return 51
        if precip < 1.0:
            return 53
        if precip < 1.3:
            return 55
        if precip < 2.5:
            return 61
        if precip < 7.6:
            return 63
        return 65
    visibility = _finite(visibility_m)
    if visibility is not None and visibility <= 1000.0:
        return 45
    cloud = _finite(cloud_cover_pct)
    if cloud is None:
        return None
    if cloud < 20.0:
        return 0
    if cloud < 50.0:
        return 1
    if cloud < 80.0:
        return 2
    return 3


def hourly_accumulation_increments(
    accumulations: Iterable[Any], *, material_reset_mm: float = 0.01
) -> np.ndarray:
    """Convert a run's accumulated values to hourly increments.

    Small floating-point reversals are clamped to zero. A material reset is
    missing, so the extractor never silently turns a reset into precipitation.
    """
    values = np.asarray(list(accumulations), dtype=float)
    output = np.full(values.shape, np.nan, dtype=float)
    previous: float | None = None
    for index, value in enumerate(values):
        if not math.isfinite(float(value)):
            previous = None
            continue
        value = float(value)
        if previous is None and index == 0:
            output[index] = max(0.0, value)
        elif previous is not None:
            delta = value - previous
            if delta >= 0.0:
                output[index] = delta
            elif delta >= -abs(float(material_reset_mm)):
                output[index] = 0.0
        previous = value
    return output


def solar_elevation_degrees(valid_time: Any, latitude: float, longitude: float) -> float:
    """Approximate apparent solar elevation using NOAA's dependency-free formula."""
    instant = pd.Timestamp(valid_time)
    if instant.tzinfo is None:
        instant = instant.tz_localize("UTC")
    instant = instant.tz_convert("UTC")
    day = instant.dayofyear
    fractional_hour = instant.hour + instant.minute / 60.0 + instant.second / 3600.0
    gamma = 2.0 * math.pi / 365.0 * (day - 1 + (fractional_hour - 12.0) / 24.0)
    equation = 229.18 * (
        0.000075 + 0.001868 * math.cos(gamma) - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma) - 0.040849 * math.sin(2 * gamma)
    )
    declination = (
        0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma) + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma) + 0.00148 * math.sin(3 * gamma)
    )
    minutes = fractional_hour * 60.0
    true_solar_minutes = (minutes + equation + 4.0 * float(longitude)) % 1440.0
    hour_angle = true_solar_minutes / 4.0 - 180.0
    lat_rad = math.radians(float(latitude))
    hour_rad = math.radians(hour_angle)
    cos_zenith = (
        math.sin(lat_rad) * math.sin(declination)
        + math.cos(lat_rad) * math.cos(declination) * math.cos(hour_rad)
    )
    zenith = math.degrees(math.acos(max(-1.0, min(1.0, cos_zenith))))
    return 90.0 - zenith


def is_daylight(valid_time: Any, latitude: float, longitude: float) -> bool:
    return solar_elevation_degrees(valid_time, latitude, longitude) >= -0.833


def weather_description(code: Any) -> str | None:
    try:
        return WEATHER_DESCRIPTIONS.get(int(code))
    except (TypeError, ValueError):
        return None


def weather_icon_path(code: Any, daylight: Any) -> Path | None:
    try:
        category = WMO_CATEGORY.get(int(code))
    except (TypeError, ValueError):
        return None
    if category is None:
        return None
    if category in _DAY_NIGHT_ASSETS:
        name = _DAY_NIGHT_ASSETS[category][0 if bool(daylight) else 1]
    else:
        name = _STATIC_ASSETS[category]
    return ICON_DIR / name


def conventional_degree_label(value: Any) -> str:
    number = _finite(value)
    if number is None:
        return "—"
    rounded = math.floor(number + 0.5) if number >= 0.0 else math.ceil(number - 0.5)
    return f"{rounded}°"


def derive_windsurfice_weather(payload: dict[str, Any]) -> int | None:
    """Coarse classifier for fields present in the Windsurfice payload."""
    lower = {str(key).lower(): value for key, value in payload.items()}
    return derive_weather_code(
        cloud_cover_pct=lower.get("clouds", lower.get("cloud_cover")),
        precipitation_mm=lower.get("rain", lower.get("precipitation")),
    )


def build_weather_timeline(
    db_path: Path,
    *,
    site: str,
    target_times: Iterable[Any],
    fallback_temperature_c: Iterable[Any],
    fallback_weather_code: Iterable[Any],
    available_at: Any,
    latitude: float | None = None,
    longitude: float | None = None,
) -> pd.DataFrame:
    """Select a complete P1 run, with atomic per-hour Windsurfice fallback.

    P1 eligibility is based on its persisted fetch time and exact valid-hour
    coverage.  Temperature and code are accepted only as a pair from a single
    source; there is no interpolation or mixing between providers.
    """
    targets = pd.DatetimeIndex(pd.to_datetime(list(target_times), utc=True))
    fallback_t = np.asarray(list(fallback_temperature_c), dtype=float)
    fallback_c = np.asarray(list(fallback_weather_code), dtype=float)
    if len(targets) != len(fallback_t) or len(targets) != len(fallback_c):
        raise ValueError("Weather fallback arrays must match target_times.")
    chosen_t = fallback_t.copy()
    chosen_c = fallback_c.copy()
    chosen_source = np.where(
        np.isfinite(chosen_t) & np.isfinite(chosen_c), "windsurfice", None
    ).astype(object)
    invalid_fallback = ~(np.isfinite(chosen_t) & np.isfinite(chosen_c))
    chosen_t[invalid_fallback] = np.nan
    chosen_c[invalid_fallback] = np.nan
    chosen_run = np.full(len(targets), None, dtype=object)

    diagnostic_columns = [
        "total_cloud_cover_pct", "low_cloud_cover_pct",
        "medium_cloud_cover_pct", "high_cloud_cover_pct",
        "total_precip_hourly_mm", "snowfall_hourly_cm", "visibility_m",
    ]
    diagnostics = {name: np.full(len(targets), np.nan) for name in diagnostic_columns}
    if len(targets) and Path(db_path).exists():
        conn = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True)
        try:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='harmonie_knmi_features'"
            ).fetchone()
            columns = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(harmonie_knmi_features)").fetchall()
            } if exists else set()
            required = {"site", "run_ts", "fetched_ts", "target_ts", "temperature_2m_c", "weather_code"}
            if required.issubset(columns):
                selected_diagnostics = [name for name in diagnostic_columns if name in columns]
                rows = pd.read_sql_query(
                    "SELECT run_ts, fetched_ts, target_ts, temperature_2m_c, weather_code"
                    + "".join(f", {name}" for name in selected_diagnostics)
                    + " FROM harmonie_knmi_features WHERE site = ?",
                    conn,
                    params=(site,),
                )
            else:
                rows = pd.DataFrame()
        finally:
            conn.close()
        if not rows.empty:
            rows["target_dt"] = pd.to_datetime(rows["target_ts"], utc=True, errors="coerce")
            rows["run_dt"] = pd.to_datetime(rows["run_ts"], utc=True, errors="coerce")
            rows["fetched_dt"] = pd.to_datetime(rows["fetched_ts"], utc=True, errors="coerce")
            cutoff = pd.Timestamp(available_at)
            cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
            rows = rows[
                rows["target_dt"].isin(targets)
                & rows["fetched_dt"].le(cutoff)
            ].copy()
            selected = pd.DataFrame(index=targets, columns=rows.columns)
            elapsed_targets = targets[targets <= cutoff]
            for target in elapsed_targets:
                candidates = rows[rows["target_dt"].eq(target)].sort_values(
                    ["run_dt", "fetched_dt"]
                )
                if not candidates.empty:
                    selected.loc[target] = candidates.iloc[-1]
            future_targets = targets[targets > cutoff]
            if len(future_targets):
                for _, group in rows.sort_values(
                    ["run_dt", "fetched_dt"], ascending=False
                ).groupby("run_dt", sort=False):
                    latest = group.sort_values("fetched_dt").drop_duplicates(
                        "target_dt", keep="last"
                    )
                    if future_targets.isin(latest["target_dt"]).all():
                        selected.loc[future_targets] = latest.set_index("target_dt").reindex(
                            future_targets
                        )
                        break
            if selected["target_ts"].notna().any():
                p1_t = pd.to_numeric(selected["temperature_2m_c"], errors="coerce").to_numpy(float)
                p1_c = pd.to_numeric(selected["weather_code"], errors="coerce").to_numpy(float)
                use_p1 = np.isfinite(p1_t) & np.isfinite(p1_c)
                chosen_t[use_p1] = p1_t[use_p1]
                chosen_c[use_p1] = p1_c[use_p1]
                chosen_source[use_p1] = "knmi_p1"
                p1_runs = pd.to_datetime(selected["run_ts"], utc=True, errors="coerce")
                chosen_run[use_p1] = [
                    value.isoformat() if not pd.isna(value) else None
                    for value in p1_runs[use_p1]
                ]
                for name in diagnostics:
                    if name in selected:
                        diagnostics[name] = pd.to_numeric(selected[name], errors="coerce").to_numpy(float)

    if latitude is None or longitude is None:
        latitude, longitude = SITE_COORDINATES.get(site, (52.168, 4.437))
    return pd.DataFrame(
        {
            "forecast_temperature_c": chosen_t,
            "weather_code": pd.array(chosen_c, dtype="Int64"),
            "weather_run_utc": chosen_run,
            "weather_source": chosen_source,
            "is_daylight": [is_daylight(value, latitude, longitude) for value in targets],
            **diagnostics,
        },
        index=targets,
    )


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
