#!/usr/bin/env python3
"""Analyze a wingfoil GPX/KML/FIT activity and emit rider-portal artifacts.

The pipeline is intentionally dependency-light:
  GPX/KML/FIT -> canonical dataframe -> samples -> runs/falls -> website assets

Speed is derived from GPS distance and timestamps because GPS exports often do
not include a speed extension.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import pandas as pd
from fitparse import FitFile


ANALYSIS_VERSION = "0.1.0"
GPX_NS = {"gpx": "http://www.topografix.com/GPX/1/1"}
KML_NS = {
    "kml": "http://www.opengis.net/kml/2.2",
    "gx": "http://www.google.com/kml/ext/2.2",
}
MPS_TO_KMH = 3.6
MPS_TO_KNOTS = 1.9438444924406
SEMICIRCLE_DEGREES = 180 / 2**31
CANONICAL_COLUMNS = [
    "timestamp",
    "lat",
    "lon",
    "altitude_m",
    "speed_mps",
    "distance_m",
    "heart_rate_bpm",
]
GENERATED_ARTIFACTS = [
    "summary.json",
    "runs.csv",
    "map.svg",
    "map.html",
    "run_distance_distribution.svg",
    "run_speed_distribution.svg",
    "run_wind_angle_distribution.svg",
    "run_speed.svg",
]
ARTIFACTS = {
    "summary_json": "summary.json",
    "runs_csv": "runs.csv",
    "map_svg": "map.svg",
    "map_html": "map.html",
    "run_distance_distribution_svg": "run_distance_distribution.svg",
    "run_speed_distribution_svg": "run_speed_distribution.svg",
    "run_wind_angle_distribution_svg": "run_wind_angle_distribution.svg",
    "run_speed_profile_svg": "run_speed.svg",
}
SUPPORTED_EXTENSIONS = {".gpx", ".kml", ".fit"}

DEFAULT_SETTINGS = {
    "run_speed_threshold_mps": 4.5,
    "min_run_duration_s": 7.0,
    "max_run_gap_s": 2.0,
    "merge_runs_without_stop": True,
    "run_stop_threshold_mps": 1.0,
    "fall_speed_threshold_mps": 1.0,
    "fall_window_s": 10.0,
    "min_fall_gap_s": 15.0,
    "water_speed_threshold_mps": 1.0,
    "smooth_window": 5,
    "min_count_for_histogram": 12,
}

SPEED_COLORMAP = [
    (0.00, "#440154"),
    (0.20, "#414487"),
    (0.40, "#2a788e"),
    (0.60, "#22a884"),
    (0.80, "#7ad151"),
    (1.00, "#fde725"),
]


@dataclass(frozen=True)
class Sample:
    index: int
    lat: float
    lon: float
    ele_m: float | None
    time: datetime
    dt_s: float
    segment_distance_m: float
    speed_mps: float
    smooth_speed_mps: float
    in_run: bool = False
    heart_rate_bpm: int | None = None


@dataclass(frozen=True)
class Run:
    run_id: int
    start_index: int
    end_index: int
    start_time: datetime
    end_time: datetime
    duration_s: float
    distance_m: float
    mean_speed_mps: float
    median_speed_mps: float
    max_speed_mps: float
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    mean_bearing_deg: float | None = None
    angle_to_wind_deg: float | None = None
    wind_angle_class: str | None = None
    end_reason: str = "unknown"
    merged_raw_run_count: int = 1


@dataclass(frozen=True)
class Fall:
    fall_id: int
    index: int
    time: datetime
    lat: float
    lon: float
    speed_before_mps: float
    speed_after_mps: float


@dataclass(frozen=True)
class WindContext:
    wind_direction_deg: float | None = None
    wind_speed_kts: float | None = None
    spot_name: str | None = None
    wind_source: str | None = None


class AnalysisError(Exception):
    """Raised when an activity file cannot be analyzed."""


@dataclass(frozen=True)
class AnalysisConfig:
    run_speed_threshold_mps: float
    min_run_duration_s: float
    max_run_gap_s: float
    merge_runs_without_stop: bool
    run_stop_threshold_mps: float
    fall_speed_threshold_mps: float
    fall_window_s: float
    min_fall_gap_s: float
    water_speed_threshold_mps: float
    smooth_window: int
    min_count_for_histogram: int
    fall_requires_previous_detected_run: bool = True


@dataclass(frozen=True)
class AnalysisResult:
    source_activity: Path
    samples: list[Sample]
    runs: list[Run]
    falls: list[Fall]
    water_time_s: float
    wind_context: WindContext
    config: AnalysisConfig
    warnings: list[str]


def default_analysis_config(**overrides: object) -> AnalysisConfig:
    values = DEFAULT_SETTINGS | overrides
    return AnalysisConfig(
        run_speed_threshold_mps=float(values["run_speed_threshold_mps"]),
        min_run_duration_s=float(values["min_run_duration_s"]),
        max_run_gap_s=float(values["max_run_gap_s"]),
        merge_runs_without_stop=bool(values["merge_runs_without_stop"]),
        run_stop_threshold_mps=float(values["run_stop_threshold_mps"]),
        fall_speed_threshold_mps=float(values["fall_speed_threshold_mps"]),
        fall_window_s=float(values["fall_window_s"]),
        min_fall_gap_s=float(values["min_fall_gap_s"]),
        water_speed_threshold_mps=float(values["water_speed_threshold_mps"]),
        smooth_window=int(values["smooth_window"]),
        min_count_for_histogram=int(values["min_count_for_histogram"]),
    )


def config_to_dict(config: AnalysisConfig) -> dict[str, object]:
    return {
        "run_speed_threshold_mps": config.run_speed_threshold_mps,
        "min_run_duration_s": config.min_run_duration_s,
        "max_run_gap_s": config.max_run_gap_s,
        "merge_runs_without_stop": config.merge_runs_without_stop,
        "run_stop_threshold_mps": config.run_stop_threshold_mps,
        "fall_speed_threshold_mps": config.fall_speed_threshold_mps,
        "fall_window_s": config.fall_window_s,
        "min_fall_gap_s": config.min_fall_gap_s,
        "water_speed_threshold_mps": config.water_speed_threshold_mps,
        "smooth_window": config.smooth_window,
        "min_count_for_histogram": config.min_count_for_histogram,
        "fall_requires_previous_detected_run": config.fall_requires_previous_detected_run,
    }


def parse_time(value: str) -> datetime:
    value = value.strip()
    if "." in value:
        prefix, suffix = value.split(".", 1)
        digits = "".join(ch for ch in suffix if ch.isdigit())
        tail = suffix[len(digits) :]
        value = f"{prefix}.{digits[:6]}{tail}"
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius_m * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def normalize_degrees(value: float) -> float:
    return value % 360


def angle_difference_degrees(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


def cardinal_direction(degrees: float | None) -> str | None:
    if degrees is None:
        return None
    labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return labels[round(normalize_degrees(degrees) / 45) % 8]


def classify_wind_angle(angle_deg: float | None) -> str | None:
    if angle_deg is None:
        return None
    if angle_deg <= 45:
        return "upwind"
    if angle_deg >= 135:
        return "downwind"
    return "crosswind"


def semicircles_to_degrees(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) * SEMICIRCLE_DEGREES


def first_present(*values: object) -> object | None:
    for value in values:
        if value is not None:
            return value
    return None


def canonicalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in CANONICAL_COLUMNS:
        if column not in df.columns:
            df[column] = None
    df = df[CANONICAL_COLUMNS]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for column in ["lat", "lon", "altitude_m", "speed_mps", "distance_m", "heart_rate_bpm"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["timestamp", "lat", "lon"]).sort_values("timestamp").reset_index(drop=True)
    return df


def warn_if_timestamps_irregular(df: pd.DataFrame, warnings: list[str]) -> None:
    if len(df) < 3:
        return
    deltas = df["timestamp"].diff().dt.total_seconds().iloc[1:]
    positive_deltas = deltas[deltas > 0]
    non_positive_count = int((deltas <= 0).sum())
    if positive_deltas.empty:
        warnings.append(
            "GPS timestamps were not strictly increasing; speed and distance were reconstructed where needed. "
            "(no positive time deltas between GPS records)"
        )
        return
    median_delta = float(positive_deltas.median())
    tolerance = max(2.0, median_delta * 2.0)
    irregular_mask = positive_deltas.sub(median_delta).abs() > tolerance
    irregular_count = int(irregular_mask.sum())
    if non_positive_count or irregular_count:
        max_delta = float(positive_deltas.max())
        details = (
            f"median interval {median_delta:.2f}s, max interval {max_delta:.2f}s, "
            f"{irregular_count} variable interval(s), {non_positive_count} non-increasing interval(s)"
        )
        warnings.append(
            "GPS timestamps were somewhat irregular; speed and distance were reconstructed where needed. "
            f"({details})"
        )


def dataframe_to_samples(df: pd.DataFrame, smooth_window: int, warnings: list[str]) -> list[Sample]:
    df = canonicalize_dataframe(df)
    warn_if_timestamps_irregular(df, warnings)
    samples: list[Sample] = []
    speeds: list[float] = []

    if df.empty:
        return samples

    provided_speed = df["speed_mps"]
    provided_distance = df["distance_m"]
    speed_reconstructed = provided_speed.isna().any()
    distance_reconstructed = provided_distance.isna().any()

    for index, row in df.iterrows():
        lat = float(row["lat"])
        lon = float(row["lon"])
        ele_m = None if pd.isna(row["altitude_m"]) else float(row["altitude_m"])
        time = row["timestamp"].to_pydatetime()
        heart_rate_bpm = None if pd.isna(row["heart_rate_bpm"]) else int(row["heart_rate_bpm"])

        if index == 0:
            dt_s = 0.0
            distance_m = 0.0
        else:
            previous = df.iloc[index - 1]
            prev_time = previous["timestamp"].to_pydatetime()
            dt_s = max((time - prev_time).total_seconds(), 0.0)
            current_distance = row["distance_m"]
            previous_distance = previous["distance_m"]
            if not pd.isna(current_distance) and not pd.isna(previous_distance):
                distance_m = float(current_distance - previous_distance)
                if distance_m < 0:
                    distance_reconstructed = True
                    distance_m = haversine_m(float(previous["lat"]), float(previous["lon"]), lat, lon)
            else:
                distance_reconstructed = True
                distance_m = haversine_m(float(previous["lat"]), float(previous["lon"]), lat, lon)

        if not pd.isna(row["speed_mps"]):
            speed_mps = float(row["speed_mps"])
        else:
            speed_reconstructed = True
            speed_mps = distance_m / dt_s if dt_s > 0 else 0.0

        speeds.append(speed_mps)
        samples.append(
            Sample(
                index=index,
                lat=lat,
                lon=lon,
                ele_m=ele_m,
                time=time,
                dt_s=dt_s,
                segment_distance_m=distance_m,
                speed_mps=speed_mps,
                smooth_speed_mps=0.0,
                heart_rate_bpm=heart_rate_bpm,
            )
        )

    if speed_reconstructed:
        warnings.append("speed had to be reconstructed from GPS distance and timestamps for records with missing speed")
    if distance_reconstructed:
        warnings.append("distance had to be reconstructed from GPS coordinates for records with missing or unusable distance")

    half_window = max(smooth_window // 2, 0)
    smoothed: list[Sample] = []
    for sample in samples:
        start = max(sample.index - half_window, 0)
        end = min(sample.index + half_window + 1, len(speeds))
        smooth_speed = statistics.median(speeds[start:end]) if end > start else sample.speed_mps
        smoothed.append(
            Sample(
                **{
                    **sample.__dict__,
                    "smooth_speed_mps": smooth_speed,
                }
            )
        )
    return smoothed


def load_gpx_activity(path: Path) -> pd.DataFrame:
    tree = ET.parse(path)
    root = tree.getroot()
    rows = []

    for trkpt in root.findall(".//gpx:trkpt", GPX_NS):
        time_el = trkpt.find("gpx:time", GPX_NS)
        if time_el is None or not time_el.text:
            continue
        ele_el = trkpt.find("gpx:ele", GPX_NS)
        hr_el = trkpt.find(".//{http://www.garmin.com/xmlschemas/TrackPointExtension/v1}hr")
        rows.append(
            {
                "timestamp": parse_time(time_el.text),
                "lat": float(trkpt.attrib["lat"]),
                "lon": float(trkpt.attrib["lon"]),
                "altitude_m": float(ele_el.text) if ele_el is not None and ele_el.text else None,
                "speed_mps": None,
                "distance_m": None,
                "heart_rate_bpm": int(hr_el.text) if hr_el is not None and hr_el.text else None,
            }
        )

    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS)


def load_kml_activity(path: Path) -> pd.DataFrame:
    tree = ET.parse(path)
    root = tree.getroot()
    track = root.find(".//gx:Track", KML_NS)
    if track is None:
        raise ValueError("KML file does not contain a gx:Track with timestamped coordinates")

    times = [parse_time(when.text) for when in track.findall("kml:when", KML_NS) if when.text]
    coords = []
    for coord in track.findall("gx:coord", KML_NS):
        if not coord.text:
            continue
        parts = coord.text.split()
        if len(parts) < 2:
            continue
        lon = float(parts[0])
        lat = float(parts[1])
        ele_m = float(parts[2]) if len(parts) >= 3 else None
        coords.append((lat, lon, ele_m))

    if len(times) != len(coords):
        raise ValueError(f"KML gx:Track has {len(times)} timestamps but {len(coords)} coordinates")

    rows = [
        {
            "timestamp": time,
            "lat": lat,
            "lon": lon,
            "altitude_m": ele_m,
            "speed_mps": None,
            "distance_m": None,
            "heart_rate_bpm": None,
        }
        for (lat, lon, ele_m), time in zip(coords, times)
    ]
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS)


def load_fit_activity(path: Path) -> pd.DataFrame:
    rows = []
    for message in FitFile(str(path)).get_messages("record"):
        record = {field.name: field.value for field in message}
        timestamp = record.get("timestamp")
        lat = semicircles_to_degrees(record.get("position_lat"))
        lon = semicircles_to_degrees(record.get("position_long"))
        if timestamp is None or lat is None or lon is None:
            continue
        rows.append(
            {
                "timestamp": timestamp,
                "lat": lat,
                "lon": lon,
                "altitude_m": first_present(record.get("enhanced_altitude"), record.get("altitude")),
                "speed_mps": first_present(record.get("enhanced_speed"), record.get("speed")),
                "distance_m": record.get("distance"),
                "heart_rate_bpm": record.get("heart_rate"),
            }
        )
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS)


def load_activity(path: Path, smooth_window: int) -> tuple[list[Sample], list[str]]:
    suffix = path.suffix.lower()
    warnings: list[str] = []
    if suffix == ".gpx":
        df = load_gpx_activity(path)
    elif suffix == ".kml":
        df = load_kml_activity(path)
    elif suffix == ".fit":
        df = load_fit_activity(path)
        if df.empty:
            warnings.append("FIT file has no GPS records")
    else:
        raise ValueError(f"Unsupported activity format: {path.suffix}. Expected .gpx, .kml, or .fit")

    return dataframe_to_samples(df, smooth_window, warnings), warnings


def detect_runs(
    samples: list[Sample],
    run_speed_threshold_mps: float,
    min_run_duration_s: float,
    max_gap_s: float,
    merge_runs_without_stop: bool,
    run_stop_threshold_mps: float,
) -> tuple[list[Sample], list[Run]]:
    if not samples:
        return [], []

    candidate_ranges: list[tuple[int, int]] = []
    start: int | None = None
    last_fast: int | None = None

    for sample in samples:
        is_fast = sample.smooth_speed_mps >= run_speed_threshold_mps
        if is_fast:
            if start is None:
                start = sample.index
            last_fast = sample.index
            continue

        if start is not None and last_fast is not None:
            gap_s = (sample.time - samples[last_fast].time).total_seconds()
            if gap_s > max_gap_s:
                candidate_ranges.append((start, last_fast))
                start = None
                last_fast = None

    if start is not None and last_fast is not None:
        candidate_ranges.append((start, last_fast))

    raw_runs: list[Run] = []
    for start_index, end_index in candidate_ranges:
        run = build_run(len(raw_runs) + 1, samples, start_index, end_index, min_run_duration_s)
        if run is not None:
            raw_runs.append(run)

    runs = (
        merge_runs_without_true_stop(samples, raw_runs, min_run_duration_s, run_stop_threshold_mps)
        if merge_runs_without_stop
        else raw_runs
    )

    run_indices: set[int] = set()
    for run in runs:
        run_indices.update(range(run.start_index, run.end_index + 1))

    marked = [
        Sample(
            **{
                **sample.__dict__,
                "in_run": sample.index in run_indices,
            }
        )
        for sample in samples
    ]
    return marked, runs


def build_run(run_id: int, samples: list[Sample], start_index: int, end_index: int, min_run_duration_s: float) -> Run | None:
    start_sample = samples[start_index]
    end_sample = samples[end_index]
    duration_s = (end_sample.time - start_sample.time).total_seconds()
    if duration_s < min_run_duration_s:
        return None

    run_samples = samples[start_index : end_index + 1]
    moving_samples = run_samples[1:] if len(run_samples) > 1 else run_samples
    distance_m = sum(sample.segment_distance_m for sample in moving_samples)
    speeds = [sample.speed_mps for sample in moving_samples if sample.dt_s > 0]
    if not speeds:
        return None

    return Run(
        run_id=run_id,
        start_index=start_index,
        end_index=end_index,
        start_time=start_sample.time,
        end_time=end_sample.time,
        duration_s=duration_s,
        distance_m=distance_m,
        mean_speed_mps=statistics.mean(speeds),
        median_speed_mps=statistics.median(speeds),
        max_speed_mps=max(speeds),
        start_lat=start_sample.lat,
        start_lon=start_sample.lon,
        end_lat=end_sample.lat,
        end_lon=end_sample.lon,
    )


def merge_runs_without_true_stop(
    samples: list[Sample],
    runs: list[Run],
    min_run_duration_s: float,
    run_stop_threshold_mps: float,
) -> list[Run]:
    if len(runs) < 2:
        return runs

    merged_ranges: list[tuple[int, int, int]] = []
    start_index = runs[0].start_index
    end_index = runs[0].end_index
    raw_run_count = 1

    for next_run in runs[1:]:
        gap_samples = samples[end_index + 1 : next_run.start_index]
        has_true_stop = any(sample.smooth_speed_mps < run_stop_threshold_mps for sample in gap_samples)
        if not has_true_stop:
            end_index = next_run.end_index
            raw_run_count += 1
            continue

        merged_ranges.append((start_index, end_index, raw_run_count))
        start_index = next_run.start_index
        end_index = next_run.end_index
        raw_run_count = 1

    merged_ranges.append((start_index, end_index, raw_run_count))

    merged_runs: list[Run] = []
    for start_index, end_index, raw_count in merged_ranges:
        run = build_run(len(merged_runs) + 1, samples, start_index, end_index, min_run_duration_s)
        if run is not None:
            merged_runs.append(Run(**{**run.__dict__, "merged_raw_run_count": raw_count}))
    return merged_runs


def detect_falls(
    samples: list[Sample],
    run_speed_threshold_mps: float,
    fall_speed_threshold_mps: float,
    fall_window_s: float,
    min_fall_gap_s: float,
) -> list[Fall]:
    falls: list[Fall] = []
    last_fall_time: datetime | None = None

    for sample in samples:
        if sample.smooth_speed_mps >= fall_speed_threshold_mps:
            continue

        window_samples = [
            previous
            for previous in samples[: sample.index]
            if 0 < (sample.time - previous.time).total_seconds() <= fall_window_s
        ]
        fast_before = [
            previous.smooth_speed_mps
            for previous in window_samples
            if previous.in_run and previous.smooth_speed_mps >= run_speed_threshold_mps
        ]
        if not fast_before:
            continue

        if last_fall_time is not None and (sample.time - last_fall_time).total_seconds() < min_fall_gap_s:
            continue

        falls.append(
            Fall(
                fall_id=len(falls) + 1,
                index=sample.index,
                time=sample.time,
                lat=sample.lat,
                lon=sample.lon,
                speed_before_mps=max(fast_before),
                speed_after_mps=sample.smooth_speed_mps,
            )
        )
        last_fall_time = sample.time

    return falls


def classify_run_end_reasons(
    samples: list[Sample],
    runs: list[Run],
    falls: list[Fall],
    fall_window_s: float,
    run_stop_threshold_mps: float,
) -> list[Run]:
    classified: list[Run] = []
    for index, run in enumerate(runs):
        end_sample = samples[run.end_index]
        fall_after_run = next(
            (
                fall
                for fall in falls
                if 0 < (fall.time - end_sample.time).total_seconds() <= fall_window_s
            ),
            None,
        )
        if fall_after_run is not None:
            end_reason = "fall"
        elif index == len(runs) - 1:
            end_reason = "session_end"
        else:
            next_run = runs[index + 1]
            gap_samples = samples[run.end_index + 1 : next_run.start_index]
            has_true_stop = any(sample.smooth_speed_mps < run_stop_threshold_mps for sample in gap_samples)
            end_reason = "pause" if has_true_stop else "slow_transition"
        classified.append(Run(**{**run.__dict__, "end_reason": end_reason}))
    return classified


def compute_water_time_s(samples: list[Sample], water_speed_threshold_mps: float) -> float:
    return sum(
        sample.dt_s
        for sample in samples
        if not sample.in_run and sample.smooth_speed_mps < water_speed_threshold_mps
    )


def summarize(values: Iterable[float]) -> dict[str, float | None]:
    items = list(values)
    if not items:
        return {"mean": None, "median": None, "max": None}
    return {
        "mean": statistics.mean(items),
        "median": statistics.median(items),
        "max": max(items),
    }


def fmt_dt(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def format_duration_s(seconds: float) -> str:
    safe = max(int(round(seconds)), 0)
    hours, remainder = divmod(safe, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {remaining_seconds}s"
    return f"{remaining_seconds}s"


def rounded_or_none(value: float | None, digits: int) -> float | None:
    return round(value, digits) if value is not None else None


def run_to_dict(run: Run) -> dict[str, float | int | str | None]:
    return {
        "run_id": run.run_id,
        "start_time": fmt_dt(run.start_time),
        "end_time": fmt_dt(run.end_time),
        "duration_s": round(run.duration_s, 2),
        "distance_m": round(run.distance_m, 2),
        "distance_km": round(run.distance_m / 1000, 3),
        "mean_speed_mps": round(run.mean_speed_mps, 3),
        "mean_speed_kmh": round(run.mean_speed_mps * MPS_TO_KMH, 2),
        "mean_speed_knots": round(run.mean_speed_mps * MPS_TO_KNOTS, 2),
        "median_speed_mps": round(run.median_speed_mps, 3),
        "median_speed_kmh": round(run.median_speed_mps * MPS_TO_KMH, 2),
        "median_speed_knots": round(run.median_speed_mps * MPS_TO_KNOTS, 2),
        "max_speed_mps": round(run.max_speed_mps, 3),
        "max_speed_kmh": round(run.max_speed_mps * MPS_TO_KMH, 2),
        "max_speed_knots": round(run.max_speed_mps * MPS_TO_KNOTS, 2),
        "start_lat": run.start_lat,
        "start_lon": run.start_lon,
        "end_lat": run.end_lat,
        "end_lon": run.end_lon,
        "mean_bearing_deg": rounded_or_none(run.mean_bearing_deg, 1),
        "angle_to_wind_deg": rounded_or_none(run.angle_to_wind_deg, 1),
        "wind_angle_class": run.wind_angle_class,
        "end_reason": run.end_reason,
        "merged_raw_run_count": run.merged_raw_run_count,
    }


def fall_to_dict(fall: Fall) -> dict[str, float | int | str]:
    return {
        "fall_id": fall.fall_id,
        "time": fmt_dt(fall.time),
        "lat": fall.lat,
        "lon": fall.lon,
        "speed_before_mps": round(fall.speed_before_mps, 3),
        "speed_after_mps": round(fall.speed_after_mps, 3),
    }


def write_runs_csv(path: Path, runs: list[Run]) -> None:
    rows = [run_to_dict(run) for run in runs]
    fieldnames = list(rows[0].keys()) if rows else list(run_to_dict(empty_run()).keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def empty_run() -> Run:
    now = datetime.fromtimestamp(0, timezone.utc)
    return Run(0, 0, 0, now, now, 0, 0, 0, 0, 0, 0, 0, 0, 0)


def svg_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def color_for_speed(speed_mps: float, min_speed_mps: float, max_speed_mps: float) -> str:
    if max_speed_mps <= min_speed_mps:
        return SPEED_COLORMAP[-1][1]
    ratio = min(max((speed_mps - min_speed_mps) / (max_speed_mps - min_speed_mps), 0.0), 1.0)
    for index in range(1, len(SPEED_COLORMAP)):
        left_pos, left_color = SPEED_COLORMAP[index - 1]
        right_pos, right_color = SPEED_COLORMAP[index]
        if ratio <= right_pos:
            local = (ratio - left_pos) / max(right_pos - left_pos, 1e-9)
            return interpolate_hex(left_color, right_color, local)
    return SPEED_COLORMAP[-1][1]


def interpolate_hex(left: str, right: str, ratio: float) -> str:
    left_rgb = tuple(int(left[i : i + 2], 16) for i in (1, 3, 5))
    right_rgb = tuple(int(right[i : i + 2], 16) for i in (1, 3, 5))
    rgb = tuple(round(a + (b - a) * ratio) for a, b in zip(left_rgb, right_rgb))
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def bearing_degrees(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def weighted_mean_bearing_deg(samples: list[Sample]) -> float | None:
    x = 0.0
    y = 0.0
    total_weight = 0.0
    for previous, sample in zip(samples, samples[1:]):
        if sample.segment_distance_m <= 0:
            continue
        bearing = math.radians(bearing_degrees(previous.lat, previous.lon, sample.lat, sample.lon))
        x += math.cos(bearing) * sample.segment_distance_m
        y += math.sin(bearing) * sample.segment_distance_m
        total_weight += sample.segment_distance_m
    if total_weight == 0:
        return None
    return normalize_degrees(math.degrees(math.atan2(y, x)))


def build_wind_context(
    wind_json: str | Path | None = None,
    wind_direction_deg: float | None = None,
    wind_speed_kts: float | None = None,
    spot_name: str | None = None,
) -> WindContext:
    payload: dict[str, object] = {}
    if wind_json:
        with Path(wind_json).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

    direction = wind_direction_deg if wind_direction_deg is not None else payload.get("wind_direction_deg")
    speed = wind_speed_kts if wind_speed_kts is not None else payload.get("wind_speed_kts")
    resolved_spot_name = spot_name if spot_name is not None else payload.get("spot_name")
    wind_source = payload.get("wind_source")
    if wind_json and not wind_source:
        wind_source = "wind_json"
    if wind_direction_deg is not None or wind_speed_kts is not None or spot_name is not None:
        wind_source = wind_source or "cli"

    return WindContext(
        wind_direction_deg=normalize_degrees(float(direction)) if direction is not None else None,
        wind_speed_kts=float(speed) if speed is not None else None,
        spot_name=str(resolved_spot_name) if resolved_spot_name else None,
        wind_source=str(wind_source) if wind_source else None,
    )


def load_wind_context(args: argparse.Namespace) -> WindContext:
    return build_wind_context(
        wind_json=args.wind_json,
        wind_direction_deg=args.wind_direction_deg,
        wind_speed_kts=args.wind_speed_kts,
        spot_name=args.spot_name,
    )


def wind_context_to_dict(wind_context: WindContext) -> dict[str, float | str | None]:
    return {
        "wind_direction_deg": round(wind_context.wind_direction_deg, 1) if wind_context.wind_direction_deg is not None else None,
        "wind_direction_cardinal": cardinal_direction(wind_context.wind_direction_deg),
        "wind_speed_kts": round(wind_context.wind_speed_kts, 1) if wind_context.wind_speed_kts is not None else None,
        "spot_name": wind_context.spot_name,
        "wind_source": wind_context.wind_source,
    }


def add_wind_metrics_to_runs(samples: list[Sample], runs: list[Run], wind_context: WindContext) -> list[Run]:
    enriched = []
    for run in runs:
        run_samples = samples[run.start_index : run.end_index + 1]
        mean_bearing = weighted_mean_bearing_deg(run_samples)
        angle_to_wind = (
            angle_difference_degrees(mean_bearing, wind_context.wind_direction_deg)
            if mean_bearing is not None and wind_context.wind_direction_deg is not None
            else None
        )
        enriched.append(
            Run(
                **{
                    **run.__dict__,
                    "mean_bearing_deg": mean_bearing,
                    "angle_to_wind_deg": angle_to_wind,
                    "wind_angle_class": classify_wind_angle(angle_to_wind),
                }
            )
        )
    return enriched


def bearing_bins(samples: list[Sample], min_speed_mps: float, bin_count: int = 16) -> list[dict[str, float]]:
    bins = [0.0 for _ in range(bin_count)]
    for previous, sample in zip(samples, samples[1:]):
        if sample.smooth_speed_mps < min_speed_mps or sample.segment_distance_m <= 0:
            continue
        bearing = bearing_degrees(previous.lat, previous.lon, sample.lat, sample.lon)
        bucket = int((bearing / 360) * bin_count) % bin_count
        bins[bucket] += sample.segment_distance_m

    max_distance = max(bins, default=0.0) or 1.0
    return [
        {
            "bearing": index * 360 / bin_count,
            "distance_m": round(distance_m, 2),
            "ratio": round(distance_m / max_distance, 4),
        }
        for index, distance_m in enumerate(bins)
    ]


def project_points(samples: list[Sample], width: int, height: int, pad: int) -> list[tuple[float, float]]:
    min_lat = min(sample.lat for sample in samples)
    max_lat = max(sample.lat for sample in samples)
    min_lon = min(sample.lon for sample in samples)
    max_lon = max(sample.lon for sample in samples)
    lat_span = max(max_lat - min_lat, 1e-9)
    lon_span = max(max_lon - min_lon, 1e-9)
    usable_w = width - 2 * pad
    usable_h = height - 2 * pad

    points = []
    for sample in samples:
        x = pad + ((sample.lon - min_lon) / lon_span) * usable_w
        y = height - pad - ((sample.lat - min_lat) / lat_span) * usable_h
        points.append((x, y))
    return points


def write_map_svg(path: Path, samples: list[Sample], runs: list[Run], falls: list[Fall]) -> None:
    width, height, pad = 1000, 700, 42
    if not samples:
        path.write_text(blank_svg(width, height, "No GPS points"), encoding="utf-8")
        return

    points = project_points(samples, width, height, pad)
    route_polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    run_speeds = [sample.smooth_speed_mps for sample in samples if sample.in_run]
    min_run_speed = min(run_speeds, default=0.0)
    max_run_speed = max(run_speeds, default=1.0)
    fall_by_index = {fall.index: fall for fall in falls}

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Wingfoil activity map">',
        '<rect width="100%" height="100%" fill="#07111f"/>',
        '<rect x="20" y="20" width="960" height="660" rx="8" fill="#0f1c30" stroke="rgba(255,255,255,0.14)"/>',
        '<g stroke-linecap="round" stroke-linejoin="round">',
        f'<polyline points="{route_polyline}" fill="none" stroke="#94a3b8" stroke-width="4" opacity="0.52"/>',
    ]

    for run in runs:
        run_points = points[run.start_index : run.end_index + 1]
        if len(run_points) < 2:
            continue
        color = color_for_speed(run.mean_speed_mps, min_run_speed, max_run_speed)
        polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in run_points)
        parts.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="7"/>')

    parts.append("</g>")
    if points:
        start_x, start_y = points[0]
        end_x, end_y = points[-1]
        parts.append(f'<circle cx="{start_x:.1f}" cy="{start_y:.1f}" r="7" fill="#34d399" stroke="#ffffff" stroke-width="2"><title>Start</title></circle>')
        parts.append(f'<circle cx="{end_x:.1f}" cy="{end_y:.1f}" r="7" fill="#0f172a" stroke="#ffffff" stroke-width="2"><title>End</title></circle>')

    for index, fall in fall_by_index.items():
        x, y = points[index]
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="#fb7185" stroke="#ffffff" stroke-width="2">'
            f"<title>Fall {fall.fall_id}: {svg_escape(fmt_dt(fall.time))}</title></circle>"
        )

    parts.extend(
        [
            '<g font-family="Inter, system-ui, sans-serif" font-size="18" fill="#cbd5e1">',
            '<text x="42" y="52" font-size="26" font-weight="700" fill="#f8fafc">Runs map</text>',
            '<text x="42" y="80">Full track, colored runs, fall markers</text>',
            "</g>",
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts), encoding="utf-8")


def write_map_html(
    path: Path,
    source_activity: Path,
    samples: list[Sample],
    runs: list[Run],
    falls: list[Fall],
    wind_context: WindContext,
    water_time_s: float,
) -> None:
    if not samples:
        path.write_text("<!doctype html><title>Wingfoil map</title><p>No GPS points</p>", encoding="utf-8")
        return

    run_speeds = [sample.smooth_speed_mps for sample in samples if sample.in_run]
    min_speed_mps = min(run_speeds, default=0.0)
    max_speed_mps = max(run_speeds, default=max((sample.smooth_speed_mps for sample in samples), default=1.0))
    run_ranges = [(run.start_index, run.end_index) for run in runs]
    falls_payload = [fall_to_dict(fall) for fall in falls]
    segments = []
    for previous, sample in zip(samples, samples[1:]):
        in_run = sample.in_run and previous.in_run
        speed_mps = sample.smooth_speed_mps
        segments.append(
            {
                "from": [previous.lat, previous.lon],
                "to": [sample.lat, sample.lon],
                "speed_mps": round(speed_mps, 3),
                "speed_kmh": round(speed_mps * MPS_TO_KMH, 2),
                "speed_knots": round(speed_mps * MPS_TO_KNOTS, 2),
                "color": color_for_speed(speed_mps, min_speed_mps, max_speed_mps) if in_run else "#b7c0c7",
                "weight": 5 if in_run else 2,
                "opacity": 0.92 if in_run else 0.42,
                "in_run": in_run,
            }
        )

    elapsed_time_s = (samples[-1].time - samples[0].time).total_seconds() if len(samples) >= 2 else 0.0
    total_distance_m = sum(sample.segment_distance_m for sample in samples)
    foil_time_s = sum(sample.dt_s for sample in samples if sample.in_run)
    foil_distance_m = sum(sample.segment_distance_m for sample in samples if sample.in_run)
    avg_speed_on_foil_mps = foil_distance_m / foil_time_s if foil_time_s > 0 else None
    avg_run_distance_m = statistics.mean(run.distance_m for run in runs) if runs else None
    max_speed_track_mps = max((sample.smooth_speed_mps for sample in samples), default=0.0)
    wind_payload = wind_context_to_dict(wind_context)
    map_wind_payload = {
        "wind_direction_deg": wind_payload["wind_direction_deg"],
        "wind_direction_cardinal": wind_payload["wind_direction_cardinal"],
        "wind_speed_kts": wind_payload["wind_speed_kts"],
    }
    wind_label = None
    if wind_context.wind_direction_deg is not None:
        wind_label = f"{wind_payload['wind_direction_cardinal']} {wind_payload['wind_direction_deg']:.0f}°"
        if wind_context.wind_speed_kts is not None:
            wind_label += f", {wind_payload['wind_speed_kts']:.0f} kt"

    payload = {
        "source_filename": source_activity.name,
        "source_format": source_activity.suffix.lower().lstrip("."),
        "center": [statistics.mean(sample.lat for sample in samples), statistics.mean(sample.lon for sample in samples)],
        "bounds": [[min(sample.lat for sample in samples), min(sample.lon for sample in samples)], [max(sample.lat for sample in samples), max(sample.lon for sample in samples)]],
        "activity": {
            "sample_count": len(samples),
            "run_count": len(runs),
            "fall_count": len(falls),
            "elapsed_time_s": round(elapsed_time_s, 1),
            "total_distance_km": round(total_distance_m / 1000, 2),
            "max_speed_kmh": round(max_speed_track_mps * MPS_TO_KMH, 1),
            "avg_run_distance_km": round(avg_run_distance_m / 1000, 2) if avg_run_distance_m is not None else None,
            "avg_speed_on_foil_kmh": round(avg_speed_on_foil_mps * MPS_TO_KMH, 1) if avg_speed_on_foil_mps is not None else None,
            "water_time_s": round(water_time_s, 1),
            "water_time_formatted": format_duration_s(water_time_s),
            "foil_time_s": round(foil_time_s, 1),
            "foil_time_formatted": format_duration_s(foil_time_s),
        },
        "wind_context": map_wind_payload,
        "wind_label": wind_label,
        "segments": segments,
        "runs": [run_to_dict(run) | {"start_index": run.start_index, "end_index": run.end_index} for run in runs],
        "run_ranges": run_ranges,
        "falls": falls_payload,
        "start": [samples[0].lat, samples[0].lon],
        "end": [samples[-1].lat, samples[-1].lon],
        "speed_min_kmh": round(min_speed_mps * MPS_TO_KMH, 1),
        "speed_max_kmh": round(max_speed_mps * MPS_TO_KMH, 1),
        "bearing_bins": bearing_bins(samples, min_speed_mps=max(min_speed_mps, 4.5)),
    }
    data_json = json.dumps(payload)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Cache-Control" content="no-store, max-age=0">
  <meta http-equiv="Pragma" content="no-cache">
  <title>Wingfoil activity map</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    :root {{
      --navy: #07111f;
      --panel: rgba(8, 18, 34, 0.9);
      --panel-soft: rgba(15, 28, 48, 0.82);
      --text: #f8fafc;
      --muted: #b8c4d5;
	      --line: rgba(255, 255, 255, 0.14);
	      --accent: #38bdf8;
	      --wind: #e879f9;
	      --red: #fb7185;
	    }}
    html, body {{ margin: 0; }}
    body {{
      background: var(--navy);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}
    .analysis-page {{ background: var(--navy); box-sizing: border-box; overflow: hidden; width: 100%; }}
    #map {{ height: clamp(420px, 52vw, 620px); width: 100%; }}
    .leaflet-container {{ background: #07111f; font-family: inherit; }}
    .leaflet-control-zoom a,
    .leaflet-control-layers,
    .leaflet-control-scale-line {{
      background: rgba(8, 18, 34, 0.88);
      border-color: var(--line);
      box-shadow: none;
      color: var(--text);
    }}
    .leaflet-control-attribution {{ background: rgba(8, 18, 34, 0.72); color: var(--muted); }}
    .leaflet-control-attribution a {{ color: #bfdbfe; }}
    .map-overlay {{
      background: rgba(8, 18, 34, 0.72);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 8px;
      box-shadow: 0 10px 28px rgba(0, 0, 0, 0.22);
      color: var(--text);
      padding: 8px 10px;
      backdrop-filter: blur(12px);
    }}
    .speed-ramp {{
      width: 118px;
      height: 7px;
      border-radius: 999px;
      background: linear-gradient(90deg, #440154, #414487, #2a788e, #22a884, #7ad151, #fde725);
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.16);
      margin: 5px 0 4px;
    }}
    .speed-title {{ color: var(--text); font-size: 11px; font-weight: 800; letter-spacing: 0.04em; }}
    .legend-row {{ align-items: center; color: var(--muted); display: flex; justify-content: space-between; gap: 12px; font-size: 10px; }}
    .wind-panel {{ max-width: 178px; }}
    .wind-content {{ align-items: center; display: flex; gap: 8px; }}
    .wind-label {{ font-size: 12px; font-weight: 800; white-space: nowrap; }}
    .wind-arrow svg {{ display: block; }}
    .dashboard {{
      box-sizing: border-box;
      display: grid;
      gap: 14px;
      grid-template-columns: minmax(0, 1.25fr) minmax(280px, 0.75fr);
      margin: 0;
      max-width: none;
      padding: 14px;
      width: 100%;
    }}
    .card {{
      background: #0f1c30;
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 16px 48px rgba(0, 0, 0, 0.22);
      min-width: 0;
      padding: 14px;
    }}
    .card-title {{ color: var(--text); font-size: 16px; font-weight: 850; margin: 0; }}
    .card-subtitle {{ color: var(--muted); font-size: 12px; margin-top: 4px; }}
    .stat-grid {{ display: grid; gap: 10px; grid-template-columns: repeat(4, minmax(0, 1fr)); margin-top: 16px; }}
    .stat {{ background: var(--panel-soft); border-radius: 8px; padding: 12px; }}
    .stat-value {{ font-size: 20px; font-weight: 850; line-height: 1.05; overflow-wrap: anywhere; }}
    .stat-label {{ color: var(--muted); font-size: 10px; font-weight: 750; letter-spacing: 0.06em; margin-top: 6px; text-transform: uppercase; }}
	    .rose svg {{ display: block; }}
	    .rose-label {{ font-size: 11px; fill: #cbd5e1; font-weight: 700; }}
	    .rose-legend {{ align-items: center; display: flex; gap: 12px; margin-top: 10px; color: var(--muted); font-size: 11px; font-weight: 700; }}
	    .rose-legend-item {{ align-items: center; display: inline-flex; gap: 5px; }}
	    .rose-legend-swatch {{ border-radius: 999px; display: inline-block; height: 3px; width: 18px; }}
	    .rose-legend-swatch.gps {{ background: #38bdf8; }}
	    .rose-legend-swatch.wind {{ background: var(--wind); }}
	    .course-rose-card {{ align-items: center; display: flex; flex-direction: column; }}
    .course-rose-viz {{ margin-top: 12px; }}
    @media (max-width: 720px) {{
      #map {{ height: clamp(340px, 70vw, 460px); }}
      .dashboard {{ grid-template-columns: 1fr; padding: 12px; }}
      .card {{ padding: 14px; }}
      .stat-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }}
      .stat-value {{ font-size: 17px; }}
      .speed-ramp {{ width: 96px; }}
    }}
  </style>
</head>
<body>
<main id="analysis-content" class="analysis-page">
  <div id="map"></div>
  <section class="dashboard" aria-label="Activity dashboard">
    <article class="card stats-card">
      <h1 class="card-title">Session stats</h1>
      <div class="stat-grid" id="statsGrid"></div>
    </article>
    <article class="card course-rose-card">
      <h2 class="card-title">Course rose</h2>
      <div class="card-subtitle">GPS travel bearings</div>
      <div class="course-rose-viz rose" id="courseRoseViz"></div>
    </article>
  </section>
</main>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const data = {data_json};
const map = L.map("map", {{ preferCanvas: true }});
const satellite = L.tileLayer(
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}",
  {{ maxZoom: 19, attribution: "Tiles &copy; Esri" }}
).addTo(map);
const labels = L.tileLayer(
  "https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{{z}}/{{y}}/{{x}}",
  {{ maxZoom: 19, attribution: "Labels &copy; Esri" }}
).addTo(map);
const lineLayer = L.layerGroup().addTo(map);
const fallLayer = L.layerGroup().addTo(map);
map.fitBounds(data.bounds, {{ padding: [28, 28] }});

for (const segment of data.segments) {{
  const line = L.polyline([segment.from, segment.to], {{
    color: segment.color,
    weight: segment.weight,
    opacity: segment.opacity,
    lineCap: "round"
  }});
  if (segment.in_run) {{
    line.bindTooltip(`${{segment.speed_kmh}} km/h<br>${{segment.speed_knots}} kt`, {{ sticky: true }});
  }}
  line.addTo(lineLayer);
}}

L.circleMarker(data.start, {{ radius: 7, color: "#ffffff", weight: 2, fillColor: "#10b981", fillOpacity: 1 }})
  .bindPopup("Start").addTo(map);
L.circleMarker(data.end, {{ radius: 7, color: "#ffffff", weight: 2, fillColor: "#111827", fillOpacity: 1 }})
  .bindPopup("End").addTo(map);

for (const fall of data.falls) {{
  L.circleMarker([fall.lat, fall.lon], {{
    radius: 8,
    color: "#ffffff",
    weight: 2,
    fillColor: "#dc2626",
    fillOpacity: 0.95
  }}).bindPopup(`Fall ${{fall.fall_id}}<br>${{fall.time}}<br>${{fall.speed_after_mps}} m/s`).addTo(fallLayer);
}}
window.__wingfoilMapDebug = {{
  segmentCount: data.segments.length,
  fallCount: data.falls.length,
  runCount: data.runs.length,
  bounds: data.bounds,
  sourceFilename: data.source_filename,
  windContext: data.wind_context
}};

function escapeHtml(value) {{
  return String(value).replace(/[&<>"']/g, function(ch) {{
    return ({{ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }})[ch];
  }});
}}

function formatDuration(seconds) {{
  const safe = Math.max(0, Math.round(seconds || 0));
  const hours = Math.floor(safe / 3600);
  const minutes = Math.floor((safe % 3600) / 60);
  if (hours > 0) return `${{hours}}h ${{minutes}}m`;
  return `${{minutes}}m`;
}}

function statCard(value, label) {{
  return `<div class="stat"><div class="stat-value">${{escapeHtml(value)}}</div><div class="stat-label">${{escapeHtml(label)}}</div></div>`;
}}

function renderStats() {{
  const avgRun = data.activity.avg_run_distance_km === null || data.activity.avg_run_distance_km === undefined
    ? "n/a"
    : `${{data.activity.avg_run_distance_km}} km`;
  const avgFoilSpeed = data.activity.avg_speed_on_foil_kmh === null || data.activity.avg_speed_on_foil_kmh === undefined
    ? "n/a"
    : `${{data.activity.avg_speed_on_foil_kmh}} km/h`;
  document.getElementById("statsGrid").innerHTML = [
    statCard(formatDuration(data.activity.elapsed_time_s), "elapsed time"),
    statCard(data.activity.foil_time_formatted, "time on foil"),
    statCard(data.activity.water_time_formatted, "time in water"),
    statCard(`${{data.activity.total_distance_km}} km`, "total distance"),
    statCard(data.activity.run_count, "runs"),
    statCard(data.activity.fall_count, "falls"),
    statCard(avgRun, "avg run distance"),
    statCard(avgFoilSpeed, "avg speed on foil")
  ].join("");
}}

if (data.wind_label) {{
  const wind = L.control({{ position: "bottomleft" }});
  wind.onAdd = function() {{
    const div = L.DomUtil.create("div", "map-overlay wind-panel");
  const hasDirection = data.wind_context.wind_direction_deg !== null && data.wind_context.wind_direction_deg !== undefined;
  const arrowRotation = hasDirection ? (data.wind_context.wind_direction_deg + 180) : 0;
  div.innerHTML = `
    <div class="wind-content">
      <div class="wind-arrow">
        <svg width="34" height="34" viewBox="0 0 34 34" aria-hidden="true">
          <circle cx="17" cy="17" r="15" fill="rgba(232,121,249,0.12)" stroke="rgba(232,121,249,0.42)"/>
          ${{hasDirection ? `<g transform="rotate(${{arrowRotation}} 17 17)"><line x1="17" y1="25" x2="17" y2="8" stroke="#e879f9" stroke-width="3" stroke-linecap="round"/><path d="M11 13 L17 7 L23 13" fill="none" stroke="#e879f9" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></g>` : ""}}
        </svg>
      </div>
      <div class="wind-label">${{escapeHtml(data.wind_label)}}</div>
    </div>
  `;
  return div;
  }};
  wind.addTo(map);
}}

const legend = L.control({{ position: "bottomright" }});
legend.onAdd = function() {{
  const div = L.DomUtil.create("div", "map-overlay");
  div.innerHTML = `
    <div class="speed-title">Speed</div>
    <div class="speed-ramp"></div>
    <div class="legend-row"><span>${{data.speed_min_kmh}} km/h</span><span>${{data.speed_max_kmh}} km/h</span></div>
  `;
  return div;
}};
legend.addTo(map);

function renderCourseRose() {{
  const size = 170;
  const center = size / 2;
  const maxR = 58;
  const bins = data.bearing_bins;
  const hasWindDirection = data.wind_context.wind_direction_deg !== null && data.wind_context.wind_direction_deg !== undefined;
  const petals = bins.map((bin, i) => {{
    const angle = (bin.bearing - 90) * Math.PI / 180;
    const r = 12 + bin.ratio * maxR;
    const x = center + Math.cos(angle) * r;
    const y = center + Math.sin(angle) * r;
    return `<line x1="${{center}}" y1="${{center}}" x2="${{x.toFixed(1)}}" y2="${{y.toFixed(1)}}" stroke="#38bdf8" stroke-width="7" stroke-linecap="round" opacity="0.84"><title>${{Math.round(bin.bearing)}} deg: ${{Math.round(bin.distance_m)}} m</title></line>`;
  }}).join("");
  let windArrow = "";
  if (hasWindDirection) {{
    const windAngle = (data.wind_context.wind_direction_deg - 90) * Math.PI / 180;
    const outerR = 64;
    const innerR = 24;
    const startX = center + Math.cos(windAngle) * outerR;
    const startY = center + Math.sin(windAngle) * outerR;
    const endX = center + Math.cos(windAngle) * innerR;
    const endY = center + Math.sin(windAngle) * innerR;
    windArrow = `
      <line x1="${{startX.toFixed(1)}}" y1="${{startY.toFixed(1)}}" x2="${{endX.toFixed(1)}}" y2="${{endY.toFixed(1)}}" stroke="#e879f9" stroke-width="4" stroke-linecap="round" marker-end="url(#windArrowHead)">
        <title>Wind direction: from ${{Math.round(data.wind_context.wind_direction_deg)}} deg</title>
      </line>`;
  }}
  document.getElementById("courseRoseViz").innerHTML = `
    <svg width="${{size}}" height="${{size}}" viewBox="0 0 ${{size}} ${{size}}" aria-label="Course rose with GPS bearings${{hasWindDirection ? " and wind direction" : ""}}">
      <defs>
        <marker id="windArrowHead" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="4.8" markerHeight="4.8" orient="auto">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#e879f9"></path>
        </marker>
      </defs>
      <circle cx="${{center}}" cy="${{center}}" r="64" fill="rgba(15,28,48,0.72)" stroke="rgba(255,255,255,0.16)"/>
      <line x1="${{center}}" y1="18" x2="${{center}}" y2="${{size - 18}}" stroke="rgba(255,255,255,0.18)"/>
      <line x1="18" y1="${{center}}" x2="${{size - 18}}" y2="${{center}}" stroke="rgba(255,255,255,0.18)"/>
      ${{petals}}
      ${{windArrow}}
      <circle cx="${{center}}" cy="${{center}}" r="4" fill="#f8fafc"/>
      <text x="${{center}}" y="14" text-anchor="middle" class="rose-label">N</text>
      <text x="${{center}}" y="${{size - 6}}" text-anchor="middle" class="rose-label">S</text>
      <text x="${{size - 10}}" y="${{center + 4}}" text-anchor="middle" class="rose-label">E</text>
      <text x="10" y="${{center + 4}}" text-anchor="middle" class="rose-label">W</text>
    </svg>
    <div class="rose-legend">
      <span class="rose-legend-item"><span class="rose-legend-swatch gps"></span>GPS bearings</span>
      ${{hasWindDirection ? '<span class="rose-legend-item"><span class="rose-legend-swatch wind"></span>Wind direction</span>' : ''}}
    </div>
  `;
}}

renderStats();
renderCourseRose();

const analysisContent = document.getElementById("analysis-content");
let heightPostTimer = null;
let lastPostedHeight = 0;

function contentHeight() {{
  if (!analysisContent) return 0;
  return Math.ceil(analysisContent.getBoundingClientRect().height);
}}

function postFrameHeight() {{
  if (window.parent === window) return;
  const height = contentHeight();
  if (!Number.isFinite(height) || height <= 0) return;
  if (Math.abs(height - lastPostedHeight) < 2) return;
  lastPostedHeight = height;
  window.parent.postMessage({{
    type: "wingfoil-analysis-height",
    source: "wingfoil-analysis-map",
    measuredElement: "analysis-content",
    height: height
  }}, window.location.origin);
}}

function scheduleFrameHeightPost(delay = 80) {{
  window.clearTimeout(heightPostTimer);
  heightPostTimer = window.setTimeout(postFrameHeight, delay);
}}

window.addEventListener("DOMContentLoaded", () => scheduleFrameHeightPost(0));
window.addEventListener("load", () => scheduleFrameHeightPost(0));
window.addEventListener("resize", () => scheduleFrameHeightPost(120));
if (window.ResizeObserver && analysisContent) {{
  const observer = new ResizeObserver(() => scheduleFrameHeightPost(80));
  observer.observe(analysisContent);
}}
map.whenReady(() => {{
  map.invalidateSize();
  scheduleFrameHeightPost(120);
}});
window.setTimeout(() => scheduleFrameHeightPost(0), 350);
window.setTimeout(() => scheduleFrameHeightPost(0), 1200);
</script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def blank_svg(width: int, height: int, label: str) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#07111f"/>',
            f'<rect x="18" y="18" width="{width - 36}" height="{height - 36}" rx="8" fill="#0f1c30" stroke="rgba(255,255,255,0.14)"/>',
            f'<text x="{width / 2}" y="{height / 2}" text-anchor="middle" font-family="Inter, system-ui, sans-serif" font-size="24" fill="#cbd5e1">{svg_escape(label)}</text>',
            "</svg>",
        ]
    )


def write_distribution_svg(path: Path, values: list[float], title: str, unit: str, min_count_for_hist: int) -> None:
    width, height = 780, 460
    margin = {"top": 64, "right": 34, "bottom": 64, "left": 68}
    if not values:
        path.write_text(blank_svg(width, height, f"No {title.lower()} data"), encoding="utf-8")
        return

    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{svg_escape(title)}">',
        '<rect width="100%" height="100%" fill="#07111f"/>',
        f'<rect x="18" y="18" width="{width - 36}" height="{height - 36}" rx="8" fill="#0f1c30" stroke="rgba(255,255,255,0.14)"/>',
        f'<text x="{margin["left"]}" y="42" font-family="Inter, system-ui, sans-serif" font-size="22" font-weight="800" fill="#f8fafc">{svg_escape(title)}</text>',
        f'<line x1="{margin["left"]}" y1="{margin["top"] + plot_h * 0.5:.1f}" x2="{margin["left"] + plot_w}" y2="{margin["top"] + plot_h * 0.5:.1f}" stroke="#243447" stroke-width="1"/>',
    ]

    if len(values) >= min_count_for_hist:
        bins = min(12, max(5, round(math.sqrt(len(values)))))
        min_v, max_v = min(values), max(values)
        span = max(max_v - min_v, 1e-9)
        counts = [0 for _ in range(bins)]
        for value in values:
            bucket = min(int((value - min_v) / span * bins), bins - 1)
            counts[bucket] += 1
        max_count = max(counts) or 1
        bar_gap = 6
        bar_w = (plot_w - bar_gap * (bins - 1)) / bins
        for i, count in enumerate(counts):
            bar_h = plot_h * count / max_count
            x = margin["left"] + i * (bar_w + bar_gap)
            y = margin["top"] + plot_h - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="#38bdf8" rx="4"/>')
        parts.append(axis_label(margin["left"], height - 26, f"{min_v:.1f}-{max_v:.1f} {unit}"))
        parts.append(axis_label(18, margin["top"] + 18, "count", rotate=True))
    else:
        max_v = max(values) or 1.0
        bar_gap = 8
        bar_w = min(54, (plot_w - bar_gap * (len(values) - 1)) / len(values))
        for i, value in enumerate(values):
            bar_h = plot_h * value / max_v
            x = margin["left"] + i * (bar_w + bar_gap)
            y = margin["top"] + plot_h - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="#34d399" rx="4"/>')
            parts.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{height - 34}" text-anchor="middle" font-family="Inter, system-ui, sans-serif" font-size="12" fill="#b8c4d5">#{i + 1}</text>'
            )
        parts.append(axis_label(margin["left"], height - 26, f"run ({unit})"))

    axis_x1 = margin["left"]
    axis_y = margin["top"] + plot_h
    parts.append(f'<line x1="{axis_x1}" y1="{axis_y}" x2="{margin["left"] + plot_w}" y2="{axis_y}" stroke="#475569" stroke-width="1"/>')
    parts.append(f'<line x1="{axis_x1}" y1="{margin["top"]}" x2="{axis_x1}" y2="{axis_y}" stroke="#475569" stroke-width="1"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def axis_label(x: float, y: float, text: str, rotate: bool = False) -> str:
    transform = f' transform="rotate(-90 {x} {y})"' if rotate else ""
    return f'<text x="{x}" y="{y}"{transform} font-family="Inter, system-ui, sans-serif" font-size="14" fill="#b8c4d5">{svg_escape(text)}</text>'


def write_run_speed_svg(path: Path, samples: list[Sample], runs: list[Run]) -> None:
    width, height = 900, 520
    margin = {"top": 72, "right": 44, "bottom": 86, "left": 82}
    if not runs:
        path.write_text(blank_svg(width, height, "No run speed data"), encoding="utf-8")
        return

    profiles: list[list[tuple[float, float]]] = []
    for run in runs:
        run_samples = samples[run.start_index : run.end_index + 1]
        cumulative_distance_m = 0.0
        profile = [(0.0, 0.0)]
        for sample in run_samples[1:]:
            cumulative_distance_m += sample.segment_distance_m
            profile.append((cumulative_distance_m, sample.smooth_speed_mps * MPS_TO_KMH))
        if len(profile) > 1:
            profile.append((cumulative_distance_m, 0.0))
            profiles.append(profile)

    if not profiles:
        path.write_text(blank_svg(width, height, "No run speed data"), encoding="utf-8")
        return

    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    max_distance_m = max((distance for profile in profiles for distance, _speed in profile), default=1.0) or 1.0
    max_speed_kmh = max((speed for profile in profiles for _distance, speed in profile), default=1.0) or 1.0
    tick_count = 6

    def nice_step(max_value: float) -> float:
        raw_step = max_value / (tick_count - 1)
        exponent = math.floor(math.log10(raw_step)) if raw_step > 0 else 0
        base = 10**exponent
        for multiplier in (1.0, 2.0, 2.5, 5.0, 10.0):
            step = multiplier * base
            if raw_step <= step:
                return step
        return 10.0 * base

    distance_step_m = nice_step(max_distance_m)
    speed_step_kmh = nice_step(max_speed_kmh)
    axis_max_distance_m = distance_step_m * (tick_count - 1)
    axis_max_speed_kmh = speed_step_kmh * (tick_count - 1)

    def point(distance_m: float, speed_kmh: float) -> tuple[float, float]:
        x = margin["left"] + (distance_m / axis_max_distance_m) * plot_w
        y = margin["top"] + plot_h - (speed_kmh / axis_max_speed_kmh) * plot_h
        return x, y

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Run speed profiles">',
        '<rect width="100%" height="100%" fill="#07111f"/>',
        f'<rect x="18" y="18" width="{width - 36}" height="{height - 36}" rx="8" fill="#0f1c30" stroke="rgba(255,255,255,0.14)"/>',
        f'<text x="{margin["left"]}" y="42" font-family="Inter, system-ui, sans-serif" font-size="22" font-weight="800" fill="#f8fafc">Run speed profiles</text>',
        f'<text x="{margin["left"]}" y="62" font-family="Inter, system-ui, sans-serif" font-size="13" fill="#b8c4d5">Speed profile for {len(profiles)} detected runs</text>',
    ]

    for tick in range(tick_count):
        distance_value = distance_step_m * tick
        x, _ = point(distance_value, 0.0)
        parts.append(f'<line x1="{x:.1f}" y1="{margin["top"]}" x2="{x:.1f}" y2="{margin["top"] + plot_h}" stroke="#1f2d3f" stroke-width="1"/>')
        parts.append(f'<line x1="{x:.1f}" y1="{margin["top"] + plot_h}" x2="{x:.1f}" y2="{margin["top"] + plot_h + 5}" stroke="#64748b" stroke-width="1"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{margin["top"] + plot_h + 24}" text-anchor="middle" font-family="Inter, system-ui, sans-serif" font-size="12" fill="#cbd5e1">{distance_value:.0f}</text>'
        )

        speed_value = speed_step_kmh * tick
        _, y = point(0.0, speed_value)
        parts.append(f'<line x1="{margin["left"]}" y1="{y:.1f}" x2="{margin["left"] + plot_w}" y2="{y:.1f}" stroke="#243447" stroke-width="1"/>')
        parts.append(f'<line x1="{margin["left"] - 5}" y1="{y:.1f}" x2="{margin["left"]}" y2="{y:.1f}" stroke="#64748b" stroke-width="1"/>')
        parts.append(
            f'<text x="{margin["left"] - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="Inter, system-ui, sans-serif" font-size="12" fill="#cbd5e1">{speed_value:.0f}</text>'
        )

    for index, profile in enumerate(profiles):
        color = color_for_speed(index, 0, max(len(profiles) - 1, 1))
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (point(distance, speed) for distance, speed in profile))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.2" opacity="0.86" stroke-linecap="round" stroke-linejoin="round"/>')

    axis_x = margin["left"]
    axis_y = margin["top"] + plot_h
    parts.extend(
        [
            f'<line x1="{axis_x}" y1="{axis_y}" x2="{margin["left"] + plot_w}" y2="{axis_y}" stroke="#475569" stroke-width="1"/>',
            f'<line x1="{axis_x}" y1="{margin["top"]}" x2="{axis_x}" y2="{axis_y}" stroke="#475569" stroke-width="1"/>',
            axis_label(margin["left"] + plot_w / 2 - 42, height - 28, "distance [m]"),
            axis_label(24, margin["top"] + 48, "speed [km/h]", rotate=True),
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts), encoding="utf-8")


def write_wind_angle_svg(path: Path, runs: list[Run], wind_context: WindContext) -> None:
    width, height = 780, 460
    if wind_context.wind_direction_deg is None:
        path.write_text(blank_svg(width, height, "Wind context unavailable"), encoding="utf-8")
        return

    totals = {"upwind": 0.0, "crosswind": 0.0, "downwind": 0.0}
    for run in runs:
        if run.wind_angle_class in totals:
            totals[run.wind_angle_class] += run.distance_m

    if not any(totals.values()):
        path.write_text(blank_svg(width, height, "No run wind-angle data"), encoding="utf-8")
        return

    margin = {"top": 72, "right": 46, "bottom": 72, "left": 78}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    max_distance = max(totals.values()) or 1.0
    bar_gap = 34
    bar_w = (plot_w - bar_gap * 2) / 3
    colors = {"upwind": "#38bdf8", "crosswind": "#34d399", "downwind": "#fde047"}
    wind_cardinal = cardinal_direction(wind_context.wind_direction_deg) or ""
    wind_speed = (
        f", {wind_context.wind_speed_kts:.0f} kt"
        if wind_context.wind_speed_kts is not None
        else ""
    )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Run distance by wind angle class">',
        '<rect width="100%" height="100%" fill="#07111f"/>',
        f'<rect x="18" y="18" width="{width - 36}" height="{height - 36}" rx="8" fill="#0f1c30" stroke="rgba(255,255,255,0.14)"/>',
        f'<text x="{margin["left"]}" y="42" font-family="Inter, system-ui, sans-serif" font-size="22" font-weight="800" fill="#f8fafc">Run distance by wind angle</text>',
        f'<text x="{margin["left"]}" y="62" font-family="Inter, system-ui, sans-serif" font-size="13" fill="#b8c4d5">External wind from {wind_context.wind_direction_deg:.0f}° {wind_cardinal}{wind_speed}</text>',
        f'<line x1="{margin["left"]}" y1="{margin["top"] + plot_h * 0.5:.1f}" x2="{margin["left"] + plot_w}" y2="{margin["top"] + plot_h * 0.5:.1f}" stroke="#243447" stroke-width="1"/>',
    ]

    for i, label in enumerate(["upwind", "crosswind", "downwind"]):
        value = totals[label]
        bar_h = plot_h * value / max_distance
        x = margin["left"] + i * (bar_w + bar_gap)
        y = margin["top"] + plot_h - bar_h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{colors[label]}" rx="5"/>')
        parts.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{y - 10:.1f}" text-anchor="middle" font-family="Inter, system-ui, sans-serif" font-size="13" font-weight="700" fill="#f8fafc">{value / 1000:.2f} km</text>'
        )
        parts.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{height - 38}" text-anchor="middle" font-family="Inter, system-ui, sans-serif" font-size="14" font-weight="700" fill="#b8c4d5">{label}</text>'
        )

    axis_y = margin["top"] + plot_h
    parts.append(f'<line x1="{margin["left"]}" y1="{axis_y}" x2="{margin["left"] + plot_w}" y2="{axis_y}" stroke="#475569" stroke-width="1"/>')
    parts.append(f'<line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{axis_y}" stroke="#475569" stroke-width="1"/>')
    parts.append(axis_label(24, margin["top"] + 40, "run distance", rotate=True))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_summary_json(
    path: Path,
    source_activity: Path,
    samples: list[Sample],
    runs: list[Run],
    falls: list[Fall],
    water_time_s: float,
    wind_context: WindContext,
    config: dict[str, object],
    warnings: list[str],
) -> None:
    total_distance_m = sum(sample.segment_distance_m for sample in samples)
    moving_time_s = sum(sample.dt_s for sample in samples if sample.smooth_speed_mps >= config["water_speed_threshold_mps"])
    foil_time_s = sum(sample.dt_s for sample in samples if sample.in_run)
    foil_distance_m = sum(sample.segment_distance_m for sample in samples if sample.in_run)
    avg_speed_on_foil_mps = foil_distance_m / foil_time_s if foil_time_s > 0 else None
    elapsed_time_s = (samples[-1].time - samples[0].time).total_seconds() if len(samples) >= 2 else 0
    distance_stats = summarize(run.distance_m for run in runs)
    speed_stats = summarize(run.mean_speed_mps for run in runs)
    avg_run_distance_m = distance_stats["mean"]
    wind_payload = wind_context_to_dict(wind_context)
    end_reason_counts = {reason: sum(1 for run in runs if run.end_reason == reason) for reason in sorted({run.end_reason for run in runs})}
    raw_run_count = sum(run.merged_raw_run_count for run in runs)

    payload = {
        "source_filename": source_activity.name,
        "source_format": source_activity.suffix.lower().lstrip("."),
        "activity_type": source_activity.suffix.lower().lstrip("."),
        "analysis_status": "ok",
        "analysis_version": ANALYSIS_VERSION,
        "generated_at": fmt_dt(datetime.now(timezone.utc)),
        "wind_direction_deg": wind_payload["wind_direction_deg"],
        "wind_direction_cardinal": wind_payload["wind_direction_cardinal"],
        "wind_speed_kts": wind_payload["wind_speed_kts"],
        "spot_name": wind_payload["spot_name"],
        "wind_source": wind_payload["wind_source"],
        "wind_context": wind_payload,
        "config": config,
        "warnings": warnings,
        "activity": {
            "start_time": fmt_dt(samples[0].time) if samples else None,
            "end_time": fmt_dt(samples[-1].time) if samples else None,
            "sample_count": len(samples),
            "elapsed_time_s": round(elapsed_time_s, 2),
            "total_distance_m": round(total_distance_m, 2),
            "moving_time_s": round(moving_time_s, 2),
            "water_time_s": round(water_time_s, 2),
            "water_time_formatted": format_duration_s(water_time_s),
            "foil_time_s": round(foil_time_s, 2),
            "foil_time_formatted": format_duration_s(foil_time_s),
            "avg_run_distance_m": round(avg_run_distance_m, 2) if avg_run_distance_m is not None else None,
            "avg_run_distance_km": round(avg_run_distance_m / 1000, 3) if avg_run_distance_m is not None else None,
            "avg_speed_on_foil_mps": round(avg_speed_on_foil_mps, 3) if avg_speed_on_foil_mps is not None else None,
            "avg_speed_on_foil_kmh": round(avg_speed_on_foil_mps * MPS_TO_KMH, 2) if avg_speed_on_foil_mps is not None else None,
        },
        "runs_summary": {
            "count": len(runs),
            "raw_detected_count_before_merge": raw_run_count,
            "end_reason_counts": end_reason_counts,
            "distance_m": {key: round(value, 2) if value is not None else None for key, value in distance_stats.items()},
            "mean_speed_mps": {key: round(value, 3) if value is not None else None for key, value in speed_stats.items()},
            "mean_speed_kmh": {
                key: round(value * MPS_TO_KMH, 2) if value is not None else None for key, value in speed_stats.items()
            },
            "mean_speed_knots": {
                key: round(value * MPS_TO_KNOTS, 2) if value is not None else None for key, value in speed_stats.items()
            },
        },
        "falls_summary": {
            "count": len(falls),
            "requires_previous_detected_run": True,
        },
        "runs": [run_to_dict(run) for run in runs],
        "falls": [fall_to_dict(fall) for fall in falls],
        "artifacts": ARTIFACTS,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def clear_generated_artifacts(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    old_run_word = "la" + "p"
    legacy_artifacts = [artifact_name.replace("run", old_run_word) for artifact_name in GENERATED_ARTIFACTS]
    for artifact_name in set(GENERATED_ARTIFACTS + legacy_artifacts):
        artifact_path = output_dir / artifact_name
        if artifact_path.exists() and artifact_path.is_file():
            artifact_path.unlink()


def validate_activity_path(activity_path: Path) -> None:
    if not activity_path.exists():
        raise AnalysisError(f"Input activity file does not exist: {activity_path}")
    if not activity_path.is_file():
        raise AnalysisError(f"Input activity path is not a file: {activity_path}")
    if activity_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise AnalysisError(f"Unsupported activity format: {activity_path.suffix}. Expected one of: {supported}")


def analyze_activity(
    activity_path: str | Path,
    wind_context: WindContext | None = None,
    config: AnalysisConfig | None = None,
) -> AnalysisResult:
    source_activity = Path(activity_path)
    validate_activity_path(source_activity)
    wind_context = wind_context or WindContext()
    config = config or default_analysis_config()

    try:
        samples, warnings = load_activity(source_activity, config.smooth_window)
    except AnalysisError:
        raise
    except Exception as exc:
        raise AnalysisError(f"Could not parse activity file {source_activity.name}: {exc}") from exc
    samples, runs = detect_runs(
        samples,
        run_speed_threshold_mps=config.run_speed_threshold_mps,
        min_run_duration_s=config.min_run_duration_s,
        max_gap_s=config.max_run_gap_s,
        merge_runs_without_stop=config.merge_runs_without_stop,
        run_stop_threshold_mps=config.run_stop_threshold_mps,
    )
    if not runs:
        warnings.append("no runs detected under current thresholds")
    falls = detect_falls(
        samples,
        run_speed_threshold_mps=config.run_speed_threshold_mps,
        fall_speed_threshold_mps=config.fall_speed_threshold_mps,
        fall_window_s=config.fall_window_s,
        min_fall_gap_s=config.min_fall_gap_s,
    )
    runs = classify_run_end_reasons(samples, runs, falls, config.fall_window_s, config.run_stop_threshold_mps)
    runs = add_wind_metrics_to_runs(samples, runs, wind_context)
    water_time_s = compute_water_time_s(samples, config.water_speed_threshold_mps)

    return AnalysisResult(
        source_activity=source_activity,
        samples=samples,
        runs=runs,
        falls=falls,
        water_time_s=water_time_s,
        wind_context=wind_context,
        config=config,
        warnings=warnings,
    )


def write_analysis_outputs(result: AnalysisResult, output_dir: str | Path, clear_outputs: bool = True) -> None:
    output_path = Path(output_dir)
    if clear_outputs:
        clear_generated_artifacts(output_path)
    else:
        output_path.mkdir(parents=True, exist_ok=True)

    config_payload = config_to_dict(result.config)
    write_summary_json(
        output_path / "summary.json",
        result.source_activity,
        result.samples,
        result.runs,
        result.falls,
        result.water_time_s,
        result.wind_context,
        config_payload,
        result.warnings,
    )
    write_runs_csv(output_path / "runs.csv", result.runs)
    write_map_svg(output_path / "map.svg", result.samples, result.runs, result.falls)
    write_map_html(output_path / "map.html", result.source_activity, result.samples, result.runs, result.falls, result.wind_context, result.water_time_s)
    write_distribution_svg(
        output_path / "run_distance_distribution.svg",
        [run.distance_m for run in result.runs],
        "Run distance distribution",
        "m",
        result.config.min_count_for_histogram,
    )
    write_distribution_svg(
        output_path / "run_speed_distribution.svg",
        [run.mean_speed_mps * MPS_TO_KMH for run in result.runs],
        "Run mean speed distribution",
        "km/h",
        result.config.min_count_for_histogram,
    )
    write_run_speed_svg(output_path / "run_speed.svg", result.samples, result.runs)
    write_wind_angle_svg(output_path / "run_wind_angle_distribution.svg", result.runs, result.wind_context)


def analysis_stats(result: AnalysisResult) -> dict[str, object]:
    samples = result.samples
    total_distance_m = sum(sample.segment_distance_m for sample in samples)
    elapsed_time_s = (samples[-1].time - samples[0].time).total_seconds() if len(samples) >= 2 else 0.0
    max_speed_mps = max((sample.speed_mps for sample in samples), default=0.0)
    foil_time_s = sum(sample.dt_s for sample in samples if sample.in_run)
    foil_distance_m = sum(sample.segment_distance_m for sample in samples if sample.in_run)
    avg_speed_on_foil_mps = foil_distance_m / foil_time_s if foil_time_s > 0 else None
    bounds = (
        [[min(sample.lat for sample in samples), min(sample.lon for sample in samples)], [max(sample.lat for sample in samples), max(sample.lon for sample in samples)]]
        if samples
        else None
    )
    return {
        "start_time": fmt_dt(samples[0].time) if samples else None,
        "end_time": fmt_dt(samples[-1].time) if samples else None,
        "duration_s": round(elapsed_time_s, 2),
        "distance_m": round(total_distance_m, 2),
        "distance_km": round(total_distance_m / 1000, 3),
        "avg_speed_on_foil_kmh": round(avg_speed_on_foil_mps * MPS_TO_KMH, 2) if avg_speed_on_foil_mps is not None else None,
        "max_speed_kmh": round(max_speed_mps * MPS_TO_KMH, 2),
        "track_point_count": len(samples),
        "run_count": len(result.runs),
        "fall_count": len(result.falls),
        "water_time_s": round(result.water_time_s, 2),
        "bounds": bounds,
    }


def analysis_result_to_dict(result: AnalysisResult, output_dir: str | Path | None = None) -> dict[str, object]:
    artifact_paths = None
    if output_dir is not None:
        output_path = Path(output_dir)
        artifact_paths = {key: str(output_path / filename) for key, filename in ARTIFACTS.items()}

    payload: dict[str, object] = {
        "status": "ok",
        "analysis_version": ANALYSIS_VERSION,
        "input_filename": result.source_activity.name,
        "input_type": result.source_activity.suffix.lower().lstrip("."),
        "summary_json": ARTIFACTS["summary_json"],
        "map_html": ARTIFACTS["map_html"],
        "map_svg": ARTIFACTS["map_svg"],
        "runs_csv": ARTIFACTS["runs_csv"],
        "artifacts": ARTIFACTS,
        "plots": {
            "speed_distribution_svg": ARTIFACTS["run_speed_distribution_svg"],
            "distance_distribution_svg": ARTIFACTS["run_distance_distribution_svg"],
            "run_speed_profile_svg": ARTIFACTS["run_speed_profile_svg"],
            "wind_angle_distribution_svg": ARTIFACTS["run_wind_angle_distribution_svg"],
        },
        "stats": analysis_stats(result),
        "warnings": result.warnings,
    }
    if artifact_paths is not None:
        payload["artifact_paths"] = artifact_paths
    return payload


def analyze_session_file(
    input_file: str | Path,
    output_dir: str | Path,
    wind_context: WindContext | None = None,
    wind_json: str | Path | None = None,
    wind_direction_deg: float | None = None,
    wind_speed_kts: float | None = None,
    spot_name: str | None = None,
    config: AnalysisConfig | None = None,
    clear_outputs: bool = True,
    raise_on_error: bool = True,
) -> dict[str, object]:
    try:
        resolved_wind_context = wind_context or build_wind_context(
            wind_json=wind_json,
            wind_direction_deg=wind_direction_deg,
            wind_speed_kts=wind_speed_kts,
            spot_name=spot_name,
        )
        result = analyze_activity(input_file, wind_context=resolved_wind_context, config=config)
        write_analysis_outputs(result, output_dir, clear_outputs=clear_outputs)
        return analysis_result_to_dict(result, output_dir)
    except Exception as exc:
        if raise_on_error:
            raise
        return {
            "status": "error",
            "analysis_version": ANALYSIS_VERSION,
            "input_filename": Path(input_file).name,
            "input_type": Path(input_file).suffix.lower().lstrip("."),
            "error": str(exc),
            "warnings": [],
        }


def build_config_from_args(args: argparse.Namespace) -> AnalysisConfig:
    return default_analysis_config(
        run_speed_threshold_mps=args.run_speed_threshold_mps,
        min_run_duration_s=args.min_run_duration_s,
        max_run_gap_s=args.max_run_gap_s,
        merge_runs_without_stop=args.merge_runs_without_stop,
        run_stop_threshold_mps=args.run_stop_threshold_mps,
        fall_speed_threshold_mps=args.fall_speed_threshold_mps,
        fall_window_s=args.fall_window_s,
        min_fall_gap_s=args.min_fall_gap_s,
        water_speed_threshold_mps=args.water_speed_threshold_mps,
        smooth_window=args.smooth_window,
        min_count_for_histogram=args.min_count_for_histogram,
    )


def analyze(args: argparse.Namespace) -> AnalysisResult:
    output_dir = Path(args.output_dir)
    activity = args.input_file or args.activity
    if not activity:
        raise AnalysisError("No input activity provided. Use an activity path or --input.")
    result = analyze_activity(
        activity_path=activity,
        wind_context=load_wind_context(args),
        config=build_config_from_args(args),
    )
    write_analysis_outputs(result, output_dir)

    print(f"Wrote analysis to {output_dir}")
    print(f"Runs: {len(result.runs)}")
    print(f"Falls: {len(result.falls)}")
    print(f"Water time: {result.water_time_s:.1f}s")
    for warning in result.warnings:
        print(f"Warning: {warning}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze wingfoil runs and falls from a GPX, KML, or FIT file.")
    parser.add_argument("activity", nargs="?", help="Input GPX, KML, or FIT file")
    parser.add_argument("--input", dest="input_file", help="Input GPX, KML, or FIT file")
    parser.add_argument("-o", "--output-dir", "--out-dir", default="outputs/wingfoil_analysis", help="Directory for JSON, CSV, and SVG artifacts")
    parser.add_argument("--run-speed-threshold-mps", type=float, default=DEFAULT_SETTINGS["run_speed_threshold_mps"], help="Minimum smoothed speed for foiling/run state")
    parser.add_argument("--min-run-duration-s", type=float, default=DEFAULT_SETTINGS["min_run_duration_s"], help="Minimum duration above threshold to count as a run")
    parser.add_argument("--max-run-gap-s", type=float, default=DEFAULT_SETTINGS["max_run_gap_s"], help="Merge short below-threshold gaps inside a run")
    parser.add_argument(
        "--merge-runs-without-stop",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SETTINGS["merge_runs_without_stop"],
        help="Merge adjacent detected runs when the gap has no true stop below --run-stop-threshold-mps",
    )
    parser.add_argument("--run-stop-threshold-mps", type=float, default=DEFAULT_SETTINGS["run_stop_threshold_mps"], help="Gap speed below this value marks a true stop between runs")
    parser.add_argument("--fall-speed-threshold-mps", type=float, default=DEFAULT_SETTINGS["fall_speed_threshold_mps"], help="Speed below this after a detected run counts as fall candidate")
    parser.add_argument("--fall-window-s", type=float, default=DEFAULT_SETTINGS["fall_window_s"], help="Lookback window for detected-run-to-low-speed fall transition")
    parser.add_argument("--min-fall-gap-s", type=float, default=DEFAULT_SETTINGS["min_fall_gap_s"], help="Minimum seconds between counted falls")
    parser.add_argument("--water-speed-threshold-mps", type=float, default=DEFAULT_SETTINGS["water_speed_threshold_mps"], help="Non-run time below this speed is counted as water time")
    parser.add_argument("--smooth-window", type=int, default=DEFAULT_SETTINGS["smooth_window"], help="Median smoothing window in samples for GPS-derived speed")
    parser.add_argument("--min-count-for-histogram", type=int, default=DEFAULT_SETTINGS["min_count_for_histogram"], help="Use histogram instead of per-run bars above this run count")
    parser.add_argument("--wind-json", help="Optional JSON file with spot_name, wind_direction_deg, wind_speed_kts, and wind_source")
    parser.add_argument("--wind-direction-deg", type=float, help="Optional external wind direction in degrees, using meteorological 'from' direction")
    parser.add_argument("--wind-speed-kts", type=float, help="Optional external wind speed in knots")
    parser.add_argument("--spot-name", help="Optional spot name for map and summary context")
    return parser


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in {"analyze", "analyse"}:
        argv = argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.activity and not args.input_file:
        parser.error("an input activity is required; pass a positional path or --input")
    analyze(args)


if __name__ == "__main__":
    main()
