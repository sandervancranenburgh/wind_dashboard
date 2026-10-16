"""Keep the latest observed current-day export, outside Git, for every site.

The caller supplies a site-isolated artifact directory. This module never
removes files; historical compaction is a separate, reviewed operation.
"""
from __future__ import annotations

import csv
import io
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"refusing to overwrite a symlink: {path}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def csv_day_and_observations(content: bytes, local_tz: str) -> tuple[str, set[str]]:
    rows = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
    if not rows or "time_local" not in rows[0]:
        raise ValueError("daily current-day CSV must contain time_local rows")
    days: set[str] = set()
    observations: set[str] = set()
    for row in rows:
        instant = datetime.fromisoformat(row["time_local"].replace("Z", "+00:00"))
        if instant.tzinfo is None:
            raise ValueError("daily CSV timestamps must include their UTC offset")
        days.add(instant.astimezone(ZoneInfo(local_tz)).date().isoformat())
        value = row.get("actual_wind_speed", "").strip().lower()
        if value and value not in {"nan", "none", "null"}:
            observations.add(instant.astimezone(timezone.utc).isoformat())
    if len(days) != 1:
        raise ValueError("daily current-day CSV must describe exactly one local day")
    return next(iter(days)), observations


def update_daily_archive(
    *, out_dir: Path, current_csv: Path, current_png: Path,
    local_tz: str = "Europe/Amsterdam", now_utc: datetime | None = None,
    render_png: Callable[[Path], None] | None = None,
) -> dict[str, str]:
    """Replace one daily PNG and CSV after every successful render.

    Use the table's target date, never wall-clock midnight or the forecast's
    issue time. Refuse a stale table that would discard archived observations.
    The canonical live files, rider files and model/evaluation files are untouched.
    """
    csv_bytes = current_csv.read_bytes()
    day, observations = csv_day_and_observations(csv_bytes, local_tz)
    now = now_utc or datetime.now(timezone.utc)
    if datetime.fromisoformat(day).date() > now.astimezone(ZoneInfo(local_tz)).date():
        raise ValueError("cannot archive a future current-day table")
    destination_csv = out_dir / "current_day_csv_archive" / f"current_day_predictions_{day}.csv"
    destination_png = out_dir / "current_day_plot_archive" / f"current_day_predictions_{day}.png"
    if destination_csv.exists():
        existing_day, existing_observations = csv_day_and_observations(destination_csv.read_bytes(), local_tz)
        if existing_day != day or not existing_observations.issubset(observations):
            return {"status": "preserved_more_complete_archive", "date": day}
    if render_png is None:
        png_bytes = current_png.read_bytes()
    else:
        with tempfile.TemporaryDirectory(prefix="wind-daily-plot-") as tmp:
            image = Path(tmp) / "daily.png"
            render_png(image)
            png_bytes = image.read_bytes()
    if not png_bytes:
        raise ValueError("cannot archive an empty plot")
    atomic_write(destination_png, png_bytes)
    atomic_write(destination_csv, csv_bytes)
    prune_closed_intraday_cache(out_dir, now.astimezone(ZoneInfo(local_tz)).date(), local_tz)
    return {"status": "updated", "date": day, "png": str(destination_png), "csv": str(destination_csv)}


def prune_closed_intraday_cache(out_dir: Path, today, local_tz: str) -> None:
    """Expire only this implementation's cache after a daily pair is retained.

    Legacy archives and unknown files are never removed by this operation.
    Today's hourly branches remain available for dashboard rendering.
    """
    cache = out_dir / "intraday_prediction_cache"
    if cache.is_symlink():
        raise ValueError("intraday cache must not be a symlink")
    if not cache.exists():
        return
    for directory in cache.iterdir():
        if directory.is_symlink() or not directory.is_dir() or not re.fullmatch(r"\d{8}", directory.name):
            continue
        try:
            day = datetime.strptime(directory.name, "%Y%m%d").date()
        except ValueError:
            continue
        if day >= today:
            continue
        daily_csv = out_dir / "current_day_csv_archive" / f"current_day_predictions_{day}.csv"
        daily_png = out_dir / "current_day_plot_archive" / f"current_day_predictions_{day}.png"
        if any(p.is_symlink() or not p.is_file() for p in [daily_csv, daily_png]):
            continue
        try:
            archived_day, archived_observations = csv_day_and_observations(daily_csv.read_bytes(), local_tz)
        except (ValueError, KeyError, UnicodeError):
            continue
        if archived_day != day.isoformat() or not archived_observations:
            continue
        for path in directory.iterdir():
            expected = path.name == "latest.csv" or re.fullmatch(
                rf"\d{{8}}-\d{{6}}_[+-]\d{{4}}_target_{directory.name}\.csv", path.name)
            if not expected or path.is_symlink() or not path.is_file():
                continue
            try:
                cached_day, cached_observations = csv_day_and_observations(path.read_bytes(), local_tz)
            except (ValueError, KeyError, UnicodeError):
                continue
            if cached_day == archived_day and cached_observations.issubset(archived_observations):
                path.unlink()
        if not any(directory.iterdir()):
            directory.rmdir()
