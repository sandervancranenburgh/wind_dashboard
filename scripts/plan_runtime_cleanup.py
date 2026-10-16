#!/usr/bin/env python3
"""Inventory redundant plot/CSV archives. This command never deletes files.

Only named runtime archive directories are scanned. In particular, rider
uploads/analyses, live website assets, canonical databases, model checkpoints
and model-gate evidence are outside its scope.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from next_day_wind_model.daily_archive import csv_day_and_observations
from next_day_wind_model.site_registry import load_site_registry

PLOT = re.compile(r"^(\d{8})-(\d{6})_(current_day|next_day)_predictions(_mobile)?\.png$")
DAILY_PLOT = re.compile(r"^(current_day|next_day)_predictions(_mobile)?_(\d{4}-\d{2}-\d{2})\.png$")
SNAPSHOT = re.compile(r"^(?:\d{8}-\d{6}(?:_[+-]\d{4})?|dayahead)_target_(\d{8})\.csv$")


def record(path: Path, reason: str) -> dict:
    stat = path.lstat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "inode": stat.st_ino, "reason": reason}


def plan_cleanup(roots: list[Path], *, today: date, local_tz: str = "Europe/Amsterdam") -> dict:
    report: dict = {"schema_version": 1, "mode": "read-only", "today_local": today.isoformat(),
                    "roots": [str(p) for p in roots], "candidates": [], "protected": [], "blocked": []}
    def add(kind: str, path: Path, reason: str) -> None:
        report[kind].append(record(path, reason))

    def safe_file(path: Path, root: Path) -> bool:
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            add("protected", path, "symlink, non-regular file or path outside artifact root")
            return False
        return True

    for root in roots:
        if not root.exists():
            continue
        if root.is_symlink():
            raise ValueError(f"artifact root must not be a symlink: {root}")
        for name in ["current_day_plot_archive", "next_day_plot_archive"]:
            directory = root / name
            if directory.is_symlink():
                raise ValueError(f"archive directory must not be a symlink: {directory}")
            groups: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
            for path in sorted(directory.glob("*.png")):
                if not safe_file(path, root):
                    continue
                match = PLOT.fullmatch(path.name)
                daily = DAILY_PLOT.fullmatch(path.name)
                if daily:
                    try:
                        day = date.fromisoformat(daily[3])
                    except ValueError:
                        add("protected", path, "unrecognized date")
                        continue
                    if name == "next_day_plot_archive" and daily[1] == "next_day" and day < today:
                        add("candidates", path, "closed-day next-day image; not used by the rider portal")
                    else:
                        add("protected", path, "permanent daily current-day plot or active day")
                elif match and match[3] == name.removesuffix("_plot_archive"):
                    try:
                        day = datetime.strptime(match[1], "%Y%m%d").date()
                        datetime.strptime(match[2], "%H%M%S")
                    except ValueError:
                        add("protected", path, "unrecognized date or time")
                        continue
                    if day >= today:
                        add("protected", path, "current/future-day archive")
                    elif match[3] == "next_day":
                        add("candidates", path, "reconstructible next-day image; not used by the rider portal")
                    else:
                        groups[(match[1], match[3], match[4] or "")].append(path)
                else:
                    add("protected", path, "unrecognized plot name")
            for paths in groups.values():
                add("protected", paths[-1], "last timestamped plot per day/type; retain until daily promotion is verified")
                for path in paths[:-1]:
                    add("candidates", path, "superseded timestamped current-day plot")

        for name in ["current_day_snapshots", "dayahead_snapshots"]:
            directory = root / name
            if directory.is_symlink():
                raise ValueError(f"snapshot directory must not be a symlink: {directory}")
            groups = defaultdict(list)
            for path in sorted(directory.glob("*.csv")):
                if not safe_file(path, root):
                    continue
                match = SNAPSHOT.fullmatch(path.name)
                try:
                    day = datetime.strptime(match[1], "%Y%m%d").date() if match else None
                except ValueError:
                    day = None
                if day is None or day >= today:
                    add("protected", path, "unknown CSV or active prediction day")
                else:
                    groups[day].append(path)
            for day, paths in groups.items():
                if name == "dayahead_snapshots":
                    legacy = [p for p in paths if not p.name.startswith("dayahead_")]
                    first = min(legacy or paths)
                    add("protected", first, "earliest day-ahead forecast: historical MAE fallback, not a final observation CSV")
                    for path in paths:
                        if path != first:
                            add("candidates", path, "later day-ahead duplicate; evaluator uses the earliest issue")
                    continue
                daily_csv = root / "current_day_csv_archive" / f"current_day_predictions_{day}.csv"
                daily_png = root / "current_day_plot_archive" / f"current_day_predictions_{day}.png"
                observations: set[str] | None = None
                if daily_csv.is_file() and not daily_csv.is_symlink() and daily_png.is_file() and not daily_png.is_symlink():
                    try:
                        archived_day, observations = csv_day_and_observations(daily_csv.read_bytes(), local_tz)
                        if archived_day != day.isoformat() or not observations:
                            observations = None
                    except (ValueError, KeyError, UnicodeError):
                        pass
                for path in paths:
                    if observations is None:
                        add("blocked", path, "first finalize and verify the daily observed CSV plus portal PNG")
                        continue
                    try:
                        snapshot_day, old_observations = csv_day_and_observations(path.read_bytes(), local_tz)
                        if snapshot_day != day.isoformat() or not old_observations.issubset(observations):
                            raise ValueError("daily archive lacks snapshot observations")
                    except (ValueError, KeyError, UnicodeError):
                        add("blocked", path, "unrecognized CSV or observations missing from the daily export")
                    else:
                        add("candidates", path, "closed-day working snapshot; observed daily CSV/PNG protected")
    report["summary"] = {kind: {"files": len(report[kind]), "bytes": sum(p["bytes"] for p in report[kind])}
                         for kind in ["candidates", "protected", "blocked"]}
    report["excluded"] = ["rider_activity_uploads", "rider_activity_analysis", "live website outputs",
                          "canonical databases", "model checkpoints", "model-gate evaluation evidence",
                          "permanent daily CSV archives", "active intraday prediction caches"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--today", type=date.fromisoformat,
                        default=datetime.now(ZoneInfo("Europe/Amsterdam")).date())
    args = parser.parse_args()
    artifacts = REPO_ROOT / "next_day_wind_model/artifacts"
    roots = args.artifact_root or [artifacts, *(artifacts / site.site_id for site in load_site_registry().sites)]
    report = plan_cleanup(roots, today=args.today)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents overwriting another inventory or a runtime file.
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report["summary"], indent=2))
    print(f"Read-only plan: {args.output}")


if __name__ == "__main__":
    main()
