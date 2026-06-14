#!/usr/bin/env python3
"""Plan cleanup of redundant timestamped PNG plot archives.

By default this is a dry run: it reports old timestamped archive PNGs that are
older than the latest file for the same archive directory, local date, and plot
type. Stable daily archive names are ignored.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TIMESTAMPED_ARCHIVE_RE = re.compile(r"^(?P<date>\d{8})-(?P<time>\d{6})_(?P<plot>.+)\.png$")


def default_archive_dirs() -> list[Path]:
    artifacts = REPO_ROOT / "next_day_wind_model" / "artifacts"
    return [
        artifacts / "current_day_plot_archive",
        artifacts / "next_day_plot_archive",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan redundant timestamped PNG archive cleanup.")
    parser.add_argument(
        "--archive-dir",
        action="append",
        type=Path,
        default=None,
        help="Archive directory to scan. May be passed more than once. Defaults to current/next-day artifact archives.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete files that would otherwise be reported. Omit for dry-run planning.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print totals, not every kept/redundant file path.",
    )
    return parser.parse_args()


def collect_timestamped_archives(archive_dirs: list[Path]) -> dict[tuple[Path, str, str], list[Path]]:
    grouped: dict[tuple[Path, str, str], list[Path]] = defaultdict(list)
    for archive_dir in archive_dirs:
        if not archive_dir.exists():
            print(f"SKIP missing {archive_dir}")
            continue
        for path in archive_dir.glob("*.png"):
            match = TIMESTAMPED_ARCHIVE_RE.fullmatch(path.name)
            if not match:
                continue
            key = (archive_dir, match.group("date"), match.group("plot"))
            grouped[key].append(path)
    return grouped


def main() -> int:
    args = parse_args()
    archive_dirs = args.archive_dir or default_archive_dirs()
    grouped = collect_timestamped_archives(archive_dirs)
    keep: list[Path] = []
    redundant: list[Path] = []

    for paths in grouped.values():
        ordered = sorted(paths, key=lambda p: p.name)
        if not ordered:
            continue
        keep.append(ordered[-1])
        redundant.extend(ordered[:-1])

    redundant_size = sum(path.stat().st_size for path in redundant if path.exists())
    keep_size = sum(path.stat().st_size for path in keep if path.exists())
    mode = "DELETE" if args.delete else "DRY-RUN"
    print(f"mode={mode}")
    print(f"groups={len(grouped)} keep_files={len(keep)} redundant_files={len(redundant)}")
    print(f"keep_bytes={keep_size} redundant_bytes={redundant_size}")

    if not args.summary_only:
        for path in keep:
            print("KEEP", path)
    for path in redundant:
        if not args.summary_only:
            print(("DELETE" if args.delete else "WOULD_DELETE"), path)
        if args.delete:
            path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
