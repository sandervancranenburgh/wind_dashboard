#!/usr/bin/env python3
"""Reconstruct a closed day's observed PNG/CSV into an isolated staging tree.

No production outputs are overwritten and SQLite is opened read-only. The
result must be reviewed before its daily pair is promoted into the portal's
site-specific archive. This is deliberately separate from cleanup planning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def finalize_day(*, source: Path, db: Path, output: Path, site: str, day: date) -> dict:
    # Plotting only: do not import/run model training or database migrations.
    os.environ.setdefault("WIND_PLOT_RENDERER_ONLY", "1")
    import pandas as pd
    from next_day_wind_model.daily_archive import csv_day_and_observations, update_daily_archive
    from next_day_wind_model.measured_update import compose_current_day_table, load_current_day_observations, _table_csv_bytes
    from next_day_wind_model.site_registry import load_site_registry
    from next_day_wind_model.update_model_and_predict import (
        _build_current_day_plot_frame, load_current_day_prediction_history, save_current_day_plot,
        compute_current_day_table_mae,
    )

    definition = load_site_registry().site(site)
    tz = ZoneInfo(definition.timezone)
    if day >= datetime.now(tz).date():
        raise ValueError("only closed local days may be finalized")
    source = source.resolve(strict=True)
    output = output.resolve()
    production = REPO_ROOT.parent / "wind_fetcher2"
    if output.is_relative_to(source) or output == db.resolve() or output.is_relative_to(production):
        raise ValueError("finalization requires an isolated output directory")
    site_output = output / site
    csv_name = f"current_day_predictions_{day}.csv"
    png_name = f"current_day_predictions_{day}.png"
    if (site_output / "current_day_csv_archive" / csv_name).exists() or (site_output / "current_day_plot_archive" / png_name).exists():
        raise FileExistsError("staged daily output already exists; use a new staging directory")
    candidates = list((source / "current_day_snapshots").glob(f"*_target_{day:%Y%m%d}.csv"))
    candidates += [source / "current_day_predictions.csv",
                   source / "current_day_csv_archive" / csv_name,
                   source / "intraday_prediction_cache" / f"{day:%Y%m%d}" / "latest.csv"]
    valid = []
    for path in candidates:
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(source):
            continue
        try:
            target, observed = csv_day_and_observations(path.read_bytes(), definition.timezone)
        except (ValueError, KeyError, UnicodeError):
            continue
        if target == day.isoformat():
            valid.append((len(observed), path.stat().st_mtime_ns, path))
    if not valid:
        raise ValueError("no recognizable current-day forecast table; retain legacy snapshots")
    snapshot = max(valid, key=lambda entry: (entry[0], entry[1]))[2]
    end = datetime.combine(day + timedelta(days=1), time.min, tzinfo=tz).astimezone(timezone.utc) - timedelta(microseconds=1)
    observations = load_current_day_observations(db, site=site, local_timezone=definition.timezone, now_utc=end)
    if observations.empty:
        raise ValueError("canonical database has no observations for this site/day; retain old files")
    frame = compose_current_day_table(pd.read_csv(snapshot), observations, local_timezone=definition.timezone,
                                     now_utc=end, build_plot_frame=_build_current_day_plot_frame)
    csv_bytes = _table_csv_bytes(frame)
    target, archived_observations = csv_day_and_observations(csv_bytes, definition.timezone)
    expected = {instant.to_pydatetime().astimezone(timezone.utc).isoformat() for instant in observations.index}
    if target != day.isoformat() or not expected.issubset(archived_observations):
        raise ValueError("final CSV does not cover every canonical observation")
    # Reconstruction can differ from the original image because code/models
    # changed. The original last plot stays protected by the cleanup planner.
    history = load_current_day_prediction_history(source, day, definition.timezone, max_snapshots=16)
    with tempfile.TemporaryDirectory(prefix="wind-finalize-day-") as tmp:
        root = Path(tmp)
        current_csv, current_png = root / "current.csv", root / "current.png"
        current_csv.write_bytes(csv_bytes)
        save_current_day_plot(frame, current_png, definition.timezone,
                              prediction_generated_at_utc=None, prediction_updated_at_utc=None,
                              model_trained_at_utc=None,
                              plot_updated_at_utc=end.isoformat(),
                              prior_prediction_tables=history, spot_name=definition.display_name,
                              live_monitoring_metric=compute_current_day_table_mae(frame), full_local_day=True)
        archive = update_daily_archive(out_dir=site_output, current_csv=current_csv, current_png=current_png,
                                       local_tz=definition.timezone)
    report = {
        "site": site, "date": day.isoformat(), "source_snapshot": str(snapshot),
        "canonical_observations": len(expected), "all_canonical_observations_exported": True,
        "latest_observation_utc": max(expected), "reconstructed_plot": True,
        "files": {kind: {"path": archive[kind], "sha256": hashlib.sha256(Path(archive[kind]).read_bytes()).hexdigest()}
                  for kind in ["png", "csv"]},
    }
    report_path = site_output / f"daily_archive_verification_{day}.json"
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-artifacts", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--day", type=date.fromisoformat, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize_day(source=args.source_artifacts, db=args.db, output=args.output_dir,
                                  site=args.site, day=args.day), indent=2))


if __name__ == "__main__":
    main()
