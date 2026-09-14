from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_pipeline import (
    DatasetConfig,
    _current_rss_mib,
    _log_rss,
    _peak_rss_mib,
    build_all_direction_training_arrays,
    build_all_training_arrays,
    load_training_forecast_lookup,
    load_training_observations,
)
from intraday_model import (
    _build_intraday_anchor_contexts,
    build_intraday_training_xy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only RSS benchmark for forecast training-data construction.",
    )
    parser.add_argument("--db", required=True, help="SQLite database path; always opened immutable/read-only.")
    parser.add_argument("--site", default="valkenburgsemeer")
    parser.add_argument("--model", default="HARMONIE")
    parser.add_argument("--window-hours", type=int, default=72)
    parser.add_argument("--target-hours", type=int, default=24)
    parser.add_argument(
        "--stage",
        choices=("loader", "next-day", "intraday", "all"),
        default="all",
    )
    parser.add_argument("--chunk-size", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_path = Path(args.db).resolve()
    cfg = DatasetConfig(
        site=args.site,
        model=args.model,
        window_hours=args.window_hours,
        target_hours=args.target_hours,
    )
    _log_rss("benchmark_process_start", stage=args.stage)

    observations = load_training_observations(db_path, cfg, read_only=True)
    forecast_lookup = load_training_forecast_lookup(
        db_path,
        cfg,
        chunk_size=args.chunk_size,
        read_only=True,
    )
    summary: dict[str, object] = {
        "stage": args.stage,
        "forecast_rows": forecast_lookup.loaded_row_count,
        "forecast_sql_queries": forecast_lookup.sql_query_count,
        "forecast_compact_mib": round(forecast_lookup.compact_nbytes / (1024.0 * 1024.0), 3),
        "observation_raw_rows": observations.attrs.get("raw_row_count"),
        "observation_hourly_rows": len(observations),
    }

    if args.stage in {"next-day", "all"}:
        speed = build_all_training_arrays(
            db_path,
            cfg,
            target_mode="residual",
            forecast_lookup=forecast_lookup,
            observations=observations,
        )
        direction = build_all_direction_training_arrays(
            db_path,
            cfg,
            forecast_lookup=forecast_lookup,
            observations=observations,
        )
        summary.update(
            {
                "speed_samples": len(speed["X_all"]),
                "speed_shape": list(speed["X_all"].shape),
                "direction_samples": len(direction["X_all"]),
                "direction_shape": list(direction["X_all"].shape),
            }
        )
        _log_rss(
            "benchmark_next_day_arrays_complete",
            speed_samples=len(speed["X_all"]),
            direction_samples=len(direction["X_all"]),
        )

    if args.stage in {"intraday", "all"}:
        contexts = _build_intraday_anchor_contexts(
            db_path,
            cfg,
            forecast_lookup=forecast_lookup,
            observations=observations,
        )
        intraday_X, intraday_y = build_intraday_training_xy(
            db_path,
            cfg,
            contexts=contexts,
        )
        summary.update(
            {
                "intraday_contexts": len(contexts),
                "intraday_shape": list(intraday_X.shape),
                "intraday_target_shape": list(intraday_y.shape),
            }
        )
        _log_rss(
            "benchmark_intraday_arrays_complete",
            contexts=len(contexts),
            samples=len(intraday_X),
        )

    summary["current_rss_mib"] = round(_current_rss_mib(), 3)
    summary["peak_rss_mib"] = round(_peak_rss_mib(), 3)
    print("TRAINING_MEMORY_BENCHMARK " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
