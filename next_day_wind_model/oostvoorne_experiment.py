"""Offline, read-only Oostvoorne experiment using ordinary HARMONIE forecasts.

Outputs are always experimental, never published, and never written to the
production prediction log. The supplied database is checked byte-for-byte
before and after the run.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from next_day_wind_model.data_pipeline import (
    DatasetConfig,
    _apply_standardizer,
    _fit_standardizer,
    _fit_target_scaler,
    build_all_direction_training_arrays,
    build_all_training_arrays,
    load_training_forecast_lookup,
    load_training_observations,
)
from next_day_wind_model.intraday_model import (
    build_intraday_holdout_context_split,
    build_intraday_holdout_evaluation_frame,
    save_intraday_model,
    train_intraday_model,
)
from next_day_wind_model.site_paths import (
    CHECKPOINT_SCHEMA_VERSION,
    build_artifact_manifest,
    sha256_file,
    write_artifact_manifest,
)
from next_day_wind_model.train_lstm import NextDayLSTM, TargetAwareNextDayLSTM
from next_day_wind_model.update_model_and_predict import (
    _eval_start_index,
    _predict_direction_batch,
    _predict_speed_batch,
    _save_model,
    train_with_validation,
)


SITE_ID = "oostvoorne"
FORECAST_MODEL = "HARMONIE"


def metric_pair(predicted: np.ndarray, actual: np.ndarray, *, circular: bool = False) -> dict[str, float | int]:
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    actual = np.asarray(actual, dtype=float).reshape(-1)
    valid = np.isfinite(predicted) & np.isfinite(actual)
    if not np.any(valid):
        return {"n": 0, "mae": float("nan"), "rmse": float("nan")}
    error = predicted[valid] - actual[valid]
    if circular:
        error = ((error + 180.0) % 360.0) - 180.0
    return {
        "n": int(np.sum(valid)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
    }


def day_block_bootstrap(
    frame: pd.DataFrame,
    *,
    prediction_col: str,
    baseline_col: str,
    actual_col: str,
    time_col: str,
    circular: bool = False,
    iterations: int = 1000,
    seed: int = 3407,
) -> dict[str, object]:
    work = frame[[prediction_col, baseline_col, actual_col, time_col]].dropna().copy()
    work["day"] = pd.to_datetime(work[time_col], utc=True).dt.floor("D")
    days = work["day"].drop_duplicates().to_numpy()
    if len(days) < 2:
        raise ValueError("day-block bootstrap requires at least two realized days")
    rng = np.random.default_rng(seed)
    samples: list[tuple[float, float]] = []
    groups = {day: work[work["day"] == day] for day in days}
    for _ in range(int(iterations)):
        chosen = rng.choice(days, size=len(days), replace=True)
        sample = pd.concat([groups[day] for day in chosen], ignore_index=True)
        model = metric_pair(sample[prediction_col], sample[actual_col], circular=circular)
        baseline = metric_pair(sample[baseline_col], sample[actual_col], circular=circular)
        samples.append((float(baseline["mae"]) - float(model["mae"]), float(baseline["rmse"]) - float(model["rmse"])))
    values = np.asarray(samples)
    return {
        "unit": "UTC target day",
        "days": int(len(days)),
        "iterations": int(iterations),
        "mae_improvement_model_vs_harmonie": {
            "estimate": float(metric_pair(work[baseline_col], work[actual_col], circular=circular)["mae"] - metric_pair(work[prediction_col], work[actual_col], circular=circular)["mae"]),
            "ci95": [float(value) for value in np.quantile(values[:, 0], [0.025, 0.975])],
        },
        "rmse_improvement_model_vs_harmonie": {
            "estimate": float(metric_pair(work[baseline_col], work[actual_col], circular=circular)["rmse"] - metric_pair(work[prediction_col], work[actual_col], circular=circular)["rmse"]),
            "ci95": [float(value) for value in np.quantile(values[:, 1], [0.025, 0.975])],
        },
    }


def summarize_by_horizon(frame: pd.DataFrame, buckets: list[tuple[int, int]], *, circular: bool = False) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for lower, upper in buckets:
        selected = frame[(frame["horizon_hr"] >= lower) & (frame["horizon_hr"] <= upper)]
        rows.append(
            {
                "horizon_bucket": f"{lower}-{upper}",
                "model": metric_pair(selected["prediction_value"], selected["actual_value"], circular=circular),
                "harmonie": metric_pair(selected["harmonie_value"], selected["actual_value"], circular=circular),
            }
        )
    return rows


def _flatten_next_day(
    *,
    arrays: dict,
    start: int,
    predictions: np.ndarray,
    actual: np.ndarray,
    baseline: np.ndarray,
) -> pd.DataFrame:
    target_times = np.asarray(arrays["target_times_all"])[start:]
    horizons = np.asarray(arrays["target_horizon_hr_all"])[start:]
    return pd.DataFrame(
        {
            "target_time_utc": pd.to_datetime(target_times.reshape(-1), utc=True),
            "horizon_hr": horizons.reshape(-1).astype(float),
            "prediction_value": predictions.reshape(-1).astype(float),
            "harmonie_value": baseline.reshape(-1).astype(float),
            "actual_value": actual.reshape(-1).astype(float),
        }
    ).dropna()


def _assert_read_only_database(db_path: Path) -> None:
    uri = f"file:{db_path.resolve()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("SELECT 1").fetchone()
    finally:
        connection.close()


def run_experiment(args: argparse.Namespace) -> Path:
    db_path = args.db.resolve()
    _assert_read_only_database(db_path)
    database_sha256_before = sha256_file(db_path)
    output_dir = args.output_root / SITE_ID / "experiments" / args.run_id
    if output_dir.exists():
        raise FileExistsError(f"experiment output already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")
    cfg = DatasetConfig(site=SITE_ID, model=FORECAST_MODEL, window_hours=72, target_hours=24)
    observations = load_training_observations(db_path, cfg, read_only=True)
    lookup = load_training_forecast_lookup(db_path, cfg, read_only=True)
    speed = build_all_training_arrays(
        db_path, cfg, target_mode="residual", forecast_lookup=lookup, observations=observations, read_only=True
    )
    direction = build_all_direction_training_arrays(
        db_path, cfg, forecast_lookup=lookup, observations=observations, read_only=True
    )

    speed_start = _eval_start_index(len(speed["X_all"]), args.holdout_fraction, args.min_holdout_samples)
    direction_start = _eval_start_index(len(direction["X_all"]), args.holdout_fraction, args.min_holdout_samples)
    speed_raw = speed["X_all"] * np.asarray(speed["x_std"]).reshape(1, 1, -1) + np.asarray(speed["x_mean"]).reshape(1, 1, -1)
    direction_raw = direction["X_all"] * np.asarray(direction["x_std"]).reshape(1, 1, -1) + np.asarray(direction["x_mean"]).reshape(1, 1, -1)
    eps = 0.1
    speed_target = np.log(np.asarray(speed["y_actual_all_raw"]) + eps) - np.log(np.asarray(speed["y_forecast_all_raw"]) + eps)
    direction_target = np.asarray(direction["y_all"]) * float(direction["y_std"][0]) + float(direction["y_mean"][0])

    speed_x_mean, speed_x_std = _fit_standardizer(speed_raw[:speed_start])
    speed_y_mean, speed_y_std = _fit_target_scaler(speed_target[:speed_start])
    speed_x_train = _apply_standardizer(speed_raw[:speed_start], speed_x_mean, speed_x_std).astype(np.float32)
    speed_x_eval = _apply_standardizer(speed_raw[speed_start:], speed_x_mean, speed_x_std).astype(np.float32)
    speed_y_train = ((speed_target[:speed_start] - speed_y_mean) / speed_y_std).astype(np.float32)
    speed_model = TargetAwareNextDayLSTM(speed_x_train.shape[2], speed_y_train.shape[1], history_hours=72).to(device)
    speed_model, speed_stats = train_with_validation(
        speed_model, speed_x_train, speed_y_train, args.batch_size, args.epochs, args.validation_split, "oostvoorne-speed", device
    )
    speed_predictions = _predict_speed_batch(
        speed_model, speed_x_eval, np.asarray(speed["y_forecast_all_raw"])[speed_start:], float(speed_y_mean), float(speed_y_std),
        "constrained_logratio", eps, None, None, device,
    )
    next_speed = _flatten_next_day(
        arrays=speed, start=speed_start, predictions=speed_predictions,
        actual=np.asarray(speed["y_actual_all_raw"])[speed_start:], baseline=np.asarray(speed["y_forecast_all_raw"])[speed_start:],
    )

    direction_x_mean, direction_x_std = _fit_standardizer(direction_raw[:direction_start])
    direction_y_mean, direction_y_std = _fit_target_scaler(direction_target[:direction_start])
    direction_x_train = _apply_standardizer(direction_raw[:direction_start], direction_x_mean, direction_x_std).astype(np.float32)
    direction_x_eval = _apply_standardizer(direction_raw[direction_start:], direction_x_mean, direction_x_std).astype(np.float32)
    direction_y_train = ((direction_target[:direction_start] - direction_y_mean) / direction_y_std).astype(np.float32)
    direction_model = NextDayLSTM(direction_x_train.shape[2], direction_y_train.shape[1]).to(device)
    direction_model, direction_stats = train_with_validation(
        direction_model, direction_x_train, direction_y_train, args.batch_size, args.epochs, args.validation_split, "oostvoorne-direction", device
    )
    direction_predictions = _predict_direction_batch(
        direction_model, direction_x_eval, np.asarray(direction["y_forecast_all_raw"])[direction_start:],
        float(direction_y_mean), float(direction_y_std), device,
    )
    next_direction = _flatten_next_day(
        arrays=direction, start=direction_start, predictions=direction_predictions,
        actual=np.asarray(direction["y_actual_all_raw"])[direction_start:], baseline=np.asarray(direction["y_forecast_all_raw"])[direction_start:],
    )

    intraday_train, intraday_holdout = build_intraday_holdout_context_split(
        db_path, cfg, args.holdout_fraction, args.min_intraday_holdout_contexts,
        forecast_lookup=lookup, observations=observations, read_only=True,
    )
    intraday_bundle, intraday_stats = train_intraday_model(
        db_path, cfg, device, args.intraday_epochs, args.batch_size, args.validation_split,
        contexts=intraday_train, forecast_lookup=lookup, observations=observations, read_only=True,
    )
    intraday = build_intraday_holdout_evaluation_frame(intraday_bundle, intraday_holdout, device)

    next_speed.to_csv(output_dir / "next_day_speed_rows.csv", index=False)
    next_direction.to_csv(output_dir / "next_day_direction_rows.csv", index=False)
    intraday.to_csv(output_dir / "intraday_speed_rows.csv", index=False)
    _save_model(
        output_dir / "experimental_next_day_speed.pt", speed_model, speed_x_train.shape[2], speed_y_train.shape[1],
        "wind_speed", "constrained_logratio", "linear", "TargetAwareNextDayLSTM", 72,
        {"artifact_schema_version": CHECKPOINT_SCHEMA_VERSION, "site_id": SITE_ID, "forecast_model": FORECAST_MODEL, "status": "experimental"},
    )
    _save_model(
        output_dir / "experimental_next_day_direction.pt", direction_model, direction_x_train.shape[2], direction_y_train.shape[1],
        "wind_direction", "residual", "linear", extra={
            "artifact_schema_version": CHECKPOINT_SCHEMA_VERSION, "site_id": SITE_ID, "forecast_model": FORECAST_MODEL, "status": "experimental"
        },
    )
    save_intraday_model(
        output_dir / "experimental_intraday_speed.pt", intraday_bundle,
        {"artifact_schema_version": CHECKPOINT_SCHEMA_VERSION, "site_id": SITE_ID, "forecast_model": FORECAST_MODEL, "status": "experimental"},
    )

    summary = {
        "status": "experimental",
        "production_eligible": False,
        "site_id": SITE_ID,
        "forecast_model": FORECAST_MODEL,
        "p1_features_included": False,
        "split": "chronological",
        "next_day_speed": {
            "model": metric_pair(next_speed["prediction_value"], next_speed["actual_value"]),
            "harmonie": metric_pair(next_speed["harmonie_value"], next_speed["actual_value"]),
            "by_horizon": summarize_by_horizon(next_speed, [(1, 6), (7, 12), (13, 18), (19, 24)]),
            "day_block_bootstrap": day_block_bootstrap(next_speed, prediction_col="prediction_value", baseline_col="harmonie_value", actual_col="actual_value", time_col="target_time_utc", iterations=args.bootstrap_iterations, seed=args.seed),
            "training": speed_stats,
        },
        "intraday_speed": {
            "model": metric_pair(intraday["prediction_value"], intraday["actual_value"]),
            "harmonie": metric_pair(intraday["harmonie_value"], intraday["actual_value"]),
            "by_horizon": summarize_by_horizon(intraday, [(1, 3), (4, 6), (7, 12)]),
            "day_block_bootstrap": day_block_bootstrap(intraday, prediction_col="prediction_value", baseline_col="harmonie_value", actual_col="actual_value", time_col="target_time_utc", iterations=args.bootstrap_iterations, seed=args.seed + 1),
            "training": intraday_stats,
        },
        "next_day_direction_diagnostic": {
            "model": metric_pair(next_direction["prediction_value"], next_direction["actual_value"], circular=True),
            "harmonie": metric_pair(next_direction["harmonie_value"], next_direction["actual_value"], circular=True),
            "by_horizon": summarize_by_horizon(next_direction, [(1, 6), (7, 12), (13, 18), (19, 24)], circular=True),
            "day_block_bootstrap": day_block_bootstrap(next_direction, prediction_col="prediction_value", baseline_col="harmonie_value", actual_col="actual_value", time_col="target_time_utc", circular=True, iterations=args.bootstrap_iterations, seed=args.seed + 2),
            "training": direction_stats,
        },
        "database_sha256_before": database_sha256_before,
    }
    database_sha256_after = sha256_file(db_path)
    summary["database_sha256_after"] = database_sha256_after
    summary["database_unchanged"] = database_sha256_before == database_sha256_after
    if not summary["database_unchanged"]:
        raise RuntimeError("database changed during read-only experiment")
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    manifest_files = [path for path in output_dir.iterdir() if path.is_file()]
    manifest = build_artifact_manifest(
        artifact_dir=output_dir, site_id=SITE_ID, forecast_model=FORECAST_MODEL,
        status="experimental", production_eligible=False, files=manifest_files,
        training_data_start_utc=str(pd.to_datetime(speed["timestamps"][0], utc=True)),
        training_data_end_utc=str(pd.to_datetime(speed["timestamps"][speed_start - 1], utc=True)),
        extra={"p1_features_included": False, "database_unchanged": True},
    )
    write_artifact_manifest(output_dir, manifest)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True, help="Restored backup; opened read-only")
    parser.add_argument("--output-root", type=Path, default=Path("next_day_wind_model/artifacts"))
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--intraday-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-split", type=float, default=0.15)
    parser.add_argument("--holdout-fraction", type=float, default=0.15)
    parser.add_argument("--min-holdout-samples", type=int, default=60)
    parser.add_argument("--min-intraday-holdout-contexts", type=int, default=48)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    output = run_experiment(args)
    print(json.dumps({"output": str(output), "status": "experimental", "production_eligible": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
