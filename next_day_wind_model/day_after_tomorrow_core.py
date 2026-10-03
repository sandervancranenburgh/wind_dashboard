"""Shared D+2 sampling, masked model training and evaluation primitives."""
from __future__ import annotations

import argparse
import copy
import html
import json
import math
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from contextlib import closing
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from next_day_wind_model import data_pipeline as dp
from next_day_wind_model.ecmwf_dashboard import MPS_TO_KNOT, load_ecmwf_plot_data
from next_day_wind_model.train_lstm import NextDayLSTM, TargetAwareNextDayLSTM
from next_day_wind_model.update_model_and_predict import (
    fit_speed_regime_calibration,
    _predict_direction_batch,
    _predict_speed_batch,
    apply_speed_regime_calibration,
    build_prediction_table,
    save_prediction_plot,
    save_wind_direction_performance_spider_plot,
)

SITE = "valkenburgsemeer"
TZ = "Europe/Amsterdam"
TARGET_HOURS = 15
HISTORY_HOURS = 72
EPS = 0.2
SEED = 42
EC_FEATURES = ["ecmwf_u_knots", "ecmwf_v_knots", "ecmwf_speed_knots", "ecmwf_gust_knots"]


def utc(value: str | datetime | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        raise ValueError("Issue time must include a timezone or UTC offset")
    return ts.tz_convert("UTC")


def calendar_targets(issue: pd.Timestamp, offset: int = 2) -> pd.DatetimeIndex:
    """Civil-calendar arithmetic, including 23/25-hour DST days."""
    day = utc(issue).tz_convert(TZ).date() + timedelta(days=offset)
    start = pd.Timestamp(datetime.combine(day, time(8)), tz=TZ)
    return pd.date_range(start, periods=TARGET_HOURS, freq="h").tz_convert("UTC")


def read_only(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def retain_snapshot_tables(path: Path, tables: tuple[str, ...]) -> None:
    """Discard unrelated tables from our local copy, never from a live source."""
    with closing(sqlite3.connect(path)) as conn:
        existing = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        for name in existing:
            if name not in tables and not name.startswith("sqlite_"):
                quoted = name.replace('"', '""')
                conn.execute(f'DROP TABLE "{quoted}"')
        conn.commit()
        conn.execute("VACUUM")


def snapshot(source: Path, destination: Path, tables: tuple[str, ...] | None = None) -> None:
    """Online backup includes committed WAL; only the destination is writable."""
    if destination.exists() or source.resolve() == destination.resolve():
        raise FileExistsError(f"Snapshot destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_suffix(destination.suffix + ".partial")
    if staging.exists():
        staging.unlink()
    with closing(read_only(source)) as src, closing(sqlite3.connect(staging)) as dst:
        src.execute("BEGIN")
        src.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()
        src.backup(dst, pages=4096, sleep=0.01)
    if tables is not None:
        retain_snapshot_tables(staging, tables)
    staging.replace(destination)


def validate_output(output: Path, db: Path, archive: Path, reference: Path | None) -> None:
    protected = [db.resolve().parent, archive.resolve().parent]
    if reference:
        protected.append(reference.resolve())
    # main is the production checkout in this repository. Refuse output anywhere
    # in that worktree, including nonstandard artifacts directories.
    result = subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=REPO_ROOT,
                            capture_output=True, text=True, check=False)
    if result.returncode == 0:
        for block in result.stdout.split("\n\n"):
            fields = dict(line.split(" ", 1) for line in block.splitlines() if " " in line)
            if fields.get("branch") == "refs/heads/main":
                protected.append(Path(fields["worktree"]).resolve())
    if any(output == path or path in output.parents for path in protected):
        raise ValueError("Output must be outside production checkout and source runtime directories")


def partial_run(lookup: dp.TrainingForecastLookup, targets: pd.DatetimeIndex,
                issue: pd.Timestamp) -> pd.DataFrame:
    """One latest available run, preserving absent targets (no vintage mixing)."""
    cutoff = dp._target_ms(issue)
    eligible = np.flatnonzero((lookup.run_available_ts <= cutoff)
                              & (lookup.run_ts[lookup.run_starts] <= cutoff))
    columns = ["run_ts", "fetched_ts", *lookup.float_columns]
    frame = pd.DataFrame(np.nan, index=targets, columns=columns)
    if not len(eligible):
        return frame
    run = int(eligible[-1])
    start, end = int(lookup.run_starts[run]), int(lookup.run_ends[run])
    # pandas 3 date_range defaults to microsecond storage; .asi8 is not
    # necessarily nanoseconds. Timestamp.value is always nanoseconds.
    target_ms = np.asarray([dp._target_ms(ts) for ts in targets], dtype=np.int64)
    positions = np.searchsorted(lookup.target_ts[start:end], target_ms)
    inside = positions < end - start
    for output_idx in np.flatnonzero(inside):
        row = start + int(positions[output_idx])
        if lookup.target_ts[row] != target_ms[output_idx] or lookup.fetched_ts[row] > cutoff:
            continue
        frame.iloc[output_idx] = [lookup.run_ts[row], lookup.fetched_ts[row],
                                 *[values[row] for values in lookup.float_columns.values()]]
    return frame


class ECMWFArchive:
    """Small in-memory archive; eligibility includes completion and row fetch times."""

    def __init__(self, path: Path | None):
        self.runs: list[tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]] = []
        self.reason = "archive unavailable"
        if path is None or not path.is_file():
            return
        with closing(read_only(path)) as conn:
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not {"forecast_collection_runs", "forecast_points"}.issubset(tables):
                self.reason = "archive tables unavailable"
                return
            runs = conn.execute("""SELECT run_time, completed_time FROM forecast_collection_runs
                WHERE site=? AND provider='ECMWF' AND model='IFS' AND status='complete'
                AND completed_time IS NOT NULL ORDER BY julianday(run_time)""", (SITE,)).fetchall()
            for run, completed in runs:
                points = pd.read_sql_query("""SELECT valid_time, fetched_time, u10_mps, v10_mps,
                    wind_gust_mps FROM forecast_points WHERE site=? AND provider='ECMWF'
                    AND model='IFS' AND run_time=? ORDER BY julianday(valid_time)""", conn,
                                           params=(SITE, run))
                points.index = pd.to_datetime(points.pop("valid_time"), utc=True)
                points["fetched_time"] = pd.to_datetime(points["fetched_time"], utc=True)
                self.runs.append((utc(run), utc(completed), points))
        self.reason = None if self.runs else "no completed runs"

    def features(self, issue: pd.Timestamp, targets: pd.DatetimeIndex) -> tuple[np.ndarray, dict]:
        values = np.full((len(targets), len(EC_FEATURES)), np.nan, dtype=np.float32)
        eligible = [run for run in self.runs if run[0] <= issue and run[1] <= issue]
        if not eligible:
            return values, {"reason": self.reason or "no completed run before issue"}
        run, completed, points = eligible[-1]
        points = points[points.fetched_time <= issue]
        # Native exact points are valid; interpolation needs two finite brackets
        # no more than three hours apart, so missing native points remain gaps.
        for i, target in enumerate(targets):
            pos = points.index.searchsorted(target)
            if pos < len(points) and points.index[pos] == target:
                interpolated = points.iloc[pos][["u10_mps", "v10_mps", "wind_gust_mps"]].to_numpy(float)
            elif pos == 0 or pos == len(points):
                continue
            else:
                left, right = points.index[pos - 1], points.index[pos]
                if right - left > pd.Timedelta(hours=3):
                    continue
                fraction = (target - left) / (right - left)
                a = points.iloc[pos - 1][["u10_mps", "v10_mps", "wind_gust_mps"]].to_numpy(float)
                b = points.iloc[pos][["u10_mps", "v10_mps", "wind_gust_mps"]].to_numpy(float)
                interpolated = a + fraction * (b - a)
            u, v, gust = interpolated * MPS_TO_KNOT
            values[i] = [u, v, np.hypot(u, v), gust]
        return values, {"run_time": run.isoformat(), "completed_time": completed.isoformat(),
                        "direction_deg": ((270 - np.degrees(np.arctan2(values[:, 1], values[:, 0]))) % 360).tolist()}


def observations(path: Path) -> tuple[pd.DataFrame, pd.Timestamp]:
    with closing(read_only(path)) as conn:
        rows = conn.execute("SELECT ts, wind_speed, wind_gust, wind_dir, payload FROM observations WHERE site=? ORDER BY ts", (SITE,)).fetchall()
    if not rows:
        raise ValueError("No observations for Valkenburgse Meer")
    records = []
    for ts, speed, gust, direction, payload_raw in rows:
        payload = json.loads(payload_raw) if payload_raw else {}
        def value(fallback, keys):
            found = dp._extract_first(payload, keys)
            return fallback if found is None else found
        speed = value(speed, ["AverageWind", "wind_speed", "WindSpeedAvg"])
        direction = value(direction, ["WindDirection", "wind_dir", "direction"])
        angle = np.deg2rad(float(direction)) if direction is not None else np.nan
        records.append((ts, speed, np.sin(angle), np.cos(angle)))
    frame = pd.DataFrame(records, columns=["ts", "actual_avg", "dir_sin", "dir_cos"])
    last = pd.to_datetime(frame.ts.max(), unit="ms", utc=True)
    frame.index = pd.to_datetime(frame.pop("ts"), unit="ms", utc=True)
    hourly = frame.resample("h").mean(numeric_only=True)
    hourly["actual_dir"] = np.degrees(np.arctan2(hourly.dir_sin, hourly.dir_cos)) % 360
    return hourly[["actual_avg", "actual_dir"]], last


@dataclass
class Sample:
    issue: pd.Timestamp
    targets: pd.DatetimeIndex
    speed_x: np.ndarray
    direction_x: np.ndarray
    frame: pd.DataFrame
    actual: np.ndarray
    actual_dir: np.ndarray
    forecast_mask: np.ndarray
    speed_mask: np.ndarray
    direction_mask: np.ndarray
    ec: np.ndarray
    ec_provenance: dict

    @property
    def day(self) -> str:
        return self.targets[0].tz_convert(TZ).date().isoformat()

    @property
    def label_end(self) -> pd.Timestamp:
        # Hourly targets aggregate observations from [hour, hour+1).
        return self.targets[-1] + pd.Timedelta(hours=1)


def make_sample(lookup: dp.TrainingForecastLookup, obs: pd.DataFrame,
                archive: ECMWFArchive, issue: pd.Timestamp, rejection: dict | None = None) -> Sample | None:
    issue = utc(issue)
    targets = calendar_targets(issue)
    history_times = pd.date_range(end=issue.floor("h") - pd.Timedelta(hours=1), periods=HISTORY_HOURS, freq="h")
    # Only completed history hours are used, not the unfinished issue hour.
    history = dp._build_training_history_forecast_frame(lookup, history_times, dp._target_ms(issue))
    if history is None:
        if rejection is not None:
            rejection["reason"] = "missing_cutoff_eligible_72h_history"
        return None
    frame = partial_run(lookup, targets, issue)
    forecast_mask = np.isfinite(frame[["forecast_avg", "forecast_dir"]].to_numpy()).all(axis=1)
    if not forecast_mask.any():
        if rejection is not None:
            rejection["reason"] = "no_cutoff_eligible_D2_targets"
        return None
    frame["forecast_max"] = frame.forecast_max.fillna(frame.forecast_avg)
    padded = frame.copy()
    # Neutral input padding is accompanied by an availability feature. It is
    # never exposed as a baseline, target, prediction, or interpolation.
    padded.loc[~forecast_mask, list(lookup.float_columns)] = 0.0
    padded["horizon_hr"] = (targets - issue).total_seconds() / 3600
    built = dp._build_feature_sequence(history, padded, feature_schema="speed_v2")
    direction = dp._build_feature_sequence(history, padded, feature_schema="direction_v2")
    if built is None or direction is None:
        if rejection is not None:
            rejection["reason"] = "missing_required_features"
        return None
    speed_x = np.column_stack([built[0], np.r_[np.ones(HISTORY_HOURS), forecast_mask.astype(float)]]).astype(np.float32)
    labels = obs.reindex(targets)
    actual = labels.actual_avg.to_numpy(np.float32)
    actual_dir = labels.actual_dir.to_numpy(np.float32)
    ec, provenance = archive.features(issue, targets)
    return Sample(issue, targets, speed_x, direction[0], frame, actual, actual_dir,
                  forecast_mask, forecast_mask & np.isfinite(actual) & (actual >= 0),
                  forecast_mask & np.isfinite(actual_dir), ec, provenance)


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Index before arithmetic so NaNs in excluded labels cannot poison gradients.
    if not bool(mask.any()):
        raise ValueError("A training batch has no supervised target hours")
    return (prediction[mask] - target[mask]).square().mean()


def purged_split(samples: list[Sample], cutoff: pd.Timestamp, validation_fraction: float = .2) -> tuple[list[Sample], list[Sample]]:
    eligible = [s for s in samples if s.label_end <= cutoff and s.issue < cutoff]
    dates = sorted({s.day for s in eligible})
    if len(dates) < 3:
        raise ValueError("At least three completed target dates are required for fitting")
    count = max(1, int(math.ceil(len(dates) * validation_fraction)))
    validation_dates = set(dates[-count:])
    validation = [s for s in eligible if s.day in validation_dates]
    # The last training label must exist before even the earliest validation issue.
    boundary = min(s.issue for s in validation)
    training = [s for s in eligible if s.day not in validation_dates and s.label_end <= boundary]
    if not training or not validation:
        raise ValueError("Insufficient history after purging overlapping target dates")
    return training, validation


def input_array(samples: list[Sample], kind: str, ecmwf: bool) -> np.ndarray:
    array = np.stack([s.direction_x if kind == "direction" else s.speed_x for s in samples])
    if ecmwf:
        extra = np.stack([np.vstack([np.zeros((HISTORY_HOURS, len(EC_FEATURES))), s.ec]) for s in samples])
        array = np.concatenate([array, extra.astype(np.float32)], axis=2)
    return array.astype(np.float32)


def target_array(samples: list[Sample], kind: str, constraint_eps: float = EPS) -> tuple[np.ndarray, np.ndarray]:
    baseline_col = "forecast_dir" if kind == "direction" else "forecast_avg"
    baseline = np.stack([s.frame[baseline_col].to_numpy(np.float32) for s in samples])
    actual = np.stack([s.actual_dir if kind == "direction" else s.actual for s in samples])
    mask = np.stack([s.direction_mask if kind == "direction" else s.speed_mask for s in samples])
    if kind == "direction":
        target = (actual - baseline + 180) % 360 - 180
    else:
        target = np.log(actual + constraint_eps) - np.log(baseline + constraint_eps)
    return np.where(mask, target, 0).astype(np.float32), mask


def calibration_context(samples: list[Sample]) -> dict:
    return {"anchor_dir_deg": np.asarray([np.degrees(np.arctan2(s.speed_x[HISTORY_HOURS - 1, 2], s.speed_x[HISTORY_HOURS - 1, 3])) % 360 for s in samples]),
            "target_month": np.asarray([s.targets[0].month for s in samples]),
            "target_forecast_dir_deg": np.stack([s.frame.forecast_dir.to_numpy(float) for s in samples]),
            "target_times_utc": np.stack([s.targets.astype(str).to_numpy() for s in samples]),
            "target_horizon_hr": np.stack([s.frame.horizon_hr.to_numpy(float) for s in samples])}


@dataclass
class Fit:
    model: nn.Module
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float
    kind: str
    ecmwf: bool
    calibration: dict | None
    info: dict


def predict(fit: Fit, samples: list[Sample], calibrate: bool = True) -> np.ndarray:
    raw = input_array(samples, fit.kind, fit.ecmwf)
    scaled = (raw - fit.x_mean) / fit.x_std
    column = "forecast_dir" if fit.kind == "direction" else "forecast_avg"
    baseline = np.stack([s.frame[column].to_numpy(np.float32) for s in samples])
    batches = []
    for i in range(0, len(samples), 256):
        if fit.kind == "direction":
            batch = _predict_direction_batch(fit.model, scaled[i:i + 256], baseline[i:i + 256],
                                             fit.y_mean, fit.y_std, torch.device("cpu"))
        else:
            batch = _predict_speed_batch(fit.model, scaled[i:i + 256], baseline[i:i + 256],
                fit.y_mean, fit.y_std, "constrained_logratio", fit.info.get("constraint_eps", EPS), None, None, torch.device("cpu"))
        batches.append(batch)
    result = np.concatenate(batches)
    if fit.kind == "speed" and calibrate and fit.calibration:
        result = apply_speed_regime_calibration(result, baseline, fit.calibration, calibration_context(samples),
            target_mask=np.stack([s.forecast_mask for s in samples]))
    return np.where(np.stack([s.forecast_mask for s in samples]), result, np.nan)


def fit_model(samples: list[Sample], cutoff: pd.Timestamp, kind: str, ecmwf: bool,
              epochs: int, batch_size: int, seed: int, constraint_eps: float = EPS, calibration_policy: str = "legacy") -> Fit:
    if calibration_policy not in {"none", "legacy"}:
        raise ValueError("Unknown D+2 calibration policy")
    training, validation = purged_split(samples, cutoff)
    torch.manual_seed(seed)
    np.random.seed(seed)
    x_train, x_val = input_array(training, kind, ecmwf), input_array(validation, kind, ecmwf)
    x_mean, x_std = dp._fit_standardizer(x_train)
    y_train, mask_train = target_array(training, kind, constraint_eps)
    y_val, mask_val = target_array(validation, kind, constraint_eps)
    if not mask_train.any() or not mask_val.any():
        raise ValueError(f"Insufficient supervised {kind} training/validation targets")
    y_mean = float(y_train[mask_train].mean())
    y_std = float(y_train[mask_train].std()) or 1.0
    train_data = TensorDataset(torch.from_numpy((x_train - x_mean) / x_std),
                               torch.from_numpy((y_train - y_mean) / y_std), torch.from_numpy(mask_train))
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    val_x = torch.from_numpy((x_val - x_mean) / x_std)
    val_y, val_mask = torch.from_numpy((y_val - y_mean) / y_std), torch.from_numpy(mask_val)
    model = (NextDayLSTM(x_train.shape[-1], TARGET_HOURS) if kind == "direction"
             else TargetAwareNextDayLSTM(x_train.shape[-1], TARGET_HOURS, HISTORY_HOURS))
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=4)
    best, best_state, stale = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        for x, y, mask in loader:
            optimizer.zero_grad()
            loss = masked_mse(model(x), y, mask)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            # Accumulate masked SSE/count rather than averaging unequal batches.
            error, count = 0., 0
            for i in range(0, len(validation), 256):
                mask = val_mask[i:i + 256]
                pred = model(val_x[i:i + 256])
                error += float((pred[mask] - val_y[i:i + 256][mask]).square().sum())
                count += int(mask.sum())
            val_loss = error / count
        scheduler.step(val_loss)
        if val_loss < best - 1e-9:
            best, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= 8:
            break
    model.load_state_dict(best_state)
    info = {"cutoff_utc": cutoff.isoformat(), "max_label_end_utc": max(s.label_end for s in training + validation).isoformat(),
            "training_dates": sorted({s.day for s in training}), "validation_dates": sorted({s.day for s in validation}),
            "training_samples": len(training), "validation_samples": len(validation),
            "epochs_ran": epoch + 1, "best_validation_loss": best, "seed": seed,
            "model_class": type(model).__name__, "ecmwf": ecmwf, "kind": kind,
            "constraint_eps": constraint_eps, "batch_size": batch_size, "max_epochs": epochs, "calibration_policy": calibration_policy if kind == "speed" else "none"}
    fit = Fit(model, x_mean, x_std, y_mean, y_std, kind, ecmwf, None, info)
    if kind == "speed" and calibration_policy == "legacy":
        # The same three-method selector as next-day, preserving window-level
        # signals and excluding missing target observations from every fit.
        pred = predict(fit, validation, calibrate=False)
        mask = np.stack([s.speed_mask for s in validation])
        baseline = np.stack([s.frame.forecast_avg.to_numpy(float) for s in validation])
        actual = np.stack([s.actual for s in validation])
        selection = {}
        fit.calibration = fit_speed_regime_calibration(pred, baseline, actual,
            calibration_context(validation), signal="pred_max", target_mask=mask, diagnostics=selection)
        fit.info["calibration_training"] = {"dates": sorted({s.day for s in validation}),
            "months": sorted({s.day[:7] for s in validation}), "contexts": len(validation),
            "matched_hours": int(mask.sum()), "selection_rule": "lowest calibration-fitting MAE among improving candidates",
            "validation_reused_for_epoch_selection": True,
            "methods": ["threshold_v1", "contextual_linear_v2", "target_hour_ridge_v1"], "selection": selection}
    return fit


def save_fit(fit: Fit, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"experiment": "day_after_tomorrow", "state_dict": fit.model.state_dict(),
                "history_hours": HISTORY_HOURS, "target_hours": TARGET_HOURS,
                "feature_schema": "direction_v2" if fit.kind == "direction" else "speed_v2_d2_masked",
                "ecmwf_feature_names": EC_FEATURES if fit.ecmwf else [],
                "n_features": len(fit.x_mean), "target_mode": "residual" if fit.kind == "direction" else "constrained_logratio",
                "constraint_eps": fit.info.get("constraint_eps", EPS), "x_mean": torch.from_numpy(fit.x_mean), "x_std": torch.from_numpy(fit.x_std),
                "y_mean": fit.y_mean, "y_std": fit.y_std, "calibration": fit.calibration,
                "calibration_policy": fit.info.get("calibration_policy", "legacy" if getattr(fit, "calibration", None) else "none"),
                "training": fit.info}, path)
    np.savez(path.with_suffix(".scalers.npz"), x_mean=fit.x_mean, x_std=fit.x_std,
             y_mean=fit.y_mean, y_std=fit.y_std)


def load_fit(path: Path) -> Fit:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("experiment") != "day_after_tomorrow":
        raise ValueError("Not a D+2 experimental checkpoint")
    info = checkpoint["training"]
    model = (NextDayLSTM(checkpoint["n_features"], TARGET_HOURS) if info["kind"] == "direction"
             else TargetAwareNextDayLSTM(checkpoint["n_features"], TARGET_HOURS, HISTORY_HOURS))
    model.load_state_dict(checkpoint["state_dict"])
    return Fit(model, checkpoint["x_mean"].numpy(), checkpoint["x_std"].numpy(),
               checkpoint["y_mean"], checkpoint["y_std"], info["kind"], info["ecmwf"],
               checkpoint["calibration"], info)


def prediction_rows(samples: list[Sample], predictions: np.ndarray, fit: Fit | None, experiment: str,
                    direction: np.ndarray | None = None) -> list[dict]:
    rows = []
    for sample, values, sample_idx in zip(samples, predictions, range(len(samples))):
        for i, target in enumerate(sample.targets):
            f = sample.frame.iloc[i]
            rows.append({"experiment": experiment, "issue_time_utc": sample.issue.isoformat(),
                         "target_time_utc": target.isoformat(), "target_date": sample.day,
                         "issue_hour": sample.issue.tz_convert(TZ).hour, "target_hour": target.tz_convert(TZ).hour,
                         "lead_hours": (target - sample.issue).total_seconds() / 3600,
                         "harmonie_run_ts": f.run_ts, "harmonie_fetched_ts": f.fetched_ts,
                         "harmonie_horizon_hr": f.horizon_hr,
                         "ecmwf_run_time": sample.ec_provenance.get("run_time"),
                         "ecmwf_completed_time": sample.ec_provenance.get("completed_time"),
                         "model_training_cutoff_utc": None if fit is None else fit.info["cutoff_utc"],
                         "calibration_policy": None if fit is None else fit.info.get("calibration_policy", "legacy" if getattr(fit, "calibration", None) else "none"),
                         "model_latest_label_end_utc": None if fit is None else fit.info["max_label_end_utc"],
                         "forecast_available": bool(sample.forecast_mask[i]),
                         "scorable": bool(sample.speed_mask[i]),
                         "direction_scorable": bool(sample.direction_mask[i]),
                         "complete_window": bool(sample.speed_mask.all()),
                         "prediction": values[i], "actual": sample.actual[i], "harmonie": f.forecast_avg,
                         "ecmwf": sample.ec[i, 2], "harmonie_direction": f.forecast_dir,
                         "actual_direction": sample.actual_dir[i],
                         "prediction_direction": np.nan if direction is None else direction[sample_idx, i]})
    return rows


def save_samples(samples: list[Sample], path: Path) -> None:
    """Numeric/string cache only; never deserialize pickle or executable objects."""
    np.savez_compressed(path, issues=np.array([s.issue.isoformat() for s in samples]),
        speed_x=np.stack([s.speed_x for s in samples]), direction_x=np.stack([s.direction_x for s in samples]),
        frame_columns=np.array(samples[0].frame.columns, dtype=str), frames=np.stack([s.frame.to_numpy(float) for s in samples]),
        actual=np.stack([s.actual for s in samples]), actual_dir=np.stack([s.actual_dir for s in samples]),
        forecast_mask=np.stack([s.forecast_mask for s in samples]), speed_mask=np.stack([s.speed_mask for s in samples]),
        direction_mask=np.stack([s.direction_mask for s in samples]), ec=np.stack([s.ec for s in samples]),
        ec_provenance=np.array([json.dumps(s.ec_provenance) for s in samples]))


def load_samples(path: Path) -> list[Sample]:
    with np.load(path, allow_pickle=False) as archive:
        cache = {name: archive[name] for name in archive.files}
    samples = []
    for i, value in enumerate(cache["issues"]):
        issue = utc(str(value))
        targets = calendar_targets(issue)
        frame = pd.DataFrame(cache["frames"][i], index=targets, columns=cache["frame_columns"])
        samples.append(Sample(issue, targets, cache["speed_x"][i], cache["direction_x"][i], frame,
            cache["actual"][i], cache["actual_dir"][i], cache["forecast_mask"][i], cache["speed_mask"][i],
            cache["direction_mask"][i], cache["ec"][i], json.loads(str(cache["ec_provenance"][i]))))
    return samples


def metric(prediction: np.ndarray, actual: np.ndarray, circular: bool = False) -> dict:
    error = np.asarray(prediction, float) - np.asarray(actual, float)
    error = error[np.isfinite(error)]
    if circular:
        error = (error + 180) % 360 - 180
    return {"count": len(error), "mae": float(np.abs(error).mean()) if len(error) else None,
            "rmse": float(np.sqrt(np.square(error).mean())) if len(error) else None,
            "bias": float(error.mean()) if len(error) else None}


def bootstrap(frame: pd.DataFrame, prediction: str, baseline: str, iterations: int = 2000,
              actual_column: str = "actual", circular: bool = False) -> dict:
    work = frame[["target_date", prediction, baseline, actual_column]].dropna()
    if work.empty:
        return {"days": 0, "ci95": None, "estimate": None}
    baseline_error, model_error = work[baseline] - work[actual_column], work[prediction] - work[actual_column]
    if circular:
        baseline_error, model_error = (baseline_error + 180) % 360 - 180, (model_error + 180) % 360 - 180
    work = work.assign(delta=np.abs(baseline_error) - np.abs(model_error))
    groups = work.groupby("target_date").delta.agg(["sum", "count"])
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(groups), size=(iterations, len(groups)))
    estimates = groups["sum"].to_numpy()[picks].sum(axis=1) / groups["count"].to_numpy()[picks].sum(axis=1)
    return {"days": len(groups), "unit": "local target date", "iterations": iterations,
            "estimate": float(work.delta.mean()), "ci95": np.quantile(estimates, [.025, .975]).tolist()}


def summarize(frame: pd.DataFrame) -> dict:
    work = frame[frame.scorable].dropna(subset=["prediction", "harmonie", "actual"])
    result = {name: metric(work[name], work.actual) for name in ["prediction", "harmonie", "ecmwf"]}
    ec_matched = work.dropna(subset=["ecmwf"])
    result["ecmwf_matched_subset"] = {"target_dates": int(ec_matched.target_date.nunique()),
        **{name: metric(ec_matched[name], ec_matched.actual) for name in ["prediction", "harmonie", "ecmwf"]}}
    result["date_bootstrap"] = bootstrap(work, "prediction", "harmonie")
    baseline_mae, model_mae = result["harmonie"]["mae"], result["prediction"]["mae"]
    improvement = (baseline_mae - model_mae) / baseline_mae if baseline_mae else None
    result["relative_mae_improvement"] = improvement
    interval = result["date_bootstrap"]["ci95"]
    result["assessment"] = ("promising" if improvement is not None and improvement >= .01 and interval[0] > 0
                            else "worse" if improvement is not None and improvement < 0 else "inconclusive")
    result["direction"] = {name: metric(work[name], work.actual_direction, circular=True)
                           for name in ["prediction_direction", "harmonie_direction"]}
    result["complete_window"] = {name: metric(work.loc[work.complete_window, name], work.loc[work.complete_window, "actual"])
                                  for name in ["prediction", "harmonie", "ecmwf"]}
    result["target_dates"] = int(work.target_date.nunique())
    return result


def export_direction_performance(frame: pd.DataFrame, output: Path) -> pd.DataFrame:
    """Match the next-day gate: average overlapping issues per target hour first.

    Sectors use the circular mean of HARMONIE forecast directions, not realised
    or corrected directions. Only matched, scorable speed forecasts contribute.
    The champion columns retain the shared next-day renderer's CSV contract;
    here they describe the experimental dedicated D+2 model.
    """
    columns = ["sector", "n_points", "forecast_mae", "champion_mae",
               "forecast_bias_pred_minus_actual", "champion_bias_pred_minus_actual",
               "champion_mae_gain_vs_forecast"]
    numeric = ["actual", "harmonie", "prediction", "harmonie_direction"]
    work = frame.loc[frame.scorable, ["target_time_utc", *numeric]].copy()
    work[numeric] = work[numeric].apply(pd.to_numeric, errors="coerce")
    work = work.loc[np.isfinite(work[numeric]).all(axis=1)].dropna(subset=["target_time_utc"])
    targets = work.groupby("target_time_utc").mean(numeric_only=True)
    def circular_mean(values):
        radians = np.deg2rad(values.to_numpy(dtype=float))
        return np.rad2deg(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())) % 360
    targets["harmonie_direction"] = work.groupby("target_time_utc").harmonie_direction.agg(circular_mean)
    targets["n_overlaps"] = work.groupby("target_time_utc").size()
    sectors = np.array(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    indices = np.floor(((targets.harmonie_direction.to_numpy() % 360) + 22.5) / 45).astype(int) % 8
    targets["sector"] = sectors[indices]
    rows = []
    for sector in sectors:
        group = targets[targets.sector == sector]
        baseline = metric(group.harmonie, group.actual)
        model = metric(group.prediction, group.actual)
        rows.append(dict(zip(columns, [sector, len(group), baseline["mae"], model["mae"],
            baseline["bias"], model["bias"],
            None if group.empty else baseline["mae"] - model["mae"]])))
    summary = pd.DataFrame(rows, columns=columns)
    csv_path = output / "day_after_tomorrow_speed_by_direction.csv"
    summary.to_csv(csv_path, index=False)
    targets.reset_index().to_csv(output / "day_after_tomorrow_direction_eval_details.csv", index=False)
    png_path = output / "day_after_tomorrow_direction_spider.png"
    # Remove stale images when a new evaluation has fewer than three sectors.
    png_path.unlink(missing_ok=True)
    save_wind_direction_performance_spider_plot(csv_path, png_path,
        model_label="Super local model day after tomorrow (experimental)",
        title="MAE for D+2 models by forecast wind direction", annotate_overflow=True)
    dashboard = output / "dashboard"
    dashboard.mkdir(exist_ok=True)
    import shutil
    shutil.copy2(csv_path, dashboard / csv_path.name)
    (dashboard / png_path.name).unlink(missing_ok=True)
    if png_path.exists():
        shutil.copy2(png_path, dashboard / png_path.name)
    return summary
