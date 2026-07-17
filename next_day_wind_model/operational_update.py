from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo


GATE_CHILD_ENV = "WIND_PIPELINE_OPERATIONAL_GATE_CHILD"
STATE_SCHEMA_VERSION = 1
FINGERPRINT_SCHEMA_VERSION = 1
FORECAST_SOURCE = "windsurfice"

MODEL_ARTIFACT_NAMES = (
    "next_day_lstm_speed_residual.pt",
    "next_day_lstm_direction_residual.pt",
    "intraday_speed_residual.pt",
    "x_mean_speed.npy",
    "x_std_speed.npy",
    "y_mean_speed.npy",
    "y_std_speed.npy",
    "x_mean_direction.npy",
    "x_std_direction.npy",
    "y_mean_direction.npy",
    "y_std_direction.npy",
)

FORECAST_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "wind_avg": (
        "wind_avg",
        "WindForecastAvr",
        "wind_speed",
        "windspeed",
        "WS",
        "ff",
        "speed",
        "_db_wind_speed",
    ),
    "wind_min": (
        "wind_min",
        "WindForecastMin",
        "windspeed_min",
        "WS_min",
        "ff_min",
        "speed_min",
    ),
    "wind_max": (
        "wind_max",
        "WindForecastMax",
        "wind_gust",
        "gust",
        "WG",
        "fg",
        "_db_wind_gust",
    ),
    "wind_dir": (
        "wind_dir",
        "WindDirection",
        "winddirection",
        "WD",
        "DD",
        "dir",
        "direction",
        "_db_wind_dir",
    ),
    "temperature": ("temperature", "Temperature", "temp", "air_temperature"),
    "pressure": ("pressure", "Pressure", "msl_pressure", "mslp"),
    "rain": ("rain", "Rain", "precipitation", "precipitation_rate", "total_precipitation_rate"),
    "rh": ("rh", "RH", "relative_humidity"),
    "clouds": ("clouds", "Clouds", "cloud_cover"),
    "low_cloud_cover": ("low_cloud_cover",),
    "medium_cloud_cover": ("medium_cloud_cover",),
    "high_cloud_cover": ("high_cloud_cover",),
    "cloud_base": ("cloud_base",),
    "global_radiation": ("global_radiation",),
}

CURRENT_PREDICTION_COLUMNS = (
    "forecast_wind_speed",
    "forecast_wind_min",
    "forecast_wind_max",
    "forecast_wind_dir_deg",
    "lstm_pred_wind_speed_full",
    "lstm_pred_wind_dir_deg_full",
    "lstm_pred_wind_speed",
    "lstm_pred_wind_dir_deg",
)

NEXT_PREDICTION_COLUMNS = (
    "forecast_wind_speed",
    "forecast_wind_min",
    "forecast_wind_max",
    "lstm_pred_wind_speed",
    "forecast_wind_dir_deg",
    "lstm_pred_wind_dir_deg",
)


class OperationalGateError(RuntimeError):
    """Raised when the cheap gate cannot safely identify its inputs."""


@dataclass(frozen=True)
class ForecastIdentity:
    fingerprint: str
    run_ts: int
    fetched_ts: int
    source_run_ts: int | None
    row_count: int


@dataclass(frozen=True)
class CachedArtifactStatus:
    valid: bool
    fingerprint: str | None
    reason: str


@dataclass(frozen=True)
class OperationalSnapshot:
    site: str
    model: str
    observation_max_ts: int | None
    forecast: ForecastIdentity | None
    model_fingerprint: str | None
    cached_artifacts: CachedArtifactStatus


@dataclass(frozen=True)
class ExecutionDecision:
    mode: str
    reason: str
    run_full_pipeline: bool
    run_measured_only: bool


def _first_case_insensitive(record: Mapping[str, Any], aliases: Sequence[str]) -> Any:
    lower = {str(key).lower(): key for key in record}
    for alias in aliases:
        key = lower.get(alias.lower())
        if key is not None and record.get(key) is not None:
            return record.get(key)
    return None


def normalize_timestamp_ms(value: Any) -> int:
    if value is None or isinstance(value, bool):
        raise ValueError("timestamp is missing or invalid")
    if isinstance(value, (int, float, Decimal)):
        try:
            number = Decimal(str(value))
        except InvalidOperation as exc:
            raise ValueError("timestamp is not numeric") from exc
        if not number.is_finite():
            raise ValueError("timestamp is not finite")
        if abs(number) < Decimal("100000000000"):
            number *= 1000
        integral = number.to_integral_value()
        if number != integral:
            raise ValueError("timestamp does not resolve to an integer millisecond")
        return int(integral)

    text = str(value).strip()
    if not text:
        raise ValueError("timestamp is empty")
    try:
        return normalize_timestamp_ms(Decimal(text))
    except (InvalidOperation, ValueError):
        pass
    iso_text = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(iso_text)
    except ValueError as exc:
        raise ValueError(f"timestamp is not a supported ISO value: {text!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed_utc = parsed.astimezone(timezone.utc)
    return int(round(parsed_utc.timestamp() * 1000.0))


def canonical_number(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("boolean is not a forecast numeric value")
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, AttributeError) as exc:
        raise ValueError(f"invalid numeric value: {value!r}") from exc
    if not number.is_finite():
        raise ValueError("non-finite forecast numeric value")
    if number == 0:
        return "0"
    normalized = number.normalize()
    text = format(normalized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _semantic_forecast_row(record: Mapping[str, Any]) -> dict[str, Any]:
    target_value = _first_case_insensitive(
        record,
        ("target_ts", "timestamp", "time", "UnixTime", "ts", "dt"),
    )
    row: dict[str, Any] = {"target_ts": normalize_timestamp_ms(target_value)}
    for field, aliases in FORECAST_FIELD_ALIASES.items():
        row[field] = canonical_number(_first_case_insensitive(record, aliases))
    return row


def compute_forecast_fingerprint(
    rows: Iterable[Mapping[str, Any]],
    *,
    site: str,
    model: str,
    source: str = FORECAST_SOURCE,
    source_run_ts: Any | None = None,
) -> str:
    """Return a stable SHA-256 identity for one operational forecast payload.

    Fetch time, transport metadata, payload formatting, JSON key order, and row
    order are deliberately excluded. An authoritative source run is included
    when the caller can distinguish it from the local fallback fetch time.
    """
    by_target: dict[int, dict[str, Any]] = {}
    for raw in rows:
        semantic = _semantic_forecast_row(raw)
        target_ts = int(semantic["target_ts"])
        previous = by_target.get(target_ts)
        if previous is not None and previous != semantic:
            raise ValueError(f"conflicting duplicate forecast target: {target_ts}")
        by_target[target_ts] = semantic
    if not by_target:
        raise ValueError("forecast payload has no usable rows")

    document: dict[str, Any] = {
        "schema": FINGERPRINT_SCHEMA_VERSION,
        "site": str(site).strip().lower(),
        "source": str(source).strip().lower(),
        "model": str(model).strip().upper(),
        "rows": [by_target[target] for target in sorted(by_target)],
    }
    if source_run_ts is not None:
        document["source_run_ts"] = normalize_timestamp_ms(source_run_ts)
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _readonly_connection(db_path: Path) -> sqlite3.Connection:
    resolved = db_path.resolve()
    return sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)


def latest_observation_timestamp(db_path: Path, site: str) -> int | None:
    with _readonly_connection(db_path) as conn:
        row = conn.execute(
            "SELECT MAX(ts) FROM observations WHERE site = ? AND ts IS NOT NULL",
            (site,),
        ).fetchone()
    return None if row is None or row[0] is None else int(row[0])


def load_latest_forecast_identity(
    db_path: Path,
    *,
    site: str,
    model: str,
    min_target_rows: int = 24,
) -> ForecastIdentity:
    """Hash only the newest stored run, never the all-vintage history."""
    with _readonly_connection(db_path) as conn:
        latest = conn.execute(
            "SELECT MAX(run_ts) FROM forecasts WHERE site = ? AND model = ?",
            (site, model),
        ).fetchone()
        if latest is None or latest[0] is None:
            raise OperationalGateError("no forecast rows found")
        run_ts = int(latest[0])
        db_rows = conn.execute(
            """
            SELECT fetched_ts, target_ts, wind_speed, wind_gust, wind_dir, payload
            FROM forecasts
            WHERE site = ? AND model = ? AND run_ts = ?
            ORDER BY target_ts
            """,
            (site, model, run_ts),
        ).fetchall()
    if len(db_rows) < int(min_target_rows):
        raise OperationalGateError(
            f"latest forecast run is incomplete: {len(db_rows)} rows < {int(min_target_rows)}"
        )

    records: list[dict[str, Any]] = []
    fetched_values: list[int] = []
    for fetched_ts, target_ts, wind_speed, wind_gust, wind_dir, payload_raw in db_rows:
        payload: dict[str, Any] = {}
        if payload_raw:
            try:
                loaded = json.loads(payload_raw)
            except json.JSONDecodeError as exc:
                raise OperationalGateError("latest forecast payload contains invalid JSON") from exc
            if isinstance(loaded, dict):
                payload = loaded
        record = dict(payload)
        record["target_ts"] = int(target_ts)
        record["_db_wind_speed"] = wind_speed
        record["_db_wind_gust"] = wind_gust
        record["_db_wind_dir"] = wind_dir
        semantic = _semantic_forecast_row(record)
        if semantic["wind_avg"] is None or semantic["wind_dir"] is None:
            raise OperationalGateError("latest forecast run lacks required wind speed or direction")
        records.append(record)
        fetched_values.append(int(fetched_ts))

    fetched_ts = max(fetched_values)
    source_run_ts = run_ts if run_ts != fetched_ts else None
    fingerprint = compute_forecast_fingerprint(
        records,
        site=site,
        model=model,
        source_run_ts=source_run_ts,
    )
    return ForecastIdentity(
        fingerprint=fingerprint,
        run_ts=run_ts,
        fetched_ts=fetched_ts,
        source_run_ts=source_run_ts,
        row_count=len(records),
    )


def compute_model_fingerprint(model_artifact_dir: Path) -> tuple[str | None, tuple[str, ...]]:
    missing = tuple(name for name in MODEL_ARTIFACT_NAMES if not (model_artifact_dir / name).is_file())
    if missing:
        return None, missing
    digest = hashlib.sha256()
    digest.update(b"wind-model-artifacts-v1\0")
    for name in MODEL_ARTIFACT_NAMES:
        path = model_artifact_dir / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    return digest.hexdigest(), ()


def _truthy_csv(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def _csv_prediction_projection(path: Path, *, current_day: bool) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"cached CSV has no header: {path}")
        timestamp_column = "time_local" if current_day else "target_time_utc"
        value_columns = CURRENT_PREDICTION_COLUMNS if current_day else NEXT_PREDICTION_COLUMNS
        required = {timestamp_column, *value_columns}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"cached CSV is missing columns: {', '.join(sorted(missing))}")
        projected: list[dict[str, Any]] = []
        for row in reader:
            if current_day and "is_forecast_grid" in row and not _truthy_csv(row.get("is_forecast_grid")):
                continue
            projected.append(
                {
                    "timestamp": normalize_timestamp_ms(row[timestamp_column]),
                    **{column: canonical_number(row.get(column)) for column in value_columns},
                }
            )
    if not projected:
        raise ValueError(f"cached CSV has no usable prediction rows: {path}")
    return projected


def validate_cached_prediction_artifacts(
    out_dir: Path,
    *,
    local_timezone: str,
    web_out_dir: Path | None = None,
    now_utc: datetime | None = None,
) -> CachedArtifactStatus:
    paths = {
        "current_csv": out_dir / "current_day_predictions.csv",
        "next_csv": out_dir / "next_day_predictions.csv",
        "metadata": out_dir / "metadata_update.json",
        "current_png": out_dir / "current_day_predictions.png",
        "current_mobile_png": out_dir / "current_day_predictions_mobile.png",
        "next_png": out_dir / "next_day_predictions.png",
        "next_mobile_png": out_dir / "next_day_predictions_mobile.png",
    }
    if web_out_dir is not None:
        paths.update(
            {
                "web_index": web_out_dir / "index.html",
                "web_metadata": web_out_dir / "metadata_update.json",
                "web_current_csv": web_out_dir / "current_day_predictions.csv",
                "web_next_csv": web_out_dir / "next_day_predictions.csv",
                "web_current_png": web_out_dir / "current_day_predictions.png",
                "web_current_mobile_png": web_out_dir / "current_day_predictions_mobile.png",
                "web_next_png": web_out_dir / "next_day_predictions.png",
                "web_next_mobile_png": web_out_dir / "next_day_predictions_mobile.png",
                "web_current_interactive": web_out_dir / "current_day_interactive_data.json",
                "web_next_interactive": web_out_dir / "next_day_interactive_data.json",
            }
        )
    missing = [name for name, path in paths.items() if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        return CachedArtifactStatus(False, None, f"missing:{','.join(missing)}")
    try:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be an object")
        current_rows = _csv_prediction_projection(paths["current_csv"], current_day=True)
        next_rows = _csv_prediction_projection(paths["next_csv"], current_day=False)
        utc_now = now_utc or datetime.now(timezone.utc)
        expected_day = utc_now.astimezone(ZoneInfo(local_timezone)).date()
        current_days = {
            datetime.fromtimestamp(row["timestamp"] / 1000.0, tz=timezone.utc)
            .astimezone(ZoneInfo(local_timezone))
            .date()
            for row in current_rows
        }
        if expected_day not in current_days:
            return CachedArtifactStatus(False, None, "current_day_cache_is_stale")
        payload = {
            "schema": 1,
            "current": current_rows,
            "next": next_rows,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return CachedArtifactStatus(True, hashlib.sha256(canonical.encode("utf-8")).hexdigest(), "ok")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return CachedArtifactStatus(False, None, f"invalid:{type(exc).__name__}:{exc}")


def state_path_for(out_dir: Path, *, site: str, model: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{site}_{model}")
    return out_dir / "operational_update_state" / f"{safe}.json"


def load_success_state(path: Path, *, site: str, model: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != STATE_SCHEMA_VERSION:
        return None
    if payload.get("site") != site or payload.get("model") != model:
        return None
    if payload.get("status") != "success":
        return None
    return payload


def decide_execution_mode(
    snapshot: OperationalSnapshot,
    state: Mapping[str, Any] | None,
) -> ExecutionDecision:
    if state is None:
        return ExecutionDecision("recovery_missing_state", "no successful operational state", True, False)
    if snapshot.forecast is None:
        return ExecutionDecision("recovery_forecast_identity", "forecast identity unavailable", True, False)
    if snapshot.model_fingerprint is None:
        return ExecutionDecision("recovery_model_artifacts", "model artifact set is incomplete", True, False)
    if not snapshot.cached_artifacts.valid:
        return ExecutionDecision(
            "recovery_missing_cache",
            snapshot.cached_artifacts.reason,
            True,
            False,
        )
    if state.get("model_fingerprint") != snapshot.model_fingerprint:
        return ExecutionDecision("model_changed", "production model identity changed", True, False)
    if state.get("forecast_fingerprint") != snapshot.forecast.fingerprint:
        return ExecutionDecision("forecast_changed", "forecast content changed", True, False)
    if state.get("cached_prediction_fingerprint") != snapshot.cached_artifacts.fingerprint:
        return ExecutionDecision("recovery_cached_prediction_changed", "cached prediction content changed", True, False)

    previous_obs = state.get("observation_max_ts")
    current_obs = snapshot.observation_max_ts
    if previous_obs is None or current_obs is None:
        return ExecutionDecision("recovery_observation_state", "observation watermark unavailable", True, False)
    if int(current_obs) < int(previous_obs):
        return ExecutionDecision("recovery_observation_regressed", "observation watermark regressed", True, False)
    if int(current_obs) > int(previous_obs):
        return ExecutionDecision("measured_only", "new observations with unchanged forecast/model", False, True)
    return ExecutionDecision("no_change", "observation, forecast, model, and cache are unchanged", False, False)


def write_bytes_if_changed(path: Path, content: bytes) -> bool:
    try:
        if path.is_file() and path.read_bytes() == content:
            return False
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
    return True


def write_success_state(
    path: Path,
    *,
    snapshot: OperationalSnapshot,
    execution_mode: str,
    successful_at_utc: datetime | None = None,
) -> None:
    if snapshot.forecast is None or snapshot.model_fingerprint is None:
        raise ValueError("cannot mark success without forecast and model identities")
    if not snapshot.cached_artifacts.valid or snapshot.cached_artifacts.fingerprint is None:
        raise ValueError("cannot mark success without valid cached prediction artifacts")
    timestamp = (successful_at_utc or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    payload = {
        "schema_version": STATE_SCHEMA_VERSION,
        "status": "success",
        "site": snapshot.site,
        "model": snapshot.model,
        "forecast_fingerprint": snapshot.forecast.fingerprint,
        "model_fingerprint": snapshot.model_fingerprint,
        "cached_prediction_fingerprint": snapshot.cached_artifacts.fingerprint,
        "observation_max_ts": snapshot.observation_max_ts,
        "source_run_ts": snapshot.forecast.source_run_ts,
        "stored_run_ts": snapshot.forecast.run_ts,
        "forecast_fetched_ts": snapshot.forecast.fetched_ts,
        "forecast_row_count": snapshot.forecast.row_count,
        "execution_mode": execution_mode,
        "successful_at_utc": timestamp,
    }
    content = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    write_bytes_if_changed(path, content)


def local_day_utc_bounds(local_timezone: str, now_utc: datetime | None = None) -> tuple[int, int]:
    utc_now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    local_now = utc_now.astimezone(ZoneInfo(local_timezone))
    start_local = datetime.combine(local_now.date(), datetime.min.time(), tzinfo=ZoneInfo(local_timezone))
    next_day = local_now.date().fromordinal(local_now.date().toordinal() + 1)
    end_local = datetime.combine(next_day, datetime.min.time(), tzinfo=ZoneInfo(local_timezone))
    return (
        int(start_local.astimezone(timezone.utc).timestamp() * 1000),
        int(end_local.astimezone(timezone.utc).timestamp() * 1000),
    )


def _launcher_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--db", default="data/wind_data_all_sites.db")
    parser.add_argument("--site", default="valkenburgsemeer")
    parser.add_argument("--model", default="HARMONIE")
    parser.add_argument("--target-hours", type=int, default=24)
    parser.add_argument("--out-dir", default="next_day_wind_model/artifacts")
    parser.add_argument("--model-artifact-dir", default=None)
    parser.add_argument("--web-out-dir", default="next_day_wind_model/web_dashboard")
    parser.add_argument("--local-timezone", default="Europe/Amsterdam")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-prediction", action="store_true")
    parser.add_argument("--skip-data-refresh-check", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--use-existing-artifacts", action="store_true")
    parser.add_argument("--test-now-local-hour", type=int, default=None)
    parser.add_argument("--operational-measured-only", action="store_true")
    return parser


def _abbrev(value: str | None) -> str:
    return "missing" if not value else value[:12]


def _collect_snapshot(args: argparse.Namespace, *, now_utc: datetime | None = None) -> OperationalSnapshot:
    db_path = Path(args.db).resolve()
    out_dir = Path(args.out_dir)
    model_dir = Path(args.model_artifact_dir) if args.model_artifact_dir else out_dir
    observation_max_ts = latest_observation_timestamp(db_path, args.site)
    forecast = load_latest_forecast_identity(
        db_path,
        site=args.site,
        model=args.model,
        min_target_rows=max(1, int(args.target_hours)),
    )
    model_fingerprint, _missing_models = compute_model_fingerprint(model_dir)
    cached = validate_cached_prediction_artifacts(
        out_dir,
        local_timezone=args.local_timezone,
        web_out_dir=Path(args.web_out_dir),
        now_utc=now_utc,
    )
    return OperationalSnapshot(
        site=args.site,
        model=args.model,
        observation_max_ts=observation_max_ts,
        forecast=forecast,
        model_fingerprint=model_fingerprint,
        cached_artifacts=cached,
    )


def _child_command(script_path: Path, argv: Sequence[str], *, measured_only: bool) -> list[str]:
    command = [sys.executable, str(script_path), *argv]
    if measured_only and "--operational-measured-only" not in command:
        command.append("--operational-measured-only")
    return command


def _run_child(script_path: Path, argv: Sequence[str], *, measured_only: bool) -> int:
    env = os.environ.copy()
    env[GATE_CHILD_ENV] = "1"
    completed = subprocess.run(
        _child_command(script_path, argv, measured_only=measured_only),
        env=env,
        check=False,
    )
    return int(completed.returncode)


def _log_decision(decision: ExecutionDecision, snapshot: OperationalSnapshot | None) -> None:
    forecast_hash = snapshot.forecast.fingerprint if snapshot and snapshot.forecast else None
    model_hash = snapshot.model_fingerprint if snapshot else None
    obs_ts = snapshot.observation_max_ts if snapshot else None
    print(
        f"execution_mode={decision.mode} reason={json.dumps(decision.reason)} "
        f"forecast_fingerprint={_abbrev(forecast_hash)} "
        f"model_fingerprint={_abbrev(model_hash)} observation_max_ts={obs_ts}",
        flush=True,
    )


def launch_operational_update(script_path: Path, argv: Sequence[str]) -> int:
    """Cheap bootstrap for the existing updater entry point.

    The normal six-minute invocation is gated before heavyweight scientific/ML
    imports. Full training/manual modes run unchanged in a child process.
    """
    args, _unknown = _launcher_parser().parse_known_args(list(argv))
    if args.operational_measured_only:
        # This should only be visible in the child; fail safe if invoked directly.
        decision = ExecutionDecision("bypass_full", "internal measured flag outside child", True, False)
        _log_decision(decision, None)
        return _run_child(script_path, argv, measured_only=False)

    normal_six_minute = (
        bool(args.skip_training)
        and bool(args.skip_data_refresh_check)
        and not bool(args.skip_prediction)
        and not bool(args.plots_only)
        and not bool(args.use_existing_artifacts)
        and args.test_now_local_hour is None
    )

    snapshot: OperationalSnapshot | None = None
    state_path = state_path_for(Path(args.out_dir), site=args.site, model=args.model)
    if normal_six_minute:
        try:
            snapshot = _collect_snapshot(args)
            state = load_success_state(state_path, site=args.site, model=args.model)
            decision = decide_execution_mode(snapshot, state)
        except Exception as exc:
            decision = ExecutionDecision(
                "recovery_gate_error",
                f"{type(exc).__name__}:{exc}",
                True,
                False,
            )
    else:
        decision = ExecutionDecision("bypass_full", "training or explicit non-operational mode", True, False)
        try:
            snapshot = _collect_snapshot(args)
        except Exception:
            snapshot = None

    _log_decision(decision, snapshot)
    if not decision.run_full_pipeline and not decision.run_measured_only:
        return 0

    return_code = _run_child(
        script_path,
        argv,
        measured_only=decision.run_measured_only,
    )
    if return_code != 0:
        print(f"operational_state=not_advanced child_return_code={return_code}", flush=True)
        return return_code

    prediction_was_generated = not (
        bool(args.skip_prediction) or bool(args.plots_only) or bool(args.use_existing_artifacts)
    )
    if not prediction_was_generated and not decision.run_measured_only:
        return 0

    if snapshot is None:
        print("operational_state=not_advanced reason=pre_execution_snapshot_unavailable", flush=True)
        return 0

    try:
        model_dir = Path(args.model_artifact_dir) if args.model_artifact_dir else Path(args.out_dir)
        post_model_fingerprint, _missing = compute_model_fingerprint(model_dir)
        post_cache = validate_cached_prediction_artifacts(
            Path(args.out_dir),
            local_timezone=args.local_timezone,
            web_out_dir=Path(args.web_out_dir),
        )
        successful_snapshot = replace(
            snapshot,
            model_fingerprint=post_model_fingerprint,
            cached_artifacts=post_cache,
        )
        write_success_state(
            state_path,
            snapshot=successful_snapshot,
            execution_mode=decision.mode,
        )
        print(f"operational_state=advanced path={state_path}", flush=True)
    except Exception as exc:
        # Outputs succeeded. Leaving state old is fail-safe and causes a retry.
        print(f"operational_state=not_advanced reason={type(exc).__name__}:{exc}", flush=True)
    return 0
