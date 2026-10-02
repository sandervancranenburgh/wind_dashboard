"""Operational D+2 stages. Publication and scheduling belong to the caller.

Cached refresh deliberately imports no training modules or torch. Production
is opt-in via --enable-day-after-tomorrow; development uses isolated paths.
"""
from __future__ import annotations

import html
import importlib.util
import json
import math
import os
import re
import shutil
import sqlite3
import sys
import uuid
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PREFIX = "day_after_tomorrow"
SITE = "valkenburgsemeer"
TZ = "Europe/Amsterdam"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temporary, path)


def gate_split(samples, cutoff, minimum=60):
    """Whole-date gate holdout, with fitting labels purged before gate issues."""
    eligible = [s for s in samples if s.label_end <= cutoff and s.speed_mask.any()]
    dates = sorted({s.day for s in eligible})
    if len(dates) < 10:
        raise ValueError("Insufficient completed D+2 target dates")
    gate_dates = set(dates[-max(1, math.ceil(len(dates) * .15)):])
    gate = [s for s in eligible if s.day in gate_dates]
    boundary = min(s.issue for s in gate)
    fitting = [s for s in eligible if s.day not in gate_dates and s.label_end <= boundary]
    if len(gate) < minimum:
        raise ValueError(f"D+2 gate has {len(gate)} usable issue contexts; requires {minimum}")
    return fitting, gate, boundary


def promotion(actual, baseline, candidate, champion, circular=False, margin=1.):
    mask = np.isfinite(actual) & np.isfinite(baseline) & np.isfinite(candidate)
    if champion is not None:
        mask &= np.isfinite(champion)
    if not mask.any():
        raise ValueError("No common D+2 gate targets")
    def scores(values):
        error = values[mask] - actual[mask]
        if circular:
            error = (error + 180) % 360 - 180
        return {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.square(error).mean()))}
    candidate_score, baseline_score = scores(candidate), scores(baseline)
    champion_score = None if champion is None else scores(champion)
    promote = champion_score is None or candidate_score["mae"] <= champion_score["mae"] * (1 - margin / 100)
    return {"count": int(mask.sum()), "baseline": baseline_score, "candidate": candidate_score,
            "champion": champion_score, "promote": promote,
            "reason": "no_existing_champion" if champion is None else "improvement" if promote else "retain_champion"}


def historical_samples(db: Path, cutoff: pd.Timestamp, artifact: Path):
    from next_day_wind_model import day_after_tomorrow_core as core
    obs, latest = core.observations(db)
    cutoff = min(cutoff, latest)
    lookup = core.dp.load_training_forecast_lookup(db, core.dp.DatasetConfig(site=SITE), read_only=True)
    cache = artifact / "samples.npz"
    provenance = artifact / "samples_source.json"
    identity = {"database": str(db.resolve()), "schema": "d2_v1"}
    # Cached completed targets are stable. Reload the latest three dates so
    # newly completed/corrected observations near the cutoff can be included.
    samples = []
    if cache.exists() and provenance.exists() and json.loads(provenance.read_text()) == identity:
        samples = [s for s in core.load_samples(cache)
                   if s.label_end <= cutoff - pd.Timedelta(days=3)]
    start = obs.index.min().tz_convert(TZ).date() + timedelta(days=3)
    if samples:
        start = max(start, max(s.issue.tz_convert(TZ).date() for s in samples) + timedelta(days=1))
    last = cutoff.tz_convert(TZ).date() - timedelta(days=2)
    archive = core.ECMWFArchive(None)  # ECMWF is not a learned production feature.
    for day in pd.date_range(start, last, freq="D"):
        for hour in range(7, 23):
            issue = pd.Timestamp(datetime.combine(day.date(), time(hour)), tz=TZ).tz_convert("UTC")
            sample = core.make_sample(lookup, obs, archive, issue)
            if sample is not None and sample.label_end <= cutoff and sample.speed_mask.any():
                samples.append(sample)
    if samples:
        core.save_samples(samples, cache)
        write_json(provenance, identity)
    return samples


def train(samples, cutoff, artifact: Path, *, epochs=30, batch_size=32, seed=42, minimum=60, margin=1.):
    from next_day_wind_model import day_after_tomorrow_core as core
    from next_day_wind_model.update_model_and_predict import (
        append_model_gate_eval_history, save_model_gate_eval_history_plot)
    fitting, gate, boundary = gate_split(samples, cutoff, minimum)
    artifact.mkdir(parents=True, exist_ok=True)
    # Repeated training at the same issue cutoff must never overwrite a file
    # referenced by an already active manifest.
    gate_id = cutoff.strftime("%Y%m%d%H%M%S") + "_" + uuid.uuid4().hex[:8]
    state_path = artifact / "champions.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    new_state = dict(state)
    results, selected, candidates, prior_predictions, selected_fits = {}, {}, {}, {}, {}
    for kind in ["speed", "direction"]:
        supervised_gate = [s for s in gate if (s.direction_mask if kind == "direction" else s.speed_mask).any()]
        if len(supervised_gate) < minimum:
            raise ValueError(f"D+2 {kind} gate requires {minimum} supervised issue contexts")
        # Avoid batches with no directional labels while keeping the same date partitions.
        supervised_fitting = [s for s in fitting if (s.direction_mask if kind == "direction" else s.speed_mask).any()]
        previous_threads = core.torch.get_num_threads()
        try:
            core.torch.set_num_threads(min(previous_threads, 2))
            candidate = core.fit_model(supervised_fitting, boundary, kind, False, epochs, batch_size, seed)
        finally:
            core.torch.set_num_threads(previous_threads)
        candidate.info.update(model_id=f"{gate_id}_{kind}", operational=True,
                              trained_at_utc=cutoff.isoformat(), gate_dates=sorted({s.day for s in gate}))
        candidate_values = core.predict(candidate, gate)
        prior = core.load_fit(artifact / state[kind]["checkpoint"]) if kind in state else None
        if prior is not None and not prior.info.get("operational"):
            raise ValueError("Refusing an experimental checkpoint as the operational champion")
        prior_values = core.predict(prior, gate) if prior else None
        actual = np.stack([s.actual_dir if kind == "direction" else s.actual for s in gate])
        baseline = np.stack([s.frame.forecast_dir if kind == "direction" else s.frame.forecast_avg for s in gate])
        masks = np.stack([s.direction_mask if kind == "direction" else s.speed_mask for s in gate])
        actual = np.where(masks, actual, np.nan)
        score = promotion(actual, baseline, candidate_values, prior_values, kind == "direction", margin)
        path = artifact / "candidates" / f"{gate_id}_{kind}.pt"
        core.save_fit(candidate, path)
        if score["promote"]:
            new_state[kind] = {"checkpoint": str(path.relative_to(artifact)),
                               "model_id": candidate.info["model_id"], "trained_at_utc": cutoff.isoformat()}
        selected[kind] = candidate_values if score["promote"] else prior_values
        selected_fits[kind] = candidate if score["promote"] else prior
        candidates[kind], prior_predictions[kind], results[kind] = candidate_values, prior_values, score
    # Candidate files are immutable. Activate only after both fits and their
    # evaluation exports succeed, so a reporting failure also retains models.
    speed = results["speed"]
    summary = {"enabled": True, "run_utc": cutoff.isoformat(), "promotion_margin_pct": margin,
               "holdout_eval_split": .15, "holdout_eval_min_samples": minimum,
               "speed_eval_rows": speed["count"], "speed_eval_samples": len(gate),
               "speed_model_id_challenger": f"{gate_id}_speed",
               "speed_model_id_champion": state.get("speed", {}).get("model_id", f"{gate_id}_speed"),
               "speed_mae_challenger": speed["candidate"]["mae"], "speed_rmse_challenger": speed["candidate"]["rmse"],
               "speed_mae_forecast": speed["baseline"]["mae"], "speed_rmse_forecast": speed["baseline"]["rmse"],
               "speed_mae_champion": (speed["champion"] or speed["candidate"])["mae"],
               "speed_rmse_champion": (speed["champion"] or speed["candidate"])["rmse"],
               "direction_mae_challenger": results["direction"]["candidate"]["mae"],
               "direction_mae_champion": (results["direction"]["champion"] or results["direction"]["candidate"])["mae"],
               "promote_speed": speed["promote"], "promote_direction": results["direction"]["promote"],
               "speed_selected": new_state["speed"]["model_id"], "direction_selected": new_state["direction"]["model_id"],
               "gate_target_dates": sorted({s.day for s in gate}), "decisions": results}
    rows = pd.DataFrame(core.prediction_rows(gate, selected["speed"], selected_fits["speed"], "operational_gate", selected["direction"]))
    rows["speed_model_id"] = new_state["speed"]["model_id"]
    rows["direction_model_id"] = new_state["direction"]["model_id"]
    rows.to_csv(artifact / "gate_predictions.csv", index=False)
    summary["date_bootstrap"] = core.bootstrap(rows[rows.scorable], "prediction", "harmonie")
    core.export_direction_performance(rows, artifact)
    details = rows.rename(columns={"actual": "actual_wind_speed", "harmonie": "forecast_wind_speed",
                                   "harmonie_direction": "forecast_wind_dir_deg"}).copy()
    details["champion_wind_speed"] = (prior_predictions["speed"] if prior_predictions["speed"] is not None else selected["speed"]).reshape(-1)
    details["challenger_wind_speed"] = candidates["speed"].reshape(-1)
    details = details[details.scorable].groupby("target_time_utc", as_index=False).mean(numeric_only=True)
    # Restore circular direction averaging after the numeric speed aggregation.
    raw = rows[rows.scorable].groupby("target_time_utc").harmonie_direction
    circular = raw.agg(lambda x: np.degrees(np.arctan2(np.sin(np.deg2rad(x)).mean(), np.cos(np.deg2rad(x)).mean())) % 360)
    details["forecast_wind_dir_deg"] = details.target_time_utc.map(circular)
    detail_path = artifact / f"{PREFIX}_model_gate_eval_details.csv"
    details.to_csv(detail_path, index=False)
    history = artifact / f"{PREFIX}_model_gate_eval_history.csv"
    append_model_gate_eval_history(history, summary, TZ)
    save_model_gate_eval_history_plot(history, artifact / f"{PREFIX}_model_gate_eval_history.png",
                                     TZ, detail_path, horizon_label="Day-after-tomorrow")
    summary.update(spider_model_id=new_state["speed"]["model_id"],
                   evaluation_start=min(s.day for s in gate), evaluation_end=max(s.day for s in gate))
    write_json(artifact / "gate_summary.json", summary)
    write_json(state_path, new_state)
    return summary


def infer(db: Path, issue, artifact: Path, output: Path):
    from next_day_wind_model import day_after_tomorrow_core as core
    from next_day_wind_model.update_model_and_predict import build_prediction_table
    state = json.loads((artifact / "champions.json").read_text())
    speed, direction = [core.load_fit(artifact / state[kind]["checkpoint"]) for kind in ["speed", "direction"]]
    for kind, fit in [("speed", speed), ("direction", direction)]:
        if not fit.info.get("operational"):
            raise ValueError("Refusing an experimental checkpoint as the operational champion")
        if pd.Timestamp(state[kind]["trained_at_utc"]) > issue or pd.Timestamp(fit.info["max_label_end_utc"]) > issue:
            raise ValueError("D+2 champion was not available at this issue cutoff")
    lookup = core.dp.load_training_forecast_lookup(db, core.dp.DatasetConfig(site=SITE), read_only=True,
        target_start_ts_ms=int((issue - pd.Timedelta(hours=80)).timestamp() * 1000),
        target_end_ts_ms=int((issue + pd.Timedelta(days=3)).timestamp() * 1000))
    empty_obs = pd.DataFrame(columns=["actual_avg", "actual_dir"], index=pd.DatetimeIndex([], tz="UTC"))
    sample = core.make_sample(lookup, empty_obs, core.ECMWFArchive(None), issue)
    if sample is None:
        raise ValueError("No usable HARMONIE D+2 history/target coverage")
    inference = {"target_times": sample.targets.astype(str).to_numpy()}
    for source, destination in [("forecast_avg", "forecast_next24"), ("forecast_min", "forecast_min_next24"),
        ("forecast_max", "forecast_max_next24"), ("forecast_dir", "forecast_dir_next24"),
        ("forecast_temperature", "forecast_temperature_next24"), ("forecast_weather_code", "forecast_weather_code_next24")]:
        inference[destination] = sample.frame[source].to_numpy(np.float32)
    table = build_prediction_table(inference, core.predict(speed, [sample])[0], core.predict(direction, [sample])[0], TZ)
    table["issue_time_utc"] = issue.isoformat()
    table["forecast_available"] = sample.forecast_mask
    table["harmonie_run_ts"] = sample.frame.run_ts.to_numpy()
    table["harmonie_fetched_ts"] = sample.frame.fetched_ts.to_numpy()
    table["speed_model_id"] = state["speed"]["model_id"]
    table["direction_model_id"] = state["direction"]["model_id"]
    table.to_csv(output / f"{PREFIX}_predictions.csv", index=False)
    metadata = {"status": "available", "issue_time_utc": issue.isoformat(), "target_date": sample.day,
                "available_hours": int(sample.forecast_mask.sum()), "expected_hours": 15,
                "champions": state, "experimental": True,
                "harmonie_fetched_at_utc": pd.Timestamp(sample.frame.fetched_ts.max(), unit="ms", tz="UTC").isoformat()}
    write_json(output / f"{PREFIX}_metadata.json", metadata)
    return sample, table, metadata


def log_predictions(db: Path, sample, table: pd.DataFrame, metadata: dict):
    from db_store import init_db, log_prediction_batch
    rows = []
    for i, target in enumerate(sample.targets):
        if not sample.forecast_mask[i]:
            continue
        for kind, prediction, baseline in [("speed", "lstm_pred_wind_speed", "forecast_wind_speed"),
                                           ("direction", "lstm_pred_wind_dir_deg", "forecast_wind_dir_deg")]:
            value = table.iloc[i][prediction]
            if not np.isfinite(value):
                continue
            rows.append({"site": SITE, "model_type": PREFIX, "prediction_kind": "wind_" + kind,
                "model_name": "superlocal_d2", "model_version": metadata["champions"][kind]["model_id"],
                "model_artifact": metadata["champions"][kind]["checkpoint"],
                "issued_ts": int(sample.issue.timestamp() * 1000), "anchor_ts": int(sample.issue.timestamp() * 1000),
                "target_ts": int(target.timestamp() * 1000), "horizon_hr": (target - sample.issue).total_seconds() / 3600,
                "prediction_value": float(value), "harmonie_value": float(table.iloc[i][baseline]),
                "harmonie_run_ts": int(sample.frame.iloc[i].run_ts),
                "harmonie_fetched_ts": int(sample.frame.iloc[i].fetched_ts),
                "run_context": "day_after_tomorrow", "metadata_json": json.dumps({"experimental": True, "coverage": metadata["available_hours"]})})
    with sqlite3.connect(db) as conn:
        init_db(conn)
        log_prediction_batch(conn, rows)


def realized_scores(db: Path, cutoff, output: Path):
    """Materialize only D+2 rows, with speed and circular direction actuals."""
    from next_day_wind_model import day_after_tomorrow_core as core
    with core.read_only(db) as conn:
        rows = conn.execute("SELECT rowid, target_ts, prediction_kind, prediction_value, harmonie_value FROM prediction_log WHERE site=? AND model_type=? AND target_ts<?",
                            (SITE, PREFIX, int(cutoff.floor('h').timestamp() * 1000))).fetchall()
    if not rows:
        return
    observations, _ = core.observations(db)
    updates = []
    for rowid, target_ts, kind, prediction, baseline in rows:
        timestamp = pd.Timestamp(target_ts, unit="ms", tz="UTC")
        if timestamp not in observations.index:
            continue
        actual = observations.loc[timestamp, "actual_dir" if kind == "wind_direction" else "actual_avg"]
        if not np.isfinite(actual):
            continue
        model_error, baseline_error = prediction - actual, baseline - actual
        if kind == "wind_direction":
            model_error, baseline_error = (model_error + 180) % 360 - 180, (baseline_error + 180) % 360 - 180
        updates.append((float(actual), float(model_error), float(baseline_error), abs(float(model_error)),
                        abs(float(baseline_error)), float(model_error ** 2), float(baseline_error ** 2), rowid))
    with sqlite3.connect(db) as conn:
        conn.executemany("UPDATE prediction_log SET actual_value=?,model_error=?,harmonie_error=?,model_abs_error=?,harmonie_abs_error=?,model_sq_error=?,harmonie_sq_error=? WHERE rowid=?", updates)
        frame = pd.read_sql_query("SELECT prediction_kind,COUNT(*) AS n_points,AVG(model_abs_error) AS model_mae,AVG(harmonie_abs_error) AS harmonie_mae,AVG(model_error) AS model_bias,AVG(harmonie_error) AS harmonie_bias FROM prediction_log WHERE site=? AND model_type=? AND actual_value IS NOT NULL GROUP BY prediction_kind", conn, params=(SITE, PREFIX))
    frame.to_csv(output / f"{PREFIX}_realized_scores.csv", index=False)


def cached_status(output: Path, now: pd.Timestamp, enabled=True) -> dict:
    if not enabled:
        return {"status": "disabled"}
    path = output / f"{PREFIX}_metadata.json"
    if not path.exists():
        return {"status": "unavailable", "reason": "No D+2 forecast is available yet"}
    try:
        metadata = json.loads(path.read_text())
        if not isinstance(metadata, dict):
            raise ValueError("Invalid D+2 metadata object")
    except (ValueError, OSError):
        return {"status": "unavailable", "reason": "D+2 forecast metadata is unavailable"}
    if metadata.get("status") not in ["available", "stale"]:
        return metadata
    try:
        issue = pd.Timestamp(metadata["issue_time_utc"]).tz_convert(TZ)
        for key in ["target_date", "available_hours", "champions"]:
            if key not in metadata:
                raise ValueError("Incomplete D+2 metadata")
    except (ValueError, KeyError, TypeError):
        return {"status": "unavailable", "reason": "D+2 forecast metadata is incomplete"}
    local = now.tz_convert(TZ)
    # Overnight, retain the last issue but never relabel its target calendar date.
    if issue.date() != local.date() or (7 <= local.hour <= 22 and now - issue.tz_convert("UTC") >= pd.Timedelta(hours=2)):
        metadata["status"] = "stale"
    return metadata


def render_cached(output: Path, archive: Path | None, now, *, force=True):
    from next_day_wind_model.ecmwf_dashboard import load_ecmwf_plot_data
    # The measured-only updater already loaded its plotting functions without
    # torch. Reuse that module rather than importing a second full updater.
    renderer = sys.modules.get("__main__")
    if not hasattr(renderer, "save_prediction_plot"):
        renderer = sys.modules.get("next_day_wind_model.update_model_and_predict")
    if renderer is None or not hasattr(renderer, "save_prediction_plot"):
        name = "next_day_wind_model._d2_plot_renderer"
        renderer = sys.modules.get(name)
        if renderer is None:
            spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("update_model_and_predict.py"))
            renderer = importlib.util.module_from_spec(spec)
            previous = os.environ.get("WIND_PLOT_RENDERER_ONLY")
            try:
                os.environ["WIND_PLOT_RENDERER_ONLY"] = "1"
                sys.modules[name] = renderer
                spec.loader.exec_module(renderer)
            finally:
                if previous is None:
                    os.environ.pop("WIND_PLOT_RENDERER_ONLY", None)
                else:
                    os.environ["WIND_PLOT_RENDERER_ONLY"] = previous
    save_prediction_plot, _frame_to_json_records = renderer.save_prediction_plot, renderer._frame_to_json_records
    metadata = cached_status(output, now)
    csv = output / f"{PREFIX}_predictions.csv"
    if not csv.exists() or metadata.get("status") not in ["available", "stale"]:
        return metadata
    table = pd.read_csv(csv)
    for col in ["target_time_utc", "target_time_local"]:
        table[col] = pd.to_datetime(table[col], utc=True)
    targets = table.target_time_utc
    ec, ec_meta = pd.DataFrame(), {}
    if archive:
        ec, ec_meta = load_ecmwf_plot_data(Path(archive), site=SITE, cutoff_utc=now.to_pydatetime(),
            start_utc=(targets.min() - pd.Timedelta(hours=3)).to_pydatetime(),
            end_utc=(targets.max() + pd.Timedelta(hours=3)).to_pydatetime(), local_timezone=TZ)
    if force or ec_meta != metadata.get("ecmwf"):
        for mobile in [False, True]:
            save_prediction_plot(table, output / f"{PREFIX}_predictions{'_mobile' if mobile else ''}.png", TZ,
                plot_updated_at_utc=now, prediction_updated_at_utc=metadata["issue_time_utc"],
                model_trained_at_utc=metadata["champions"]["speed"]["trained_at_utc"],
                harmonie_time_utc=metadata.get("harmonie_fetched_at_utc"),
                mobile=mobile, ecmwf_speed_series=ec, spot_name="Valkenburgse Meer",
                experiment_label=f"Experimental D+2 · {metadata['status']} · coverage {metadata['available_hours']}/15 hours")
        table["target_time_local"] = table.target_time_utc.dt.tz_convert(TZ).astype(str)
        table["target_time_utc"] = table.target_time_utc.astype(str)
        from next_day_wind_model.weather_conditions import weather_description
        table["weather_description"] = table.weather_code.map(weather_description)
        for column in ["time_utc", "time_local"]:
            if column in ec:
                ec[column] = ec[column].astype(str)
        ec_rows = [] if ec.empty else _frame_to_json_records(ec)
        payload = {"plot_kind": "next_day", "timezone": TZ, "metadata": metadata,
                   "rows": _frame_to_json_records(table), "ecmwf": ec_rows}
        write_json(output / f"{PREFIX}_interactive_data.json", payload)
    metadata["ecmwf"] = json.loads(json.dumps(ec_meta), parse_constant=lambda _: None)
    write_json(output / f"{PREFIX}_metadata.json", metadata)
    return metadata


def run_stage(*, args, db_path: Path, out_dir: Path, model_artifact_dir: Path,
              training=False, prediction=False, now=None, samples=None, log=True, force_render=True):
    now = pd.Timestamp(now or datetime.now(timezone.utc)).tz_convert("UTC")
    enabled = bool(getattr(args, "enable_day_after_tomorrow", False)) and args.site == SITE
    if not enabled:
        return {"status": "disabled"}
    artifact = model_artifact_dir / PREFIX
    issue = now.floor("h")
    error = None
    try:
        artifact.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        if training:
            samples = historical_samples(db_path, issue, artifact) if samples is None else samples
            try:
                train(samples, issue, artifact, epochs=args.epochs, batch_size=args.batch_size,
                      seed=getattr(args, "seed", 42), minimum=args.challenge_min_eval_samples,
                      margin=args.promotion_margin_pct)
            except Exception as exc:
                # Keep active champions and previous successful evaluation intact.
                write_json(artifact / "last_training_attempt.json", {"status": "failed", "at": now.isoformat(), "reason": str(exc)})
                error = str(exc)
        if prediction and 7 <= issue.tz_convert(TZ).hour <= 22:
            sample, table, metadata = infer(db_path, issue, artifact, out_dir)
            if log:
                log_predictions(db_path, sample, table, metadata)
                realized_scores(db_path, issue, out_dir)
        metadata = render_cached(out_dir, getattr(args, "ecmwf_archive_db", None), now, force=force_render)
    except Exception as exc:
        error = str(exc)
        metadata = cached_status(out_dir, now)
        metadata.update(status="stale" if metadata.get("issue_time_utc") else "unavailable", reason=error)
        write_json(out_dir / f"{PREFIX}_metadata.json", metadata)
    if error:
        metadata["last_error"] = error
    gate = artifact / "gate_summary.json"
    if gate.exists():
        metadata["gate"] = json.loads(gate.read_text())
    metadata["artifact_dir"] = str(artifact)
    return metadata


def safe_stage(**kwargs):
    """Ensure even unreadable output paths cannot stop other forecast horizons."""
    try:
        return run_stage(**kwargs)
    except Exception as exc:
        print(f"D+2 stage unavailable: {type(exc).__name__}: {exc}", flush=True)
        return {"status": "unavailable", "reason": "D+2 update failed; other forecasts continue"}


def artifact_inputs(output: Path, state: dict) -> dict:
    if state.get("status") == "disabled":
        return {}
    paths = {}
    for name in [f"{PREFIX}_predictions.png", f"{PREFIX}_predictions_mobile.png", f"{PREFIX}_predictions.csv",
                 f"{PREFIX}_interactive_data.json", f"{PREFIX}_metadata.json", f"{PREFIX}_realized_scores.csv"]:
        path = output / name
        if path.exists():
            paths[name] = path
    artifact = Path(state.get("artifact_dir", output / PREFIX))
    for name in [f"{PREFIX}_direction_spider.png", f"{PREFIX}_speed_by_direction.csv",
                 f"{PREFIX}_model_gate_eval_history.png", f"{PREFIX}_model_gate_eval_history.csv",
                 f"{PREFIX}_model_gate_eval_details.csv"]:
        if (artifact / name).exists():
            paths[name] = artifact / name
    return paths


def refresh_published(*, args, db_path, out_dir, model_artifact_dir, now=None, force_render=False):
    """Called by measured-only refresh; never samples, trains, infers or logs."""
    state = safe_stage(args=args, db_path=db_path, out_dir=out_dir,
        model_artifact_dir=model_artifact_dir, now=now, force_render=force_render)
    changed = {}
    web = Path(args.web_out_dir)
    for name, source in artifact_inputs(out_dir, state).items():
        destination = web / name
        data = source.read_bytes()
        if not destination.exists() or destination.read_bytes() != data:
            destination.write_bytes(data)
            changed[name] = True
    index = web / "index.html"
    if index.exists():
        content = index.read_text()
        replacement = status_text(state)
        updated = re.sub(r'(<p id="day-after-tomorrow-status"[^>]*>).*?(</p>)',
                         lambda match: match[1] + html.escape(replacement) + match[2], content, flags=re.DOTALL)
        if updated != content:
            index.write_text(updated)
            changed["index.html"] = True
    return state, changed


def status_text(state: dict) -> str:
    if state.get("status") == "disabled":
        return ""
    if not state.get("issue_time_utc"):
        return "Experimental D+2 forecast unavailable. " + state.get("reason", "")
    issue = pd.Timestamp(state["issue_time_utc"]).tz_convert(TZ).strftime("%d %b %Y %H:%M %Z")
    return (f"Experimental · {state['status']} · issued {issue} · target {state['target_date']} "
            f"08:00–22:00 · HARMONIE coverage {state['available_hours']}/15 hours. Missing hours remain gaps.")
