"""Isolated D+2 experiment; never collects, publishes, promotes, or logs live predictions.

The issue cutoff and target calendar day are independent. Historical evaluation
uses daily expanding training with purged target dates and train-only scalers.
"""
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
    _fit_target_hour_speed_calibration,
    _predict_direction_batch,
    _predict_speed_batch,
    apply_speed_regime_calibration,
    build_prediction_table,
    save_prediction_plot,
    save_wind_direction_performance_spider_plot,
)

from next_day_wind_model.day_after_tomorrow_core import (
    SITE,
    TZ,
    TARGET_HOURS,
    HISTORY_HOURS,
    EPS,
    SEED,
    EC_FEATURES,
    utc,
    calendar_targets,
    read_only,
    retain_snapshot_tables,
    snapshot,
    validate_output,
    partial_run,
    ECMWFArchive,
    observations,
    Sample,
    make_sample,
    masked_mse,
    purged_split,
    input_array,
    target_array,
    calibration_context,
    Fit,
    predict,
    fit_model,
    save_fit,
    load_fit,
    prediction_rows,
    save_samples,
    load_samples,
    metric,
    bootstrap,
    summarize,
    export_direction_performance,
)


def replay(samples: list[Sample], eval_dates: list[str], output: Path, args: argparse.Namespace,
           name: str, ecmwf: bool = False, directions: bool = False) -> pd.DataFrame:
    rows, training_log = [], []
    log_path = output / f"{name}_training.json"
    csv_path = output / f"{name}_predictions.csv"
    if args.resume and log_path.exists() and csv_path.exists():
        training_log = json.loads(log_path.read_text())
        rows = pd.read_csv(csv_path).to_dict("records")
    completed = {entry["target_date"] for entry in training_log}
    for index, day in enumerate(eval_dates):
        if day in completed:
            continue
        evaluation = [s for s in samples if s.day == day]
        cutoff = min(s.issue for s in evaluation)
        fitting = [s for s in samples if s.label_end <= cutoff]
        fitting = [s for s in fitting if s.speed_mask.any() and (not directions or s.direction_mask.any())]
        print(f"replay={name} target_date={day} progress={index + 1}/{len(eval_dates)} fitting_samples={len(fitting)}", flush=True)
        try:
            speed = fit_model(fitting, cutoff, "speed", ecmwf, args.epochs, args.batch_size, args.seed)
            direction_fit = fit_model(fitting, cutoff, "direction", False, args.epochs, args.batch_size, args.seed) if directions else None
        except ValueError as exc:
            training_log.append({"target_date": day, "skipped": str(exc)})
            continue
        training_log.append({"target_date": day, "speed": speed.info,
                             "direction": None if direction_fit is None else direction_fit.info})
        values = predict(speed, evaluation)
        direction = predict(direction_fit, evaluation) if direction_fit else None
        rows.extend(prediction_rows(evaluation, values, speed, name, direction))
        save_fit(speed, output / "models" / f"{name}_speed.pt")
        if direction_fit:
            save_fit(direction_fit, output / "models" / f"{name}_direction.pt")
        # Persist completed dates for diagnosis if a later fit fails.
        pd.DataFrame(rows).to_csv(output / f"{name}_predictions.csv", index=False)
        log_path.write_text(json.dumps(training_log, indent=2))
    return pd.DataFrame(rows)


def local_dashboard(sample: Sample, speed: Fit, direction: Fit, output: Path,
                    archive_path: Path | None, source_dashboard: Path | None) -> dict:
    dashboard = output / "dashboard"
    dashboard.mkdir(exist_ok=True)
    inference = {"target_times": sample.targets.astype(str).to_numpy(),
                 "forecast_next24": sample.frame.forecast_avg.to_numpy(np.float32),
                 "forecast_min_next24": sample.frame.forecast_min.to_numpy(np.float32),
                 "forecast_max_next24": sample.frame.forecast_max.to_numpy(np.float32),
                 "forecast_dir_next24": sample.frame.forecast_dir.to_numpy(np.float32),
                 "forecast_temperature_next24": sample.frame.forecast_temperature.to_numpy(np.float32),
                 "forecast_weather_code_next24": sample.frame.forecast_weather_code.to_numpy(np.float32)}
    table = build_prediction_table(inference, predict(speed, [sample])[0], predict(direction, [sample])[0], TZ)
    table.to_csv(dashboard / "day_after_tomorrow_predictions.csv", index=False)
    ec, ec_meta = pd.DataFrame(), {}
    if archive_path:
        ec, ec_meta = load_ecmwf_plot_data(archive_path, site=SITE, cutoff_utc=sample.issue.to_pydatetime(),
            start_utc=(sample.targets[0] - pd.Timedelta(hours=3)).to_pydatetime(),
            end_utc=(sample.targets[-1] + pd.Timedelta(hours=3)).to_pydatetime(), local_timezone=TZ)
    coverage = int(sample.forecast_mask.sum())
    diagnostics = {}
    for mobile in [False, True]:
        name = "day_after_tomorrow_predictions" + ("_mobile" if mobile else "")
        diagnostic = {}
        fetched = sample.frame.fetched_ts.dropna()
        save_prediction_plot(table, dashboard / f"{name}.png", TZ,
            plot_updated_at_utc=sample.issue, prediction_updated_at_utc=sample.issue.isoformat(),
            model_trained_at_utc=speed.info["cutoff_utc"],
            harmonie_time_utc=None if fetched.empty else pd.to_datetime(fetched.max(), unit="ms", utc=True),
            mobile=mobile, ecmwf_speed_series=ec, spot_name="Valkenburgse Meer",
            experiment_label=f"Experimental D+2 · coverage {coverage}/15 hours", render_diagnostics=diagnostic)
        diagnostics[name] = diagnostic
    # Existing two panels are read-only copies, not regenerated or presented as
    # matching the experimental issue. Their provenance is displayed separately.
    panels = []
    if source_dashboard:
        for prefix, title in [("current_day", "Current day"), ("next_day", "Next day")]:
            desktop = source_dashboard / f"{prefix}_predictions.png"
            mobile = source_dashboard / f"{prefix}_predictions_mobile.png"
            if desktop.exists():
                import shutil
                shutil.copy2(desktop, dashboard / desktop.name)
                if mobile.exists():
                    shutil.copy2(mobile, dashboard / mobile.name)
                panels.append(f'<section><h2>{title}</h2><p>Reference panel copied from the existing dashboard; its timestamps are shown in the image.</p><picture><source media="(max-width:700px)" srcset="{mobile.name if mobile.exists() else desktop.name}"><img src="{desktop.name}" alt="{title} forecast"></picture></section>')
    panels.append('<section><h2>Day after tomorrow · experimental</h2><picture><source media="(max-width:700px)" srcset="day_after_tomorrow_predictions_mobile.png"><img src="day_after_tomorrow_predictions.png" alt="Experimental day-after-tomorrow forecast"></picture><p><a href="day_after_tomorrow_predictions.csv">Download forecast CSV</a></p></section>')
    dashboard.joinpath("index.html").write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>D+2 forecast experiment</title><style>body{{font:16px system-ui;background:#f5f7fa;color:#182536;max-width:1400px;margin:2rem auto;padding:0 1rem}}section{{background:white;padding:1rem;margin:1rem 0;border-radius:12px}}img{{width:100%;height:auto}}p{{line-height:1.5}}</style><h1>Valkenburgse Meer · forecast experiment</h1><p>Issued {html.escape(sample.issue.tz_convert(TZ).isoformat())}. Target date {sample.day}, 08:00–22:00. HARMONIE coverage: {coverage}/15 hours. Missing hours remain gaps.</p><p><a href="../report.md">Evaluation report</a></p>{''.join(panels)}</html>''')
    result = {"issue_time_utc": sample.issue.isoformat(), "target_date": sample.day,
              "available_hours": coverage, "expected_hours": 15, "ecmwf": ec_meta, "rendering": diagnostics}
    (dashboard / "metadata.json").write_text(json.dumps(result, indent=2))
    return result


def export_report(frames: dict[str, pd.DataFrame], output: Path, coverage: list[dict], manifest: dict,
                  ecmwf_status: dict, preview: dict) -> dict:
    direction_summary = export_direction_performance(frames["historical"], output)
    direction_plot = output / "day_after_tomorrow_direction_spider.png"
    index = output / "dashboard" / "index.html"
    if index.exists():
        import re
        page = re.sub(r'<section id="day-after-tomorrow-direction-evaluation">.*?</section>', '',
                      index.read_text(), flags=re.DOTALL)
        historical = frames["historical"]
        dates = historical.target_date
        description = (f"Historical evaluation: {dates.min()}–{dates.max()}, 08:00–22:00 local time. "
            "Speed MAE in knots by HARMONIE forecast direction; lower values are better. "
            "Overlapping issue forecasts are averaged per target hour, using circular direction averaging, "
            "as in the next-day evaluation. Missing forecasts and observations are excluded.")
        image = ('<img style="max-width:800px;display:block;margin:auto" '
                 'src="day_after_tomorrow_direction_spider.png" '
                 'alt="Day-after-tomorrow speed MAE by forecast wind direction">') if direction_plot.exists() else '<p>Insufficient direction sectors to draw the spider diagram.</p>'
        panel = ('<section id="day-after-tomorrow-direction-evaluation">'
                 '<h2>Day-after-tomorrow performance by wind direction · experimental</h2>'
                 f'<p>{html.escape(description)}</p>{image}'
                 '<p><a href="day_after_tomorrow_speed_by_direction.csv">Download sector MAE CSV</a></p></section>')
        index.write_text(page.replace('</html>', panel + '</html>'))
    summaries = {name: summarize(frame) for name, frame in frames.items() if not frame.empty}
    for name, frame in frames.items():
        if frame.empty:
            continue
        for group in ["issue_hour", "target_hour", "lead_hours", "target_date"]:
            rows = []
            for value, selected in frame.groupby(group):
                result = summarize(selected)
                rows.append({group: value, "count": result["prediction"]["count"],
                             "model_mae": result["prediction"]["mae"], "model_rmse": result["prediction"]["rmse"],
                             "model_bias": result["prediction"]["bias"], "harmonie_mae": result["harmonie"]["mae"],
                             "harmonie_rmse": result["harmonie"]["rmse"], "harmonie_bias": result["harmonie"]["bias"],
                             "ecmwf_mae": result["ecmwf"]["mae"], "ecmwf_rmse": result["ecmwf"]["rmse"],
                             "ecmwf_bias": result["ecmwf"]["bias"],
                             "relative_mae_improvement": result["relative_mae_improvement"],
                             "complete_window_points": result["complete_window"]["prediction"]["count"],
                             "direction_mae": result["direction"]["prediction_direction"]["mae"],
                             "direction_rmse": result["direction"]["prediction_direction"]["rmse"],
                             "direction_bias": result["direction"]["prediction_direction"]["bias"],
                             "harmonie_direction_mae": result["direction"]["harmonie_direction"]["mae"]})
            pd.DataFrame(rows).to_csv(output / f"{name}_by_{group}.csv", index=False)
    if {"overlap_harmonie", "overlap_ecmwf"}.issubset(frames) and not frames["overlap_ecmwf"].empty:
        keys = ["issue_time_utc", "target_time_utc", "target_date"]
        paired = frames["overlap_ecmwf"].merge(frames["overlap_harmonie"][keys + ["prediction"]], on=keys,
                                              suffixes=("", "_without_ecmwf"), validate="one_to_one")
        paired.to_csv(output / "ecmwf_paired_predictions.csv", index=False)
        ecmwf_status["added_value_bootstrap"] = bootstrap(paired[paired.scorable], "prediction", "prediction_without_ecmwf")
        ecmwf_status["raw_ecmwf_vs_harmonie_bootstrap"] = bootstrap(paired[paired.scorable], "ecmwf", "harmonie")
        matched = paired[paired.scorable].dropna(subset=["prediction", "prediction_without_ecmwf", "actual"])
        plain = metric(matched.prediction_without_ecmwf, matched.actual)["mae"]
        augmented = metric(matched.prediction, matched.actual)["mae"]
        gain = (plain - augmented) / plain if plain else None
        interval = ecmwf_status["added_value_bootstrap"]["ci95"]
        ecmwf_status["relative_mae_improvement"] = gain
        ecmwf_status["assessment"] = "promising" if gain is not None and gain >= .01 and interval[0] > 0 else "worse" if gain is not None and gain < 0 else "inconclusive"
    pd.DataFrame(coverage).to_csv(output / "coverage.csv", index=False)
    report = {"manifest": manifest, "summaries": summaries, "ecmwf_experiment": ecmwf_status, "preview": preview,
              "wind_direction_evaluation": {"csv": "day_after_tomorrow_speed_by_direction.csv",
                  "plot": direction_plot.name if direction_plot.exists() else None,
                  "aggregation": "Matched speed forecasts averaged per target hour; circular mean HARMONIE forecast direction; eight 45-degree compass sectors.",
                  "target_hours": int(direction_summary.n_points.sum())},
              "evaluation_method": "Daily expanding refits at each target date's earliest issue; 20% validation by local target date, purged before validation issue; all labels complete by fit cutoff.",
              "limitations": ["ECMWF overlap is short; results are preliminary.", "P5 excluded.",
                              "Missing HARMONIE hours are unscored gaps; full-window scores are reported separately.",
                              "Hourly observations use circular direction aggregation.",
                              "Observation first-arrival timestamps are not archived; eligibility conservatively uses the end of the observation hour."]}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    lines = ["# Day-after-tomorrow experiment", "", "Valkenburgse Meer · 08:00–22:00 Europe/Amsterdam · development only.", "",
             report["evaluation_method"], "", "| Experiment | Target dates | Forecast MAE (knots) | HARMONIE MAE | Improvement | Assessment |",
             "|---|---:|---:|---:|---:|---|"]
    labels = {"historical": "Dedicated D+2 model", "raw_ecmwf": "Raw ECMWF (recent matched dates)",
              "overlap_harmonie": "HARMONIE-only overlap model", "overlap_ecmwf": "Model with ECMWF features"}
    def number(value):
        return "n/a" if value is None else f"{value:.3f}"
    for name, result in summaries.items():
        gain = result["relative_mae_improvement"]
        gain_text = "n/a" if gain is None else f"{gain:.1%}"
        lines.append(f'| {labels.get(name, name)} | {result["target_dates"]} | {number(result["prediction"]["mae"])} | {number(result["harmonie"]["mae"])} | {gain_text} | {result["assessment"]} |')
    for name, result in summaries.items():
        dates = frames[name].target_date
        lines += ["", f"## {labels.get(name, name)}", "", f"Evaluation dates: {dates.min()}–{dates.max()}.", "",
                  "| Forecast | Scored hours | MAE (knots) | RMSE (knots) | Bias (knots) |",
                  "|---|---:|---:|---:|---:|"]
        for key, title in [("prediction", "Forecast"), ("harmonie", "HARMONIE")]:
            scores = result[key]
            lines.append(f'| {title} | {scores["count"]} | {number(scores["mae"])} | {number(scores["rmse"])} | {number(scores["bias"])} |')
        interval = result["date_bootstrap"]["ci95"]
        if interval:
            lines += ["", f"95% target-date bootstrap interval for MAE improvement over HARMONIE: **{interval[0]:.3f}–{interval[1]:.3f} knots**."]
        full = result["complete_window"]
        lines += ["", f'Complete 08:00–22:00 windows: {full["prediction"]["count"]} scored hours; forecast MAE {number(full["prediction"]["mae"])} versus HARMONIE {number(full["harmonie"]["mae"])} knots.']
        direction = result["direction"]
        if direction["prediction_direction"]["count"]:
            lines += ["", f'Direction MAE: {number(direction["prediction_direction"]["mae"])}° versus HARMONIE {number(direction["harmonie_direction"]["mae"])}°.']
        paired = result["ecmwf_matched_subset"]
        if paired["prediction"]["count"]:
            lines += ["", f'On the identical ECMWF-available subset ({paired["target_dates"]} target dates, {paired["prediction"]["count"]} hours): forecast MAE {number(paired["prediction"]["mae"])}, HARMONIE {number(paired["harmonie"]["mae"])}, raw ECMWF {number(paired["ecmwf"]["mae"])} knots.']
    lines += ["", "## ECMWF feature comparison", "", f'Status: **{ecmwf_status.get("status", "unknown")}**.', ""]
    if ecmwf_status.get("reason"):
        lines.append(ecmwf_status["reason"] + ".")
    lines += ["", f'Archive: {ecmwf_status.get("completed_runs", 0)} completed runs, {ecmwf_status.get("first_run")}–{ecmwf_status.get("last_run")}.',
              "", f'Initial fitting split: {len(ecmwf_status.get("initial_training_dates", []))} training dates and {len(ecmwf_status.get("initial_validation_dates", []))} validation dates after purging. At least seven training dates are required.',
              "", "ECMWF wind components, speed, and gust are implemented as candidate features. Their learned added value cannot yet be established with this split."] if ecmwf_status.get("status") == "insufficient_data" else ["", json.dumps(ecmwf_status, indent=2)]
    lines += ["", "## Speed MAE by forecast wind direction", "",
              "Same format and aggregation as the next-day evaluation: eight compass sectors, HARMONIE in grey, the dedicated D+2 model in orange, and a fixed 0–3.5-knot radial scale. Overlapping issues are averaged per target hour before scoring; forecast directions are averaged circularly. Only matched scorable hours contribute. These target-averaged MAEs can differ from the issue-level metrics above.", "",
              "[Sector MAE and bias CSV](day_after_tomorrow_speed_by_direction.csv)", ""]
    if direction_plot.exists():
        lines += ["![Day-after-tomorrow speed MAE by forecast wind direction](day_after_tomorrow_direction_spider.png)", ""]
    else:
        lines += ["Insufficient direction sectors to draw the spider diagram.", ""]
    lines += ["", "## Coverage and limitations", ""]
    lines += [f"- {text}" for text in report["limitations"]]
    lines += ["", "[Open local dashboard](dashboard/index.html)", "", "Detailed predictions, training cutoffs, and grouped metrics are in the adjacent CSV/JSON files."]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, frame in frames.items():
        if frame.empty:
            continue
        work = frame[frame.scorable].copy()
        work["improvement"] = np.abs(work.harmonie - work.actual) - np.abs(work.prediction - work.actual)
        daily = work.groupby("target_date").improvement.mean()
        ax.plot(pd.to_datetime(daily.index), daily, label=name)
    ax.axhline(0, color="gray", linewidth=1)
    ax.set(ylabel="MAE improvement over HARMONIE (knots)", title="D+2 daily forecast improvement")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output / "daily_improvement.png", dpi=150)
    plt.close(fig)
    return report


def run(args: argparse.Namespace) -> dict:
    issue = utc(args.issue_time)
    output = args.output_dir.resolve()
    validate_output(output, args.db, args.ecmwf_archive, args.reference_dashboard)
    if output.exists() and not args.resume:
        raise FileExistsError("Use a new isolated output directory for each experiment")
    output.mkdir(parents=True, exist_ok=args.resume)
    torch.set_num_threads(args.threads)
    manifest = {"site": SITE, "issue_time_utc": issue.isoformat(), "timezone": TZ,
                "source_database": str(args.db.resolve()), "source_ecmwf_archive": str(args.ecmwf_archive.resolve()),
                "epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed,
                "history_hours": HISTORY_HOURS, "target_hours": TARGET_HOURS,
                "created_at_utc": datetime.now(timezone.utc).isoformat()}
    manifest_path = output / "manifest.json"
    if args.resume and manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key in manifest:
            if key != "created_at_utc" and previous.get(key) != manifest[key]:
                raise ValueError(f"Resume settings differ: {key}")
        manifest = previous
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2))
    db = output / "snapshots" / "observations_forecasts.sqlite"
    print("Creating consistent read-only source snapshots", flush=True)
    if not (args.resume and db.exists()):
        snapshot(args.db, db, tables=("forecasts", "observations"))
    ec_path = None
    if args.ecmwf_archive.is_file():
        ec_path = output / "snapshots" / "ecmwf.sqlite"
        if not (args.resume and ec_path.exists()):
            snapshot(args.ecmwf_archive, ec_path)
    archive = ECMWFArchive(ec_path)
    cache_path = output / "samples.npz"
    if args.resume and cache_path.exists():
        samples = load_samples(cache_path)
        preview_sample = load_samples(output / "preview_sample.npz")[0]
        coverage = pd.read_csv(output / "coverage.csv").to_dict("records")
    else:
        obs, latest_obs = observations(db)
        cfg = dp.DatasetConfig(site=SITE)
        lookup = dp.load_training_forecast_lookup(db, cfg, read_only=True)
        start = obs.index.min().tz_convert(TZ).date() + timedelta(days=3)
        completed_cutoff = min(latest_obs, issue).tz_convert(TZ)
        last_target_date = completed_cutoff.date() - timedelta(days=int(completed_cutoff.hour < 23))
        samples, coverage = [], []
        for day in pd.date_range(start, last_target_date - timedelta(days=2), freq="D"):
            for hour in range(7, 23):
                cutoff = pd.Timestamp(datetime.combine(day.date(), time(hour)), tz=TZ).tz_convert("UTC")
                sample = make_sample(lookup, obs, archive, cutoff)
                if sample is not None and sample.label_end <= min(latest_obs, issue) and sample.speed_mask.any():
                    samples.append(sample)
                coverage.append({"issue_time_utc": cutoff.isoformat(), "target_date": calendar_targets(cutoff)[0].tz_convert(TZ).date().isoformat(),
                                 "forecast_hours": 0 if sample is None else int(sample.forecast_mask.sum()),
                                 "scorable_hours": 0 if sample is None else int(sample.speed_mask.sum()),
                                 "ecmwf_feature_hours": 0 if sample is None else int(np.isfinite(sample.ec).all(axis=1).sum()),
                                 "reason": "missing forecast history/targets" if sample is None else None})
            if len(coverage) % 160 == 0:
                print(f"samples={len(samples)} through_issue_date={day.date()}", flush=True)
        preview_sample = make_sample(lookup, obs, archive, issue)
        if not samples or preview_sample is None:
            raise ValueError("No usable historical or preview HARMONIE context")
        save_samples(samples, cache_path)
        save_samples([preview_sample], output / "preview_sample.npz")
        pd.DataFrame(coverage).to_csv(output / "coverage.csv", index=False)
        del lookup, obs
    dates = sorted({s.day for s in samples})
    if len(dates) < 10:
        raise ValueError("Insufficient completed target dates for historical experiment")
    evaluation_dates = dates[-int(math.ceil(len(dates) * .2)):]
    frames = {"historical": replay(samples, evaluation_dates, output, args, "historical", directions=True)}
    overlap = [s for s in samples if np.isfinite(s.ec).all()]
    overlap_dates = sorted({s.day for s in overlap})
    ec_status = {"archive_reason": archive.reason, "completed_runs": len(archive.runs),
                 "first_run": archive.runs[0][0].isoformat() if archive.runs else None,
                 "last_run": archive.runs[-1][0].isoformat() if archive.runs else None,
                 "usable_target_dates": overlap_dates, "preliminary": True}
    if len(overlap_dates) >= 5:
        raw_samples = [s for s in overlap if s.day in overlap_dates[-5:]]
        raw = pd.DataFrame(prediction_rows(raw_samples, np.stack([s.ec[:, 2] for s in raw_samples]), None, "raw_ecmwf"))
        raw.to_csv(output / "raw_ecmwf_predictions.csv", index=False)
        frames["raw_ecmwf"] = raw
        ec_status["raw_comparison"] = summarize(raw)
    if len(overlap_dates) >= 12:
        ec_eval_dates = overlap_dates[-5:]
        first_issue = min(s.issue for s in overlap if s.day == ec_eval_dates[0])
        eligible_days = {s.day for s in overlap if s.label_end <= first_issue}
        try:
            ec_train, ec_val = purged_split(overlap, first_issue)
            ec_training_days = len({s.day for s in ec_train})
            ec_status.update(initial_training_dates=sorted({s.day for s in ec_train}),
                             initial_validation_dates=sorted({s.day for s in ec_val}))
        except ValueError:
            ec_training_days = 0
        if len(eligible_days) >= 7 and ec_training_days >= 7:
            ec_status["status"] = "evaluated"
            frames["overlap_harmonie"] = replay(overlap, ec_eval_dates, output, args, "overlap_harmonie")
            frames["overlap_ecmwf"] = replay(overlap, ec_eval_dates, output, args, "overlap_ecmwf", ecmwf=True)
        else:
            ec_status.update(status="insufficient_data", reason="Fewer than seven training dates remain after validation and temporal purging before the five-date evaluation")
    else:
        ec_status.update(status="insufficient_data", reason="Fewer than seven training plus five evaluation target dates")
    if preview_sample is None:
        raise ValueError("No usable HARMONIE history/targets at requested preview issue time")
    speed_path, direction_path = output / "models" / "preview_speed.pt", output / "models" / "preview_direction.pt"
    if args.resume and speed_path.exists() and direction_path.exists():
        speed, direction = load_fit(speed_path), load_fit(direction_path)
    else:
        speed = fit_model(samples, issue, "speed", False, args.epochs, args.batch_size, args.seed)
        direction = fit_model(samples, issue, "direction", False, args.epochs, args.batch_size, args.seed)
    save_fit(speed, output / "models" / "preview_speed.pt")
    save_fit(direction, output / "models" / "preview_direction.pt")
    preview = local_dashboard(preview_sample, speed, direction, output, ec_path, args.reference_dashboard)
    # Renderer metadata can contain NaNs; normalize these before strict report serialization.
    preview = json.loads(json.dumps(preview), parse_constant=lambda _: None)
    report = export_report(frames, output, coverage, manifest, ec_status, preview)
    print(json.dumps({"output": str(output), "summaries": report["summaries"], "ecmwf": ec_status}, indent=2), flush=True)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--ecmwf-archive", required=True, type=Path)
    parser.add_argument("--issue-time", required=True, help="Timezone-aware ISO timestamp for preview and information cutoff")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference-dashboard", type=Path, help="Read-only source of current/next-day reference PNGs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true", help="Resume this runner's existing isolated snapshots/cache with identical settings")
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.threads) < 1:
        parser.error("epochs, batch-size, and threads must be positive")
    utc(args.issue_time)
    return args


if __name__ == "__main__":
    run(parse_args())
