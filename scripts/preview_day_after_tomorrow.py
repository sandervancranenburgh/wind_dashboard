#!/usr/bin/env python3
"""Isolated operational D+2 rehearsal and two-page website preview.

Never invokes collectors, production wrappers, git publication or services.
Only source reads are allowed; weather-only online snapshots are writable.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_review(output, state, previous=None):
    """Export evidence without selecting models or changing checkpoints."""
    import pandas as pd
    import matplotlib.pyplot as plt
    from next_day_wind_model import day_after_tomorrow as d2
    artifact = output / "models" / d2.PREFIX
    diagnostic_path = artifact / "calibration_diagnostic.json"
    if not diagnostic_path.exists():
        return
    diagnostic = json.loads(diagnostic_path.read_text())
    if "date_bootstrap" not in diagnostic["matched_hours"]["direction"]["prediction_direction"]:
        diagnostic = d2.export_calibration_diagnostics(artifact, diagnostic["training"], diagnostic["calibration"],
            pd.read_csv(artifact / "calibration_predictions.csv"))
    selection = diagnostic["training"].get("calibration_training", {}).get("selection")
    audit_path = artifact / "calibration_selection_audit.json"
    if not selection and not audit_path.exists() and diagnostic["training"].get("calibration_policy", "legacy") != "none":
        # Older in-flight fits may predate selection-metadata export. Reproduce
        # the selection on their exact pre-gate validation data for reporting;
        # never retrain the network or choose a different deployed calibration.
        import numpy as np
        from next_day_wind_model import day_after_tomorrow_core as core
        from next_day_wind_model.update_model_and_predict import fit_speed_regime_calibration
        fit = core.load_fit(artifact / "candidates" / (diagnostic["model_id"] + ".pt"))
        _, validation = core.purged_split(core.load_samples(artifact / "samples.npz"), core.utc(fit.info["cutoff_utc"]))
        if sorted({s.day for s in validation}) != fit.info["validation_dates"]:
            raise ValueError("Calibration audit validation dates differ from the fitted checkpoint")
        selection = {}
        recomputed = fit_speed_regime_calibration(core.predict(fit, validation, calibrate=False),
            np.stack([s.frame.forecast_avg.to_numpy(float) for s in validation]),
            np.stack([s.actual for s in validation]), core.calibration_context(validation),
            target_mask=np.stack([s.speed_mask for s in validation]), diagnostics=selection)
        if recomputed != fit.calibration:
            raise ValueError("Calibration audit differs from the fitted checkpoint")
        selection["checkpoint_calibration_reproduced"] = True
        selection["uses_gate_observations"] = False
        d2.write_json(audit_path, selection)
    elif selection:
        d2.write_json(audit_path, selection)
    if audit_path.exists():
        diagnostic["calibration_selection_audit"] = json.loads(audit_path.read_text())
    coverage = pd.read_csv(artifact / "coverage_by_month.csv")
    inventory = pd.read_csv(artifact / "coverage.csv")
    usable = inventory.loc[inventory.usable]
    metrics = diagnostic["matched_hours"]
    before = None
    if previous:
        old_path = previous / "models/day_after_tomorrow/calibration_diagnostic.json"
        if not old_path.exists():
            old_path = previous / "validation/calibration_diagnostic.json"
        if old_path.exists():
            before = json.loads(old_path.read_text())
    fit = diagnostic["training"]
    report = {"development_only": True, "issue_time_utc": state.get("issue_time_utc"),
        "target_date": state.get("target_date"), "available_forecast_hours": state.get("available_hours"),
        "coverage": {"examined_contexts": len(inventory), "usable_contexts": len(usable),
            "excluded_contexts": len(inventory) - len(usable), "matched_hours": int(inventory.scorable_hours.sum()),
            "first_usable_target_date": usable.target_date.min(), "by_month": coverage.to_dict("records")},
        "diagnostic": diagnostic, "previous_review": before,
        "evaluation_dates": [state["gate"]["evaluation_start"], state["gate"]["evaluation_end"]],
        "no_gate_parameter_tuning": True, "historical_gate_informed_calibration_policy": fit.get("calibration_policy") == "none",
        "validation_reused_for_epoch_selection_and_calibration": diagnostic.get("calibration") is not None}
    d2.write_json(output / "review_report.json", report)
    rows = []
    for name, title in [("harmonie", "HARMONIE"), ("uncalibrated", "D+2 before calibration"), ("prediction", "D+2 delivered forecast")]:
        m = metrics[name]
        rows.append(f"| {title} | {m['mae']:.3f} | {m['rmse']:.3f} | {m['bias']:+.3f} |")
    calibration = diagnostic.get("calibration")
    selected_type = calibration["type"] if calibration else "none: deliberately disabled"
    gain = metrics["prediction"]["relative_mae_improvement"]
    interval = metrics["prediction"]["date_bootstrap"]["ci95"]
    comparison = ""
    if before:
        comparison = (f"\nPrevious development run: uncalibrated MAE {before.get('matched_hours', before)['uncalibrated']['mae']:.3f}, "
            f"calibrated MAE {before.get('matched_hours', before).get('prediction', before.get('calibrated', {}))['mae']:.3f}, HARMONIE MAE {before.get('matched_hours', before)['harmonie']['mae']:.3f} knots. "
            "The new run uses a fresh snapshot and batch size 16, matching next-day; this is not a controlled attribution of each change.\n")
    monthly = pd.read_csv(artifact / "calibration_breakdowns.csv")
    monthly = monthly.loc[monthly.dimension == "month"].pivot(index="value", columns="model", values="mae")
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for name, title, color in [("harmonie", "HARMONIE", "grey"), ("uncalibrated", "Before calibration", "#1f77b4"), ("prediction", "Delivered D+2", "#ff7f0e")]:
        axes[0].plot(monthly.index, monthly[name], marker="o", label=title, color=color)
    axes[0].set_ylabel("Speed MAE (knots)")
    axes[0].legend()
    axes[0].set_title("D+2 gate: calibration diagnostic")
    corrections = diagnostic["monthly_mean_correction_knots"]
    axes[1].plot(list(corrections), list(corrections.values()), marker="o", color="#ff7f0e")
    axes[1].axhline(0, color="grey", linewidth=.8)
    axes[1].set_ylabel("Mean calibration\ncorrection (knots)")
    axes[1].set_xlabel("Local target month")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "calibration_monthly_diagnostic.png", dpi=150)
    plt.close(fig)
    validation_note = ("Validation is used for neural early stopping; speed calibration is deliberately disabled."
                       if fit.get("calibration_policy") == "none" else
                       "Validation is reused for epoch selection and calibration; it is not an independent calibration acceptance set.")
    markdown = f'''# Aligned D+2 review

Development only; production databases, models, scheduling and publication are unchanged.

[Forecasts](dashboard/index.html) · [Evaluation](dashboard/evaluation.html)

Issue: {state.get('issue_time_utc')}; target: {state.get('target_date')}, 08:00–22:00 local.
HARMONIE available for {state.get('available_hours')} of 15 forecast hours.

## Data and training

The full archive scan examined {len(inventory):,} contexts: {len(usable):,} usable and {len(inventory)-len(usable):,} excluded,
with {int(inventory.scorable_hours.sum()):,} matched target hours. First usable D+2 target: {usable.target_date.min()}.
Monthly coverage and exclusion reasons are in the model directory's `coverage_by_month.csv` and `coverage_exclusions.csv`.

Fresh model: {fit['model_id']}. Training: {fit['training_dates'][0]}–{fit['training_dates'][-1]},
{fit['training_samples']:,} contexts. Validation/calibration: {fit['validation_dates'][0]}–{fit['validation_dates'][-1]},
{fit['validation_samples']:,} contexts. Gate: {report['evaluation_dates'][0]}–{report['evaluation_dates'][1]}.
{validation_note}
Gate observations were excluded from parameter fitting. Earlier gate results informed the decision to disable calibration; this is a historical diagnostic, not fresh confirmation. Confirmation on 20 new target dates is pending.

## Holdout performance

| Forecast | MAE (knots) | RMSE (knots) | Bias (knots) |
|---|---:|---:|---:|
{chr(10).join(rows)}

Selected calibration: **{selected_type}**. Delivered speed MAE improvement over HARMONIE: **{gain:.1%}**;
95% target-date bootstrap interval for absolute improvement: **{interval[0]:.3f}–{interval[1]:.3f} knots**.
{'The delivered forecast is worse than HARMONIE.' if gain < 0 else 'The delivered forecast improves on HARMONIE in this holdout.'}
{comparison}
The JSON report includes complete-window scores, circular direction errors, fitting dates, coefficients and monthly corrections.
`calibration_predictions.csv` and `calibration_breakdowns.csv` contain provenance and grouped speed metrics.

![Calibration across target months](calibration_monthly_diagnostic.png)

## Website and deployment

D+2 uses the next-day desktop/mobile renderer. Evaluation order: current-day, next-day and D+2 spiders,
then current-day, next-day and D+2 model gates, followed by realised history and downloads.
Current/next-day images retain their actual public timestamps; current-day gate details and identities are copied reporting inputs.

Deployment requires separate approval, fresh verified backups and the repository deployment workflow.
Follow `docs/day_after_tomorrow_operations.md` for enabling D+2 and rollback; do not copy development champions into production.
'''
    (output / "review_report.md").write_text(markdown)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--ecmwf-archive", required=True, type=Path)
    parser.add_argument("--reference-dashboard", required=True, type=Path)
    parser.add_argument("--reference-model-artifacts", type=Path, help="Read-only source of actual current-day gate details and identities")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--issue-time", required=True)
    parser.add_argument("--stage", choices=["train", "predict", "refresh"], default="train")
    parser.add_argument("--day-after-tomorrow-calibration", choices=["none", "legacy"], default="none")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--speed-constraint-eps", type=float, default=.2)
    parser.add_argument("--previous-review", type=Path, help="Optional earlier development review for before/after reporting")
    parser.add_argument("--sample-cache", type=Path, help="Optional existing weather sample cache for this source archive; never model checkpoints")
    options = parser.parse_args()
    from next_day_wind_model import day_after_tomorrow_core as core
    from next_day_wind_model import day_after_tomorrow as d2
    from next_day_wind_model.update_model_and_predict import publish_web_dashboard, current_day_gate_assets
    core.validate_output(options.output_dir, options.db, options.ecmwf_archive, options.reference_dashboard)
    output = options.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    db = output / "snapshots/weather.sqlite"
    ec = output / "snapshots/ecmwf.sqlite"
    if not db.exists():
        core.snapshot(options.db, db, tables=("observations", "forecasts"))
    if not ec.exists() and options.ecmwf_archive.exists():
        core.snapshot(options.ecmwf_archive, ec)
    args = SimpleNamespace(site=d2.SITE, enable_day_after_tomorrow=True,
        epochs=options.epochs, batch_size=options.batch_size, challenge_min_eval_samples=60,
        promotion_margin_pct=1., ecmwf_archive_db=ec, seed=42, web_out_dir=output / "dashboard",
        speed_constraint_eps=options.speed_constraint_eps, day_after_tomorrow_calibration=options.day_after_tomorrow_calibration)
    artifacts, models = output / "artifacts", output / "models"
    artifacts.mkdir(exist_ok=True)
    models.mkdir(exist_ok=True)
    if options.stage == "train" and options.previous_review and not (models / d2.PREFIX / "champions.json").exists():
        prior = options.previous_review / "models" / d2.PREFIX
        target = models / d2.PREFIX
        target.mkdir(exist_ok=True)
        manifest = json.loads((prior / "champions.json").read_text())
        for entry in manifest.values():
            source = prior / entry["checkpoint"]
            destination = target / entry["checkpoint"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if source.with_suffix(".scalers.npz").exists():
                shutil.copy2(source.with_suffix(".scalers.npz"), destination.with_suffix(".scalers.npz"))
        d2.write_json(target / "champions.json", manifest)
        history = prior / f"{d2.PREFIX}_model_gate_eval_history.csv"
        if history.exists():
            shutil.copy2(history, target / history.name)
    core.torch.set_num_threads(2)
    if options.sample_cache:
        cache_artifact = models / d2.PREFIX
        cache_artifact.mkdir(exist_ok=True)
        shutil.copy2(options.sample_cache, cache_artifact / "samples.npz")
        d2.write_json(cache_artifact / "samples_source.json", {"database": str(db.resolve()), "schema": "d2_v1"})
    state = d2.run_stage(args=args, db_path=db, out_dir=artifacts, model_artifact_dir=models,
        training=options.stage == "train", prediction=options.stage in ["train", "predict"], now=core.utc(options.issue_time))
    if state.get("last_error"):
        raise RuntimeError(state["last_error"])
    current_gate = current_day_gate_assets(artifacts)
    if options.reference_model_artifacts:
        reference = options.reference_model_artifacts.resolve()
        metadata = json.loads((reference / "metadata_update.json").read_text())
        gate = metadata.get("intraday_model_selection_gate", {})
        source = metadata.get("intraday_model_gate_eval_details_csv")
        if source and gate.get("enabled"):
            source = Path(source)
            if not source.is_absolute():
                source = reference.parents[1] / source
            destination = artifacts / "intraday_model_gate_eval_details" / source.name
            destination.parent.mkdir(exist_ok=True)
            shutil.copy2(source, destination)
            d2.write_json(artifacts / "reference_current_day_gate.json", gate)
            current_gate = current_day_gate_assets(artifacts, destination, gate)
    sources = options.reference_dashboard
    paths = lambda name: sources / name if (sources / name).exists() else None
    study_assets = {}
    study = output / "research"
    for name in ["d2_historical_report.json", "next_day_historical_report.json", "affine_selection.json",
                 "d2_historical_predictions.csv", "next_day_historical_predictions.csv", "d2_historical_breakdowns.csv",
                 "next_day_historical_breakdowns.csv", "frozen_manifest.json", "confirmation_status.json", "calibration_comparison.png"]:
        if (study / name).exists():
            study_assets[f"day_after_tomorrow_study_{name}"] = study / name
    published = publish_web_dashboard(web_out_dir=output / "dashboard", local_tz=core.TZ,
        web_refresh_seconds=360, spot_name="Valkenburgse Meer",
        next_day_png=sources / "next_day_predictions.png", next_day_png_mobile=paths("next_day_predictions_mobile.png"),
        next_day_csv=sources / "next_day_predictions.csv", current_day_png=sources / "current_day_predictions.png",
        current_day_png_mobile=paths("current_day_predictions_mobile.png"), current_day_csv=sources / "current_day_predictions.csv",
        daily_mae_png=paths("daily_mae_history.png"), daily_mae_png_mobile=paths("daily_mae_history_mobile.png"), daily_mae_csv=paths("daily_mae_history.csv"),
        gate_eval_png=paths("model_gate_eval_history.png"), gate_eval_csv=paths("model_gate_eval_history.csv"),
        direction_spider_png=paths("model_gate_direction_spider.png"), direction_spider_csv=paths("model_gate_speed_by_direction.csv"),
        current_day_direction_spider_png=paths("current_day_direction_spider.png"), current_day_direction_spider_csv=paths("current_day_speed_by_direction.csv"),
        companion_app_base_url="https://portal-cityailab.tbm.tudelft.nl", day_after_tomorrow_state=state,
        day_after_tomorrow_assets={**d2.artifact_inputs(artifacts, state), **study_assets}, plot_updated_at_utc=core.utc(options.issue_time),
        **current_gate)
    result = {"day_after_tomorrow": state, "published_files": sorted(published),
              "development_only": True, "reference_forecasts": "Copied public artifacts retain their original timestamps"}
    d2.write_json(output / "rehearsal.json", result)
    write_review(output, state, options.previous_review)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
