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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--ecmwf-archive", required=True, type=Path)
    parser.add_argument("--reference-dashboard", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--issue-time", required=True)
    parser.add_argument("--stage", choices=["train", "predict", "refresh"], default="train")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sample-cache", type=Path, help="Optional existing weather sample cache for this source archive; never model checkpoints")
    options = parser.parse_args()
    from next_day_wind_model import day_after_tomorrow_core as core
    from next_day_wind_model import day_after_tomorrow as d2
    from next_day_wind_model.update_model_and_predict import publish_web_dashboard
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
        promotion_margin_pct=1., ecmwf_archive_db=ec, seed=42, web_out_dir=output / "dashboard")
    artifacts, models = output / "artifacts", output / "models"
    artifacts.mkdir(exist_ok=True)
    models.mkdir(exist_ok=True)
    core.torch.set_num_threads(2)
    if options.sample_cache:
        cache_artifact = models / d2.PREFIX
        cache_artifact.mkdir(exist_ok=True)
        shutil.copy2(options.sample_cache, cache_artifact / "samples.npz")
        d2.write_json(cache_artifact / "samples_source.json", {"database": str(db.resolve()), "schema": "d2_v1"})
    state = d2.run_stage(args=args, db_path=db, out_dir=artifacts, model_artifact_dir=models,
        training=options.stage == "train", prediction=options.stage in ["train", "predict"], now=core.utc(options.issue_time))
    sources = options.reference_dashboard
    paths = lambda name: sources / name if (sources / name).exists() else None
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
        day_after_tomorrow_assets=d2.artifact_inputs(artifacts, state), plot_updated_at_utc=core.utc(options.issue_time))
    result = {"day_after_tomorrow": state, "published_files": sorted(published),
              "development_only": True, "reference_forecasts": "Copied public artifacts retain their original timestamps"}
    d2.write_json(output / "rehearsal.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
