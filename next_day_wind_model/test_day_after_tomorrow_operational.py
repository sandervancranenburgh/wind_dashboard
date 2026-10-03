from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from next_day_wind_model import day_after_tomorrow as d2
from next_day_wind_model import day_after_tomorrow_core as core
from next_day_wind_model import update_model_and_predict as updater
from next_day_wind_model import operational_update
from next_day_wind_model.test_day_after_tomorrow_experiment import sample


def samples():
    result = []
    for day in pd.date_range("2026-07-01", periods=40):
        for hour in range(7, 23):
            s = sample(f"{day.date()}T{hour:02d}:00:00+02:00")
            s.frame["run_ts"] = int(s.issue.timestamp() * 1000) - 1000
            s.frame["fetched_ts"] = s.frame.run_ts
            s.frame["forecast_dir"] = np.arange(15) * 45 % 360
            result.append(s)
    return result


def args(**extra):
    return SimpleNamespace(site=d2.SITE, enable_day_after_tomorrow=True, epochs=1,
        batch_size=32, challenge_min_eval_samples=60, promotion_margin_pct=1., **extra)


class SharedDirectoryTests(unittest.TestCase):
    def test_hourly_issue_refresh_even_with_unchanged_forecast(self):
        with tempfile.TemporaryDirectory() as temp:
            configured = args(day_after_tomorrow_model_artifact_dir=temp)
            with patch.object(operational_update, "compute_model_fingerprint", return_value=("existing", [])):
                def identity(stamp):
                    return operational_update.operational_model_fingerprint(configured, Path(temp), now_utc=pd.Timestamp(stamp).to_pydatetime())
                self.assertEqual(identity("2026-10-03T07:00Z"), identity("2026-10-03T07:54Z"))
                self.assertNotEqual(identity("2026-10-03T07:54Z"), identity("2026-10-03T08:00Z"))
                self.assertEqual(identity("2026-10-03T20:00Z"), identity("2026-10-04T04:59Z"))
                self.assertNotEqual(identity("2026-10-04T04:59Z"), identity("2026-10-04T05:00Z"))

    def test_cli_environment_and_explicit_directory(self):
        with patch.dict(os.environ, {"WIND_DAY_AFTER_TOMORROW_MODEL_ARTIFACT_DIR": "/tmp/shared-d2", "WIND_ENABLE_DAY_AFTER_TOMORROW": "1"}):
            launch = operational_update._launcher_parser().parse_args(["--site", d2.SITE])
            self.assertTrue(launch.enable_day_after_tomorrow)
            self.assertEqual(launch.day_after_tomorrow_model_artifact_dir, "/tmp/shared-d2")
            self.assertFalse(operational_update._launcher_parser().parse_args(["--site", d2.SITE, "--no-enable-day-after-tomorrow"]).enable_day_after_tomorrow)
            with patch.object(sys, "argv", ["updater", "--site", d2.SITE]):
                self.assertEqual(updater.parse_args().day_after_tomorrow_model_artifact_dir, "/tmp/shared-d2")
            with patch.object(sys, "argv", ["updater", "--site", d2.SITE, "--day-after-tomorrow-model-artifact-dir", "/tmp/explicit-d2"]):
                self.assertEqual(updater.parse_args().day_after_tomorrow_model_artifact_dir, "/tmp/explicit-d2")
        self.assertEqual(d2.model_directory(args(), Path("legacy")), Path("legacy/day_after_tomorrow"))

    def test_daily_hourly_and_cached_refresh_read_same_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shared = root / "shared"
            shared.mkdir()
            (shared / "gate_summary.json").write_text('{"gate_id":"shared-gate"}')
            configured = args(day_after_tomorrow_model_artifact_dir=str(shared))
            with patch.object(d2, "render_cached", return_value={"status": "available"}):
                for parent in [root / "legacy", root / "site"]:
                    state = d2.run_stage(args=configured, db_path=root / "unused.db", out_dir=root / parent.name,
                        model_artifact_dir=parent, now=pd.Timestamp("2026-10-03T10:00Z"))
                    self.assertEqual(state["artifact_dir"], str(shared))
                    self.assertEqual(state["gate"]["gate_id"], "shared-gate")
            with patch.object(operational_update, "compute_model_fingerprint", return_value=("existing", [])):
                before = operational_update.operational_model_fingerprint(configured, root / "legacy")
                self.assertEqual(before, operational_update.operational_model_fingerprint(configured, root / "site"))
                (shared / "champions.json").write_text('{"speed":{"model_id":"new"}}')
                after = operational_update.operational_model_fingerprint(configured, root / "site")
                self.assertNotEqual(before, after)
                other = args(day_after_tomorrow_model_artifact_dir=str(root / "other"))
                self.assertNotEqual(after, operational_update.operational_model_fingerprint(other, root / "site"))


class GateTests(unittest.TestCase):
    def test_whole_date_holdout_and_temporal_purging(self):
        all_samples = samples()
        fitting, gate, boundary = d2.gate_split(all_samples, core.utc("2026-08-20T07:00:00+02:00"))
        train, validation = core.purged_split(fitting, boundary)
        self.assertEqual(len({s.day for s in gate}), 6)
        self.assertGreaterEqual(len(gate), 60)
        self.assertTrue(all(s.label_end <= boundary for s in fitting))
        self.assertTrue(all(s.label_end <= min(v.issue for v in validation) for s in train))
        self.assertFalse({s.day for s in gate} & {s.day for s in fitting})
        with self.assertRaises(ValueError):
            d2.gate_split(all_samples, core.utc("2026-08-20T07:00:00+02:00"), minimum=100)

    def test_promotion_masks_and_circular_error(self):
        actual = np.array([1., np.nan, 10.])
        baseline = np.array([359., 20., 12.])
        candidate = np.array([0., 30., np.nan])
        champion = np.array([359., 1., 10.])
        result = d2.promotion(actual, baseline, candidate, champion, circular=True)
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["champion"]["mae"], 2.)
        self.assertTrue(result["promote"])
        self.assertFalse(d2.promotion(np.array([0.]), np.array([4.]), np.array([1.995]), np.array([2.]))["promote"])
        self.assertTrue(d2.promotion(np.array([0.]), np.array([4.]), np.array([1.98]), np.array([2.]))["promote"])
        with self.assertRaises(ValueError):
            d2.promotion(np.array([np.nan]), baseline[:1], candidate[:1], None)

    def test_initialize_retain_promote_and_failed_fit_keep_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            artifact = Path(temp)
            stored, candidate_error = {}, [2.]
            def fit(_samples, cutoff, kind, *_args, calibration_policy="none"):
                return SimpleNamespace(kind=kind, error=candidate_error[0], info={"cutoff_utc": cutoff.isoformat(),
                    "max_label_end_utc": max(s.label_end for s in _samples).isoformat(), "calibration_policy": calibration_policy})
            def predict(fit, gate):
                return np.stack([s.actual if fit.kind == "speed" else s.actual_dir for s in gate]) + fit.error
            def save(fit, path):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("dedicated checkpoint")
                stored[path] = fit
            with patch.object(core, "fit_model", side_effect=fit), patch.object(core, "predict", side_effect=predict), \
                 patch.object(core, "save_fit", side_effect=save), patch.object(core, "load_fit", side_effect=lambda p: stored[p]), \
                 patch.object(updater, "save_model_gate_eval_history_plot"):
                first = d2.train(samples(), core.utc("2026-08-20T07:00:00+02:00"), artifact,
                                availability_clock=lambda: core.utc("2026-08-20T07:24:00+02:00"))
                self.assertTrue(all(value["available_at_utc"] == "2026-08-20T05:24:00+00:00"
                                    for value in json.loads((artifact / "champions.json").read_text()).values()))
                self.assertTrue(first["promote_speed"])
                original = (artifact / "champions.json").read_bytes()
                candidate_error[0] = 3.
                second = d2.train(samples(), core.utc("2026-08-21T07:00:00+02:00"), artifact)
                self.assertFalse(second["promote_speed"])
                self.assertEqual((artifact / "champions.json").read_bytes(), original)
                candidate_error[0] = 1.
                third = d2.train(samples(), core.utc("2026-08-22T07:00:00+02:00"), artifact)
                self.assertTrue(third["promote_speed"])
                self.assertNotEqual((artifact / "champions.json").read_bytes(), original)
                active = (artifact / "champions.json").read_bytes()
                active_paths = {artifact / value["checkpoint"] for value in json.loads(active).values()}
                saved_paths = set(stored)
                def fail_direction(_samples, cutoff, kind, *_args, calibration_policy="none"):
                    if kind == "direction":
                        raise ValueError("No directional validation labels")
                    return fit(_samples, cutoff, kind)
                with patch.object(core, "fit_model", side_effect=fail_direction):
                    with self.assertRaises(ValueError):
                        # Retry the exact issue timestamp: candidate paths must
                        # stay distinct from the already active checkpoints.
                        d2.train(samples(), core.utc("2026-08-22T07:00:00+02:00"), artifact)
                self.assertEqual((artifact / "champions.json").read_bytes(), active)
                self.assertTrue(set(stored) - saved_paths)
                self.assertFalse((set(stored) - saved_paths) & active_paths)


class StageTests(unittest.TestCase):
    def test_new_champion_cannot_be_backdated_to_training_issue(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            issue = core.utc("2026-10-03T10:00:00+02:00")
            state = {kind: {"checkpoint": kind + ".pt", "trained_at_utc": issue.isoformat(),
                    "available_at_utc": (issue + pd.Timedelta(minutes=24)).isoformat()}
                    for kind in ["speed", "direction"]}
            d2.write_json(root / "champions.json", state)
            fit = SimpleNamespace(info={"operational": True, "max_label_end_utc": (issue - pd.Timedelta(days=1)).isoformat()})
            with patch.object(core, "load_fit", return_value=fit), patch.object(core.dp, "load_training_forecast_lookup") as lookup:
                with self.assertRaisesRegex(ValueError, "not available"):
                    d2.infer(root / "unused.db", issue, root, root)
                lookup.assert_not_called()

    def test_standalone_cached_refresh_loads_no_training_stack(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            times = core.calendar_targets(core.utc("2026-10-02T15:00:00+02:00"))
            values = np.full(15, 5., np.float32)
            table = updater.build_prediction_table({"target_times": times.astype(str).to_numpy(),
                "forecast_next24": values, "forecast_min_next24": values, "forecast_max_next24": values,
                "forecast_dir_next24": values}, values, values)
            table.to_csv(output / "day_after_tomorrow_predictions.csv", index=False)
            d2.write_json(output / "day_after_tomorrow_metadata.json", {"status": "available",
                "issue_time_utc": "2026-10-02T13:00:00+00:00", "target_date": "2026-10-04",
                "available_hours": 15, "champions": {"speed": {"trained_at_utc": "2026-10-02T05:00:00+00:00"}}})
            code = '''import sys,pandas as pd
from pathlib import Path
from next_day_wind_model.day_after_tomorrow import render_cached
render_cached(Path(sys.argv[1]),None,pd.Timestamp('2026-10-02T13:06:00Z'))
assert 'torch' not in sys.modules
assert 'next_day_wind_model.day_after_tomorrow_core' not in sys.modules
assert 'next_day_wind_model.day_after_tomorrow_experiment' not in sys.modules
print('cached renderer has no training imports')'''
            result = subprocess.run([sys.executable, "-c", code, str(output)], cwd=Path(__file__).resolve().parents[1],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("no training imports", result.stdout)

    def test_refresh_never_trains_infers_or_logs(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            with patch.object(d2, "historical_samples") as history, patch.object(d2, "train") as train, \
                 patch.object(d2, "infer") as infer, patch.object(d2, "log_predictions") as log, \
                 patch.object(d2, "render_cached", return_value={"status": "unavailable"}):
                d2.run_stage(args=args(), db_path=output / "db", out_dir=output,
                    model_artifact_dir=output / "models", now=core.utc("2026-10-02T15:06:00+02:00"))
                for mock in [history, train, infer, log]:
                    mock.assert_not_called()

    def test_disabled_site_and_failed_training_remain_nonfatal(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            settings = args()
            settings.site = "oostvoorne"
            self.assertEqual(d2.run_stage(args=settings, db_path=output / "db", out_dir=output,
                model_artifact_dir=output / "models")["status"], "disabled")
            with patch.object(d2, "train", side_effect=ValueError("too few contexts")), \
                 patch.object(d2, "infer", side_effect=FileNotFoundError("no champion")):
                result = d2.run_stage(args=args(), db_path=output / "db", out_dir=output,
                    model_artifact_dir=output / "models", training=True, prediction=True, samples=[],
                    now=core.utc("2026-10-02T15:00:00+02:00"))
                self.assertEqual(result["status"], "unavailable")
                self.assertTrue((output / "models/day_after_tomorrow/last_training_attempt.json").exists())

    def test_cached_date_is_not_relabelled_at_rollover(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            d2.write_json(output / "day_after_tomorrow_metadata.json", {"status": "available",
                "issue_time_utc": "2026-10-01T20:00:00+00:00", "target_date": "2026-10-03",
                "available_hours": 15, "champions": {}})
            state = d2.cached_status(output, core.utc("2026-10-02T07:00:00+02:00"))
            self.assertEqual(state["status"], "stale")
            self.assertEqual(state["target_date"], "2026-10-03")

    def test_logging_masks_provenance_and_model_type(self):
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / "db.sqlite"
            s = sample("2026-10-02T15:00:00+02:00", available=9)
            s.frame["run_ts"] = int(s.issue.timestamp() * 1000) - 1000
            s.frame["fetched_ts"] = s.frame.run_ts
            table = pd.DataFrame({"lstm_pred_wind_speed": [10.] * 15, "forecast_wind_speed": [9.] * 15,
                                  "lstm_pred_wind_dir_deg": [1.] * 15, "forecast_wind_dir_deg": [359.] * 15})
            metadata = {"available_hours": 9, "champions": {kind: {"model_id": kind, "checkpoint": kind + ".pt"} for kind in ["speed", "direction"]}}
            d2.log_predictions(db, s, table, metadata)
            d2.log_predictions(db, s, table, metadata)
            with sqlite3.connect(db) as conn:
                rows = conn.execute("SELECT model_type,anchor_ts,issued_ts,harmonie_fetched_ts FROM prediction_log").fetchall()
            self.assertEqual(len(rows), 18)
            self.assertTrue(all(row[0] == "day_after_tomorrow" and row[1] == row[2] and row[3] <= row[2] for row in rows))

    def test_future_champion_cannot_be_used_for_an_earlier_issue(self):
        with tempfile.TemporaryDirectory() as temp:
            artifact = Path(temp)
            state = {kind: {"checkpoint": kind + ".pt", "trained_at_utc": "2026-10-02T13:00:00+00:00"}
                     for kind in ["speed", "direction"]}
            d2.write_json(artifact / "champions.json", state)
            fit = SimpleNamespace(info={"operational": True, "max_label_end_utc": "2026-08-16T21:00:00+00:00"})
            with patch.object(core, "load_fit", return_value=fit), patch.object(core.dp, "load_training_forecast_lookup") as loader:
                with self.assertRaisesRegex(ValueError, "not available"):
                    d2.infer(artifact / "db", core.utc("2026-10-01T13:00:00Z"), artifact, artifact)
                loader.assert_not_called()

    def test_realized_direction_is_circular_and_other_horizons_untouched(self):
        from db_store import init_db, log_prediction_batch
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            db = output / "db.sqlite"
            target = core.utc("2026-10-04T08:00:00+02:00")
            target_ms = int(target.timestamp() * 1000)
            with sqlite3.connect(db) as conn:
                init_db(conn)
                conn.executemany("INSERT INTO observations(site,ts,wind_speed,wind_dir) VALUES(?,?,?,?)",
                    [(d2.SITE, target_ms, 10., 359.), (d2.SITE, target_ms + 1800000, 10., 1.)])
                for model_type in [d2.PREFIX, "next_day"]:
                    log_prediction_batch(conn, [{"site": d2.SITE, "model_type": model_type,
                        "prediction_kind": "wind_direction", "issued_ts": target_ms-86400000,
                        "anchor_ts": target_ms-86400000, "target_ts": target_ms,
                        "prediction_value": 1., "harmonie_value": 359.}])
            d2.realized_scores(db, target + pd.Timedelta(minutes=30), output)
            with sqlite3.connect(db) as conn:
                self.assertIsNone(conn.execute("SELECT actual_value FROM prediction_log WHERE model_type=?", (d2.PREFIX,)).fetchone()[0])
            d2.realized_scores(db, target + pd.Timedelta(hours=1), output)
            with sqlite3.connect(db) as conn:
                actual, model_error, baseline_error = conn.execute("SELECT actual_value,model_error,harmonie_error FROM prediction_log WHERE model_type=?", (d2.PREFIX,)).fetchone()
                self.assertAlmostEqual(actual % 360, 0., places=5)
                self.assertAlmostEqual(model_error, 1., places=5)
                self.assertAlmostEqual(baseline_error, -1., places=5)
                self.assertIsNone(conn.execute("SELECT actual_value FROM prediction_log WHERE model_type='next_day'").fetchone()[0])

    def test_refresh_payload_and_csv_preserve_masked_gaps(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            targets = core.calendar_targets(core.utc("2026-10-02T15:00:00+02:00"))
            values = np.r_[np.ones(9), np.full(6, np.nan)].astype(np.float32)
            frame = updater.build_prediction_table({"target_times": targets.astype(str).to_numpy(),
                "forecast_next24": values, "forecast_min_next24": values, "forecast_max_next24": values,
                "forecast_dir_next24": values}, values, values)
            csv = output / "day_after_tomorrow_predictions.csv"
            frame.to_csv(csv, index=False)
            original = csv.read_bytes()
            d2.write_json(output / "day_after_tomorrow_metadata.json", {"status": "available",
                "issue_time_utc": "2026-10-02T13:00:00+00:00", "target_date": "2026-10-04",
                "available_hours": 9, "champions": {"speed": {"trained_at_utc": "2026-10-02T05:00:00+00:00",
                    "available_at_utc": "2026-10-02T05:24:00+00:00"}}})
            with patch.object(updater, "save_prediction_plot") as renderer:
                state = d2.render_cached(output, None, core.utc("2026-10-02T15:06:00+02:00"))
                self.assertEqual(renderer.call_count, 2)
                self.assertEqual(renderer.call_args.kwargs["model_trained_at_utc"], "2026-10-02T05:24:00+00:00")
                self.assertEqual(state["available_hours"], 9)
            payload = json.loads((output / "day_after_tomorrow_interactive_data.json").read_text())
            self.assertEqual(len(payload["rows"]), 15)
            self.assertIsNone(payload["rows"][-1]["lstm_pred_wind_speed"])
            self.assertTrue(payload["rows"][0]["target_time_utc"].startswith("2026-10-04"))
            self.assertEqual(csv.read_bytes(), original)


class PublicationTests(unittest.TestCase):
    def test_evaluation_moves_content_and_disabled_forecast_stays_two_horizons(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "source.png"
            image.write_bytes(b"fixture")
            csv = root / "next.csv"
            pd.DataFrame({"target_time_local": ["2026-10-03T08:00:00+02:00"]}).to_csv(csv, index=False)
            kwargs = dict(web_out_dir=root / "web", local_tz=core.TZ, web_refresh_seconds=360,
                next_day_png=image, next_day_png_mobile=None, next_day_csv=csv,
                current_day_png=image, current_day_png_mobile=None, current_day_csv=root / "missing.csv",
                daily_mae_png=image, daily_mae_png_mobile=image, daily_mae_csv=None,
                gate_eval_png=image, gate_eval_csv=None, direction_spider_png=image, direction_spider_csv=None,
                current_day_direction_spider_png=image, current_day_direction_spider_csv=None,
                spot_name="Valkenburgse Meer", companion_app_base_url="https://portal.example")
            state = {"status": "unavailable", "reason": "Awaiting valid champion"}
            copied = updater.publish_web_dashboard(**kwargs, day_after_tomorrow_state=state)
            index = (root / "web/index.html").read_text()
            evaluation = (root / "web/evaluation.html").read_text()
            self.assertIn('href="evaluation.html"', index)
            self.assertIn('id="day-after-tomorrow-status"', index)
            self.assertNotIn('alt="Model gate evaluation history"', index)
            self.assertNotIn('alt="Next-day prediction performance by wind direction"', index)
            self.assertIn('alt="Model gate evaluation history"', evaluation)
            self.assertIn('alt="Next-day prediction performance by wind direction"', evaluation)
            self.assertNotIn("Realised forecast MAE history", evaluation)
            self.assertNotIn("Evaluation downloads", evaluation)
            self.assertIn('href="index.html">Forecasts</a>', evaluation)
            self.assertIn('href="https://portal.example/">Rider portal</a>', evaluation)
            self.assertEqual(evaluation.count('style="height:44px"'), 2)
            self.assertIn("evaluation.html", copied)
            meta = json.loads((root / "web/metadata_update.json").read_text())
            self.assertEqual(meta["pages"], ["index.html", "evaluation.html"])
            updater.publish_web_dashboard(**kwargs, day_after_tomorrow_state={"status": "disabled"})
            index = (root / "web/index.html").read_text()
            self.assertNotIn('id="day-after-tomorrow-status"', index)
            self.assertIn("combines two local", index)

if __name__ == "__main__":
    unittest.main()
