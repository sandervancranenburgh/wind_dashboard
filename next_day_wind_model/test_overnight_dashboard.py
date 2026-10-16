from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from next_day_wind_model import data_pipeline
from next_day_wind_model.operational_update import (
    CURRENT_PREDICTION_COLUMNS, NEXT_PREDICTION_COLUMNS,
    validate_cached_prediction_artifacts,
)
from next_day_wind_model.update_model_and_predict import _format_plot_meta_text

ROOT = Path(__file__).resolve().parents[1]


class OvernightMetadataTests(unittest.TestCase):
    def metadata(self, now: str, plotted: str) -> str:
        with mock.patch("next_day_wind_model.update_model_and_predict.datetime", wraps=datetime) as clock:
            clock.now.return_value = datetime.fromisoformat(now)
            return _format_plot_meta_text(
                plotted, "2026-10-15T20:43:00Z", None, "Europe/Amsterdam",
                harmonie_time_utc="2026-10-15T20:42:00Z",
                harmonie_expected_next_at_utc="2026-10-15T21:42:00Z",
            )

    def test_penultimate_run_still_announces_the_final_daytime_poll(self):
        text = self.metadata("2026-10-15T22:48:05+02:00", "2026-10-15T20:48:05Z")
        self.assertIn("Next update: 22:54", text)

    def test_last_evening_plot_announces_midnight_and_morning_fetch(self):
        text = self.metadata("2026-10-15T22:54:00+02:00", "2026-10-15T20:54:00Z")
        self.assertIn("Next update: 16 October 00:00", text)
        self.assertIn("Next expected update: ~16 October 07:00", text)
        self.assertIn("Next expected fetch: ~16 October 07:00", text)

    def test_midnight_refresh_announces_morning_and_preserves_fetch_date(self):
        text = self.metadata("2026-10-16T00:00:15+02:00", "2026-10-15T22:00:15Z")
        self.assertIn("Last plot update: 00:00 - Next update: 07:00", text)
        self.assertIn("Last prediction update: 15 October 22:43 - Next expected update: ~07:00", text)
        self.assertIn("Last HARMONIE fetch: 15 October 22:42 - Next expected fetch: ~07:00", text)

    def test_manual_early_morning_refresh_does_not_predict_an_overnight_fetch(self):
        text = self.metadata("2026-10-16T06:30:00+02:00", "2026-10-16T04:30:00Z")
        self.assertIn("Next update: 07:00", text)
        self.assertIn("Next expected fetch: ~07:00", text)


class NextDayCalendarTests(unittest.TestCase):
    def test_tomorrow_uses_issue_date_with_yesterdays_observations(self):
        # Exercise feature construction with a real isolated observation DB;
        # provide forecast frames for the exact windows requested by inference.
        cases = [
            ("2026-10-15T20:54:00Z", "2026-10-15T22:00:00Z", "2026-10-17"),
            ("2026-10-15T20:54:00Z", "2026-10-16T04:30:00Z", "2026-10-17"),
            ("2026-03-28T21:54:00Z", "2026-03-28T23:00:00Z", "2026-03-30"),
            ("2026-10-24T20:54:00Z", "2026-10-24T22:00:00Z", "2026-10-26"),
        ]
        def context(**kwargs):
            def frame(times):
                values = {name: 1.0 for name in data_pipeline._legacy_feature_cols()}
                values.update({name: 1.0 for name in (
                    "forecast_min", "forecast_temperature", "forecast_weather_code",
                    "run_ts", "fetched_ts", "horizon_hr",
                )})
                return pd.DataFrame(values, index=times)
            return {"history_frame": frame(kwargs["history_times"]),
                    "target_frame": frame(kwargs["target_times"])}

        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "observations.db"
            import sqlite3
            with sqlite3.connect(db) as conn:
                conn.execute("CREATE TABLE observations (site TEXT, ts INTEGER)")
            for last_obs, issue, tomorrow in cases:
                with self.subTest(issue=issue):
                    with sqlite3.connect(db) as conn:
                        conn.execute("DELETE FROM observations")
                        conn.execute("INSERT INTO observations VALUES (?, ?)",
                                     ("valkenburgsemeer", int(pd.Timestamp(last_obs).timestamp() * 1000)))
                    width = len(data_pipeline._legacy_feature_cols())
                    with mock.patch.object(data_pipeline, "build_inference_forecast_context", side_effect=context):
                        result = data_pipeline.build_next_day_inference_input(
                            db, data_pipeline.DatasetConfig(), np.zeros(width), np.ones(width),
                            reference_time_utc=issue,
                        )
                    local_start = pd.Timestamp(result["prediction_day_start_local"])
                    self.assertEqual(str(local_start.date()), tomorrow)
                    self.assertEqual(local_start.hour, 0)
                    self.assertEqual(result["X_input"].shape, (1, 72, width))


class MidnightPipelineTests(unittest.TestCase):
    def test_refresh_only_skips_network_and_keeps_the_updater_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / "scripts"
            scripts.mkdir()
            for name in ("run_wind_dashboard_pipeline.sh", "run_wind_updater_locked.sh"):
                shutil.copy2(ROOT / "scripts" / name, scripts / name)
            capture = root / "commands.jsonl"
            recorder = root / "record-python"
            import sys
            recorder.write_text(
                "#!" + sys.executable + "\nimport json, os, sys\n"
                "with open(os.environ['TEST_COMMANDS'], 'a') as f: "
                "f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            )
            recorder.chmod(0o755)
            env = dict(os.environ, WIND_FETCHER_PYTHON_BIN=str(recorder),
                       WIND_UPDATER_LOCK_FILE=str(root / "updater.lock"), TEST_COMMANDS=str(capture))
            env.pop("WIND_DASHBOARD_PIPELINE_LOCK_HELD", None)
            env.pop("WIND_USE_LEGACY_MODEL_ARTIFACTS", None)
            for options, expected_count in ((["--refresh-only"], 1), ([], 2)):
                with self.subTest(options=options):
                    capture.unlink(missing_ok=True)
                    subprocess.run(["bash", str(scripts / "run_wind_dashboard_pipeline.sh"), *options],
                                   env=env, check=True, capture_output=True, text=True)
                    commands = [json.loads(line) for line in capture.read_text().splitlines()]
                    self.assertEqual(len(commands), expected_count)
                    if not options:
                        self.assertEqual(commands[0], ["source_fetch.py", "data", "--all-sites"])
                    updater = commands[-1]
                    self.assertEqual(updater[0], "next_day_wind_model/update_model_and_predict.py")
                    self.assertIn("--skip-training", updater)
                    self.assertIn("--skip-data-refresh-check", updater)
                    self.assertNotIn("--plots-only", updater)
                    self.assertTrue((root / "updater.lock").exists())

    def test_cache_rejects_yesterdays_current_day_and_wrong_tomorrow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("current_day_predictions.png", "current_day_predictions_mobile.png",
                         "next_day_predictions.png", "next_day_predictions_mobile.png"):
                (root / name).write_bytes(b"test")
            (root / "metadata_update.json").write_text("{}")
            def write_csv(name, time_col, timestamp, columns):
                with (root / name).open("w") as handle:
                    writer = csv.DictWriter(handle, fieldnames=[time_col, *columns])
                    writer.writeheader()
                    writer.writerow({time_col: timestamp, **{col: 1 for col in columns}})
            write_csv("current_day_predictions.csv", "time_local", "2026-10-15T12:00:00+02:00",
                      CURRENT_PREDICTION_COLUMNS)
            write_csv("next_day_predictions.csv", "target_time_utc", "2026-10-16T10:00:00Z",
                      NEXT_PREDICTION_COLUMNS)
            def validate():
                return validate_cached_prediction_artifacts(
                    root, local_timezone="Europe/Amsterdam",
                    now_utc=datetime(2026, 10, 15, 22, tzinfo=timezone.utc))
            self.assertEqual(validate().reason, "current_day_cache_is_stale")
            write_csv("current_day_predictions.csv", "time_local", "2026-10-16T12:00:00+02:00",
                      CURRENT_PREDICTION_COLUMNS)
            self.assertEqual(validate().reason, "next_day_cache_is_stale")
            write_csv("next_day_predictions.csv", "target_time_utc", "2026-10-17T10:00:00Z",
                      NEXT_PREDICTION_COLUMNS)
            self.assertTrue(validate().valid)
