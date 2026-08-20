from __future__ import annotations

import sqlite3
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

import pandas as pd
from PIL import Image, ImageDraw

import source_fetch
from next_day_wind_model.update_model_and_predict import (
    _current_day_direction_arrow_rows,
    _estimate_next_harmonie_arrival,
    _format_plot_meta_text,
    _load_latest_harmonie_metadata_time,
    _load_recent_harmonie_arrivals,
    auto_push_dashboard_changes,
)
from scripts.regenerate_favicons import remove_exterior_white_matte


class SourceFetchSequencingTests(unittest.TestCase):
    def test_second_site_failure_propagates_to_the_combined_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                source_fetch,
                "fetch_site",
                side_effect=[None, RuntimeError("second site failed")],
            ) as fetch_site:
                with self.assertRaisesRegex(RuntimeError, "second site failed"):
                    source_fetch.main([directory])

        self.assertEqual(
            [call.args[0].site for call in fetch_site.call_args_list],
            ["valkenburgsemeer", "oostvoorne"],
        )


class UpdateMetadataTests(unittest.TestCase):
    def test_process_times_and_next_updates_use_their_configured_cadences(self) -> None:
        tz = ZoneInfo("Europe/Amsterdam")
        today = datetime.now(tz)

        def utc_at(hour: int, minute: int) -> str:
            return today.replace(hour=hour, minute=minute, second=0, microsecond=0).astimezone(
                timezone.utc
            ).isoformat()

        text = _format_plot_meta_text(
            utc_at(21, 14),
            utc_at(21, 1),
            utc_at(7, 15),
            "Europe/Amsterdam",
            harmonie_time_utc=utc_at(21, 0),
            harmonie_time_kind="fetched",
            plot_update_interval_minutes=6,
            harmonie_update_interval_minutes=60,
        )

        self.assertIn("Last plot update: 21:14 - Next update: 21:20", text)
        self.assertIn("Last prediction update: 21:01 - Next expected update: ~22:00", text)
        self.assertIn("Last HARMONIE fetch: 21:00 - Next expected fetch: ~22:00", text)

    def test_prediction_next_update_is_unknown_without_harmonie_arrival_time(self) -> None:
        tz = ZoneInfo("Europe/Amsterdam")
        today = datetime.now(tz)
        prediction_time = today.replace(hour=21, minute=1, second=0, microsecond=0).astimezone(
            timezone.utc
        )

        text = _format_plot_meta_text(
            prediction_time,
            prediction_time,
            None,
            "Europe/Amsterdam",
            harmonie_time_utc=None,
            harmonie_update_interval_minutes=60,
        )

        self.assertIn(
            "Last prediction update: 21:01 - Next expected update: unknown",
            text,
        )

    def test_production_forecast_fetch_precedes_shadow_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "fixture.db"
            conn = sqlite3.connect(db_path)
            conn.executescript(
                """
                CREATE TABLE forecasts (
                    site TEXT, model TEXT, run_ts INTEGER, fetched_ts INTEGER, target_ts INTEGER
                );
                CREATE TABLE harmonie_knmi_features (
                    site TEXT, fetched_ts TEXT, run_ts TEXT, horizon_hr INTEGER
                );
                """
            )
            conn.execute(
                "INSERT INTO forecasts VALUES (?, ?, ?, ?, ?)",
                ("site", "HARMONIE", 1_787_250_060_000, 1_787_250_061_000, 1_787_250_060_000),
            )
            conn.execute(
                "INSERT INTO harmonie_knmi_features VALUES (?, ?, ?, ?)",
                ("site", "2026-08-20T22:00:00Z", "2026-08-20T21:00:00Z", 0),
            )
            conn.commit()
            conn.close()

            value, kind = _load_latest_harmonie_metadata_time(db_path, "site")

        self.assertEqual(kind, "fetched")
        self.assertEqual(value, datetime.fromtimestamp(1_787_250_061, tz=timezone.utc))

    def test_recent_arrival_phase_estimates_normal_cadence_for_both_lines(self) -> None:
        arrivals = [
            "2026-08-20T16:43:00+00:00",
            "2026-08-20T17:42:00+00:00",
            "2026-08-20T18:44:00+00:00",
            "2026-08-20T19:42:00+00:00",
        ]

        estimate = _estimate_next_harmonie_arrival(arrivals, arrivals[-1])

        self.assertEqual(estimate.method, "median_recent_arrival_phase")
        self.assertEqual(
            estimate.expected_at_utc,
            datetime(2026, 8, 20, 20, 42, 30, tzinfo=timezone.utc),
        )
        text = _format_plot_meta_text(
            "2026-08-20T19:48:00+00:00",
            "2026-08-20T19:47:00+00:00",
            None,
            "Europe/Amsterdam",
            harmonie_time_utc=arrivals[-1],
            harmonie_expected_next_at_utc=estimate.expected_at_utc,
        )
        self.assertIn("Last prediction update: 21:47 - Next expected update: ~22:42", text)
        self.assertIn("Last HARMONIE fetch: 21:42 - Next expected fetch: ~22:42", text)

    def test_one_delayed_cycle_does_not_pull_the_median_estimate(self) -> None:
        arrivals = [
            "2026-08-20T15:42:00+00:00",
            "2026-08-20T16:43:00+00:00",
            "2026-08-20T17:58:00+00:00",
            "2026-08-20T18:42:00+00:00",
            "2026-08-20T19:44:00+00:00",
        ]

        estimate = _estimate_next_harmonie_arrival(arrivals, arrivals[-1])

        self.assertEqual(estimate.method, "median_recent_arrival_phase")
        self.assertEqual(
            estimate.expected_at_utc,
            datetime(2026, 8, 20, 20, 43, tzinfo=timezone.utc),
        )

    def test_insufficient_arrival_history_falls_back_to_nominal_interval(self) -> None:
        arrivals = [
            "2026-08-20T18:43:00+00:00",
            "2026-08-20T19:42:00+00:00",
        ]

        estimate = _estimate_next_harmonie_arrival(arrivals, arrivals[-1])

        self.assertEqual(estimate.method, "nominal_interval_fallback")
        self.assertEqual(
            estimate.expected_at_utc,
            datetime(2026, 8, 20, 20, 42, tzinfo=timezone.utc),
        )

    def test_repeated_poll_contents_count_as_one_arrival(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "fixture.db"
            conn = sqlite3.connect(db_path)
            conn.execute(
                """
                CREATE TABLE forecasts (
                    site TEXT, model TEXT, run_ts INTEGER, fetched_ts INTEGER,
                    target_ts INTEGER, wind_speed REAL, wind_gust REAL,
                    wind_dir REAL, payload TEXT
                )
                """
            )
            poll_times = [
                "2026-08-20T18:42:00+00:00",
                "2026-08-20T18:48:00+00:00",
                "2026-08-20T19:42:00+00:00",
                "2026-08-20T19:48:00+00:00",
            ]
            for poll_index, poll_time in enumerate(poll_times):
                fetched_ms = int(pd.Timestamp(poll_time).timestamp() * 1000)
                forecast_version = 0 if poll_index < 2 else 1
                for target_index in range(24):
                    target_ms = int(
                        (pd.Timestamp("2026-08-21T00:00:00+00:00") + pd.Timedelta(hours=target_index)).timestamp()
                        * 1000
                    )
                    conn.execute(
                        "INSERT INTO forecasts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            "site",
                            "HARMONIE",
                            fetched_ms,
                            fetched_ms,
                            target_ms,
                            10.0 + forecast_version,
                            14.0 + forecast_version,
                            220.0,
                            "{}",
                        ),
                    )
            conn.commit()
            conn.close()

            arrivals = _load_recent_harmonie_arrivals(db_path, "site")

        self.assertEqual(
            arrivals,
            [
                datetime(2026, 8, 20, 18, 42, tzinfo=timezone.utc),
                datetime(2026, 8, 20, 19, 42, tzinfo=timezone.utc),
            ],
        )


class MeasuredDirectionArrowTests(unittest.TestCase):
    def test_nearest_real_observation_per_hour_is_selected_separately(self) -> None:
        times = pd.to_datetime(
            [
                "2026-08-20T08:00:00Z",
                "2026-08-20T08:02:10Z",
                "2026-08-20T08:58:40Z",
                "2026-08-20T09:00:00Z",
                "2026-08-20T09:03:10Z",
            ],
            utc=True,
        )
        table = pd.DataFrame(
            {
                "time_local": times,
                "is_forecast_grid": [True, False, False, True, False],
                "is_actual_observation": [False, True, True, False, True],
                "forecast_wind_dir_deg": [180.0, None, None, 190.0, None],
                "lstm_pred_wind_dir_deg": [185.0, None, None, 195.0, None],
                "lstm_pred_wind_dir_deg_full": [185.0, None, None, 195.0, None],
                "actual_wind_dir_deg": [None, 170.0, 175.0, None, 180.0],
            }
        )

        forecast_rows, actual_rows = _current_day_direction_arrow_rows(table)

        self.assertEqual(list(forecast_rows["forecast_wind_dir_deg"]), [180.0, 190.0])
        self.assertEqual(list(actual_rows["actual_wind_dir_deg"]), [170.0, 175.0])
        self.assertTrue(actual_rows["is_actual_observation"].all())


class PublishSafetyTests(unittest.TestCase):
    def test_auto_publish_refuses_a_different_checked_out_branch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / ".git").mkdir()
            completed = subprocess.CompletedProcess([], 0, stdout="feature-branch\n", stderr="")
            with mock.patch("subprocess.run", return_value=completed):
                result = auto_push_dashboard_changes(repo, repo / "docs", "origin", "main")
        self.assertEqual(result["reason"], "current_branch_mismatch")
        self.assertEqual(result["current_branch"], "feature-branch")


class FaviconTransparencyTests(unittest.TestCase):
    def test_exterior_white_becomes_transparent_but_interior_white_stays_opaque(self) -> None:
        source = Image.new("RGB", (32, 32), "white")
        draw = ImageDraw.Draw(source)
        draw.rounded_rectangle((2, 2, 29, 29), radius=7, fill=(0, 30, 65))
        draw.ellipse((13, 13, 18, 18), fill="white")

        cleaned = remove_exterior_white_matte(source)

        self.assertEqual(cleaned.getpixel((0, 0))[3], 0)
        self.assertEqual(cleaned.getpixel((15, 15)), (255, 255, 255, 255))
        self.assertEqual(cleaned.getpixel((15, 3))[3], 255)

    def test_cleanup_is_idempotent_for_an_already_transparent_master(self) -> None:
        source = Image.new("RGB", (9, 9), "white")
        pixels = source.load()
        for y in range(1, 8):
            for x in range(1, 8):
                pixels[x, y] = (0, 32, 64)
        once = remove_exterior_white_matte(source)
        twice = remove_exterior_white_matte(once)
        self.assertEqual(once.mode, twice.mode)
        self.assertEqual(once.size, twice.size)
        self.assertEqual(once.tobytes(), twice.tobytes())


if __name__ == "__main__":
    unittest.main()
