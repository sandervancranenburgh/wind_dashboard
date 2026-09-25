from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("WIND_PLOT_RENDERER_ONLY", "1")

from next_day_wind_model.ecmwf_dashboard import (
    ECMWF_FORECAST_COLOR,
    ECMWF_FORECAST_LINEWIDTH,
    load_ecmwf_plot_data,
)
from next_day_wind_model.operational_update import (
    CachedArtifactStatus,
    ExecutionDecision,
    ForecastIdentity,
    OperationalSnapshot,
    decide_execution_mode,
)
from next_day_wind_model.update_model_and_predict import (
    save_current_day_plot,
    save_prediction_plot,
)


LOCAL_TZ = "Europe/Amsterdam"
DAY = "2026-09-16"


def current_table() -> pd.DataFrame:
    times = pd.date_range(f"{DAY} 08:00", f"{DAY} 22:00", freq="1h", tz=LOCAL_TZ)
    count = len(times)
    actual = np.full(count, np.nan)
    actual[:4] = [6.0, 7.0, 8.0, 9.0]
    actual_dir = np.full(count, np.nan)
    actual_dir[:4] = [210.0, 215.0, 220.0, 225.0]
    return pd.DataFrame(
        {
            "time_local": times,
            "is_forecast_grid": True,
            "is_actual_observation": np.isfinite(actual),
            "forecast_wind_speed": np.linspace(7.0, 10.0, count),
            "forecast_wind_min": np.linspace(6.0, 9.0, count),
            "forecast_wind_max": np.linspace(8.0, 11.0, count),
            "forecast_wind_dir_deg": np.linspace(180.0, 240.0, count),
            "lstm_pred_wind_speed_full": np.linspace(7.5, 10.5, count),
            "lstm_pred_wind_dir_deg_full": np.linspace(185.0, 245.0, count),
            "lstm_pred_wind_speed": np.linspace(7.5, 10.5, count),
            "lstm_pred_wind_dir_deg": np.linspace(185.0, 245.0, count),
            "actual_wind_speed": actual,
            "actual_wind_min": actual - 1.0,
            "actual_wind_max": actual + 1.0,
            "actual_wind_dir_deg": actual_dir,
            "is_future": times >= pd.Timestamp(f"{DAY} 12:00", tz=LOCAL_TZ),
            "hour_local": times.strftime("%H"),
            "minute_local": times.minute,
            "forecast_temperature_c": np.linspace(12.0, 18.0, count),
            "weather_code": [0, 1, 2, 3, 45, 51, 53, 55, 61, 63, 65, 71, 73, 75, 0],
            "weather_source": "windsurfice",
            "is_daylight": [True] * 12 + [False] * 3,
        }
    )


def next_table() -> pd.DataFrame:
    times = pd.date_range("2026-09-17 08:00", "2026-09-17 22:00", freq="1h", tz=LOCAL_TZ)
    count = len(times)
    return pd.DataFrame(
        {
            "target_time_utc": times.tz_convert("UTC"),
            "target_time_local": times,
            "forecast_wind_speed": np.linspace(8.0, 11.0, count),
            "forecast_wind_min": np.linspace(7.0, 10.0, count),
            "forecast_wind_max": np.linspace(9.0, 12.0, count),
            "lstm_pred_wind_speed": np.linspace(8.5, 11.5, count),
            "forecast_wind_dir_deg": np.linspace(180.0, 240.0, count),
            "lstm_pred_wind_dir_deg": np.linspace(185.0, 245.0, count),
            "forecast_temperature_c": np.linspace(11.0, 17.0, count),
            "weather_code": [3, 2, 1, 0, 51, 53, 55, 61, 63, 65, 71, 73, 75, 45, 0],
            "weather_source": "windsurfice",
            "is_daylight": [True] * 12 + [False] * 3,
        }
    )


def current_kwargs() -> dict:
    return {
        "local_tz": LOCAL_TZ,
        "prediction_generated_at_utc": "2026-09-16T07:30:00Z",
        "prediction_updated_at_utc": "2026-09-16T07:31:00Z",
        "model_trained_at_utc": "2026-09-15T05:25:00Z",
        "harmonie_time_utc": "2026-09-16T07:25:00Z",
        "spot_name": "Valkenburgse meer",
        "plot_updated_at_utc": "2026-09-16T08:00:00Z",
        "prior_prediction_tables": [],
        "live_monitoring_metric": {
            "available": True,
            "mae_superlocal": 1.0,
            "mae_harmonie": 2.0,
            "measurement_point_count": 4,
        },
    }


def next_kwargs() -> dict:
    return {
        "local_tz": LOCAL_TZ,
        "plot_updated_at_utc": "2026-09-16T08:00:00Z",
        "prediction_updated_at_utc": "2026-09-16T07:31:00Z",
        "model_trained_at_utc": "2026-09-15T05:25:00Z",
        "harmonie_time_utc": "2026-09-16T07:25:00Z",
        "spot_name": "Valkenburgse meer",
    }


def bounds_overlap(first: list[float], second: list[float]) -> bool:
    return not (
        first[2] <= second[0]
        or second[2] <= first[0]
        or first[3] <= second[1]
        or second[3] <= first[1]
    )


def assert_header_bounds_do_not_overlap(
    case: unittest.TestCase, diagnostics: dict[str, object]
) -> None:
    bounds = diagnostics["header_bounds_figure"]
    names = list(bounds)
    for index, first_name in enumerate(names):
        for second_name in names[index + 1 :]:
            case.assertFalse(
                bounds_overlap(bounds[first_name], bounds[second_name]),
                f"{first_name} overlaps {second_name}: {bounds}",
            )


def overlay_for(day: str, *, extreme: bool = False) -> pd.DataFrame:
    times = pd.to_datetime(
        [
            f"{day}T06:00:00Z",
            f"{day}T09:00:00Z",
            f"{day}T12:00:00Z",
            f"{day}T15:00:00Z",
            f"{day}T18:00:00Z",
            f"{day}T21:00:00Z",
        ],
        utc=True,
    )
    values = [7.0, 8.0, 9.0, 10.0, 99.0 if extreme else 9.0, 8.0]
    return pd.DataFrame({"time_local": times, "wind_speed_knots": values})


class RendererIntegrationTests(unittest.TestCase):
    def test_current_day_solid_style_post_xmax_and_unchanged_layout(self) -> None:
        table = current_table()
        overlay = overlay_for(DAY, extreme=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference: dict[str, object] = {}
            development: dict[str, object] = {}
            save_current_day_plot(
                table,
                root / "reference.png",
                render_diagnostics=reference,
                **current_kwargs(),
            )
            save_current_day_plot(
                table,
                root / "ecmwf.png",
                ecmwf_speed_series=overlay,
                ecmwf_metadata_text="Last ECMWF fetch: 09:48 - Next expected fetch: ~16:00",
                render_diagnostics=development,
                **current_kwargs(),
            )

        self.assertEqual(development["ecmwf_color"], ECMWF_FORECAST_COLOR)
        self.assertEqual(development["ecmwf_linestyle"], "-")
        self.assertEqual(development["ecmwf_linewidth"], ECMWF_FORECAST_LINEWIDTH)
        self.assertEqual(development["ecmwf_marker"], "None")
        self.assertEqual(development["ecmwf_legend_marker"], "None")
        plotted = pd.to_datetime(development["ecmwf_times_local"], utc=True)
        self.assertGreater(plotted.max(), pd.Timestamp("2026-09-16T20:00:00Z"))
        self.assertEqual(reference["x_limits"], development["x_limits"])
        self.assertEqual(reference["speed_y_limits"], development["speed_y_limits"])
        self.assertEqual(reference["speed_y_ticks"], development["speed_y_ticks"])
        self.assertEqual(reference["axes_positions"], development["axes_positions"])
        self.assertEqual(
            reference["axis_line_colors"][1:], development["axis_line_colors"][1:]
        )
        self.assertEqual(
            reference["axis_annotation_counts"][1:],
            development["axis_annotation_counts"][1:],
        )
        self.assertGreater(
            development["metadata_position_axes"][1],
            reference["metadata_position_axes"][1],
        )

    def test_next_day_uses_same_solid_color_and_extreme_does_not_change_axis(self) -> None:
        table = next_table()
        overlay = overlay_for("2026-09-17", extreme=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference: dict[str, object] = {}
            development: dict[str, object] = {}
            save_prediction_plot(
                table,
                root / "reference.png",
                render_diagnostics=reference,
                **next_kwargs(),
            )
            save_prediction_plot(
                table,
                root / "ecmwf.png",
                ecmwf_speed_series=overlay,
                render_diagnostics=development,
                **next_kwargs(),
            )

        self.assertEqual(development["ecmwf_color"], ECMWF_FORECAST_COLOR)
        self.assertEqual(development["ecmwf_linestyle"], "-")
        self.assertEqual(development["ecmwf_linewidth"], ECMWF_FORECAST_LINEWIDTH)
        self.assertEqual(development["ecmwf_marker"], "None")
        self.assertGreater(max(development["ecmwf_x_data"]), development["x_limits"][1])
        plotted = pd.to_datetime(development["ecmwf_times_local"], utc=True)
        self.assertGreater(plotted.max(), pd.Timestamp("2026-09-17T20:00:00Z"))
        self.assertEqual(reference["x_limits"], development["x_limits"])
        self.assertEqual(reference["y_limits"], development["y_limits"])
        self.assertEqual(reference["y_ticks"], development["y_ticks"])
        self.assertEqual(reference["axes_position"], development["axes_position"])
        self.assertEqual(
            development["legend_labels"],
            [
                "Super local wind prediction - avg speed",
                "Harmonie model - avg speed",
                "ECMWF forecast",
                "Harmonie model - max speed",
            ],
        )

    def test_renderers_remain_backward_compatible_without_ecmwf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current_diagnostics: dict[str, object] = {}
            next_diagnostics: dict[str, object] = {}
            save_current_day_plot(
                current_table(),
                root / "current.png",
                render_diagnostics=current_diagnostics,
                **current_kwargs(),
            )
            save_prediction_plot(
                next_table(),
                root / "next.png",
                render_diagnostics=next_diagnostics,
                **next_kwargs(),
            )
        self.assertFalse(current_diagnostics["ecmwf_plotted"])
        self.assertFalse(next_diagnostics["ecmwf_plotted"])
        self.assertNotIn("ECMWF forecast", current_diagnostics["legend_labels"])
        self.assertNotIn("ECMWF forecast", next_diagnostics["legend_labels"])
        self.assertEqual(current_diagnostics["axis_count"], 4)
        self.assertEqual(
            current_diagnostics["axis_roles"],
            ["wind_speed", "weather", "variability", "direction"],
        )
        self.assertEqual(current_diagnostics["weather_cell_count"], 14)
        self.assertEqual(current_diagnostics["weather_icon_count"], 14)
        self.assertEqual(current_diagnostics["direction_arrow_count"], 34)
        self.assertEqual(next_diagnostics["axis_count"], 3)
        self.assertEqual(
            next_diagnostics["axis_roles"],
            ["wind_speed", "weather", "direction"],
        )
        self.assertEqual(next_diagnostics["weather_cell_count"], 14)
        self.assertEqual(next_diagnostics["weather_icon_count"], 14)
        self.assertEqual(next_diagnostics["weather_background_count"], 14)
        self.assertEqual(next_diagnostics["weather_separator_count"], 15)
        self.assertEqual(current_diagnostics["weather_icon_zoom"], 0.36)
        self.assertEqual(next_diagnostics["weather_icon_zoom"], 0.36)
        self.assertEqual(current_diagnostics["weather_icon_y"], 0.20)
        self.assertEqual(next_diagnostics["weather_icon_y"], 0.25)
        self.assertEqual(current_diagnostics["weather_icon_temperature_overlap_count"], 0)
        self.assertEqual(next_diagnostics["weather_icon_temperature_overlap_count"], 0)
        expected_hours = [f"{hour:02d}h" for hour in range(8, 23)]
        self.assertEqual(current_diagnostics["x_tick_labels"], expected_hours)
        self.assertEqual(next_diagnostics["x_tick_labels"], expected_hours)
        self.assertEqual(current_diagnostics["x_axis_labels"], ["", "", "", ""])
        self.assertEqual(next_diagnostics["x_axis_labels"], ["", "", ""])
        self.assertNotIn("unknown", next_diagnostics["model_id_text"].lower())
        self.assertGreaterEqual(
            next_diagnostics["header_bounds_figure"]["metadata"][1],
            next_diagnostics["axes_position"][1]
            + next_diagnostics["axes_position"][3],
        )
        self.assertEqual(current_diagnostics["spot_name"], "Valkenburgse meer")
        self.assertEqual(next_diagnostics["spot_name"], "Valkenburgse meer")
        assert_header_bounds_do_not_overlap(self, current_diagnostics)
        assert_header_bounds_do_not_overlap(self, next_diagnostics)
        self.assertEqual(next_diagnostics["direction_arrow_count"], 30)

    def test_mobile_weather_cells_remain_readable_and_aligned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current_diagnostics: dict[str, object] = {}
            next_diagnostics: dict[str, object] = {}
            save_current_day_plot(
                current_table(), root / "current-mobile.png", mobile=True,
                render_diagnostics=current_diagnostics, **current_kwargs(),
            )
            save_prediction_plot(
                next_table(), root / "next-mobile.png", mobile=True,
                render_diagnostics=next_diagnostics, **next_kwargs(),
            )
            self.assertGreater((root / "current-mobile.png").stat().st_size, 20_000)
            self.assertGreater((root / "next-mobile.png").stat().st_size, 20_000)
        self.assertEqual(len(current_diagnostics["weather_cell_boundaries"]), 15)
        self.assertEqual(len(next_diagnostics["weather_cell_boundaries"]), 15)
        self.assertEqual(current_diagnostics["weather_icon_count"], 14)
        self.assertEqual(next_diagnostics["weather_icon_count"], 14)
        self.assertEqual(current_diagnostics["subplot_hspace"], 0.10)
        self.assertEqual(current_diagnostics["weather_cell_count"], 14)
        self.assertEqual(next_diagnostics["weather_cell_count"], 14)
        self.assertEqual(next_diagnostics["axis_count"], 3)
        self.assertEqual(next_diagnostics["weather_background_count"], 14)
        self.assertEqual(next_diagnostics["weather_separator_count"], 15)
        self.assertEqual(current_diagnostics["weather_icon_zoom"], 0.29)
        self.assertEqual(next_diagnostics["weather_icon_zoom"], 0.29)
        self.assertEqual(current_diagnostics["weather_icon_y"], 0.20)
        self.assertEqual(next_diagnostics["weather_icon_y"], 0.32)
        self.assertEqual(current_diagnostics["weather_icon_temperature_overlap_count"], 0)
        self.assertEqual(next_diagnostics["weather_icon_temperature_overlap_count"], 0)
        self.assertNotIn("unknown", next_diagnostics["model_id_text"].lower())
        self.assertEqual(current_diagnostics["direction_arrow_count"], 34)
        self.assertEqual(next_diagnostics["direction_arrow_count"], 30)
        assert_header_bounds_do_not_overlap(self, current_diagnostics)
        assert_header_bounds_do_not_overlap(self, next_diagnostics)


class ArchiveSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.database = Path(self.temporary.name) / "ecmwf.sqlite"
        connection = sqlite3.connect(self.database)
        connection.executescript(
            """
            CREATE TABLE forecast_collection_runs (
                provider TEXT, model TEXT, run_time TEXT, site TEXT, status TEXT,
                completed_time TEXT
            );
            CREATE TABLE forecast_points (
                provider TEXT, model TEXT, run_time TEXT, valid_time TEXT,
                wind_speed_mps REAL, fetched_time TEXT, grid_latitude REAL,
                grid_longitude REAL, model_version TEXT, resolution TEXT,
                source TEXT, site TEXT
            );
            """
        )
        self.connection = connection

    def tearDown(self) -> None:
        self.connection.close()
        self.temporary.cleanup()

    def add_run(self, run: datetime, completed: datetime, speed: float) -> None:
        run_iso = run.isoformat().replace("+00:00", "Z")
        completed_iso = completed.isoformat().replace("+00:00", "Z")
        self.connection.execute(
            "INSERT INTO forecast_collection_runs VALUES (?,?,?,?,?,?)",
            ("ECMWF", "IFS", run_iso, "valkenburgsemeer", "complete", completed_iso),
        )
        for lead in range(0, 61, 3):
            valid = run + timedelta(hours=lead)
            self.connection.execute(
                "INSERT INTO forecast_points VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "ECMWF",
                    "IFS",
                    run_iso,
                    valid.isoformat().replace("+00:00", "Z"),
                    speed + lead / 100.0,
                    completed_iso,
                    52.25,
                    4.5,
                    "50r1",
                    "0.25 degrees",
                    "ecmwf_open_data",
                    "valkenburgsemeer",
                ),
            )
        self.connection.commit()

    def test_new_arrival_advances_run_metadata_and_preserves_single_vintage(self) -> None:
        for day in (12, 13, 14):
            run = datetime(2026, 9, day, 0, tzinfo=timezone.utc)
            self.add_run(run, run + timedelta(hours=8), float(day))
        old_run = datetime(2026, 9, 15, 0, tzinfo=timezone.utc)
        new_run = datetime(2026, 9, 15, 6, tzinfo=timezone.utc)
        self.add_run(old_run, old_run + timedelta(hours=8), 5.0)
        self.add_run(new_run, new_run + timedelta(hours=7), 9.0)

        common = {
            "archive_db": self.database,
            "site": "valkenburgsemeer",
            "start_utc": datetime(2026, 9, 16, 0, tzinfo=timezone.utc),
            "end_utc": datetime(2026, 9, 17, 23, tzinfo=timezone.utc),
            "local_timezone": LOCAL_TZ,
        }
        old_points, old_metadata = load_ecmwf_plot_data(
            cutoff_utc=datetime(2026, 9, 15, 12, tzinfo=timezone.utc),
            **common,
        )
        new_points, new_metadata = load_ecmwf_plot_data(
            cutoff_utc=datetime(2026, 9, 15, 14, tzinfo=timezone.utc),
            **common,
        )

        self.assertEqual(old_metadata["run_time_utc"], "2026-09-15T00:00:00Z")
        self.assertEqual(new_metadata["run_time_utc"], "2026-09-15T06:00:00Z")
        self.assertNotEqual(old_metadata["metadata_line"], new_metadata["metadata_line"])
        self.assertGreater(
            pd.Timestamp(new_metadata["next_expected_fetch_utc"]),
            pd.Timestamp(new_metadata["information_cutoff_utc"]),
        )
        self.assertEqual(
            new_metadata["arrival_estimate_method"],
            "median_recent_complete_run_latency",
        )
        self.assertLessEqual(new_metadata["rows_loaded"], 17)
        self.assertGreater(
            float(new_points["wind_speed_knots"].median()),
            float(old_points["wind_speed_knots"].median()),
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_diag: dict[str, object] = {}
            new_diag: dict[str, object] = {}
            save_current_day_plot(
                current_table(),
                root / "old.png",
                ecmwf_speed_series=old_points,
                ecmwf_metadata_text=str(old_metadata["metadata_line"]),
                render_diagnostics=old_diag,
                **current_kwargs(),
            )
            save_current_day_plot(
                current_table(),
                root / "new.png",
                ecmwf_speed_series=new_points,
                ecmwf_metadata_text=str(new_metadata["metadata_line"]),
                render_diagnostics=new_diag,
                **current_kwargs(),
            )
        for series_name in ("measured_wind", "superlocal", "harmonie"):
            pd.testing.assert_frame_equal(
                old_diag["series"][series_name],
                new_diag["series"][series_name],
            )
        self.assertEqual(old_diag["active_anchor_local"], new_diag["active_anchor_local"])


class OperationalGateTests(unittest.TestCase):
    def test_new_ecmwf_identity_uses_lightweight_stage(self) -> None:
        snapshot = OperationalSnapshot(
            site="valkenburgsemeer",
            model="HARMONIE",
            observation_max_ts=200,
            forecast=ForecastIdentity("a" * 64, 10, 10, None, 24),
            model_fingerprint="b" * 64,
            cached_artifacts=CachedArtifactStatus(True, "c" * 64, "ok"),
            ecmwf_run_identity="new",
        )
        state = {
            "schema_version": 1,
            "fingerprint_version": 1,
            "status": "success",
            "site": "valkenburgsemeer",
            "model": "HARMONIE",
            "forecast_fingerprint": "a" * 64,
            "model_fingerprint": "b" * 64,
            "cached_prediction_fingerprint": "c" * 64,
            "observation_max_ts": 200,
            "ecmwf_run_identity": "old",
        }
        decision: ExecutionDecision = decide_execution_mode(snapshot, state)
        self.assertEqual(decision.mode, "ecmwf_changed")
        self.assertFalse(decision.run_full_pipeline)
        self.assertTrue(decision.run_measured_only)


if __name__ == "__main__":
    unittest.main()
