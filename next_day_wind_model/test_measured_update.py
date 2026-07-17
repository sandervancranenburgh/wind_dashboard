from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

from next_day_wind_model.measured_update import (
    FORECAST_COLUMNS,
    compose_current_day_table,
    load_current_day_observations,
    run_measured_only_stage,
)
from next_day_wind_model.operational_update import local_day_utc_bounds


NOW_UTC = datetime(2026, 7, 17, 8, 5, tzinfo=timezone.utc)


def _create_observation_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE observations(
            site TEXT NOT NULL, ts INTEGER NOT NULL, wind_speed REAL,
            wind_gust REAL, wind_dir REAL, payload TEXT,
            PRIMARY KEY(site, ts)
        )
        """
    )
    rows = [
        ("site", 1784235600000, 1.0, 2.0, 90.0, {}),  # prior local day
        (
            "site",
            1784275200000,
            5.0,
            8.0,
            180.0,
            {"AverageWind": 5.0, "MinWind": 3.0, "MaxWind": 8.0, "WindDirection": 180.0},
        ),
        (
            "site",
            1784275500000,
            6.0,
            9.0,
            190.0,
            {"AverageWind": 6.0, "MinWind": 4.0, "MaxWind": 9.0, "WindDirection": 190.0},
        ),
        ("site", 1784329200000, 9.0, 10.0, 200.0, {}),  # next local day
    ]
    conn.executemany(
        "INSERT INTO observations VALUES(?,?,?,?,?,?)",
        [(site, ts, avg, gust, direction, json.dumps(payload)) for site, ts, avg, gust, direction, payload in rows],
    )
    conn.commit()
    conn.close()


def _cached_table() -> pd.DataFrame:
    times = pd.date_range("2026-07-17T00:00:00+02:00", periods=24, freq="1h")
    data = {
        "time_local": times,
        "is_forecast_grid": True,
        "is_actual_observation": False,
        "is_future": times >= pd.Timestamp("2026-07-17T11:00:00+02:00"),
    }
    for index, column in enumerate(FORECAST_COLUMNS):
        data[column] = np.arange(24, dtype=float) + index
    data["actual_wind_speed"] = np.nan
    data["actual_wind_min"] = np.nan
    data["actual_wind_max"] = np.nan
    data["actual_wind_dir_deg"] = np.nan
    data["hour_local"] = times.strftime("%H")
    data["minute_local"] = times.minute
    return pd.DataFrame(data)


def _fixture_build_plot_frame(
    dense_times: pd.DatetimeIndex,
    forecast_columns: dict[str, np.ndarray],
    actual_day_raw: pd.DataFrame,
    *,
    now_local,
    future_start,
) -> pd.DataFrame:
    plot_times = dense_times.union(actual_day_raw.index).sort_values()
    result = pd.DataFrame(
        {
            "time_local": plot_times,
            "is_forecast_grid": plot_times.isin(dense_times),
            "is_actual_observation": plot_times.isin(actual_day_raw.index),
            "is_future": plot_times >= pd.Timestamp(future_start),
        }
    )
    for column, values in forecast_columns.items():
        result[column] = pd.Series(values, index=dense_times).reindex(plot_times).to_numpy()
    mapping = {
        "actual_wind_speed": "actual_avg",
        "actual_wind_min": "actual_min",
        "actual_wind_max": "actual_max",
        "actual_wind_dir_deg": "actual_dir",
    }
    for output, source in mapping.items():
        result[output] = actual_day_raw[source].reindex(plot_times).to_numpy()
    result["hour_local"] = result["time_local"].dt.strftime("%H")
    result["minute_local"] = result["time_local"].dt.minute
    return result


class BoundedObservationTests(unittest.TestCase):
    def test_loads_only_current_local_day(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observations.db"
            _create_observation_db(path)
            frame = load_current_day_observations(
                path,
                site="site",
                local_timezone="Europe/Amsterdam",
                now_utc=NOW_UTC,
            )
            self.assertEqual(len(frame), 2)
            self.assertEqual(frame.index.min().date().isoformat(), "2026-07-17")
            self.assertEqual(frame.index.max().date().isoformat(), "2026-07-17")
            self.assertEqual(frame.iloc[-1]["actual_min"], 4.0)

    def test_dst_bounds_are_23_and_25_hours(self) -> None:
        spring_start, spring_end = local_day_utc_bounds(
            "Europe/Amsterdam", datetime(2026, 3, 29, 12, tzinfo=timezone.utc)
        )
        fall_start, fall_end = local_day_utc_bounds(
            "Europe/Amsterdam", datetime(2026, 10, 25, 12, tzinfo=timezone.utc)
        )
        self.assertEqual((spring_end - spring_start) // 3_600_000, 23)
        self.assertEqual((fall_end - fall_start) // 3_600_000, 25)


class CompositionTests(unittest.TestCase):
    def test_reuses_cached_orange_prediction_and_includes_new_observations(self) -> None:
        cached = _cached_table()
        observations = pd.DataFrame(
            {"actual_avg": [6.0], "actual_min": [4.0], "actual_max": [9.0], "actual_dir": [190.0]},
            index=pd.DatetimeIndex([pd.Timestamp("2026-07-17T10:05:00+02:00")]),
        )
        captured: dict[str, np.ndarray] = {}

        def capture(*args, **kwargs):
            captured.update({key: value.copy() for key, value in args[1].items()})
            return _fixture_build_plot_frame(*args, **kwargs)

        result = compose_current_day_table(
            cached,
            observations,
            local_timezone="Europe/Amsterdam",
            now_utc=NOW_UTC,
            build_plot_frame=capture,
        )
        original = cached["lstm_pred_wind_speed"].to_numpy(dtype=float)
        reused = captured["lstm_pred_wind_speed"]
        future_mask = pd.DatetimeIndex(cached["time_local"]) >= pd.Timestamp("2026-07-17T10:00:00+02:00")
        np.testing.assert_allclose(reused[future_mask], original[future_mask], equal_nan=True)
        self.assertIn(pd.Timestamp("2026-07-17T10:05:00+02:00"), set(result["time_local"]))
        self.assertEqual(result["actual_wind_speed"].dropna().iloc[-1], 6.0)


class MeasuredStageTests(unittest.TestCase):
    def _prepare(self, root: Path) -> tuple[Path, Path, Path, SimpleNamespace]:
        db_path = root / "fixture.db"
        _create_observation_db(db_path)
        out_dir = root / "artifacts"
        web_dir = root / "web"
        out_dir.mkdir()
        web_dir.mkdir()
        table = _cached_table().copy()
        table["time_local"] = pd.to_datetime(table["time_local"]).dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        table.to_csv(out_dir / "current_day_predictions.csv", index=False)
        (out_dir / "metadata_update.json").write_text(
            json.dumps(
                {
                    "prediction_generated_at_utc": "2026-07-17T07:00:00+00:00",
                    "prediction_updated_at_utc": "2026-07-17T07:00:00+00:00",
                    "model_last_trained_at_utc": "2026-07-17T05:00:00+00:00",
                    "current_day_live_monitoring_metric": {"available": False},
                }
            ),
            encoding="utf-8",
        )
        (web_dir / "index.html").write_text(
            '<p class="meta" data-dashboard-version="old-version">Last updated: old</p>\n'
            '<div data-json-url="current_day_interactive_data.json?v=old-token"></div>\n'
            '<source srcset="current_day_predictions_mobile.png?v=old-token">\n'
            '<img src="current_day_predictions.png?v=old-token">\n'
            '<img src="next_day_predictions.png?v=keep-next-token">\n'
            '<script>const dashboardRefresh = { currentVersion: "old-version" };</script>\n',
            encoding="utf-8",
        )
        args = SimpleNamespace(
            site="site",
            local_timezone="Europe/Amsterdam",
            web_out_dir=str(web_dir),
            git_auto_push_pages=False,
            git_remote="origin",
            git_branch="main",
        )
        return db_path, out_dir, web_dir, args

    @staticmethod
    def _save_plot(table, path, local_tz, **kwargs) -> None:
        marker = b"mobile" if kwargs.get("mobile") else b"desktop"
        observation_count = int(pd.to_numeric(table["actual_wind_speed"], errors="coerce").notna().sum())
        Path(path).write_bytes(marker + str(observation_count).encode("ascii"))

    @staticmethod
    def _write_interactive(*, web_out_dir, **kwargs):
        destination = Path(web_out_dir) / "current_day_interactive_data.json"
        source_csv = Path(kwargs["current_day_csv"]).read_bytes()
        destination.write_bytes(b"interactive:" + source_csv)
        return {"current_day_json": destination.name}

    def test_stage_skips_expensive_work_and_avoids_unchanged_rewrites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path, out_dir, web_dir, args = self._prepare(Path(directory))
            git_push = mock.Mock(side_effect=AssertionError("git must not run when disabled"))
            callbacks = dict(
                args=args,
                db_path=db_path,
                out_dir=out_dir,
                build_plot_frame=_fixture_build_plot_frame,
                save_current_day_plot=self._save_plot,
                load_prediction_history=lambda **kwargs: [],
                write_interactive_assets=self._write_interactive,
                load_harmonie_metadata=lambda *args: (None, "fetched"),
                auto_push=git_push,
                now_utc=NOW_UTC,
            )
            first = run_measured_only_stage(**callbacks)
            self.assertEqual(first["observation_rows"], 2)
            self.assertTrue(any(first["web_changes"].values()))
            self.assertTrue((web_dir / "current_day_predictions.csv").is_file())
            output = pd.read_csv(out_dir / "current_day_predictions.csv")
            self.assertEqual(pd.to_numeric(output["actual_wind_speed"], errors="coerce").notna().sum(), 2)
            index_html = (web_dir / "index.html").read_text(encoding="utf-8")
            self.assertNotIn("old-token", index_html)
            self.assertNotIn('currentVersion: "old-version"', index_html)
            self.assertIn("next_day_predictions.png?v=keep-next-token", index_html)
            self.assertTrue(first["web_changes"]["index.html"])

            watched = [
                out_dir / "current_day_predictions.csv",
                out_dir / "current_day_predictions.png",
                web_dir / "current_day_predictions.csv",
                web_dir / "metadata_update.json",
                web_dir / "index.html",
            ]
            mtimes = {path: path.stat().st_mtime_ns for path in watched}
            second = run_measured_only_stage(**callbacks)
            self.assertFalse(any(second["artifact_changes"].values()))
            self.assertFalse(any(second["web_changes"].values()))
            self.assertEqual(mtimes, {path: path.stat().st_mtime_ns for path in watched})

            metadata = json.loads((web_dir / "metadata_update.json").read_text(encoding="utf-8"))
            generated_at = metadata["generated_at_utc"]
            index_path = web_dir / "index.html"
            stale_index = index_path.read_text(encoding="utf-8").replace(
                f'data-dashboard-version="{generated_at}"',
                'data-dashboard-version="stale-version"',
                1,
            )
            index_path.write_text(stale_index, encoding="utf-8")
            recovered = run_measured_only_stage(**callbacks)
            self.assertFalse(any(recovered["artifact_changes"].values()))
            self.assertTrue(recovered["web_changes"]["index.html"])
            self.assertFalse(recovered["web_changes"]["metadata_update.json"])
            recovered_metadata = json.loads(
                (web_dir / "metadata_update.json").read_text(encoding="utf-8")
            )
            self.assertEqual(recovered_metadata["generated_at_utc"], generated_at)
            git_push.assert_not_called()

    def test_meaningful_new_measurement_changes_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db_path, out_dir, _web_dir, args = self._prepare(root)
            callbacks = dict(
                args=args,
                db_path=db_path,
                out_dir=out_dir,
                build_plot_frame=_fixture_build_plot_frame,
                save_current_day_plot=self._save_plot,
                load_prediction_history=lambda **kwargs: [],
                write_interactive_assets=self._write_interactive,
                load_harmonie_metadata=lambda *args: (None, "fetched"),
                auto_push=mock.Mock(),
                now_utc=NOW_UTC,
            )
            run_measured_only_stage(**callbacks)
            conn = sqlite3.connect(db_path)
            conn.execute(
                "INSERT INTO observations VALUES(?,?,?,?,?,?)",
                (
                    "site",
                    1784275560000,
                    7.0,
                    10.0,
                    200.0,
                    json.dumps({"AverageWind": 7.0, "MinWind": 5.0, "MaxWind": 10.0}),
                ),
            )
            conn.commit()
            conn.close()
            result = run_measured_only_stage(**callbacks)
            self.assertTrue(result["artifact_changes"]["current_day_predictions.csv"])
            self.assertEqual(result["observation_rows"], 3)


if __name__ == "__main__":
    unittest.main()
