from __future__ import annotations

import inspect
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from next_day_wind_model import data_pipeline

sys.modules.setdefault("data_pipeline", data_pipeline)
from next_day_wind_model import intraday_model


SITE = "valkenburgsemeer"
MODEL = "HARMONIE"


class ForecastQueryScopeTests(unittest.TestCase):
    def setUp(self) -> None:
        data_pipeline._load_inference_vintage_lookup_bundle.cache_clear()

    def tearDown(self) -> None:
        data_pipeline._load_inference_vintage_lookup_bundle.cache_clear()

    @staticmethod
    def _create_forecast_db(path: Path) -> dict[str, int]:
        target_ts = 1_800_000_000_000
        history_ts = target_ts - 3_600_000
        unrelated_ts = target_ts + 3_600_000
        anchor_ts = history_ts
        old_run_ts = anchor_ts - 7_200_000
        new_run_ts = anchor_ts - 3_600_000

        conn = sqlite3.connect(path)
        try:
            conn.execute(
                """
                CREATE TABLE forecasts (
                    site TEXT NOT NULL,
                    model TEXT NOT NULL,
                    run_ts INTEGER NOT NULL,
                    fetched_ts INTEGER NOT NULL,
                    target_ts INTEGER NOT NULL,
                    horizon_hr INTEGER,
                    wind_speed REAL,
                    wind_gust REAL,
                    wind_dir REAL,
                    payload TEXT,
                    PRIMARY KEY (site, model, run_ts, target_ts)
                )
                """
            )
            conn.execute("CREATE INDEX idx_fc_site_model_run ON forecasts(site, model, run_ts)")
            conn.execute("CREATE INDEX idx_fc_site_model_target ON forecasts(site, model, target_ts)")

            payload = json.dumps(
                {
                    "WindForecastAvr": 10.0,
                    "WindForecastMin": 8.0,
                    "WindForecastMax": 12.0,
                    "WindDirection": 180.0,
                }
            )
            rows = []
            for run_ts, fetched_ts, speed in (
                (old_run_ts, anchor_ts - 2_000, 9.0),
                (new_run_ts, anchor_ts - 1_000, 10.0),
            ):
                for row_target_ts in (history_ts, target_ts):
                    rows.append(
                        (
                            SITE,
                            MODEL,
                            run_ts,
                            fetched_ts,
                            row_target_ts,
                            int((row_target_ts - run_ts) / 3_600_000),
                            speed,
                            speed + 2.0,
                            180.0,
                            payload,
                        )
                    )
            # The newest run has one unrelated row that arrived after the
            # anchor. The legacy full-run availability rule must still reject
            # that run for the future target, even though this row is outside
            # the bounded inference target range.
            rows.append(
                (
                    SITE,
                    MODEL,
                    new_run_ts,
                    anchor_ts + 1_000,
                    unrelated_ts,
                    int((unrelated_ts - new_run_ts) / 3_600_000),
                    11.0,
                    13.0,
                    180.0,
                    payload,
                )
            )
            conn.executemany(
                """
                INSERT INTO forecasts(
                    site, model, run_ts, fetched_ts, target_ts, horizon_hr,
                    wind_speed, wind_gust, wind_dir, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
        finally:
            conn.close()

        return {
            "history_ts": history_ts,
            "target_ts": target_ts,
            "anchor_ts": anchor_ts,
            "old_run_ts": old_run_ts,
        }

    def test_target_bounds_are_in_sql_and_reduce_loaded_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "forecast.db"
            times = self._create_forecast_db(db_path)
            conn = sqlite3.connect(db_path)
            statements: list[str] = []
            conn.set_trace_callback(statements.append)
            try:
                bounded = data_pipeline.load_forecast_vintages(
                    conn,
                    SITE,
                    MODEL,
                    target_start_ts_ms=times["history_ts"],
                    target_end_ts_ms=times["target_ts"],
                )
            finally:
                conn.close()

        self.assertEqual(len(bounded), 4)
        self.assertEqual(set(bounded["target_ts"]), {times["history_ts"], times["target_ts"]})
        select_sql = next(statement for statement in statements if "SELECT run_ts" in statement)
        self.assertIn("target_ts >=", select_sql)
        self.assertIn("target_ts <=", select_sql)

    def test_bounded_context_matches_legacy_and_preserves_full_run_availability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "forecast.db"
            times = self._create_forecast_db(db_path)
            history = pd.to_datetime([times["history_ts"]], unit="ms", utc=True)
            targets = pd.to_datetime([times["target_ts"]], unit="ms", utc=True)
            anchor = pd.to_datetime(times["anchor_ts"], unit="ms", utc=True)

            legacy_bundle = data_pipeline._load_inference_vintage_lookup_bundle(str(db_path), SITE, MODEL)
            expected_history = data_pipeline._build_history_forecast_frame(
                legacy_bundle["target_lookup"], history, times["anchor_ts"]
            )
            expected_target = data_pipeline._select_latest_complete_run_frame(
                legacy_bundle["run_entries"], targets, times["anchor_ts"]
            )

            actual = data_pipeline.build_inference_forecast_context(
                db_path=db_path,
                cfg=data_pipeline.DatasetConfig(site=SITE, model=MODEL, window_hours=1, target_hours=1),
                anchor_time=anchor,
                history_times=history,
                target_times=targets,
            )
            bounded_bundle = data_pipeline._load_inference_vintage_lookup_bundle(
                str(db_path), SITE, MODEL, times["history_ts"], times["target_ts"]
            )

        self.assertEqual(legacy_bundle["loaded_row_count"], 5)
        self.assertEqual(bounded_bundle["loaded_row_count"], 4)
        self.assertEqual(set(bounded_bundle["target_lookup"]), {times["history_ts"], times["target_ts"]})
        pd.testing.assert_frame_equal(actual["history_frame"], expected_history)
        pd.testing.assert_frame_equal(actual["target_frame"], expected_target)
        self.assertEqual(int(actual["target_frame"]["run_ts"].iloc[0]), times["old_run_ts"])


class TrainingForecastLoaderTests(unittest.TestCase):
    @staticmethod
    def _create_training_db(path: Path, n_hours: int = 180) -> dict[str, int]:
        hour_ms = 3_600_000
        base_ts = 1_800_000_000_000
        conn = sqlite3.connect(path)
        try:
            conn.executescript(
                """
                CREATE TABLE forecasts (
                    site TEXT NOT NULL,
                    model TEXT NOT NULL,
                    run_ts INTEGER NOT NULL,
                    fetched_ts INTEGER NOT NULL,
                    target_ts INTEGER NOT NULL,
                    horizon_hr INTEGER,
                    wind_speed REAL,
                    wind_gust REAL,
                    wind_dir REAL,
                    payload TEXT,
                    PRIMARY KEY (site, model, run_ts, target_ts)
                );
                CREATE INDEX idx_fc_site_model_run
                    ON forecasts(site, model, run_ts);
                CREATE INDEX idx_fc_site_model_target
                    ON forecasts(site, model, target_ts);
                CREATE TABLE observations (
                    site TEXT NOT NULL,
                    ts INTEGER NOT NULL,
                    wind_speed REAL,
                    wind_gust REAL,
                    wind_dir REAL,
                    payload TEXT,
                    PRIMARY KEY (site, ts)
                );
                """
            )
            observation_rows = []
            for hour in range(n_hours):
                target_ts = base_ts + hour * hour_ms
                actual_avg = 7.0 + hour * 0.01
                actual_max = actual_avg + 2.0
                actual_dir = float((170 + hour) % 360)
                observation_rows.append(
                    (
                        SITE,
                        target_ts,
                        actual_avg,
                        actual_max,
                        actual_dir,
                        json.dumps(
                            {
                                "AverageWind": actual_avg,
                                "MaxWind": actual_max,
                                "WindDirection": actual_dir,
                            }
                        ),
                    )
                )
            conn.executemany(
                """
                INSERT INTO observations(site, ts, wind_speed, wind_gust, wind_dir, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                observation_rows,
            )

            forecast_rows = []
            run_offsets = list(range(-6, n_hours, 6))
            for run_number, run_hour in enumerate(run_offsets):
                run_ts = base_ts + run_hour * hour_ms
                fetched_ts = run_ts + 1_000
                for target_hour in range(n_hours):
                    target_ts = base_ts + target_hour * hour_ms
                    forecast_avg = 6.0 + target_hour * 0.01 + run_number * 0.001
                    forecast_min = forecast_avg - 1.0
                    forecast_max = forecast_avg + 2.0
                    forecast_dir = float((150 + target_hour + run_number) % 360)
                    payload = json.dumps(
                        {
                            "WindForecastAvr": forecast_avg,
                            "WindForecastMin": forecast_min,
                            "WindForecastMax": forecast_max,
                            "WindDirection": forecast_dir,
                            "Temperature": 280.0 + target_hour * 0.01,
                            "Pressure": 101000.0 + run_number,
                            "Rain": target_hour * 0.001,
                            "RH": 70.0 + run_number * 0.01,
                            "Clouds": float(target_hour % 100),
                            "low_cloud_cover": 10.0 + run_number,
                            "medium_cloud_cover": 20.0 + run_number,
                            "high_cloud_cover": 30.0 + run_number,
                            "cloud_base": 500.0 + target_hour,
                            "global_radiation": 100.0 + target_hour,
                        }
                    )
                    forecast_rows.append(
                        (
                            SITE,
                            MODEL,
                            run_ts,
                            fetched_ts,
                            target_ts,
                            int((target_ts - run_ts) / hour_ms),
                            forecast_avg,
                            forecast_max,
                            forecast_dir,
                            payload,
                        )
                    )

            delayed_run_ts = base_ts + 60 * hour_ms
            delayed_target_ts = base_ts + n_hours * hour_ms
            forecast_rows.append(
                (
                    SITE,
                    MODEL,
                    delayed_run_ts,
                    base_ts + (n_hours + 24) * hour_ms,
                    delayed_target_ts,
                    int((delayed_target_ts - delayed_run_ts) / hour_ms),
                    12.0,
                    14.0,
                    220.0,
                    json.dumps(
                        {
                            "WindForecastAvr": 12.0,
                            "WindForecastMin": 11.0,
                            "WindForecastMax": 14.0,
                            "WindDirection": 220.0,
                        }
                    ),
                )
            )
            forecast_rows.append(
                (
                    "other-site",
                    "OTHER",
                    base_ts,
                    base_ts,
                    base_ts,
                    0,
                    99.0,
                    100.0,
                    0.0,
                    "{}",
                )
            )
            conn.executemany(
                """
                INSERT INTO forecasts(
                    site, model, run_ts, fetched_ts, target_ts, horizon_hr,
                    wind_speed, wind_gust, wind_dir, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                forecast_rows,
            )
            conn.commit()
        finally:
            conn.close()
        return {
            "base_ts": base_ts,
            "last_ts": base_ts + (n_hours - 1) * hour_ms,
            "expected_rows": len(run_offsets) * n_hours,
            "delayed_run_ts": delayed_run_ts,
            "delayed_available_ts": base_ts + (n_hours + 24) * hour_ms,
        }

    def _load_legacy_and_compact(
        self,
        db_path: Path,
        cfg: data_pipeline.DatasetConfig,
        times: dict[str, int],
    ) -> tuple[dict, data_pipeline.TrainingForecastLookup, pd.DataFrame]:
        data_pipeline._load_inference_vintage_lookup_bundle.cache_clear()
        legacy = data_pipeline._load_inference_vintage_lookup_bundle(
            str(db_path),
            cfg.site,
            cfg.model,
            times["base_ts"],
            times["last_ts"],
        )
        compact = data_pipeline.load_training_forecast_lookup(
            db_path,
            cfg,
            target_start_ts_ms=times["base_ts"],
            target_end_ts_ms=times["last_ts"],
            chunk_size=17,
        )
        observations = data_pipeline.load_training_observations(db_path, cfg)
        return legacy, compact, observations

    def test_training_loader_streams_and_filters_in_sql(self) -> None:
        source = inspect.getsource(data_pipeline.load_training_forecast_lookup)
        self.assertIn("fetchmany(", source)
        self.assertNotIn("fetchall(", source)
        self.assertIn('"site = ?"', source)
        self.assertIn('"model = ?"', source)

        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "training.db"
            times = self._create_training_db(db_path)
            cfg = data_pipeline.DatasetConfig(site=SITE, model=MODEL)
            compact = data_pipeline.load_training_forecast_lookup(
                db_path,
                cfg,
                target_start_ts_ms=times["base_ts"],
                target_end_ts_ms=times["last_ts"],
                chunk_size=13,
            )
            full_history = data_pipeline.load_training_forecast_lookup(
                db_path,
                cfg,
                chunk_size=13,
            )

        self.assertEqual(compact.loaded_row_count, times["expected_rows"])
        self.assertEqual(compact.sql_query_count, 2)
        self.assertEqual(compact.target_start_ts_ms, times["base_ts"])
        self.assertEqual(compact.target_end_ts_ms, times["last_ts"])
        self.assertEqual(full_history.loaded_row_count, times["expected_rows"] + 1)
        self.assertEqual(full_history.sql_query_count, 1)

    def test_compact_rows_and_full_run_availability_match_legacy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "training.db"
            times = self._create_training_db(db_path)
            cfg = data_pipeline.DatasetConfig(site=SITE, model=MODEL)
            legacy, compact, _ = self._load_legacy_and_compact(db_path, cfg, times)
            conn = sqlite3.connect(db_path)
            try:
                legacy_rows = data_pipeline.load_forecast_vintages(
                    conn,
                    SITE,
                    MODEL,
                    target_start_ts_ms=times["base_ts"],
                    target_end_ts_ms=times["last_ts"],
                )
            finally:
                conn.close()

        self.assertEqual(len(legacy_rows), compact.loaded_row_count)
        np.testing.assert_array_equal(legacy_rows["run_ts"].to_numpy(np.int64), compact.run_ts)
        np.testing.assert_array_equal(legacy_rows["fetched_ts"].to_numpy(np.int64), compact.fetched_ts)
        np.testing.assert_array_equal(legacy_rows["target_ts"].to_numpy(np.int64), compact.target_ts)
        for name in data_pipeline._TRAINING_FLOAT_COLUMNS:
            np.testing.assert_allclose(
                pd.to_numeric(legacy_rows[name], errors="coerce").to_numpy(np.float32),
                compact.float_columns[name],
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )
        legacy_available = np.asarray(
            [entry["available_ts"] for entry in legacy["run_entries"]],
            dtype=np.int64,
        )
        np.testing.assert_array_equal(legacy_available, compact.run_available_ts)
        delayed_idx = int(np.flatnonzero(compact.run_ts[compact.run_starts] == times["delayed_run_ts"])[0])
        self.assertEqual(int(compact.run_available_ts[delayed_idx]), times["delayed_available_ts"])

    def test_compact_partial_run_selection_matches_legacy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "forecast.db"
            times = ForecastQueryScopeTests._create_forecast_db(db_path)
            cfg = data_pipeline.DatasetConfig(site=SITE, model=MODEL, window_hours=1, target_hours=1)
            compact = data_pipeline.load_training_forecast_lookup(
                db_path,
                cfg,
                target_start_ts_ms=times["history_ts"],
                target_end_ts_ms=times["target_ts"],
                chunk_size=2,
            )
            context = data_pipeline.build_training_forecast_context(
                compact,
                anchor_time=pd.to_datetime(times["anchor_ts"], unit="ms", utc=True),
                history_times=pd.to_datetime([times["history_ts"]], unit="ms", utc=True),
                target_times=pd.to_datetime([times["target_ts"]], unit="ms", utc=True),
            )

        self.assertEqual(int(context["target_frame"]["run_ts"].iloc[0]), times["old_run_ts"])

    def test_speed_direction_anchors_targets_and_splits_match_legacy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "training.db"
            times = self._create_training_db(db_path)
            cfg = data_pipeline.DatasetConfig(
                site=SITE,
                model=MODEL,
                window_hours=72,
                target_hours=24,
            )
            legacy, compact, observations = self._load_legacy_and_compact(db_path, cfg, times)
            legacy_speed = data_pipeline.build_all_training_arrays(
                db_path,
                cfg,
                target_mode="residual",
                forecast_lookup=legacy,
                observations=observations,
            )
            compact_speed = data_pipeline.build_all_training_arrays(
                db_path,
                cfg,
                target_mode="residual",
                forecast_lookup=compact,
                observations=observations,
            )
            legacy_direction = data_pipeline.build_all_direction_training_arrays(
                db_path,
                cfg,
                forecast_lookup=legacy,
                observations=observations,
            )
            compact_direction = data_pipeline.build_all_direction_training_arrays(
                db_path,
                cfg,
                forecast_lookup=compact,
                observations=observations,
            )

        numeric_keys = [
            "X_all",
            "y_all",
            "y_actual_all_raw",
            "y_forecast_all_raw",
            "target_forecast_dir_all_raw",
            "target_run_ts_all",
            "target_fetched_ts_all",
            "target_horizon_hr_all",
            "anchor_forecast_dir_all_raw",
            "x_mean",
            "x_std",
            "y_mean",
            "y_std",
        ]
        for legacy_arrays, compact_arrays in (
            (legacy_speed, compact_speed),
            (legacy_direction, compact_direction),
        ):
            self.assertEqual(legacy_arrays["feature_cols"], compact_arrays["feature_cols"])
            np.testing.assert_array_equal(legacy_arrays["timestamps"], compact_arrays["timestamps"])
            np.testing.assert_array_equal(legacy_arrays["target_times_all"], compact_arrays["target_times_all"])
            for key in numeric_keys:
                if key in legacy_arrays:
                    np.testing.assert_allclose(
                        legacy_arrays[key],
                        compact_arrays[key],
                        rtol=0.0,
                        atol=0.0,
                        equal_nan=True,
                    )
                    np.testing.assert_array_equal(
                        np.isnan(legacy_arrays[key]),
                        np.isnan(compact_arrays[key]),
                    )

            n_samples = len(legacy_arrays["timestamps"])
            for split_fraction in (0.80, 0.85):
                split_idx = int(n_samples * split_fraction)
                np.testing.assert_array_equal(
                    legacy_arrays["timestamps"][:split_idx],
                    compact_arrays["timestamps"][:split_idx],
                )
                np.testing.assert_array_equal(
                    legacy_arrays["timestamps"][split_idx:],
                    compact_arrays["timestamps"][split_idx:],
                )

    def test_next_day_reuses_supplied_forecasts_and_observations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "training.db"
            times = self._create_training_db(db_path)
            cfg = data_pipeline.DatasetConfig(site=SITE, model=MODEL)
            _, compact, observations = self._load_legacy_and_compact(db_path, cfg, times)
            with (
                mock.patch.object(
                    data_pipeline,
                    "load_training_forecast_lookup",
                    side_effect=AssertionError("unexpected forecast SQL"),
                ),
                mock.patch.object(
                    data_pipeline,
                    "load_training_observations",
                    side_effect=AssertionError("unexpected observation SQL"),
                ),
            ):
                speed = data_pipeline.build_all_training_arrays(
                    db_path,
                    cfg,
                    target_mode="residual",
                    forecast_lookup=compact,
                    observations=observations,
                )
                direction = data_pipeline.build_all_direction_training_arrays(
                    db_path,
                    cfg,
                    forecast_lookup=compact,
                    observations=observations,
                )

        self.assertGreater(len(speed["X_all"]), 0)
        self.assertEqual(len(speed["X_all"]), len(direction["X_all"]))

    def test_intraday_uses_shared_lookup_without_per_anchor_sql(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "training.db"
            times = self._create_training_db(db_path)
            cfg = data_pipeline.DatasetConfig(site=SITE, model=MODEL)
            _, compact, observations = self._load_legacy_and_compact(db_path, cfg, times)
            with (
                mock.patch.object(
                    intraday_model,
                    "load_training_forecast_lookup",
                    side_effect=AssertionError("unexpected forecast SQL"),
                ),
                mock.patch.object(
                    intraday_model,
                    "load_training_observations",
                    side_effect=AssertionError("unexpected observation SQL"),
                ),
            ):
                contexts = intraday_model._build_intraday_anchor_contexts(
                    db_path,
                    cfg,
                    forecast_lookup=compact,
                    observations=observations,
                )

        self.assertGreaterEqual(len(contexts), 50)
        anchors = pd.DatetimeIndex([context["anchor_time"] for context in contexts])
        self.assertTrue(anchors.is_monotonic_increasing)


if __name__ == "__main__":
    unittest.main()
