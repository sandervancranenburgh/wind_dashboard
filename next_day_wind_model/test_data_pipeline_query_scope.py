from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from next_day_wind_model import data_pipeline


SITE = "valkenburgsemeer"
MODEL = "HARMONIE"


class ForecastQueryScopeTests(unittest.TestCase):
    def setUp(self) -> None:
        data_pipeline._load_vintage_lookup_bundle.cache_clear()

    def tearDown(self) -> None:
        data_pipeline._load_vintage_lookup_bundle.cache_clear()

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

            legacy_bundle = data_pipeline._load_vintage_lookup_bundle(str(db_path), SITE, MODEL)
            expected_history = data_pipeline._build_history_forecast_frame(
                legacy_bundle["target_lookup"], history, times["anchor_ts"]
            )
            expected_target = data_pipeline._select_latest_complete_run_frame(
                legacy_bundle["run_entries"], targets, times["anchor_ts"]
            )

            actual = data_pipeline.build_anchor_forecast_context(
                db_path=db_path,
                cfg=data_pipeline.DatasetConfig(site=SITE, model=MODEL, window_hours=1, target_hours=1),
                anchor_time=anchor,
                history_times=history,
                target_times=targets,
            )
            bounded_bundle = data_pipeline._load_vintage_lookup_bundle(
                str(db_path), SITE, MODEL, times["history_ts"], times["target_ts"]
            )

        self.assertEqual(legacy_bundle["loaded_row_count"], 5)
        self.assertEqual(bounded_bundle["loaded_row_count"], 4)
        self.assertEqual(set(bounded_bundle["target_lookup"]), {times["history_ts"], times["target_ts"]})
        pd.testing.assert_frame_equal(actual["history_frame"], expected_history)
        pd.testing.assert_frame_equal(actual["target_frame"], expected_target)
        self.assertEqual(int(actual["target_frame"]["run_ts"].iloc[0]), times["old_run_ts"])


if __name__ == "__main__":
    unittest.main()
