from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from next_day_wind_model.gustiness_experiment import (
    ExperimentConfig,
    add_hourly_proxy_features,
    build_session_dataset,
    connect_immutable,
    grouped_oof_predictions,
    load_point_in_time_rows,
)


class GustinessExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "gustiness.sqlite"
        conn = sqlite3.connect(self.db_path)
        conn.executescript(
            """
            CREATE TABLE surf_experiences (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                spot TEXT NOT NULL,
                date TEXT NOT NULL,
                start_ts INTEGER NOT NULL,
                end_ts INTEGER NOT NULL,
                perceived_wind_variability TEXT
            );
            CREATE TABLE harmonie_knmi_features (
                site TEXT NOT NULL,
                run_ts TEXT NOT NULL,
                fetched_ts TEXT NOT NULL,
                target_ts TEXT NOT NULL,
                u_10m_mps REAL,
                v_10m_mps REAL,
                wind_speed_10m_mps REAL,
                wind_gust_10m_mps REAL,
                u_50m_mps REAL,
                v_50m_mps REAL,
                wind_speed_50m_mps REAL,
                u_100m_mps REAL,
                v_100m_mps REAL,
                wind_speed_100m_mps REAL,
                dir_shear_10m_100m REAL
            );
            """
        )
        start = int(pd.Timestamp("2026-06-01T12:00:00Z").timestamp() * 1000)
        end = int(pd.Timestamp("2026-06-01T13:00:00Z").timestamp() * 1000)
        conn.execute(
            "INSERT INTO surf_experiences VALUES (?, ?, ?, ?, ?, ?, ?)",
            (1, 10, "Valkenburgse meer", "2026-06-01", start, end, "gusty"),
        )
        rows = []
        for target, mean_speed, gust in (
            ("2026-06-01T12:00:00+00:00", 5.0, 8.8),
            ("2026-06-01T13:00:00+00:00", 7.0, 10.8),
        ):
            rows.extend(
                [
                    (
                        "valkenburgsemeer", "2026-06-01T08:00:00+00:00",
                        "2026-06-01T09:00:00Z", target,
                        mean_speed, 0.0, mean_speed, gust,
                        mean_speed + 1.0, 0.0, mean_speed + 1.0,
                        mean_speed + 2.0, 0.0, mean_speed + 2.0, 5.0,
                    ),
                    (
                        "valkenburgsemeer", "2026-06-01T10:00:00+00:00",
                        "2026-06-01T11:00:00Z", target,
                        mean_speed + 0.5, 0.0, mean_speed + 0.5, gust + 0.5,
                        mean_speed + 1.5, 0.0, mean_speed + 1.5,
                        mean_speed + 2.5, 0.0, mean_speed + 2.5, 7.0,
                    ),
                    (
                        "valkenburgsemeer", "2026-06-01T12:00:00+00:00",
                        "2026-06-01T12:30:00Z", target,
                        20.0, 0.0, 20.0, 30.0,
                        21.0, 0.0, 21.0, 22.0, 0.0, 22.0, 20.0,
                    ),
                ]
            )
        conn.executemany(
            "INSERT INTO harmonie_knmi_features VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        conn.close()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_database_connection_is_immutable(self) -> None:
        conn = connect_immutable(self.db_path)
        with self.assertRaises(sqlite3.OperationalError):
            conn.execute("INSERT INTO surf_experiences VALUES (2, 10, 'x', '2026-01-01', 1, 2, 'steady')")
        conn.close()

    def test_point_in_time_loader_excludes_future_vintage(self) -> None:
        cfg = ExperimentConfig(bootstrap_repeats=10)
        rows = load_point_in_time_rows(self.db_path, cfg)
        self.assertEqual(len(rows), 2)
        self.assertTrue((rows["fetched_ts"] == "2026-06-01T11:00:00Z").all())
        self.assertLess(float(rows["wind_speed_10m_mps"].max()), 20.0)

    def test_proxy_formulas_and_session_weighting(self) -> None:
        cfg = ExperimentConfig(facraf=3.8, bootstrap_repeats=10)
        rows = load_point_in_time_rows(self.db_path, cfg)
        hourly = add_hourly_proxy_features(rows, cfg)
        self.assertTrue(np.allclose(hourly["gust_excess"], 3.8))
        self.assertTrue(np.allclose(hourly["tke_proxy"], 1.0))
        expected_ti = np.sqrt(2.0 / 3.0) / hourly["wind_speed_10m_mps"].to_numpy()
        self.assertTrue(np.allclose(hourly["ti_proxy"], expected_ti))
        self.assertTrue(np.allclose(hourly["overlap_seconds"], 1800.0))

        sessions = build_session_dataset(hourly, cfg)
        self.assertEqual(len(sessions), 1)
        self.assertAlmostEqual(float(sessions.iloc[0]["gust_excess_mean"]), 3.8)
        self.assertEqual(int(sessions.iloc[0]["target"]), 1)
        self.assertTrue(str(sessions.iloc[0]["session_key"]).startswith("S"))

    def test_grouped_predictions_hold_out_complete_dates(self) -> None:
        rows = []
        for idx in range(12):
            target = idx % 2
            rows.append(
                {
                    "session_key": f"S{idx:02d}",
                    "date_group": f"D{idx:02d}",
                    "rider_group": "R001" if idx < 6 else "R002",
                    "target": target,
                    "signal": float(target * 4 + idx / 100),
                }
            )
        sessions = pd.DataFrame(rows)
        predictions = grouped_oof_predictions(
            sessions,
            feature_sets={"test_signal": ["signal"]},
            seed=7,
        )
        self.assertFalse(predictions["prob_test_signal"].isna().any())
        self.assertGreater(
            float(predictions.loc[predictions["target"] == 1, "prob_test_signal"].mean()),
            float(predictions.loc[predictions["target"] == 0, "prob_test_signal"].mean()),
        )


if __name__ == "__main__":
    unittest.main()
