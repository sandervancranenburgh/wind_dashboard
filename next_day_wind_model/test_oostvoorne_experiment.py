from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from next_day_wind_model.oostvoorne_experiment import day_block_bootstrap, metric_pair, summarize_by_horizon


class OostvoorneExperimentMetricTests(unittest.TestCase):
    def test_circular_metrics_wrap_north(self) -> None:
        result = metric_pair(np.array([359.0, 1.0]), np.array([1.0, 359.0]), circular=True)
        self.assertEqual(result["mae"], 2.0)
        self.assertEqual(result["rmse"], 2.0)

    def test_day_block_bootstrap_is_deterministic_and_day_based(self) -> None:
        frame = pd.DataFrame(
            {
                "target_time_utc": pd.to_datetime(
                    ["2026-01-01T01:00Z", "2026-01-01T02:00Z", "2026-01-02T01:00Z", "2026-01-02T02:00Z"]
                ),
                "prediction_value": [1.0, 2.0, 2.0, 3.0],
                "harmonie_value": [3.0, 4.0, 4.0, 5.0],
                "actual_value": [1.0, 2.0, 2.0, 3.0],
                "horizon_hr": [1.0, 2.0, 1.0, 2.0],
            }
        )
        result = day_block_bootstrap(
            frame,
            prediction_col="prediction_value",
            baseline_col="harmonie_value",
            actual_col="actual_value",
            time_col="target_time_utc",
            iterations=20,
            seed=1,
        )
        self.assertEqual(result["days"], 2)
        self.assertEqual(result["mae_improvement_model_vs_harmonie"]["estimate"], 2.0)
        buckets = summarize_by_horizon(frame, [(1, 1), (2, 2)])
        self.assertEqual([row["model"]["n"] for row in buckets], [2, 2])


if __name__ == "__main__":
    unittest.main()
