from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from next_day_wind_model.ecmwf_dashboard import MPS_TO_KNOT, ECMWF_FORECAST_COLOR
from next_day_wind_model.ecmwf_evaluation import (
    SECTORS, load_ecmwf_evaluation, attach_ecmwf,
    add_sector_metrics, coverage_text, report_is_current,
)
from next_day_wind_model.direction_performance_plot import save_direction_plot


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "archive.sqlite"
        with sqlite3.connect(self.path) as conn:
            conn.executescript("""
                CREATE TABLE forecast_collection_runs
                    (provider, model, site, status, run_time, completed_time);
                CREATE TABLE forecast_points
                    (provider, model, site, run_time, valid_time, fetched_time, wind_speed_mps);
            """)
        self.add_run("2026-09-01T00:00Z", "2026-09-01T02:00Z")

    def add_run(self, run, completed, *, site="test", status="complete",
                speeds=(3., 6., 9.), fetched="2026-09-01T01:00Z"):
        with sqlite3.connect(self.path) as conn:
            conn.execute("INSERT INTO forecast_collection_runs VALUES ('ECMWF','IFS',?,?,?,?)",
                         (site, status, run, completed))
            for hour, speed in zip([6, 9, 12], speeds):
                conn.execute("INSERT INTO forecast_points VALUES ('ECMWF','IFS',?,?,?,?,?)",
                    (site, run, f"2026-09-03T{hour:02d}:00Z", fetched, speed))

    def load(self, targets, issue="2026-09-01T07:00Z"):
        return load_ecmwf_evaluation(self.path, site="test",
            issue_times=[issue] * len(targets), target_times=targets)

    def test_scalar_interpolation_exact_points_units_and_no_extrapolation(self):
        values = self.load([f"2026-09-03T{h:02d}:00Z" for h in [5, 6, 7, 8, 9, 13]])
        np.testing.assert_allclose(values.ecmwf_speed_knots.iloc[1:5], np.array([3, 4, 5, 6]) * MPS_TO_KNOT)
        self.assertTrue(values.ecmwf_speed_knots.iloc[[0, 5]].isna().all())
        self.assertEqual(values.ecmwf_run_time.iloc[2], "2026-09-01T00:00Z")
        self.assertTrue(pd.isna(values.ecmwf_missing_reason.iloc[2]))
        # This fixture has no gust or vector columns: speed evaluation needs neither.

    def test_run_and_completion_eligibility_site_isolation_and_input_order(self):
        self.add_run("2026-09-01T06:00Z", "2026-09-01T08:00Z", speeds=(20.,) * 3)
        self.add_run("2026-09-01T12:00Z", "2026-09-01T01:00Z", speeds=(30.,) * 3)
        self.add_run("2026-09-01T06:00Z", "2026-09-01T06:30Z", site="elsewhere", speeds=(40.,) * 3)
        self.add_run("2026-09-01T05:00Z", "2026-09-01T05:30Z", status="partial", speeds=(50.,) * 3)
        values = load_ecmwf_evaluation(self.path, site="test",
            issue_times=["2026-09-01T09:00Z", "2026-09-01T07:00Z", "2026-09-01T09:00Z"],
            target_times=["2026-09-03T06:00Z"] * 3)
        np.testing.assert_allclose(values.ecmwf_speed_knots, np.array([20, 3, 20]) * MPS_TO_KNOT)
        self.assertTrue(self.load(["2026-09-03T06:00Z"], issue="2026-09-01T01:30Z").ecmwf_speed_knots.isna().all())

    def test_late_fetched_point_cannot_be_used_or_bridged(self):
        with sqlite3.connect(self.path) as conn:
            conn.execute("UPDATE forecast_points SET fetched_time='2026-09-01T08:00Z' WHERE valid_time='2026-09-03T09:00Z'")
        frame = self.load(["2026-09-03T07:00Z", "2026-09-03T09:00Z", "2026-09-03T12:00Z"])
        self.assertTrue(frame.ecmwf_speed_knots.iloc[:2].isna().all())
        self.assertEqual(frame.ecmwf_missing_reason.iloc[0], "point fetched after issue")
        self.assertAlmostEqual(frame.ecmwf_speed_knots.iloc[2], 9 * MPS_TO_KNOT)

    def test_missing_native_point_and_nonfinite_speed_preserve_gaps(self):
        with sqlite3.connect(self.path) as conn:
            conn.execute("DELETE FROM forecast_points WHERE valid_time='2026-09-03T09:00Z'")
        frame = self.load(["2026-09-03T08:00Z", "2026-09-03T12:00Z"])
        self.assertTrue(pd.isna(frame.ecmwf_speed_knots.iloc[0]))
        self.assertEqual(frame.ecmwf_missing_reason.iloc[0], "native gap exceeds three hours")
        with sqlite3.connect(self.path) as conn:
            conn.execute("UPDATE forecast_points SET wind_speed_mps=NULL WHERE valid_time='2026-09-03T12:00Z'")
        self.assertTrue(self.load(["2026-09-03T12:00Z"]).ecmwf_speed_knots.isna().all())

    def test_missing_selected_run_point_does_not_fall_back_to_older_run(self):
        self.add_run("2026-09-01T06:00Z", "2026-09-01T06:30Z", speeds=(20.,) * 3)
        with sqlite3.connect(self.path) as conn:
            conn.execute("DELETE FROM forecast_points WHERE run_time='2026-09-01T06:00Z' AND valid_time='2026-09-03T09:00Z'")
        self.assertTrue(self.load(["2026-09-03T08:00Z"]).ecmwf_speed_knots.isna().all())

    def test_missing_or_corrupt_archive_replaces_stale_results(self):
        cases = pd.DataFrame({"issue_time_utc": ["2026-09-01T07:00Z"],
            "target_time_utc": ["2026-09-03T07:00Z"], "ecmwf_speed_knots": [99.]})
        for path in [None, self.path.parent / "absent.sqlite"]:
            self.assertTrue(attach_ecmwf(cases, path, site="test").ecmwf_speed_knots.isna().all())
        corrupt = self.path.parent / "corrupt.sqlite"
        corrupt.write_text("invalid SQLite database")
        result = attach_ecmwf(cases, corrupt, site="test")
        self.assertTrue(result.ecmwf_speed_knots.isna().all())
        self.assertEqual(result.ecmwf_missing_reason.iloc[0], "archive unreadable")
        self.assertFalse((self.path.parent / "absent.sqlite").exists())

    def test_read_only_and_bounded_point_queries(self):
        original = sqlite3.connect
        queries = []
        def connect(*args, **kwargs):
            self.assertIn("mode=ro", args[0])
            self.assertTrue(kwargs["uri"])
            conn = original(*args, **kwargs)
            conn.set_trace_callback(queries.append)
            return conn
        with patch("next_day_wind_model.ecmwf_evaluation.sqlite3.connect", side_effect=connect):
            self.load(["2026-09-03T07:00Z"])
        self.assertTrue(any("query_only=ON" in q for q in queries))
        points = next(q for q in queries if "FROM forecast_points" in q)
        self.assertIn("julianday(valid_time) >=", points)
        self.assertIn("julianday(valid_time) <=", points)

    def test_next_day_export_keeps_issues_and_averages_before_absolute_error(self):
        from next_day_wind_model.update_model_and_predict import save_next_day_direction_evaluation
        target = "2026-09-03T08:00Z"
        self.add_run("2026-09-01T06:00Z", "2026-09-01T08:00Z", speeds=(10.,) * 3)
        raw = pd.DataFrame({"issue_time_utc": ["2026-09-01T07:00Z", "2026-09-01T09:00Z"],
            "target_time_utc": pd.to_datetime([target] * 2),
            "actual_wind_speed": [7.5 * MPS_TO_KNOT] * 2,
            "forecast_wind_speed": [10., 14.], "forecast_wind_dir_deg": [359., 1.],
            "challenger_wind_speed": [9., 13.], "champion_wind_speed": [8., 12.]})
        output = Path(self.temp.name) / "next_day"
        paths = save_next_day_direction_evaluation(raw, output, "first", site="test", ecmwf_archive_db=self.path)
        summary = pd.read_csv(paths["speed_eval_direction_stable_csv"]).set_index("sector")
        self.assertEqual(summary.index.tolist(), ["N"])
        self.assertAlmostEqual(summary.loc["N", "ecmwf_mae"], 0.)
        self.assertEqual(summary.loc["N", "ecmwf_n_points"], 1)
        issues = pd.read_csv(paths["speed_eval_issues_csv"])
        self.assertEqual(len(issues), 2)
        self.assertEqual(issues.ecmwf_run_time.tolist(), ["2026-09-01T00:00Z", "2026-09-01T06:00Z"])
        saved = pd.read_csv(paths["speed_eval_details_csv"])
        self.assertEqual(saved.n_overlaps.iloc[0], 2)
        self.assertAlmostEqual(saved.forecast_wind_speed.iloc[0], 12.)
        self.assertAlmostEqual(saved.champion_wind_speed.iloc[0], 10.)
        legacy = summary.drop(columns=[c for c in summary if c.startswith("ecmwf_")])
        paths = save_next_day_direction_evaluation(raw, output, "absent", site="test")
        absent = pd.read_csv(paths["speed_eval_direction_stable_csv"]).set_index("sector")
        pd.testing.assert_frame_equal(absent[legacy.columns], legacy)
        self.assertEqual(absent.ecmwf_n_points.sum(), 0)

    def test_current_day_export_retains_first_issue_cases_and_prior_scores(self):
        from next_day_wind_model.update_model_and_predict import save_current_day_direction_performance_csv
        from next_day_wind_model.data_pipeline import DatasetConfig
        database = self.path.parent / "observations.sqlite"
        def ms(value):
            return int(pd.Timestamp(value).timestamp() * 1000)
        target = ms("2026-09-03T08:00Z")
        with sqlite3.connect(database) as conn:
            conn.executescript("""
                CREATE TABLE prediction_log(site,model_type,prediction_kind,anchor_ts,issued_ts,
                    target_ts,horizon_hr,prediction_value,harmonie_value,actual_value,harmonie_run_ts);
                CREATE TABLE forecasts(site,model,run_ts,target_ts,wind_dir);
            """)
            conn.execute("INSERT INTO forecasts VALUES ('test','HARMONIE',1,?,90)", (target,))
            for anchor, issued, prediction in [("2026-09-01T07:00Z", "2026-09-01T07:01Z", 9.),
                    ("2026-09-01T07:00Z", "2026-09-01T07:02Z", 99.),
                    ("2026-09-01T08:00Z", "2026-09-01T08:01Z", 11.)]:
                conn.execute("INSERT INTO prediction_log VALUES ('test','intraday','wind_speed',?,?,?,?,?,?,?,1)",
                    (ms(anchor), ms(issued), target, 1., prediction, 12., 10.))
        report = self.path.parent / "current_day.csv"
        cfg = DatasetConfig(site="test")
        save_current_day_direction_performance_csv(database, cfg, report, self.path)
        enriched = pd.read_csv(report)
        east = enriched.set_index("sector").loc["E"]
        self.assertEqual(east.n_points, 2)
        self.assertEqual(east.superlocal_mae, 1.)
        self.assertEqual(east.forecast_mae, 2.)
        self.assertEqual(east.ecmwf_n_points, 2)
        self.assertEqual(east.ecmwf_n_target_hours, 1)
        issues = pd.read_csv(report.with_name("current_day_direction_eval_issues.csv"))
        self.assertEqual(len(issues), 2)
        save_current_day_direction_performance_csv(database, cfg, report)
        absent = pd.read_csv(report)
        columns = [c for c in enriched if not c.startswith("ecmwf_")]
        pd.testing.assert_frame_equal(enriched[columns], absent[columns])
        self.assertEqual(absent.ecmwf_n_points.sum(), 0)

    def test_d2_export_reports_scalar_speed_without_changing_legacy_features(self):
        from next_day_wind_model.day_after_tomorrow_core import export_direction_performance
        rows = pd.DataFrame({"issue_time_utc": ["2026-09-01T07:00Z"] * 3,
            "target_time_utc": [f"2026-09-03T{h:02d}:00Z" for h in [6, 7, 8]],
            "scorable": [True] * 3, "actual": [10.] * 3, "harmonie": [12.] * 3,
            "prediction": [11.] * 3, "harmonie_direction": [0., 90., 180.],
            "ecmwf": [99.] * 3})
        output = Path(self.temp.name) / "d2"
        output.mkdir()
        legacy = export_direction_performance(rows, output)
        enriched = attach_ecmwf(rows, self.path, site="test")
        pd.testing.assert_series_equal(enriched.ecmwf, rows.ecmwf)
        report = export_direction_performance(enriched, output)
        columns = [c for c in legacy if not c.startswith("ecmwf_")]
        pd.testing.assert_frame_equal(legacy[columns], report[columns])
        self.assertEqual(report.ecmwf_n_points.sum(), 3)
        east = report.set_index("sector").loc["E"]
        self.assertAlmostEqual(east.ecmwf_mae, abs(4 * MPS_TO_KNOT - 10.))


class SummaryAndPlotTests(unittest.TestCase):
    def test_overlap_average_before_error_and_unchanged_existing_statistics(self):
        summary = pd.DataFrame({"sector": SECTORS, "forecast_mae": np.arange(8.),
                                "champion_mae": np.arange(8.) / 2, "n_points": [7] * 8})
        cases = pd.DataFrame({"target_time_utc": ["2026-09-15T08:00Z"] * 3 + ["2026-09-15T09:00Z"],
            "actual": [10.] * 4, "ecmwf_speed_knots": [12., 8., np.nan, 14.],
            "sector": ["N"] * 3 + ["E"]})
        sectors = pd.Series(["N", "E"], index=["2026-09-15T08:00Z", "2026-09-15T09:00Z"])
        result = add_sector_metrics(summary, cases, sectors)
        pd.testing.assert_frame_equal(result[summary.columns], summary)
        self.assertEqual(result.ecmwf_n_points.sum(), 2)
        self.assertEqual(result.set_index("sector").loc["N", "ecmwf_mae"], 0.)
        self.assertEqual(result.set_index("sector").loc["E", "ecmwf_mae"], 4.)
        current = add_sector_metrics(summary, cases)
        self.assertEqual(current.set_index("sector").loc["N", "ecmwf_mae"], 2.)
        self.assertEqual(current.ecmwf_n_points.sum(), 3)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "report.csv"
            result.to_csv(path, index=False)
            self.assertTrue(report_is_current(path))
            self.assertIn("2 target hours", coverage_text(path))
            self.assertIn("15 Sep 2026", coverage_text(path))
            summary.to_csv(path, index=False)
            self.assertFalse(report_is_current(path))
            self.assertEqual(coverage_text(path), "")

    def test_compass_gaps_weighted_ecmwf_legend_overflow_and_legacy_fallback(self):
        frame = pd.DataFrame({"sector": SECTORS, "forecast_mae": [2.] * 8,
            "champion_mae": [1.] * 8, "n_points": [10] * 8,
            "ecmwf_mae": [1., np.nan, 5., np.nan, np.nan, np.nan, np.nan, np.nan],
            "ecmwf_n_points": [3, 0, 1, 0, 0, 0, 0, 0]})
        with tempfile.TemporaryDirectory() as temp:
            path, png = Path(temp) / "report.csv", Path(temp) / "plot.png"
            frame.to_csv(path, index=False)
            with patch("next_day_wind_model.direction_performance_plot.plt.close") as close:
                save_direction_plot(path, png, model_column="champion_mae", model_label="Super local", title="Test")
            fig = close.call_args.args[0]
            ax = fig.axes[0]
            self.assertEqual([t.get_text() for t in ax.get_xticklabels()], SECTORS)
            curve = next(line for line in ax.lines if line.get_color() == ECMWF_FORECAST_COLOR)
            self.assertEqual(curve.get_label(), "ECMWF (2.00 kts)")
            self.assertTrue(np.isnan(curve.get_ydata()[1]))
            self.assertAlmostEqual(curve.get_xdata()[2], np.pi / 2)
            self.assertTrue(any(t.get_text() == "5.00" for t in ax.texts))
            plt.close(fig)
            frame["ecmwf_mae"] = np.nan
            frame.to_csv(path, index=False)
            with patch("next_day_wind_model.direction_performance_plot.plt.close") as close:
                save_direction_plot(path, png, model_column="champion_mae", model_label="Super local", title="Test")
            fig = close.call_args.args[0]
            self.assertEqual(len(fig.axes[0].lines), 2)
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
